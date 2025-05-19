import os
import torch
import time 
from itertools import count
import torch.nn.functional as F
from collections import deque
import math
import cv2 # type: ignore
import numpy as np
import random

from plotting import plot_codebook_usage, plot_input_vs_recon
from utils import VC, create_vectorized_envs, log, sample_memory, create_eval_env
from model import VQVAE

EPS_START, EPS_END, EPS_DECAY = 1, 0.05, 100000

def preprocess_frames(observations, device, crop_size=(34, 194, 0, 160), target_size=(84, 84), normalize=True):
    def process(obs):
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) # To grayscale
        frame = frame[crop_size[0]:crop_size[1],crop_size[2]:crop_size[3]]
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)  # Resize
        if normalize:
            return frame.astype(np.float32) / 255 # normalize
        else:
            return frame
    return torch.from_numpy(np.stack([process(obs) for obs in observations])).to(device)

def _tensorize(e1, e2, e3, device):
    return (torch.tensor([e1], device=device), torch.tensor([e2], device=device), torch.tensor([e3], device=device))

def to_tensor(items1, items2, items3, device):
    ts = zip(*[_tensorize(e1, e2, e3, device) for e1, e2, e3 in zip(items1, items2, items3)])
    return [torch.stack(t) for t in ts]

def reset_or_fire(envs, i, infos, device, frame_stacks):
    env = envs.envs[i]
    if infos['lives'][i] == 0:
        obs, _, = env.reset()
    else:
        obs, _, _, _, _ = env.step(get_fire_action(env))
    obs = preprocess_frames([obs], device)
    frame_stacks[i] = deque([obs[0]]*4, maxlen=4)
    return torch.stack(list(frame_stacks[i]), dim=0) # (num_envs, 4, 84, 84)

def warmup(game, memory, device, log_dir, num_steps=10000, num_envs=6):
    envs, action_space, _, _ = create_vectorized_envs(game, num_envs=num_envs) # (num_envs, H, W, 3)
    log(log_dir, "\tWarmup...", console_log=True, no_log=True)

    st = time.time()
    obs, _, = envs.reset()
    obs = preprocess_frames(obs, device) # (num_envs, 84, 84)
    frame_stacks = [deque([obs[i]]*4, maxlen=4) for i in range(num_envs)] # (num_envs, 4, 84, 84)
    stacked_obs = torch.stack([torch.stack(list(fs), dim=0) for fs in frame_stacks]) # (num_envs, 4, 84, 84)

    steps = 0
    while steps < num_steps:
        actions = action_space.sample()
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        next_obs = preprocess_frames(next_obs, device) # (num_envs, 84, 84)
        rewards, actions, dones = to_tensor(rewards, actions, [t1 or t2 for t1, t2 in zip(terms, truncs)], device) # (8, 1)

        for i in range(num_envs):
            frame_stacks[i].append(next_obs[i])
            single_ssp = torch.stack(list(frame_stacks[i]), dim=0) # (4, 84, 84)
            single_ss = stacked_obs[i] # (4, 84, 84)
            memory.append(single_ss, actions[i], single_ssp, rewards[i], dones[i])
            stacked_obs[i] = single_ssp # num_envs x (4, 84, 84)

            if dones[i].item():
                stacked_obs[i] = reset_or_fire(envs, i, infos, device, frame_stacks)

        steps += 1*num_envs

    log(log_dir, f"\tWarmup completed in: {time.time() - st:.4f}, Total Steps: {steps}", console_log=True, no_log=True)
    envs.close()

def estimate_codebook_usage_probs(model, x):
    z_e = model.encoder(x)
    _, _, indices = model.quantizer(z_e)
    flat_indices = indices.view(-1)
    counts = torch.bincount(flat_indices, minlength=model.quantizer.num_embeddings).float()
    return counts / counts.sum()

def entropy_bonus(probs, scale=0.05):
    return -scale * (-torch.sum(probs * torch.log(probs + 1e-8)))

def usage_penalty(model, probs, scale=0.05):
    return scale * F.kl_div((probs + 1e-8).log(), torch.full_like(probs, 1.0 / model.quantizer.num_embeddings), reduction='sum')

def train_VQ_VAE(model, memory, optimizer, log_dir, max_iterations=10000, batch_size=32, theta=5e-4, N=500):
    ast = time.time()
    log(log_dir, "\tTraining Model...", console_log=True, no_log=True)
    recon_loss_list, vq_loss_list, entropy_bonus_list, usage_penalty_list = [], [], [], []

    model.train()

    for iteration in count():
        st = time.time()

        x, _, _, _, _ = sample_memory(memory, batch_size)
        x_r, vq_loss = model(x)
        # probs = estimate_codebook_usage_probs(model, x)
        # eb = entropy_bonus(probs)
        # up = usage_penalty(model, probs)

        recon_loss = F.mse_loss(x_r, x, reduction='sum')
        loss = recon_loss + vq_loss # + up + eb

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss_list.append(recon_loss.item())
        vq_loss_list.append(vq_loss.item())
        # entropy_bonus_list.append(eb)
        # usage_penalty_list.append(up)

        # log(log_dir, f"\t\tTraining Round: {iteration}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Usage Penalty: {up:.4f}, Entropy Bonus: {eb:.4f}, Duration: {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
        log(log_dir, f"\t\tTraining Round: {iteration}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Duration: {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
        if iteration > max_iterations - 1 or (len(vq_loss_list) > N and (abs(recon_loss_list[-N] + vq_loss_list[-N] - loss.item()) < theta)): # if max iterations reached or loss does not improve => break
            break

    log(log_dir, f"\tModel training completed in: {time.time() - ast}", console_log=True, no_log=True)
    return recon_loss_list, vq_loss_list, entropy_bonus_list, usage_penalty_list

def initial_model_training(memory, args, epoch, seed, log_dir, device, usage_log=None):
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss = train_VQ_VAE(model, memory, optimizer, log_dir, max_iterations=args.initial_iterations, batch_size=args.batch_size, N=args.initial_iterations)
    torch.save(model, os.path.join(os.path.join(log_dir, f"seed_{seed}"), f'model{epoch}.pth'))
    plot_input_vs_recon(model, memory, args, epoch, log_dir, seed)
    plot_codebook_usage(model, memory, log_dir, epoch, seed, usage_log=usage_log)
    return model, optimizer, loss

def additional_model_training(model, optimizer, memory, args, epoch, seed, log_dir, usage_log=None):
    loss = train_VQ_VAE(model, memory, optimizer, log_dir, max_iterations=args.max_iterations, batch_size=args.batch_size)
    torch.save(model, os.path.join(os.path.join(log_dir, f"seed_{seed}"), f'model{epoch}.pth'))
    plot_input_vs_recon(model, memory, args, epoch, log_dir, seed)
    plot_codebook_usage(model, memory, log_dir, epoch, seed, usage_log=usage_log)
    return loss

def get_fire_action(env, fire_action=[]):
    if fire_action:
        return fire_action[0]
    
    for i in range(env.action_space.n):
        if env.unwrapped.get_action_meanings()[i] == "FIRE":
            fire_action.append(i)
            return fire_action[0]
    
def eval_planner(mdp, args, video, seed, device, epoch, log_dir):
    log(log_dir, "\tEvaluating Model...", console_log=True, no_log=True)
    env, action_space, s, _ = create_eval_env(args.env_name, seed=seed, video=video) # Already wrappped, produces gray scaled (84, 84) frames
    s = torch.from_numpy(s).to(device) # (84, 84)
    frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)

    total_reward = 0
    steps = 0
    st = time.time()
    while True:
        s = torch.stack(list(frame_stack), dim=0) # (4, 84, 84)
        a = mdp.get_action(s, action_space)
        s, r, term, trun, info = env.step(a)
        s = torch.from_numpy(s).to(device) # (84, 84)
        frame_stack.append(s) # (4, 84, 84)
        
        total_reward += r
        steps += 1
        lives = info["lives"]
        log(log_dir, f"\t\tSteps Taken: {steps}, Action: {a}, Total Reward: {total_reward}, Lives: {lives}, Duration: {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
        if term or trun:
            if info['lives'] == 0:
                break
            else:
                s, _, _, _, _ = env.step(get_fire_action(env))
                s = torch.from_numpy(s).to(device) # (84, 84)
                frame_stack = deque([s]*4, maxlen=4) # reset framebuffer

    path = os.path.join(f"seed_{seed}", f"eval_epoch_{epoch}_reward_{total_reward}.mp4")
    video.save(path)
    video.reset()
    env.close()
    log(log_dir, f"\tEvaluation completed in: {time.time() - st:.4f}, Total Reward: {total_reward}", console_log=True, no_log=True)
    return total_reward

def select_action(mdp, action_space, obs, num_envs, disable_eps_greedy=False):
    if not disable_eps_greedy:
        VC.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * VC.steps_done / EPS_DECAY)
    VC.steps_done += 1*num_envs
    
    if disable_eps_greedy or random.random() > VC.eps_threshold:
        return np.stack([mdp.get_action(obs[i], action_space[i]) for i in range(num_envs)]) # explotive choice
    else:
        return np.stack(action_space.sample()) # random eps-greedy action

def collect_transitions(mdp, game, memory, num_transitions, device, log_dir, episode_reward_list=None, num_envs=6, disable_eps_greedy=True):
    log(log_dir, "\tCollecting Transitions...", console_log=True, no_log=True)
    envs, action_space, _, _ = create_vectorized_envs(game, num_envs=num_envs) # (num_envs, H, W, 3)

    transitions = []
    total_reward_list = []
    st = time.time()
    obs, _, = envs.reset()
    obs = preprocess_frames(obs, device) # (num_envs, 84, 84)
    frame_stacks = [deque([obs[i]]*4, maxlen=4) for i in range(num_envs)] # (num_envs, 4, 84, 84)
    stacked_obs = torch.stack([torch.stack(list(fs), dim=0) for fs in frame_stacks]) # (num_envs, 4, 84, 84)

    steps, total_avg_reward = 0, 0
    while steps < num_transitions:
        actions = select_action(mdp, action_space, stacked_obs, num_envs, disable_eps_greedy=disable_eps_greedy) # (num_evns, )
        next_obs, rewards, terms, truncs, infos = envs.step(actions)
        next_obs = preprocess_frames(next_obs, device) # (num_envs, 84, 84)
        rewards, actions, dones = to_tensor(rewards, actions, [t1 or t2 for t1, t2 in zip(terms, truncs)], device) # (8, 1)

        for i in range(num_envs):
            frame_stacks[i].append(next_obs[i])
            single_ssp = torch.stack(list(frame_stacks[i]), dim=0) # (4, 84, 84)
            single_ss = stacked_obs[i] # (4, 84, 84)
            memory.append(single_ss, actions[i], single_ssp, rewards[i], dones[i]) 
            transitions.append((single_ss, actions[i], single_ssp, rewards[i], dones[i])) # store transitions for MDP update
            stacked_obs[i] = single_ssp # num_envs x (84, 84)

            if dones[i].item():
                stacked_obs[i] = reset_or_fire(envs, i, infos, device, frame_stacks)

        total_avg_reward += sum(r.item() for r in rewards)
        steps += 1*num_envs

    total_reward_list.append(total_avg_reward / num_envs)
    eps_thres = VC.eps_threshold if not disable_eps_greedy else "None"
    log(log_dir, f"\tTransitions collected in: {time.time() - st:.4f}, Total Steps: {len(transitions)}, Epsilon: {eps_thres}", console_log=True, no_log=True)
    episode_reward_list += total_reward_list
    envs.close()
    return transitions