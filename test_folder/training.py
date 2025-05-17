import os
import torch
import time 
from itertools import count
import torch.nn.functional as F
from collections import deque
import math
import random

from plotting import plot_codebook_usage, plot_input_vs_recon
from utils import Transition, VC, log, sample_memory, create_env
from mdp import discretize
from model import VQVAE

EPS_START, EPS_END, EPS_DECAY = 1, 0.05, 100000

def convert_to_tensor(next_obs, action, reward, truncated, terminated, device):
    return (torch.from_numpy(next_obs).to(device), # (84, 84)
            torch.tensor([action], device=device), # (1)
            torch.tensor([reward], device=device), # (1)
            torch.tensor([truncated or terminated], device=device) # (1)
            )

def warmup(env, memory, device, log_dir, num_steps=10000): # modified method based on version of code from github
    log(log_dir, "\tWarming up...", console_log=True, no_log=True)

    steps_taken = 0
    st = time.time()
    while steps_taken < num_steps:
        s, _ = env.reset()
        s = torch.from_numpy(s).to(device) # (84, 84)
        frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)
        s = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)

        while True:
            a = env.action_space.sample() # select random action
            sp, r, term, trun, _ = env.step(a)
            sp, a, r, d = convert_to_tensor(sp, a, r, trun, term, device)
            frame_stack.append(sp)
            sp = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)
            memory.append(s, a, sp, r, d)
            s = sp

            steps_taken += 1
            if term or trun or steps_taken >= num_steps:
                break

    log(log_dir, f"\tWarmup completed in: {time.time() - st:.4f}, Collected Observations: {num_steps}", console_log=VC.debug_mode, no_log=True)

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
        probs = estimate_codebook_usage_probs(model, x)
        eb = entropy_bonus(probs)
        up = usage_penalty(model, probs)

        recon_loss = F.mse_loss(x_r, x, reduction='sum')
        loss = recon_loss + vq_loss + up + eb

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss_list.append(recon_loss.item())
        vq_loss_list.append(vq_loss.item())
        entropy_bonus_list.append(eb)
        usage_penalty_list.append(up)

        log(log_dir, f"\t\tTraining Round: {iteration}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Usage Penalty: {up:.4f}, Entropy Bonus: {eb:.4f}, Duration: {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
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

def select_action_eval(model, action_space, pi, s):
    model.eval()
    ds = discretize(model, s)
    if ds in pi:
        return pi.get(ds)
    else:
        return action_space.sample()

def eval_planner(model, pi, args, video, seed, device, epoch, log_dir):
    model.eval()
    log(log_dir, "\tEvaluating Model...", console_log=True, no_log=True)
    env, action_space, s, info = create_env(args.env_name, seed=seed, video=video)
    s = torch.from_numpy(s).to(device) # (84, 84)
    frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)

    total_reward = 0
    steps = 0
    st = time.time()
    while True:
        s = torch.stack(list(frame_stack), dim=0).unsqueeze(0) # (1, 4, 84, 84)
        a = select_action_eval(model, action_space, pi, s)
        s, r, term, trun, info = env.step(a)
        s = torch.from_numpy(s).to(device=device) # (1, 84, 84)
        
        frame_stack.append(s) # (4, 84, 84)
        total_reward += r

        steps += 1
        lives = info["lives"]
        log(log_dir, f"\t\tSteps Taken: {steps}, Lives: {lives}, Total Reward: {total_reward}, Duration: {time.time() - st:.4f}", console_log=VC.debug_mode, no_log=True)
        if term or trun:
            if info["lives"] == 0:
                break
            else:
                s, info = env.reset(seed=seed)
                s = torch.from_numpy(s).to(device=device) # (1, 84, 84)
                frame_stack = deque([s] * 4, maxlen=4) # (1, 84, 84)- # reset frame stack to match env being reset

    path = os.path.join(f"seed_{seed}", f"eval_epoch_{epoch}_reward_{total_reward}.mp4")
    video.save(path)
    video.reset()
    env.close()
    log(log_dir, f"\tEvaluation completed in: {time.time() - st:.4f}, Total Reward: {total_reward}", console_log=True, no_log=True)
    return total_reward

def select_action(model, action_space, pi, s, disable_eps_greedy=False):
    model.eval()

    if not disable_eps_greedy: # Don't do epsilon calculations if it is disabled
        VC.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * VC.steps_done / EPS_DECAY)
    
    VC.steps_done += 1
    
    if disable_eps_greedy or random.random() > VC.eps_threshold:
        ds = discretize(model, s)
        return pi.get(ds) if ds in pi.keys() else action_space.sample()
    else:
        return action_space.sample() # random eps-greedy action
    
def collect_transitions(model, game, pi, memory, num_transitions, device, log_dir, episode_reward_list=None):
    ast = time.time()
    log(log_dir, "\tCollecting Transitions...", console_log=True, no_log=True)
    model.eval()

    transitions = []
    total_reward_list = []
    
    while True:
        env, action_space, s, info = create_env(game) # (84, 84)
        s = torch.from_numpy(s).to(device) # (84, 84)
        frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)

        steps = 0
        st = time.time()
        total_reward = 0

        while True:
            s = torch.stack(list(frame_stack), dim=0).unsqueeze(0) # (1, 4, 84, 84)
            a = select_action(model, action_space, pi, s, disable_eps_greedy=True) # Disable eps-greedy policy behaviour 
            # a = select_action(model, action_space, pi, s, disable_eps_greedy=False) # Enable eps-greedy policy bahaviour
            sp, r, term, trun, info = env.step(a)
            sp, a, r, d = convert_to_tensor(sp, a, r, trun, term, device)
            frame_stack.append(sp)
            sp = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)

            memory.append(s, a, sp, r, d)
            transitions.append(Transition(s, a, sp, r, d)) # store transitions for MDP update
            total_reward += r.item()
            lives = info["lives"]
            steps += 1

            log(log_dir, f"\t\t\tSteps Taken: {steps}, Lives: {lives}, Epsilon: {VC.eps_threshold}, Reward: {r.item()}, Elapsed Time: {time.time() - st:.4f}",console_log=VC.debug_mode, no_log=True)

            if term or trun:
                if info["lives"] == 0:
                    break
                else:
                    s, info = env.reset()
                    s = torch.from_numpy(s).to(device)
                    frame_stack = deque([s] * 4, maxlen=4)

        log(log_dir, f"\t\tEpisode completed in: {time.time() - st:.4f}, Steps Taken: {steps}, Epsilon: {VC.eps_threshold}, Total Reward: {total_reward}", console_log=VC.debug_mode, no_log=True)
        
        total_reward_list.append(total_reward)

        if len(transitions) >= num_transitions:
            break

    log(log_dir, f"\tTransitions collected in: {time.time() - ast:.4f}, Total Steps: {len(transitions)}", console_log=True, no_log=True)
    episode_reward_list += total_reward_list
    return transitions