import random
import torch
import os
from collections import deque
from itertools import count
import torch.nn.functional as F
import math
import time

from model import VQVAE
from utils import VALUE_CONTAINER, create_env, create_async_vector_env, log, sample_memory
from plotting import plot_codebook_usage, plot_input_vs_recon

# Hyperparameters
EPS_START, EPS_END, EPS_DECAY = 1, 0.05, 100000
VALUE_CONTAINER.eps_threshold = EPS_START

def convert_to_tensor(next_obs, actions, rewards, terms, truncs, device):
    return (
            torch.from_numpy(next_obs).to(device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(truncs | terms, dtype=torch.bool, device=device)
            )

def warmup(env_name, device, log_dir, num_steps=10000):
    env, action_space, _, _ = create_env(env_name)
    log(log_dir, "\tWarming up...", console_log=True, no_log=True)

    transitions = []
    steps_taken = 0
    st = time.time()
    while steps_taken < num_steps:
        s, _ = env.reset()
        s = torch.from_numpy(s).to(device) # (84, 84)
        frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)
        s = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)

        while True:
            a = action_space.sample() # select random action
            sp, r, term, trun, _ = env.step(a)
            sp, a, r, d = convert_to_tensor(sp, a, r, trun, term, device)
            frame_stack.append(sp)
            sp = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)
            transitions.append((s, a, sp, r, d))
            s = sp

            steps_taken += 1
            if term or trun or steps_taken >= num_steps:
                break
    VALUE_CONTAINER.num_actions = action_space.n
    env.close()
    log(log_dir, f"\tWarmup completed in: {time.time() - st:.4f}, Collected Observations: {num_steps}", console_log=VALUE_CONTAINER.debug_mode, no_log=True)
    return transitions

def estimate_code_usage_probs(model, x):
    z_e = model.encoder(x)
    _, _, indices = model.quantizer(z_e)
    counts = torch.bincount(indices.view(-1), minlength=model.quantizer.num_embeddings).float()
    return counts / counts.sum()

def entropy_bonus(probs, scale=0.05):
    return -scale * -torch.sum(probs * torch.log(probs + 1e-8))

def usage_penalty(num_embeddings, probs, scale=0.05):
    uniform = torch.full_like(probs, 1.0 / num_embeddings)
    kl_div = F.kl_div((probs + 1e-8).log(), uniform, reduction='sum')
    return scale * kl_div

def train_VQ_VAE(model, memory, optimizer, log_dir, max_iterations=10000, batch_size=32, theta=5e-4, N=500):
    ast = time.time()
    recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list = [], [], [], []
    log(log_dir, "\tTraining Model...", console_log=True, no_log=True)

    model.train()
    for iteration in count():
        st = time.time()

        x, _, _, _, _ = sample_memory(memory, batch_size)
        x_r, vq_loss = model(x)
        probs = estimate_code_usage_probs(model, x)
        recon_loss = F.mse_loss(x_r, x, reduction='sum')
        up = usage_penalty(model.quantizer.num_embeddings, probs)
        eb = entropy_bonus(probs)
        loss = recon_loss + vq_loss + up + eb

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss_list.append(recon_loss.item())
        vq_loss_list.append(vq_loss.item())
        usage_penalty_list.append(up)
        entropy_bonus_list.append(eb)

        log(log_dir, f"\t\tTraining Round: {iteration}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Usage Penalty: {up:.4f}, Entropy Bonus: {eb:.4f}, Duration: {time.time() - st:.4f}", console_log=VALUE_CONTAINER.debug_mode, no_log=True)
        if iteration > max_iterations - 1 or (len(vq_loss_list) > N and (abs(recon_loss_list[-N] + vq_loss_list[-N] - loss.item()) < theta)): # if max iterations reached or loss does not improve => break
            break

    log(log_dir, f"\tModel training completed in: {time.time() - ast}", console_log=True, no_log=True)
    return recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list

def initial_model_training(memory, args, epoch, seed, log_dir, device, usage_log=None):
    model = VQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_lists = train_VQ_VAE(model, memory, optimizer, log_dir, max_iterations=args.initial_iterations, batch_size=args.batch_size, N=args.initial_iterations)
    torch.save(model, os.path.join(os.path.join(log_dir, f"seed_{seed}"), f'model{epoch}.pth')) # save current model
    plot_input_vs_recon(model, memory, args, epoch, log_dir, seed)
    plot_codebook_usage(model, memory, log_dir, epoch, seed, usage_log=usage_log)
    return model, optimizer, loss_lists

def train_model(model, optimizer, memory, args, epoch, seed, log_dir, usage_log=None):
    loss_lists = train_VQ_VAE(model, memory, optimizer, log_dir, max_iterations=args.max_iterations, batch_size=args.batch_size, N=1000)
    torch.save(model, os.path.join(os.path.join(log_dir, f"seed_{seed}"), f'model{epoch}.pth')) # save current model
    plot_input_vs_recon(model, memory, args, epoch, log_dir, seed)
    plot_codebook_usage(model, memory, log_dir, epoch, seed, usage_log=usage_log)
    return loss_lists

def select_action_eval(mdp, action_space, pi, s):
    s_idx = mdp.get_index_if_known(s)
    if s_idx is not None:
        return pi[s_idx]
    else:
        return action_space.sample()

def eval_planner(mdp, pi, args, video, seed, device, epoch, log_dir):
    log(log_dir, "\tEvaluating Model...", console_log=True, no_log=True)
    env, action_space, s, info = create_env(args.env_name, seed=seed, video=video)
    s = torch.from_numpy(s).to(device) # (84, 84)
    frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)

    total_reward = 0
    steps = 0
    st = time.time()
    while True:
        s = torch.stack(list(frame_stack), dim=0).unsqueeze(0) # (1, 4, 84, 84)
        a = select_action_eval(mdp, action_space, pi, s)
        s, r, term, trun, info = env.step(a)
        s = torch.from_numpy(s).to(device=device) # (1, 84, 84)
        
        frame_stack.append(s) # (4, 84, 84)
        total_reward += r

        steps += 1
        lives = info["lives"]
        log(log_dir, f"\t\tSteps Taken: {steps}, Lives: {lives}, Total Reward: {total_reward}, Duration: {time.time() - st:.4f}", console_log=VALUE_CONTAINER.debug_mode, no_log=True)
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

def select_action(mdp, action_space, pi, s, disable_eps_greedy=False):
    if not disable_eps_greedy: # Don't do epsilon calculations if it is disabled
        VALUE_CONTAINER.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * VALUE_CONTAINER.steps_done / EPS_DECAY)
    
    VALUE_CONTAINER.steps_done += 1
    
    if disable_eps_greedy or random.random() > VALUE_CONTAINER.eps_threshold:
        s_idx = mdp.discretize_and_index(s)
        if s_idx < len(pi):
            return pi[s_idx] # exploitive action 
        else:
            return action_space.sample() # fallback action
    else:
        return action_space.sample() # random eps-greedy action
    
def collect_transitions(mdp, game, pi, num_transitions, device, log_dir):
    ast = time.time()
    log(log_dir, "\tCollecting Transitions...", console_log=True, no_log=True)
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
            a = select_action(mdp, action_space, pi, s, disable_eps_greedy=True) # Disable eps-greedy policy behaviour 
            # a = select_action(model, action_space, pi, s, disable_eps_greedy=False) # Enable eps-greedy policy bahaviour
            sp, r, term, trun, info = env.step(a)
            sp, a, r, d = convert_to_tensor(sp, a, r, trun, term, device)
            frame_stack.append(sp)
            sp = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)

            transitions.append((s, a, sp, r, d)) # store transitions for MDP update
            total_reward += r.item()
            lives = info["lives"]
            steps += 1

            log(log_dir, f"\t\t\tSteps Taken: {steps}, Lives: {lives}, Epsilon: {VALUE_CONTAINER.eps_threshold}, Reward: {r.item()}, Elapsed Time: {time.time() - st:.4f}",console_log=VALUE_CONTAINER.debug_mode, no_log=True)

            if term or trun:
                if info["lives"] == 0:
                    break
                else:
                    s, info = env.reset()
                    s = torch.from_numpy(s).to(device)
                    frame_stack = deque([s] * 4, maxlen=4)

        log(log_dir, f"\t\tEpisode completed in: {time.time() - st:.4f}, Steps Taken: {steps}, Epsilon: {VALUE_CONTAINER.eps_threshold}, Total Reward: {total_reward}", console_log=VALUE_CONTAINER.debug_mode, no_log=True)
        
        total_reward_list.append(total_reward)
        env.close()
        if len(transitions) >= num_transitions:
            break

    log(log_dir, f"\tTransitions collected in: {time.time() - ast:.4f}, Total Steps: {len(transitions)}", console_log=True, no_log=True)

    return transitions, total_reward_list


# import torch.multiprocessing as mp
# from collections import deque

# def worker_collect(worker_id, mdp, game, pi, num_transitions, device, output_queue):
#     transitions = []
#     total_rewards = []

#     while len(transitions) < num_transitions:
#         env, action_space, s, info = create_env(game)
#         s = torch.from_numpy(s).to(device)
#         frame_stack = deque([s] * 4, maxlen=4)

#         total_reward = 0
#         while True:
#             s = torch.stack(list(frame_stack), dim=0).unsqueeze(0)
#             a = select_action(mdp, action_space, pi, s, disable_eps_greedy=True)
#             sp, r, term, trun, info = env.step(a)
#             sp, a, r, d = convert_to_tensor(sp, a, r, trun, term, device)
#             frame_stack.append(sp)
#             sp = torch.stack(list(frame_stack), dim=0).unsqueeze(0)

#             transitions.append((s, a, sp, r, d))
#             total_reward += r.item()

#             if term or trun:
#                 if info["lives"] == 0:
#                     break
#                 else:
#                     s, info = env.reset()
#                     s = torch.from_numpy(s).to(device)
#                     frame_stack = deque([s] * 4, maxlen=4)

#         total_rewards.append(total_reward)
#         env.close()

#     output_queue.put((transitions, total_rewards))

# def collect_transitions(mdp, game, pi, num_transitions, device, log_dir, num_workers=4):
#     st = time.time()
#     mp.set_start_method("spawn", force=True)
#     transitions_per_worker = num_transitions // num_workers

#     log(log_dir, f"\t[Concurrent] Collecting {num_transitions} transitions using {num_workers} workers...", console_log=True)
    
#     output_queue = mp.Queue()
#     workers = []
#     for i in range(num_workers):
#         p = mp.Process(target=worker_collect, args=(i, mdp, game, pi, transitions_per_worker, device, output_queue))
#         p.start()
#         workers.append(p)

#     all_transitions = []
#     all_rewards = []
#     for _ in range(num_workers):
#         trans, rewards = output_queue.get()
#         all_transitions.extend(trans)
#         all_rewards.extend(rewards)

#     for p in workers:
#         p.join()

#     log(log_dir, f"\t[Concurrent] Collected {len(all_transitions)} transitions, Duration {time.time() - st:.4f}", console_log=True)
#     return all_transitions, all_rewards