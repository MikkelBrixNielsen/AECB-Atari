import os
import time
from itertools import count
import gymnasium as gym # type: ignore
import imageio # type: ignore
import matplotlib.pyplot as plt
from wrapper import AtariWrapper
import torch.nn.functional as F
from collections import namedtuple, deque, defaultdict, Counter
import torch
import numpy as np
import random
import argparse

class MDPBuilder:
    def __init__(self, encoder, quantizer):
        self.encoder = encoder
        self.quantizer = quantizer

    def discretize(self, frame):
        frame = frame.unsqueeze(0) # (1, 4, 84, 84)
        with torch.no_grad():
            z = self.encoder(frame)
            _, _, indices = self.quantizer(z)
            print(f"Unique frames: {len(torch.unique(indices))} / {self.quantizer.num_embeddings}")
            indices = indices.view(z.shape[2], z.shape[3]) # (H, W)

        return tuple(indices.view(-1).cpu().numpy())

    def build(self, transitions):
        transitions = defaultdict(Counter)  # (s, a) -> next_s -> count
        rewards = defaultdict(float)        # (s, a) -> total_reward
        dones = defaultdict(int)            # (s, a) -> # of terminal transitions
        counts = defaultdict(int)           # (s, a) -> count

        for s, a, sp, r, done in transitions:
            print("DOING MDP STUFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
            ds = self.discretize(s)
            dsp = self.discretize(sp)

            transitions[(ds, a)][dsp] += 1
            rewards[(ds, a)] += r
            counts[(ds, a)] += 1
            if done:
                dones[(ds, a)] += 1

        mdp = defaultdict(dict)

        for (s, a), next_states in transitions.items():
            print("DOING MDP STUFFF2222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222")

            total = sum(next_states.values())
            prob_list = []

            for s_prime, cnt in next_states.items():
                prob = cnt / total
                avg_reward = rewards[(s, a)] / counts[(s, a)]
                done_prob = dones[(s, a)] / counts[(s, a)]
                prob_list.append((prob, s_prime, avg_reward, done_prob > 0.5))

            mdp[s][a] = prob_list

        return mdp

class VideoRecorder: # from previous project
    def __init__(self, dir_name, fps=30):
        self.dir_name = dir_name
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record(self, frame):
        self.frames.append(frame)

    def save(self, file_name):
        path = os.path.join(self.dir_name, file_name)
        imageio.mimsave(path, self.frames, fps=self.fps, macro_block_size = None)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class MemoryBuffer:
    def __init__(self, size):
        self.size = size
        self.memory = deque(maxlen=size)

    def append(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch):
        # FIXME try doing an 80/20 split of random vs. recent frames to encourage recently learned behaviour
        return random.sample(self.memory, batch)

    def get_all(self):
        return list(self.memory)

    def __len__(self):
        return len(self.memory)


def convert_to_tensor(next_obs, action, reward, truncated, terminated, device):
    return (torch.from_numpy(next_obs).to(device), # (84, 84)
            torch.tensor([reward], device=device), # (1)
            torch.tensor([action], device=device), # (1)
            torch.tensor([truncated or terminated], device=device) # (1)
            )

def warmup(env, memory, seed, device, warmup=1000): # modified method based on version of code from github
    print("Warming up...")

    warmupstep = 0
    while True:
        obs, _ = env.reset(seed=seed)
        obs = torch.from_numpy(obs).to(device) # (84, 84)
        frame_stack = deque([obs, obs, obs, obs], maxlen=4) # (4, 84, 84)
        obs = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)

        while True:
            action = torch.tensor([[env.action_space.sample()]]).to(device)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            next_obs, action, reward, done = convert_to_tensor(next_obs, action, reward, truncated, terminated, device)
            frame_stack.append(next_obs)
            next_obs = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)
            memory.append(obs, action, next_obs, reward, done)

            obs = next_obs
            warmupstep += 1

            if terminated or truncated:
                break

        if warmupstep > warmup:
            break

def sample_memory(memory, args):
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
    return (torch.cat(batch.state), # state_batch (bs, 4, 84, 84)
            torch.cat(batch.action).unsqueeze(1), # action_batch (bs, 4, 84, 84)
            torch.cat(batch.next_state), # next_state_batch (bs, 4, 84, 84)
            torch.cat(batch.reward).unsqueeze(1), # reward_batch (bs, 4, 84, 84)
    )

def train_VQ_VAE(model, memory, optimizer, args, delta=0.0005, eta=0.0005, MAX_ITERATIONS=250):
    # FIXME Maybe include some performance / loss tracking?
    model.train()

    for iteration in count():
        st = time.time()

        state_batch, _, next_state_batch, _ = sample_memory(memory, args)
        batch = torch.cat([state_batch, next_state_batch], dim=0) # (bs*2, 4, 84, 84)
        recon, vq_loss = model(batch)
        
        # recon_loss = F.mse_loss(recon, batch, reduction='mean') # STANDARD/DEFAULT LOSS CALCULATION BASED ON VQ-VAE ARTICLE THING
        recon_loss = F.mse_loss(recon, batch, reduction='sum')




        # Output from model
        # recon_sample = recon[0].detach().cpu()

        # if iteration == MAX_ITERATIONS:
        #     fig, axs = plt.subplots(1, 4, figsize=(10, 3))
        #     for i in range(4):
        #         axs[i].imshow(recon_sample[i], cmap='gray')
        #         axs[i].set_title(f'Recon {i}')
        #         axs[i].axis('off')
        #     plt.suptitle("VQ-VAE Reconstruction")
        #     plt.show()


        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # FIXME: Make this into a methods which also logs to a file
        print(f"\tIteration {iteration}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Duration: {time.time() - st:.4f}")

        if iteration > MAX_ITERATIONS or (recon_loss < delta and vq_loss < eta): # if max iterations reached oconvergence => break
            break

def value_iteration(mdp, gamma=0.99, theta=1e-6):
    V = defaultdict(float)
    pi = {}

    while True:
        delta = 0
        for state in mdp:
            action_values = []
            for action, outcomes in mdp[state].items():
                value = 0
                for prob, next_state, reward, done in outcomes:
                    value += prob * (reward + gamma * V[next_state] * (not done))
                action_values.append((value, action))

            if action_values:
                best_value, best_action = max(action_values)
                delta = max(delta, abs(V[state] - best_value))
                V[state] = best_value
                pi[state] = best_action

        if delta < theta:
            break

    return V, pi

def select_action(model, obs, pi, n_action):
    model.eval()
    with torch.no_grad():
        z = model.encoder(obs)
        _, _, indices = model.quantizer(z)
        indices = indices.view(z.shape[2], z.shape[3])
        state = tuple(indices.view(-1).cpu().numpy())  # flatten to hashable tuple
        return pi.get(state, random.randint(0, n_action - 1))  # fallback

def eval_planner(model, pi, env_name, n_action, seed, video, device, epoch, log_dir):
    env, _, obs, info = create_env(env_name, seed, video=video)
    obs = torch.from_numpy(obs).to(device) # (84, 84)
    frame_stack = deque([obs, obs, obs, obs], maxlen=4) # (4, 84, 84)

    total_reward = 0
    video.reset()

    while True:
        obs = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)
        action = select_action(model, obs, pi, n_action)
        next_obs, reward, _, _, info = env.step(action)
        frame_stack.append(torch.from_numpy(next_obs).to(device))
        total_reward += reward

        if info["lives"] == 0:
            break
    
    env.close()

    subdir = f"seed_{seed}"
    full_path = os.path.join(log_dir, subdir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    filename = f"eval_{epoch}_{total_reward}.mp4"
    video.save(os.path.join(subdir, filename))

    print(f"\tSeed {seed} - total reward: {total_reward}")

# interact with env, inlcuding option to provide an epsilon threshold for eps-greedy action selection (to manage exploration)
def interact_with_env(model, pi, env, n_action, memory, seed, device, eps_threshold=.25):
    obs, _ = env.reset(seed=seed)
    obs = torch.from_numpy(obs).to(device) # (84, 84)
    frame_stack = deque([obs, obs, obs, obs], maxlen=4) # (4, 84, 84)
    obs = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)
    
    steps = 0
    while True:
        # epsilon threshold to manage exploration / exploitation
        action = select_action(model, obs, pi, n_action) if random.random() > eps_threshold else random.randint(0, n_action - 1) 
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs, reward, action, done = convert_to_tensor(next_obs, action, reward, truncated, terminated, device)
        frame_stack.append(next_obs)
        next_obs = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)
        memory.append(obs, action, next_obs, reward, done)
        obs = next_obs
        steps += 1

        if info["lives"] == 0:
            break

    return steps

def create_argparser(): # modified from previous project
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["breakout", "tennis", "space_invaders", "boxing", "pong"], help="env name")
    parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
    parser.add_argument('--batch-size', default=32, type=int, help="batch size")
    parser.add_argument('--eval-cycle', default=50, type=int, help="evaluation cycle")
    parser.add_argument('--episodes', default=50, type=int, help="number of epsiodes")
    return parser.parse_args()

def create_env(game, seed, video=None): # from previous project
    game_envs = {
        "breakout": "BreakoutNoFrameskip-v4",
        "tennis": "TennisNoFrameskip-v4",
        "space_invaders": "SpaceInvadersNoFrameskip-v4",
        "boxing": "BoxingNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4"
    }

    env = gym.make(game_envs.get(game, "BoxingNoFrameskip-v4"))
    env = AtariWrapper(env) if not video else AtariWrapper(env, video=video)
    obs, info = env.reset(seed=seed)
    n_action = env.action_space.n
    
    return env, int(n_action), obs, info

def make_log_dir(args): # modified from previous project
    log_dir = os.path.join(f"log_{args.env_name}", "VQ_VAE")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "log.txt")
    return log_dir, log_path

def plot_runs(runs, log_dir):
    # FIXME make plot and save it in corresponding directory
    pass
