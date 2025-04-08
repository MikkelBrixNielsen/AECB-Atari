import os
import time
import gymnasium as gym # type: ignore
import imageio # type: ignore
import matplotlib.pyplot as plt
from wrapper import AtariWrapper
import torch.nn.functional as F
from collections import namedtuple, deque, defaultdict, Counter
import torch
import random
import argparse

class MDPBuilder:
    def __init__(self, encoder, quantizer):
        self.encoder = encoder
        self.quantizer = quantizer

    def discretize(self, frame):
        frame = frame.unsqueeze(0) # (1, 1, 84, 84)
        with torch.no_grad():
            z = self.encoder(frame)
            _, _, indices = self.quantizer(z)
            indices = indices.view(z.shape[2], z.shape[3]) # (H, W)

        return tuple(indices.view(-1).cpu().numpy()) # hashable

    def build(self, transitions):
        transitions = defaultdict(Counter)  # (s, a) -> next_s -> count
        rewards = defaultdict(float)        # (s, a) -> total_reward
        dones = defaultdict(int)            # (s, a) -> # of terminal transitions
        counts = defaultdict(int)           # (s, a) -> count

        for s, a, sp, r, done in transitions:
            ds = self.discretize(s)
            dsp = self.discretize(sp)

            transitions[(ds, a)][dsp] += 1
            rewards[(ds, a)] += r
            counts[(ds, a)] += 1
            if done:
                dones[(ds, a)] += 1

        mdp = defaultdict(dict)

        for (s, a), next_states in transitions.items():
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

class MemoryBuffer():
    # Cyclic memory buffer
    def __init__(self, size):
        self.size = size
        self.memory = [None for _ in range(size)]
        self.ptr = 0

    def append(self, *args):
        self.memory[self.ptr] = Transition(*args)
        self.ptr = (self.ptr + 1) % self.size

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def get_all(self):
        return [t for t in self.memory if t]

def convert_to_tensor(next_obs, action, reward, truncated, terminated, device):
    return (torch.from_numpy(next_obs).to(device).unsqueeze(0), # (1, 84, 84)
            torch.tensor([reward], device=device), # (1)
            torch.tensor([action], device=device), # (1)
            torch.tensor([truncated or terminated], device=device) # (1)
            )

def warmup(env, memory, seed, device, warmup=1000): # modified method based on version of code from github
    print("Warming up...")

    warmupstep = 0
    while True:
        obs, _ = env.reset(seed=seed)
        obs = torch.from_numpy(obs).to(device).unsqueeze(0) # (1, 84, 84)

        while True:
            action = torch.tensor([[env.action_space.sample()]]).to(device)
            next_obs, _, terminated, truncated, _ = env.step(action.item())
            next_obs, action, reward, done = convert_to_tensor(next_obs, action, reward, truncated, terminated, device)
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
    return (torch.cat(batch.state), # state_batch (bs, 1, 84, 84)
            torch.cat(batch.action), # action_batch (bs, 1, 84, 84)
            torch.cat(batch.next_state), # next_state_batch (bs, 1, 84, 84)
            torch.cat(batch.reward), # reward_batch (bs, 1, 84, 84)
    )

def train_VQ_VAE(model, memory, optimizer, args):
    # FIXME Maybe include some performance / loss tracking?
    model.train()

    for epoch in range(args.epoch):
        st = time.time()

        state_batch, _, next_state_batch, _  = sample_memory(memory, args)
        batch = torch.cat([state_batch, next_state_batch], dim=0) # (bs*2, 1, 84, 84)

        recon, vq_loss = model(batch)
        recon_loss = F.mse_loss(recon, batch)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # FIXME: Make this into a methods which also logs to a file
        print(f"Epoch {epoch}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Duration: {time.time() - st:.4f}")
    
def train_planner():
    pass

def eval_planner():
    pass

def create_argparser(): # modified from previous project
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["breakout", "tennis", "space_invaders", "boxing", "pong"], help="env name")
    parser.add_argument('--lr', default=2.5e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="training epoch")
    parser.add_argument('--batch-size', default=32, type=int, help="batch size")
    parser.add_argument('--eval-cycle', default=500, type=int, help="evaluation cycle")
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
