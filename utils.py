from itertools import count
import os
import time
import gymnasium as gym # type: ignore
import imageio # type: ignore
import matplotlib.pyplot as plt
from wrapper import AtariWrapper
import torch.nn.functional as F
from collections import namedtuple, deque
import torch
import random
import argparse

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

# Transition = namedtuple('Transition',
                        # ('state', 'action', 'next_state', 'reward', 'done')) # from previous project

Transition = namedtuple('Transition',
                        ('state', 'next_state'))

class MemoryBuffer():
    def __init__(self, size):
        self.size = size
        self.memory = [None for _ in range(size)]
        self.ptr = 0

    def append(self, *args):
        self.memory[self.ptr] = Transition(*args)
        self.ptr = (self.ptr + 1) % self.size

    def sample(self, batch):
        return random.sample(self.memory, batch)

def log_and_print(epoch, eps_threshold, steps_done, total_loss, total_reward, avg_loss, avg_reward, time, log_path): # from previous project
        output = (f"Training epoch {epoch}: "
                  f"Loss {total_loss:.2f}, "
                  f"Avgloss {avg_loss:.2f}, "
                  f"Reward {total_reward}, "
                  f"Avgreward {avg_reward:.2f}, "
                  f"Epsilon {eps_threshold:.2f}, "
                  f"TotalStep {steps_done}, "
                  f"Seconds elapsed {time:.2f}"
                  )

        print(output)
        with open(log_path,"a") as f:
            f.write(f"{output}\n")

# def convert_to_tensor(reward, done, obs, next_obs, device):
#     return (torch.tensor([reward], device=device), # reward (1)
#             torch.tensor([done], device=device), # done(1)
#             torch.stack((torch.from_numpy(next_obs).to(device), obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0) # (1, 4, 84, 84)
#             )

def warmup(env, memory, seed, device, warmup=1000): # modified method based on version of code from github
    print("Warming up...")

    warmupstep = 0
    while True:
        obs, _ = env.reset(seed=seed)
        obs = torch.from_numpy(obs).to(device).unsqueeze(0) # (1, 84, 84)

        while True:
            action = torch.tensor([[env.action_space.sample()]]).to(device)
            # next_obs, reward, terminated, truncated, _ = env.step(action.item())
            # reward, done, next_obs = convert_to_tensor(reward, terminated or truncated, obs, next_obs, device)
            # memory.append(obs, action, next_obs, reward, done)
            
            next_obs, _, terminated, truncated, _ = env.step(action.item())
            torch.from_numpy(next_obs).to(device).unsqueeze(0) # (1, 84, 84)
            memory.append(obs, next_obs)

            obs = next_obs
            warmupstep += 1

            if terminated or truncated:
                break

        if warmupstep > warmup:
            break

# def sample_memory(memory, args):
#     transitions = memory.sample(args.batch_size)
#     batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
#     return (torch.cat(batch.state), # state_batch (bs, 1, 84, 84)
#             torch.cat(batch.next_state), # next_state_batch (bs, 1, 84, 84)
#             torch.cat(batch.action), # action_batch (bs, 1)
#             torch.cat(batch.reward).unsqueeze(1), # reward_batch (bs, 1)
#             torch.cat(batch.done).unsqueeze(1), # done_batch (bs, 1)
#     )

def sample_memory(memory, args):
    transitions = memory.sample(args.batch_size)
    batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
    return (torch.cat(batch.state), # state_batch (bs, 1, 84, 84)
            torch.cat(batch.next_state), # next_state_batch (bs, 1, 84, 84)
    )

def train_VQ_VAE(model, memory, optimizer, args):
    # FIXME Maybe include some performance / loss tracking?
    model.train()

    for epoch in range(args.epoch):
        st = time.time()

        state_batch, next_state_batch = sample_memory(memory, args)
        batch = torch.cat([state_batch, next_state_batch], dim=0) # (bs*2, 1, 84, 84)

        recon, vq_loss = model(batch)
        recon_loss = F.mse_loss(recon, batch)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Duration: {time.time() - st:.4f}")
    
def eval_model():
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

def plot_runs(res_runs, log_dir):
    # FIXME make plot and save it in corresponding directory
    pass
