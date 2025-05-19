import os
import gymnasium as gym # type: ignore
import imageio # type: ignore
from wrapper import AtariWrapper
from collections import namedtuple, deque
import torch
import random
import argparse

class ValueContainer():
    def __init__(self):
        self.debug_mode = False
        self.eps_threshold = 1
        self.steps_done = 0
        self.codebook_usage = "Not defined"
        self.transition_percentage = "Not defined"
        self.write_mode = "w"
        self.called = False
    
    def enable(self):
        self.debug_mode = True
        print("Debug mode enabled")

    def get_write_mode(self):
        if not self.called:
           temp = self.write_mode
           self.write_mode = "a"
           self.called = True
           return temp
        return self.write_mode
    
# Global value container shared between files
VC = ValueContainer()

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

    def sample(self, batch_size):
        return random.sample(list(self.memory), batch_size)

    def sample_recent(self, batch_size, num_most_recent=10000):
        num = min(len(self.memory), num_most_recent)
        return random.sample(list(self.memory)[-num:], batch_size)

    def get_all(self):
        return list(self.memory)

    def __len__(self):
        return len(self.memory)

def create_argparser(): # modified from previous project
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["breakout", "tennis", "space_invaders", "boxing", "pong"], help="env name")
    parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="number of training epoch")
    parser.add_argument('--batch-size', default=32, type=int, help="batch size")
    parser.add_argument('--eval-cycle', default=10, type=int, help="epoch before retraining VQVAE")
    parser.add_argument('--transitions', default=2500, type=int, help="number of transitions to do before mdp rebuild")
    parser.add_argument('--VQVAE-cycle', default=1000, type=int, help="number of epochs before model is retrained")
    parser.add_argument('--MDP-cycle', default=1000, type=int, help="number of epochs before mdp is recreated")
    parser.add_argument('--max-iterations', default=25000, type=int, help="max iterations VQVAE runs per training cycle")
    parser.add_argument('--initial-iterations', default=10000, type=int, help="number of iterations VQVAE does in epoch 0")
    parser.add_argument('--warmup', default=10000, type=int, help="number of warmup transitions to collect")
    parser.add_argument('--min-visits', default=1, type=int, help="times (s,a)-pair has to be visited to be considered known")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()

def create_log_dir(args, seeds):
    log_dir = os.path.join(f"log_{args.env_name}", "VQ_VAE")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    for seed in seeds:
        subdir = os.path.join(log_dir, f"seed_{seed}")
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    log_path = os.path.join(log_dir, "log.txt")
    return log_dir, log_path

def log(log_dir, message, console_log=False, show_steps=False, show_eps=False, show_codebook_usage=False, show_transition_percentage=False, no_log=False):
    if show_steps:
        message = message + f", Steps done w/o warmup: {VC.steps_done}"
    if show_eps:
        message = message + f", Epsilon: {VC.eps_threshold:.4f}"
    if show_codebook_usage:
        message = message + f", {VC.codebook_usage}"
    if show_transition_percentage:
        message = message + f", {VC.transition_percentage}"
    if console_log:
        print(message)
    if no_log:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")
    with open(log_path, VC.get_write_mode()) as f:
        f.write(message + "\n")

def get_env(game):
    game_envs = {
        "breakout": "BreakoutNoFrameskip-v4",
        "tennis": "TennisNoFrameskip-v4",
        "space_invaders": "SpaceInvadersNoFrameskip-v4",
        "boxing": "BoxingNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4"
    }
    return game_envs.get(game, "BreakoutNoFrameskip-v4")

def create_eval_env(game, seed=None, video=None): # from previous project
    env = gym.make(get_env(game))
    env = AtariWrapper(env) if not video else AtariWrapper(env, video=video)
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    return env, env.action_space, obs, info

def create_vectorized_envs(game, seed=None, num_envs=8):
    envs = gym.make_vec(get_env(game), num_envs, vectorization_mode="sync")
    obs, info = envs.reset(seed=seed) if seed is not None else envs.reset()
    return envs, envs.action_space, obs, info

def compute_known_transition_percentage(N_sa, states, actions, M):
    total_possible = states * actions
    known = sum([1 for (s, a) in N_sa if N_sa[(s, a)] >= M])
    return 100.0 * known / total_possible if total_possible > 0 else 0.0

def extract_and_batch(transitions):
    batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
    return (torch.cat([t.unsqueeze(0) for t in batch.state]), # state_batch (bs, 4, 84, 84)
            torch.cat(batch.action).unsqueeze(1), # action_batch (bs, 1)
            torch.cat([t.unsqueeze(0) for t in batch.next_state]), # next_state_batch (bs, 4, 84, 84)
            torch.cat(batch.reward).unsqueeze(1), # reward_batch (bs, 1)
            torch.cat(batch.done).unsqueeze(1), # done_batch (bs, 1)
    )

def sample_memory(memory, batch_size):
   return extract_and_batch(memory.sample(batch_size))