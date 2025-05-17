import os
import random
from typing import NamedTuple
import gymnasium as gym # type: ignore
import imageio # type: ignore
from wrapper import AtariWrapper
import torch
import argparse

class ValueContainer():
    def __init__(self):
        self.debug_mode = False
        self.eps_threshold = 0
        self.num_actions = 0
        self.steps_done = 0
        self.codebook_usage = "Not defined"
        self.transition_percentage = "Not defined"
        self.write_mode = "w"
        self.called = False
    
    def enable_debug(self):
        self.debug_mode = True
        print("Debug mode enabled")
    
    def get_write_mode(self, ):
        if self.called:
           return self.write_mode
        else:
            temp = self.write_mode
            self.write_mode = "a"
            self.called = True
            return temp

VALUE_CONTAINER = ValueContainer()

class VideoRecorder:
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

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["breakout", "tennis", "space_invaders", "boxing", "pong"], help="env name")
    parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="number of training epoch")
    parser.add_argument('--batch-size', default=32, type=int, help="batch size")
    parser.add_argument('--eval-cycle', default=10, type=int, help="epoch before retraining VQVAE")
    parser.add_argument('--transitions', default=1, type=int, help="number of transitions to do before mdp rebuild")
    parser.add_argument('--VQVAE-cycle', default=1000, type=int, help="number of epochs before model is retrained")
    parser.add_argument('--MDP-cycle', default=1000, type=int, help="number of epochs before mdp is recreated")
    parser.add_argument('--max-iterations', default=5000, type=int, help="max iterations VQVAE runs per training cycle")
    parser.add_argument('--initial-iterations', default=15000, type=int, help="number of iterations VQVAE does in epoch 0")
    parser.add_argument('--warmup', default=20000, type=int, help="number of warmup transitions to collect")
    parser.add_argument('--min-visits', default=1, type=int, help="times (s,a)-pair has to be visited to be considered known")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()

def log(log_dir, message, console_log=False, show_steps=False, show_eps=False, show_codebook_usage=False, show_transition_percentage=False, no_log=False):
    if show_steps:
        message = message + f", Steps done: {VALUE_CONTAINER.steps_done}"
    if show_eps:
        message = message + f", Epsilon: {VALUE_CONTAINER.eps_threshold:.4f}"
    if show_codebook_usage:
        message = message + f", {VALUE_CONTAINER.codebook_usage}"
    if show_transition_percentage:
        message = message + f", {VALUE_CONTAINER.transition_percentage}"
    if console_log:
        print(message)
    if no_log:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.txt")
    with open(log_path, VALUE_CONTAINER.get_write_mode()) as f:
        f.write(message + "\n")

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

def get_env_name(game):
    game_envs = {
        "breakout": "BreakoutNoFrameskip-v4",
        "tennis": "TennisNoFrameskip-v4",
        "space_invaders": "SpaceInvadersNoFrameskip-v4",
        "boxing": "BoxingNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4"
    }
    return game_envs.get(game, "BreakoutNoFrameskip-v4")

def create_env(game, seed=None, video=None):
    env = gym.make(get_env_name(game))
    env = AtariWrapper(env) if not video else AtariWrapper(env, video=video)
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    action_space = env.action_space

    return env, action_space, obs, info

def create_async_vector_env(game, num_envs=1): # FIXME: BROKEN RETURNS ONLY BLACK FRAMES, WHEN STEP IS CALLED
    env_name = get_env_name(game)
    def make_env():
        return AtariWrapper(gym.make(env_name))
    return gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)]), gym.make(env_name).action_space

def extract_and_batch(transitions):
    s_batch, a_batch, sp_batch, r_batch, d_batch = zip(*transitions)
    return (
        torch.cat(s_batch, dim=0).float(),
        torch.stack(a_batch).unsqueeze(1),
        torch.cat(sp_batch, dim=0).float(),
        torch.stack(r_batch).unsqueeze(1),
        torch.stack(d_batch).unsqueeze(1)
    )

def sample_memory(memory, batch_size):
   return extract_and_batch(random.sample(memory, batch_size))