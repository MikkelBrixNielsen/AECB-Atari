import math
import os
import time
from itertools import count
import gymnasium as gym # type: ignore
import imageio # type: ignore
import matplotlib.pyplot as plt
from wrapper import AtariWrapper
import torch.nn.functional as F
from collections import namedtuple, deque, defaultdict, Counter
from torch.utils.data import DataLoader, TensorDataset

import torch
import random
import argparse

class DebugContainer():
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
    
    def enable(self):
        self.debug_mode = True
        print("Debug mode enabled")
    
    def get_mode(self):
        return self.debug_mode

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

# Hyperparameters
EPS_START, EPS_END, EPS_DECAY = 1, 0.05, 50000
eps_threshold = EPS_START
steps_done = 0

# Global variables
DEBUGGER = DebugContainer(debug_mode=False)
write_mode = 'w'

def create_argparser(): # modified from previous project
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="breakout", type=str, choices=["breakout", "tennis", "space_invaders", "boxing", "pong"], help="env name")
    parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")
    parser.add_argument('--epoch', default=10001, type=int, help="number of training epoch")
    parser.add_argument('--batch-size', default=32, type=int, help="batch size")
    parser.add_argument('--eval-cycle', default=10, type=int, help="epoch before retraining VQVAE")
    parser.add_argument('--transitions', default=1, type=int, help="number of transitions to do before mdp rebuild")
    parser.add_argument('--retrain-cycle', default=1000, type=int, help="number of epochs before model is retrained")
    parser.add_argument('--max-iterations', default=25000, type=int, help="max iterations VQVAE runs per training cycle")
    parser.add_argument('--warmup', default=10000, type=int, help="number of warmup transitions to collect")
    parser.add_argument('--min-visits', default=50, type=int, help="times (s,a)-pair has to be visited to be considered known")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()

def make_log_dir(args, seeds):
    log_dir = os.path.join(f"log_{args.env_name}", "VQ_VAE")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    for seed in seeds:
        subdir = os.path.join(log_dir, f"seed_{seed}")
        if not os.path.exists(subdir):
            os.makedirs(subdir)

    log_path = os.path.join(log_dir, "log.txt")
    return log_dir, log_path

def log(log_dir, message, console_log=False, show_steps=False, show_eps=False):
    global steps_done
    global write_mode
    global eps_threshold
    if show_steps:
        message = message + f", Steps done excluding warmup: {steps_done}"
    if show_eps:
        message = message + f", Epsilon: {eps_threshold}"
    if console_log:
        print(message)
    os.makedirs(log_dir, exist_ok=True)  # Ensure the directory exists
    log_path = os.path.join(log_dir, "log.txt")
    with open(log_path, write_mode) as f:
        f.write(message + "\n")
    write_mode = 'a'

def plot_runs(runs, log_dir):
    # FIXME make plot and save it in corresponding directory
    pass

def plot_input_vs_recon(model, memory, args, epoch, log_dir, seed=0000):
    recon = None
    with torch.no_grad():
        state_batch, _, next_state_batch, _, _ = sample_memory(memory, args)
        batch = torch.cat([state_batch, next_state_batch], dim=0) # (bs*2, 4, 84, 84)
        recon, _ = model(batch)

    batch = batch[:8]  # first 8 samples
    recon = recon[:8]

    fig, axs = plt.subplots(8, 8, figsize=(12, 12))  # 8 images × 4 channels × 2 (input+recon)
    for i in range(8):  # for each image
        for j in range(4):  # for each frame
            axs[i, j].imshow(batch[i, j].cpu(), cmap='gray')
            axs[i, j].set_title(f'In {j}')
            axs[i, j].axis('off')

            axs[i, j+4].imshow(recon[i, j].detach().cpu(), cmap='gray')
            axs[i, j+4].set_title(f'Out {j}')
            axs[i, j+4].axis('off')

    path = os.path.join(f'{log_dir}/seed_{seed}', 'log_reconstruction_images')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.tight_layout()
    plt.savefig(f"{path}/epoch_{epoch}")
    plt.close()

def plot_multiple_series(data_lists, log_dir, seed=0, labels=None, title='Plot', xlabel='X', ylabel='Y', steps=None, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    
    for i, data in enumerate(data_lists):
        # x = list(range(len(data)))
        label = labels[i] if labels and i < len(labels) else f'Series {i + 1}'
        step = steps[i] if steps and i < len(steps) else 1
        x = [n*step for n in range(len(data))]
        plt.plot(x, data, label=label, alpha=0.6)
    
    plt.title(title)
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/seed_{seed}/{title}")
    plt.close()

def plot_model_loss(recon_loss_list, vq_loss_list, seed, log_dir):
    total_loss = [sum(t) for t in zip(recon_loss_list, vq_loss_list)]
    data = [recon_loss_list, vq_loss_list, total_loss]
    titles = ["Recon Loss", "VQ Loss", "Total Loss"]
    plot_name = "VQ-VAE Loss Curves"
    plot_multiple_series(data, log_dir, seed, titles, plot_name, "Epochs", "Loss")

def plot_planner_reward(episode_reward_list, eval_reward_list, seed, log_dir):
    plot_multiple_series([episode_reward_list], log_dir, seed, "Episodic Reward", "Planner Episodic Reward", "Episode", "Reward")
    plot_multiple_series([eval_reward_list], log_dir, seed, "Evaluation Reward", "Planner Evaluation Reward", "Epochs", "Reward")

def compute_codebook_usage(model, dataset, batch_size=128):
    model.eval()
    usage_counter = Counter()
    loader = DataLoader(TensorDataset(torch.stack(dataset)), batch_size=batch_size)

    with torch.no_grad():
        for (obs_batch, ) in loader:
            obs_batch = obs_batch.to(next(model.parameters()).device)
            z_e = model.encoder(obs_batch)
            _, _, z_q_indices = model.quantizer(z_e)
            z_q_indices = z_q_indices.view(-1).cpu().numpy()
            usage_counter.update(z_q_indices.tolist())

    return usage_counter

def plot_codebook_usage(model, memory, log_dir, epoch, seed):
    num_codes = model.quantizer.num_embeddings
    dataset = []

    for s, _, sp, _, _ in memory.get_all():
        dataset.append(s)
        dataset.append(sp)
    
    usage_counter = compute_codebook_usage(model, dataset)
    usage = [usage_counter.get(i, 0) for i in range(num_codes)]
    used_codes = sum(1 for count in usage if count > 0)

    plt.figure(figsize=(12, 5))
    plt.bar(range(num_codes), usage)
    plt.xlabel("Codebook Index")
    plt.ylabel("Usage Count")
    plt.title("VQ-VAE Codebook Usage")
    path = f"{log_dir}/seed_{seed}/Codebook_Usage_epoch_{epoch}"
    plt.savefig(path)
    plt.close()
    print()

    log(log_dir, f"Codebook usage: {used_codes} / {num_codes}")

def create_env(game, seed, video=None): # from previous project
    game_envs = {
        "breakout": "BreakoutNoFrameskip-v4",
        "tennis": "TennisNoFrameskip-v4",
        "space_invaders": "SpaceInvadersNoFrameskip-v4",
        "boxing": "BoxingNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4"
    }

    env = gym.make(game_envs.get(game, "BreakoutNoFrameskip-v4"))
    env = AtariWrapper(env) if not video else AtariWrapper(env, video=video)
    obs, info = env.reset(seed=seed)
    action_space = env.action_space
    
    return env, action_space, obs, info

def convert_to_tensor(next_obs, action, reward, truncated, terminated, device):
    return (torch.from_numpy(next_obs).to(device), # (84, 84)
            torch.tensor([action], device=device), # (1)
            torch.tensor([reward], device=device), # (1)
            torch.tensor([truncated or terminated], device=device) # (1)
            )

def warmup(env, memory:MemoryBuffer, seed, device, log_dir, num_steps=10000): # modified method based on version of code from github
    log(log_dir, "\tWarming up...", console_log=True)

    steps_taken = 0
    st = time.time()
    while steps_taken < num_steps:
        s, _ = env.reset(seed=seed)
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

    log(log_dir, f"\tWarmup completed in: {time.time() - st:.4f}, Collected Observations: {num_steps}", console_log=DEBUGGER.get_mode())

def extract_and_batch(transitions):
    batch = Transition(*zip(*transitions)) # batch-array of Transitions -> Transition of batch-arrays.
    return (torch.cat(batch.state), # state_batch (bs, 4, 84, 84)
            torch.cat(batch.action).unsqueeze(1), # action_batch (bs, 1)
            torch.cat(batch.next_state), # next_state_batch (bs, 4, 84, 84)
            torch.cat(batch.reward).unsqueeze(1), # reward_batch (bs, 1)
            torch.cat(batch.done).unsqueeze(1), # done_batch (bs, 1)
    )

def sample_memory(memory, args):
   return extract_and_batch(memory.sample(args.batch_size))

def train_VQ_VAE(model, memory, optimizer, args, log_dir, theta=5e-4, N=500):
    ast = time.time()
    log(log_dir, "\tTraining Model...", console_log=True)
    recon_loss_list = []
    vq_loss_list = []

    model.train()

    for iteration in count():
        st = time.time()

        x, _, _, _, _ = sample_memory(memory, args) # bs, 4, 84, 84
        x_r, vq_loss = model(x)

        recon_loss = F.mse_loss(x_r, x, reduction='sum')
        # recon_loss = F.mse_loss(x_r, x, reduction='mean')
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recon_loss_list.append(recon_loss.item())
        vq_loss_list.append(vq_loss.item())

        log(log_dir, f"\t\tTraining Round: {iteration}, Recon Loss: {recon_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Duration: {time.time() - st:.4f}", console_log=DEBUGGER.get_mode())
        if iteration > args.max_iterations - 1 or (len(vq_loss_list) > N and (abs(recon_loss_list[-N] + vq_loss_list[-N] - loss.item()) < theta)): # if max iterations reached or loss does not improve => break
            break

    log(log_dir, f"\tModel traininig completed in: {time.time() - ast}", console_log=True)
    return recon_loss_list, vq_loss_list

def train_model_and_plot(model, memory, optimizer, args, epoch, seed, log_dir):
    lrec, lvq = train_VQ_VAE(model, memory, optimizer, args, log_dir)
    torch.save(model, os.path.join(os.path.join(log_dir, f"seed_{seed}"), f'model{epoch}.pth')) # save current model
    plot_input_vs_recon(model, memory, args, epoch, log_dir, seed)
    plot_codebook_usage(model, memory, log_dir, epoch, seed)
    return lrec, lvq

def validate_transition_probabilities(P, tolerance=1e-6, log_dir=None, console_log=False):
    invalid_pairs = []

    for (s, a), transitions in P.items():
        total_prob = sum(transitions.values())
        if abs(total_prob - 1) > tolerance and len(transitions.values()):
            invalid_pairs.append(((s, a), total_prob))

    if invalid_pairs:
        msg = f"\t\t[WARNING] {len(invalid_pairs)} (s, a) pairs have invalid transition probability sums:"
        if log_dir:
            log(log_dir, msg, console_log=console_log)
            for (s, a), total in invalid_pairs:
                log(log_dir, f"\t(s, a) = ({s}, {a}) → total probability = {total:.6f}", console_log=False)
        else:
            print(msg)
            for i, ((s, a), total) in enumerate(invalid_pairs):
                print(f"\t\t\tPair {i} total probability = {total:.6f}")
    else:
        msg = "\t\t[OK] All transition probability distributions sum to ~1.0"
        if log_dir:
            log(log_dir, msg, console_log=console_log)
        else:
            print(msg)

    return len(invalid_pairs) == 0

def discretize(model, s):
    model.eval()
    with torch.no_grad():
        z_e = model.encoder(s)
        _, _, z_q_indices = model.quantizer(z_e)
        z_q_indices = z_q_indices.cpu().numpy()

    return z_q_indices.flatten().tobytes()

def discretize_multiple(model, obs_batch):
    return [discretize(model, s.unsqueeze(0)) for s in obs_batch]

def discretized_extract_and_batch(model, transitions, batch_size=128): # Larger batch increases speed but also memory usage
    model.eval()
    ds_list, a_list, dsp_list, r_list, d_list = [], [], [], [], []

    for i in range(0, len(transitions), batch_size):
        mini_batch = transitions[i:i+batch_size]
        s_batch, a_batch, sp_batch, r_batch, d_batch = extract_and_batch(mini_batch)

        with torch.no_grad():
            ds_batch = discretize_multiple(model, s_batch)
            dsp_batch = discretize_multiple(model, sp_batch)

        ds_list.extend(ds_batch)
        a_list.extend(a_batch)
        dsp_list.extend(dsp_batch)
        r_list.extend(r_batch)
        d_list.extend(d_batch)

    return zip(ds_list, a_list, dsp_list, r_list, d_list)


def is_known(N_sa_val, M):
    return N_sa_val >= M

def identity(s1, s2):
    return int(s1 == s2)

# P(s'|s, a) = { N(s, a, s) / N(s, a)    if N(s, a) >= M  |  R(s, a) = { R_sum / N(s, a)     if N(s, a) >= M  |  D(s, a) = { 1 if D_sum / N(s, a) > 0.5 else 0     if N(s, a) >= M
#              { I[s' = s]}              otherwise        |            { R_max               otherwise        |            { 0                                     otherwise
# Note: The otherwise part of R and D is provided by the default behaviour of defaultdicts, so can be omitted 
def update_P_R_D(items, N_sas, R_sum, D_sum, states, P, R, D, M=1):
    for (s, a), total in items:
        if is_known(total, M):
            P[(s, a)] = {
                sp: N_sas[(s, a, sp)] / total
                for sp in states
                if N_sas[(s, a, sp)] > 0 # keep P as sparse as possible
            }
            R[(s, a)] = R_sum[(s, a)] / total
            D[(s, a)] = 1 if D_sum[(s, a)] / total > 0.5 else 0
    return P, R, D

def compute_P_R_D(N_sa, N_sas, R_sum, D_sum, states, M=1):
    P = defaultdict(dict)
    R = defaultdict(lambda: 1.0)    # R-MAX fallback
    D = defaultdict(int)            # defaults to 0
    return update_P_R_D(N_sa.items(), N_sas, R_sum, D_sum, states, P, R, D, M)

def create_mdp(model, actions, transitions, log_dir, M=1):
    st = time.time()
    log(log_dir, "\tCreating MDP...", console_log=True)
    processed_transitions = discretized_extract_and_batch(model, transitions)

    N_sa = Counter()
    N_sas = Counter()
    R_sum = Counter()
    D_sum = Counter()
    states = set()

    for s, a, sp, r, d in processed_transitions:
        a, r, d = a.item(), r.item(), d.item() # tensor -> value
        N_sa[(s, a)] += 1
        N_sas[(s, a, sp)] += 1
        R_sum[(s, a)] += r

        if d:
            D_sum[(s, a)] += 1

        states.update([s, sp])

    P, R, D = compute_P_R_D(N_sa, N_sas, R_sum, D_sum, states, M)

    # add self loop to unknown (s, a)-pairs
    for s in states: # observed discritized states
        for a in actions: # full action space from env
            if (s, a) not in P.keys():
                P[(s, a)] = {s: 1.0}

    if DEBUGGER.get_mode():
        validate_transition_probabilities(P, tolerance=1e-6, log_dir=None, console_log=DEBUGGER.get_mode())
    log(log_dir, f"\tMDP created in {time.time() - st:.4f}", console_log=True)

    return {
        'N_sa': N_sa,           # Count of observed (s, a)-pairs
        'N_sas': N_sas,         # Count of observed (s, a, s')-pairs
        'R_sum': R_sum,         # Total reward for all observed (s, a)-pairs
        'D_sum': D_sum,         # Total number of observed s, a)-pairs leading to a terminal state
        'P': P,                 # Estimated P(s'|s, a)
        'R': R,                 # Estimated R(s, a)
        'D': D,                 # Estimation of whether (s, a) -> terminal state
        'states': states,       # Observed discretized states 
        'actions': actions      # Iterable containing all possible actions in env
    }

def update_mdp(mdp, model, transitions, log_dir, M=1):
    st = time.time()
    log(log_dir, "\tUpdating MDP...", console_log=DEBUGGER.get_mode())
    updated_sa = set()
    processed_transitions = discretized_extract_and_batch(model, transitions)

    for s, a, sp, r, d in processed_transitions:
        a, r, d = a.item(), r.item(), d.item()
        mdp['N_sa'][(s, a)] += 1
        mdp['N_sas'][(s, a, sp)] += 1
        mdp['R_sum'][(s, a)] += r
        if d:
            mdp['D_sum'][(s, a)] += 1
        mdp['states'].update([s, sp])

        updated_sa.add((s, a))

    items = [((s, a), mdp['N_sa'][(s, a)]) for (s, a) in updated_sa]
    mdp['P'], mdp['R'], mdp['D'] = update_P_R_D(items, mdp['N_sas'], mdp['R_sum'], mdp['D_sum'], mdp['states'], mdp['P'], mdp['R'], mdp['D'], M)

    if DEBUGGER.get_mode():
        validate_transition_probabilities(mdp['P'], tolerance=1e-6, log_dir=None, console_log=DEBUGGER.get_mode())
    log(log_dir, f"\tMDP update completed in: {time.time() - st:.4f}", console_log=DEBUGGER.get_mode())

def VI(P, R, states, actions, log_dir, V=defaultdict(float), gamma=0.99, max_iterations=10000, tol=1e-6, max_patience=10):
    ast = time.time()
    log(log_dir, "\tDoing Value Iteration...", console_log=DEBUGGER.get_mode())
    Q = defaultdict(float)
    pi = {}
    patience = 0
    prev_delta = float('inf')

    for i in range(max_iterations):
        st = time.time()
        delta = 0
        for s in states:
            q_max = float('-inf')
            best_a = None
            for a in actions:
                q = sum([p * (R[(s, a)] + (gamma * V[sp])) for sp, p in P[(s, a)].items()])
                if q > q_max:
                    q_max = q
                    best_a = a
                Q[(s, a)] = q
            delta = max(delta, abs(q_max - V[s]))
            V[s] = q_max
            pi[s] = best_a

        log(log_dir, f"\t\tVI - Round: {i}, Delta: {delta}, Target: {tol}, Duration: {time.time() - st:.4f}", console_log=DEBUGGER.get_mode())
        if delta < tol or patience >= max_patience: # convergence check
            break

        if (delta == prev_delta):
            patience += 1

        prev_delta = delta

    log(log_dir, f"\tVI completed in: {time.time() - ast:.4f}", console_log=DEBUGGER.get_mode())
    return pi, V

def select_action_eval(model, action_space, pi, s):
    model.eval()
    ds = discretize(model, s)
    if ds in pi:
        return pi.get(ds)
    else:
        return action_space.sample()

def eval_planner(model, pi, args, video, seed, device, epoch, log_dir):
    model.eval()
    log(log_dir, "\tEvaluating Model...", console_log=True)
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
        log(log_dir, f"\t\tSteps Taken: {steps}, Lives: {lives}, Total Reward: {total_reward}, Duration: {time.time() - st:.4f}", console_log=DEBUGGER.get_mode())
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
    log(log_dir, f"\tEvaluation completed in: {time.time() - st:.4f}, Total Reward: {total_reward}", console_log=DEBUGGER.get_mode())
    return total_reward

def select_action(model, action_space, pi, s):
    model.eval()
    global eps_threshold
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if random.random() > eps_threshold:
        ds = discretize(model, s)
        if ds in pi:
            return pi.get(ds)
        else:
            return action_space.sample()
    else:
        return action_space.sample() # random eps-greedy action
    
def collect_transitions(model, env, pi, memory, num_transitions, device, log_dir):
    ast = time.time()
    log(log_dir, "\tCollecting Transitions...", console_log=True)
    model.eval()
    global eps_threshold
    global steps_done

    transitions = []
    total_reward_list = []
    
    while True:
        s, info = env.reset() # (84, 84)
        s = torch.from_numpy(s).to(device) # (84, 84)
        frame_stack = deque([s] * 4, maxlen=4) # (4, 84, 84)

        steps = 0
        st = time.time()
        total_reward = 0

        while True:
            s = torch.stack(list(frame_stack), dim=0).unsqueeze(0) # (1, 4, 84, 84)
            a = select_action(model, env.action_space, pi, s)
            sp, r, term, trun, info = env.step(a)
            sp, a, r, d = convert_to_tensor(sp, a, r, trun, term, device)
            frame_stack.append(sp)
            sp = torch.stack(list(frame_stack), dim=0).unsqueeze(0)  # (1, 4, 84, 84)

            memory.append(s, a, sp, r, d)
            transitions.append(Transition(s, a, sp, r, d)) # store transitions for MDP update
            total_reward += r.item()
            lives = info["lives"]
            steps += 1

            log(log_dir, f"\t\t\tSteps Taken: {steps}, Lives: {lives}, Epsilon: {eps_threshold}, Reward: {r.item()}, Elapsed Time: {time.time() - st:.4f}",console_log=DEBUGGER.get_mode())

            if term or trun:
                if info["lives"] == 0:
                    break
                else:
                    s, info = env.reset()
                    s = torch.from_numpy(s).to(device)
                    frame_stack = deque([s] * 4, maxlen=4)

        log(log_dir, f"\t\tEpisode completed in: {time.time() - st:.4f}, Steps Taken: {steps}, Epsilon: {eps_threshold}, Total Reward: {total_reward}", console_log=DEBUGGER.get_mode())
        
        total_reward_list.append(total_reward)

        if len(transitions) >= num_transitions:
            break

    log(log_dir, f"\tTransitions collected in: {time.time() - ast:.4f}", console_log=DEBUGGER.get_mode())

    return transitions, total_reward_list[:num_transitions] # return transitions for MDP update