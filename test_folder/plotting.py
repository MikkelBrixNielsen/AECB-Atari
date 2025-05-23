import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns # type:ignore
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from utils import VC, log, sample_memory

def plot_input_vs_recon(model, memory, args, epoch, log_dir):
    recon = None
    with torch.no_grad():
        state_batch, _, next_state_batch, _, _ = sample_memory(memory, args.batch_size)
        batch = torch.cat([state_batch, next_state_batch], dim=0) # (bs*2, 4, 84, 84)
        recon, _ = model(batch)

    batch = batch[:8]  # first 8 samples
    recon = recon[:8]

    _, axs = plt.subplots(8, 8, figsize=(12, 12))  # 8 images × 4 channels × 2 (input+recon)
    for i in range(8):  # for each image
        for j in range(4):  # for each frame
            axs[i, j].imshow(batch[i, j].cpu(), cmap='gray')
            axs[i, j].set_title(f'In {j}')
            axs[i, j].axis('off')

            axs[i, j+4].imshow(recon[i, j].detach().cpu(), cmap='gray')
            axs[i, j+4].set_title(f'Out {j}')
            axs[i, j+4].axis('off')

    path = os.path.join(f'{log_dir}', 'log_reconstruction_images')
    if not os.path.exists(path):
        os.makedirs(path)

    plt.tight_layout()
    plt.savefig(f"{path}/epoch_{epoch}")
    plt.close()

def plot_multiple_series(data_lists, log_dir, labels=None, title='Plot', xlabel='X', ylabel='Y', steps=None, figsize=(8, 6), log_scale=False):
    plt.figure(figsize=figsize)
    
    for i, data in enumerate(data_lists):
        label = labels[i] if labels and i < len(labels) else f'Series {i + 1}'
        step = steps[i] if steps and i < len(steps) else 1
        x = [n*step for n in range(len(data))]
        plt.plot(x, data, label=label, alpha=0.6)
    
    plt.title(title)
    if log_scale:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/{title}")
    plt.close()

def plot_model_loss(recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list, log_dir):
    total_loss = [sum(t) for t in zip(recon_loss_list, vq_loss_list)]
    plot_multiple_series([recon_loss_list, vq_loss_list, total_loss], log_dir, ["Recon Loss", "VQ Loss", "Total Loss"], "VQ-VAE Loss Curves", "Training Steps", "Loss", log_scale=True)
    plot_multiple_series([usage_penalty_list], log_dir, title="Usage Penalty", xlabel="Training Steps", ylabel="Value")
    plot_multiple_series([entropy_bonus_list], log_dir, title="Entropy Bonus", xlabel="Training Steps", ylabel="Value")

def plot_planner_reward(eval_reward_lists, log_dir, labels):
    plot_multiple_series(eval_reward_lists, log_dir, labels, title="Planner Evaluation Reward", xlabel="Epochs", ylabel="Reward")

def plot_episodic_reward(episode_reward_list, log_dir):
    plot_multiple_series([episode_reward_list], log_dir, title="Planner Episodic Reward", xlabel="Epochs", ylabel="Reward")

def compute_codebook_usage(model, dataset, batch_size=128):
    model.eval()
    usage_counter = Counter()
    loader = DataLoader(TensorDataset(torch.stack(dataset)), batch_size=batch_size) # (4, 84, 84) -> (bs, 4, 84, 84)

    with torch.no_grad():
        for (obs_batch, ) in loader:
            obs_batch = obs_batch.to(next(model.parameters()).device)
            z_e = model.encoder(obs_batch)
            _, _, z_q_indices = model.quantizer(z_e)
            z_q_indices = z_q_indices.view(-1).cpu().numpy()
            usage_counter.update(z_q_indices.tolist())

    return usage_counter

def plot_codebook_usage(model, memory, log_dir, epoch, batch_size=5000, usage_log=None):
    num_codes = model.quantizer.num_embeddings
    dataset = []

    min_size = min(len(memory), batch_size)
    for s, _, sp, _, _ in memory.sample(min_size):
        dataset.append(s.squeeze(0))
        dataset.append(sp.squeeze(0))

    usage_counter = compute_codebook_usage(model, dataset)
    usage = [usage_counter.get(i, 0) for i in range(num_codes)]
    used_codes = sum(1 for count in usage if count > 0)
    VC.codebook_usage = f"Codebook usage: {used_codes} / {num_codes}"
    log(log_dir, f"\t" + VC.codebook_usage, console_log=VC.debug_mode, no_log=True)

    # Track over epochs
    if usage_log is not None:
        usage_log.append((epoch, used_codes))

    # Bar plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(num_codes), usage)
    plt.xlabel("Codebook Index")
    plt.ylabel("Usage Count")
    plt.title("VQ-VAE Codebook Usage")
    path = f"{log_dir}/Codebook_Usage/Codebook_Usage_epoch_{epoch}"
    plt.savefig(path)
    plt.close()

def plot_usage_log(usage_log, log_dir):
    plt.figure()
    plt.plot(*zip(*usage_log))
    plt.title("Unique Codebook Indices Used Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Number of Used Codes")
    plt.savefig(f"{log_dir}/codebook_usage_over_time.png") 

def plot_N_sa_histogram(N_sa, epoch, log_dir):
    counts = list(N_sa.values())
    plt.figure(figsize=(10, 5))
    plt.bar(list(range(len(counts))), counts)
    plt.xlabel("Visit Count per (s, a)")
    plt.ylabel("Frequency")
    plt.title("Histogram of N_sa Visit Counts")
    path = f"{log_dir}/N_sa_Histograms/N_sa_Histogram_epoch_{epoch}"
    plt.savefig(path)
    plt.yscale('log')
    plt.close()

def plot_N_sa_heatmap(mdp, epoch, log_dir):
    N_sa_matrix = np.zeros((mdp.num_states, mdp.num_actions))
    for (s, a), count in mdp.N_sa.items():
        if s not in mdp.state2idx:
            log(log_dir, "[WARNING], (s, a)-pair not in mdp.state2idx", console_log=True, no_log=True)
            continue
        i = mdp.state2idx[s]
        N_sa_matrix[i, a] = count
    plt.figure(figsize=(10, 6))
    sns.heatmap(N_sa_matrix, cmap='viridis', xticklabels=[i for i in range(mdp.num_actions)], yticklabels=False)
    plt.xlabel("Action")
    plt.ylabel("Latent Code Index (z)")
    plt.title(f"N_sa Visit Counts - Epoch {epoch}")
    plt.tight_layout()
    path = f"{log_dir}/N_sa_HeatMaps/N_sa_heatmap_epoch_{epoch}.png"
    plt.savefig(path)
    plt.close()

def plot_everything(mdp, epoch, recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list, episode_reward_list, eval_reward_list, usage_log, log_dir, SEEDS):
    plot_N_sa_histogram(mdp.N_sa, epoch, log_dir)
    plot_N_sa_heatmap(mdp, epoch, log_dir)
    plot_model_loss(recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list, log_dir)
    plot_episodic_reward(episode_reward_list, log_dir)
    plot_planner_reward(list(zip(*eval_reward_list)), log_dir, [str(seed) for seed in SEEDS])
    plot_usage_log(usage_log, log_dir)