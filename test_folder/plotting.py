import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns # type:ignore
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from utils import VC, log, sample_memory


def plot_input_vs_recon(model, memory, args, epoch, log_dir, seed=0000):
    recon = None
    with torch.no_grad():
        state_batch, _, next_state_batch, _, _ = sample_memory(memory, args.batch_size)
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
    loader = DataLoader(TensorDataset(torch.stack(dataset)), batch_size=batch_size) # (4, 84, 84) -> (bs, 4, 84, 84)

    with torch.no_grad():
        for (obs_batch, ) in loader:
            obs_batch = obs_batch.to(next(model.parameters()).device)
            z_e = model.encoder(obs_batch)
            _, _, z_q_indices = model.quantizer(z_e)
            z_q_indices = z_q_indices.view(-1).cpu().numpy()
            usage_counter.update(z_q_indices.tolist())

    return usage_counter

def plot_codebook_usage(model, memory, log_dir, epoch, seed, batch_size=5000, usage_log=None):
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
    path = f"{log_dir}/seed_{seed}/Codebook_Usage_epoch_{epoch}"
    plt.savefig(path)
    plt.close()

def plot_usage_log(usage_log, log_dir, seed):
    plt.plot(*zip(usage_log))
    plt.title("Unique Codebook Indices Used Over Time")
    plt.xlabel("Epochs")
    plt.ylabel("Used Codes")
    plt.savefig(f"{log_dir}/seed_{seed}/codebook_usage_over_time.png") 

def plot_N_sa_histogram(N_sa, log_dir, epoch, seed):
    counts = list(N_sa.values())
    plt.figure(figsize=(10, 5))
    plt.bar(list(range(len(counts))), counts)
    plt.xlabel("Visit Count per (s, a)")
    plt.ylabel("Frequency")
    plt.title("Histogram of N_sa Visit Counts")
    path = f"{log_dir}/seed_{seed}/N_sa_Histogram_epoch_{epoch}"
    plt.savefig(path)
    plt.yscale('log')
    plt.close()

def plot_N_sa_heatmap(mdp, epoch, log_dir, seed):
    N_sa_matrix = np.zeros((mdp.num_states, mdp.num_actions))
    for i, ((s,a), count) in enumerate(mdp.N_sa.items()):
        N_sa_matrix[i, a] = count
    plt.figure(figsize=(10, 6))
    sns.heatmap(N_sa_matrix, cmap='viridis', xticklabels=[i for i in range(mdp.num_actions)], yticklabels=False)
    plt.xlabel("Action")
    plt.ylabel("Latent Code Index (z)")
    plt.title(f"N_sa Visit Counts - Epoch {epoch}")
    plt.tight_layout()
    path = f"{log_dir}/seed_{seed}/N_sa_heatmap_epoch_{epoch}.png"
    plt.savefig(path)
    plt.close()