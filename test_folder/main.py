import torch
import time
from utils import MemoryBuffer, VideoRecorder, VC
from utils import create_log_dir, create_argparser, log
from training import warmup, initial_model_training, additional_model_training, eval_planner, collect_transitions
from plotting import plot_model_loss, plot_planner_reward, plot_N_sa_histogram, plot_usage_log, plot_N_sa_heatmap
from mdp import MDP

# SEEDS = [862559, 454354, 737532, 275105, 523498]
SEEDS = [275105]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARGS = create_argparser()
if ARGS.debug:
    VC.enable()
LOG_DIR, LOG_PATH = create_log_dir(ARGS, SEEDS)

def append_loss(target_list, loss_list):
    for target, values in zip(target_list, loss_list):
        target += values

def main():
    for seed in SEEDS:
        log(LOG_DIR, f"seed: {seed}, arguments: {ARGS}".replace("Namespace", ""))
        usage_log, recon_loss_list, vq_loss_list, eval_reward_list, episode_reward_list = [], [], [], [], []
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR)
        warmup(ARGS.env_name, memory, DEVICE, LOG_DIR, ARGS.warmup)
        model, optimizer, loss = initial_model_training(memory, ARGS, 0, seed, LOG_DIR, DEVICE, usage_log)
        append_loss([recon_loss_list, vq_loss_list, eval_reward_list, episode_reward_list], loss)
        mdp = MDP(model, DEVICE, LOG_DIR, min_visits=ARGS.min_visits)

        transitions = memory.get_all()
        for epoch in range(ARGS.epoch):
            st = time.time()
            mdp.update(transitions)
            mdp.solve()
            if epoch % ARGS.eval_cycle == 0:
                eval_reward_list.append(eval_planner(mdp, ARGS, video, seed, DEVICE, epoch, LOG_DIR))
                plot_N_sa_histogram(mdp.N_sa, LOG_DIR, epoch, seed)
                plot_N_sa_heatmap(mdp, epoch, LOG_DIR, seed)
            transitions = collect_transitions(mdp, ARGS.env_name, memory, ARGS, DEVICE, LOG_DIR, episode_reward_list, num_envs=6, disable_eps_greedy=False)
            log(LOG_DIR, f"Epoch: {epoch}, Duration: {time.time() - st:.4f}", console_log=True, show_steps=True, show_eps=True, show_codebook_usage=True, show_transition_percentage=True)

            if epoch % ARGS.VQVAE_cycle == 0:
                additional_model_training(model, optimizer, memory, ARGS, epoch, seed, LOG_DIR, usage_log)

            # if epoch % ARGS.MDP_cycle == 0:
            #     mdp = MDP(model, DEVICE, LOG_DIR, min_visits=ARGS.min_visits)
            #     transitions = memory.get_all()

        plot_model_loss(recon_loss_list, vq_loss_list, seed, LOG_DIR)
        plot_planner_reward(episode_reward_list, eval_reward_list, seed, LOG_DIR)
        plot_usage_log(usage_log)

if __name__ == "__main__":
    main()

# TODO: Do pooling in encoder / adapt decoder to deal with this 
# TODO: Evaluate over multiple seeds

# env_name='breakout', lr=0.0002, epoch=2500, batch_size=32, eval_cycle=10, transitions=2500, VQVAE_cycle=1000, MDP_cycle=1000, max_iterations=3, initial_iterations=5, warmup=10000, min_visits=25, debug=False)
# - try min_visits = 15 / 10 / 5 (25 explored very slowly)
# - try VQVAE og MDP cycle = 100 ?
