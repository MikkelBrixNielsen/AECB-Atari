import torch
import time
from utils import VideoRecorder, VALUE_CONTAINER, create_argparser, create_log_dir, log
from training import warmup, initial_model_training, train_model, collect_transitions, eval_planner
from plotting import plot_N_sa_histogram, plot_model_loss, plot_planner_reward, plot_usage_log
from mdp import VectorizedMDP

SEEDS = [248929]
ARGS = create_argparser()
if ARGS.debug:
    VALUE_CONTAINER.enable_debug()
VALUE_CONTAINER.warmup = ARGS.warmup
LOG_DIR, LOG_PATH = create_log_dir(ARGS, SEEDS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO = VideoRecorder(LOG_DIR)

def append_losses(target_lists, loss_lists):
    for target, value in zip(target_lists, loss_lists):
        target.append(value)

def main():
    for seed in SEEDS:
        st = time.time()
        usage_log, recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list, episode_reward_list, eval_reward_list = [], [], [], [], [], [], []
        transitions = warmup(ARGS.env_name, DEVICE, LOG_DIR, num_steps=ARGS.warmup)
        model, optimizer, loss_lists = initial_model_training(transitions, ARGS, 0, SEEDS[0], LOG_DIR, DEVICE, usage_log)
        append_losses([recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list], loss_lists)
        mdp = VectorizedMDP(model, VALUE_CONTAINER.num_actions, LOG_DIR, ARGS.batch_size)

        for epoch in range(ARGS.epoch):
            mdp.update(transitions)
            plot_N_sa_histogram(mdp.N_sa, LOG_DIR, 0, SEEDS[0])
            pi = mdp.solve(max_iters=10000)
            if epoch % ARGS.eval_cycle == 0:
                eval_reward_list.append(eval_planner(mdp, pi, ARGS, VIDEO, seed, DEVICE, epoch, LOG_DIR))
            transitions, episode_rewards = collect_transitions(mdp, ARGS.env_name, pi, ARGS.transitions, DEVICE, LOG_DIR)
            episode_reward_list + episode_rewards
            log(LOG_DIR, f"Epoch: {epoch}, Duration: {time.time() - st:.4f}", console_log=True, show_steps=True, show_eps=True, show_codebook_usage=True, show_transition_percentage=True)
            st = time.time() # reset

            # if epoch % ARGS.VQVAE_cycle == 0:
                # continue training vqvae

            # if epoch % ARGS.MDP_cycle == 0:
                # recreate instantiate MDP
        
        plot_model_loss(recon_loss_list, vq_loss_list, seed, LOG_DIR)
        plot_planner_reward(episode_reward_list, eval_reward_list, seed, LOG_DIR)
        plot_usage_log(usage_log)

if __name__ == "__main__":
    main()

    # TODO: 
        # Increase speed of training
        # Increase speed of collecting warmup episodes
        # Increase speed of episode collection

    # NOTE: MDP-cycle, min-visits, VQVAE-cycle (and by extension max-iterations) - is not used for anything