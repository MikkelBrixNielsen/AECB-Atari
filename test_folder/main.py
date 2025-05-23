import torch
import time
from utils import MemoryBuffer, VideoRecorder, VC
from utils import create_log_dir, create_argparser, append_loss, log
from training import warmup, initial_model_training, additional_model_training, eval_planner, collect_transitions
from plotting import plot_model_loss, plot_planner_reward, plot_episodic_reward, plot_N_sa_histogram, plot_usage_log, plot_N_sa_heatmap
from mdp import MDP
from model import VQVAE

SEEDS = [862559, 454354, 737532, 275105, 523498]
# SEEDS = [275105]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARGS = create_argparser()
if ARGS.debug:
    VC.enable()
LOG_DIR, LOG_PATH = create_log_dir(ARGS, SEEDS)

def main():
    log(LOG_DIR, f"seeds: {SEEDS}, arguments: {ARGS}".replace("Namespace", ""))
    usage_log, recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list, eval_reward_list, episode_reward_list = [], [], [], [], [], [], []
    memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR)
    warmup(ARGS.env_name, memory, DEVICE, LOG_DIR, ARGS.warmup)
    model, optimizer, loss = initial_model_training(memory, ARGS, 0, LOG_DIR, DEVICE, usage_log)
    model_frozen = VQVAE().to(DEVICE)
    append_loss([recon_loss_list, vq_loss_list, entropy_bonus_list, usage_penalty_list], loss)
    
    # mdp = MDP(model, DEVICE, LOG_DIR, min_visits=ARGS.min_visits)
    # transitions = memory.get_all()

    for epoch in range(ARGS.epoch):
        st = time.time()

        if epoch % ARGS.MDP_cycle == 0:
            model_frozen.load_state_dict(model.state_dict())
            mdp = MDP(model_frozen, DEVICE, LOG_DIR, min_visits=ARGS.min_visits)
            transitions = memory.get_all()

        mdp.update(transitions)
        mdp.solve()

        if epoch % ARGS.eval_cycle == 0:
            eval_reward_list.append(eval_planner(mdp, ARGS, video, SEEDS, DEVICE, epoch, LOG_DIR))
            plot_N_sa_histogram(mdp.N_sa, LOG_DIR, epoch)
            plot_N_sa_heatmap(mdp, epoch, LOG_DIR)

        transitions = collect_transitions(mdp, ARGS.env_name, memory, ARGS, DEVICE, LOG_DIR, episode_reward_list, num_envs=ARGS.num_envs, disable_eps_greedy=False)
        
        if epoch % ARGS.VQVAE_cycle == 0:
            loss = additional_model_training(model, optimizer, memory, ARGS, epoch, LOG_DIR, usage_log, newest_half=True)
            append_loss([recon_loss_list, vq_loss_list, entropy_bonus_list, usage_penalty_list], loss)
        
        log(LOG_DIR, f"Epoch: {epoch}, Duration: {time.time() - st:.4f}", console_log=True, show_steps=True, show_eps=True, show_codebook_usage=True, show_transition_percentage=True, show_training_data=True)

    plot_model_loss(recon_loss_list, vq_loss_list, usage_penalty_list, entropy_bonus_list, LOG_DIR)
    plot_episodic_reward(episode_reward_list, LOG_DIR)
    plot_planner_reward(list(zip(*eval_reward_list)), LOG_DIR, [str(seed) for seed in SEEDS])
    plot_usage_log(usage_log, LOG_DIR)

if __name__ == "__main__":
    main()


# TODO: Plot evaluation as a funciton of VC.GD_steps_done (i.e. gradient steps done) ????

# Most promissing
# Model params: latent_dim=4, num_embeddings=8, hidden_channels=64, commitment_cost=0.4
                # Encoder 84x84 -> 11x11, decoder 11x11 -> 84x84
# Program arguments: (env_name='breakout', lr=0.0002, epoch=2500, batch_size=32, eval_cycle=5, transitions=5000, episodes=2500, VQVAE_cycle=5, MDP_cycle=1000, max_iterations=2000, initial_iterations=15000, warmup=10000, min_visits=1, num_envs=6, debug=False)