import torch
from model import VQVAE
from torch import optim
from utils import MemoryBuffer, VideoRecorder, DEBUGGER
from utils import make_log_dir, create_argparser, create_env, warmup, train_model_and_plot, eval_planner, create_mdp, update_mdp, VI, collect_transitions, plot_model_loss, plot_planner_reward, plot_N_sa_histogram, plot_usage_log, log
from collections import defaultdict
import time

# Hyperparameters
GAMMA = 0.99

# Global variables
ARGS = create_argparser()
if ARGS.debug:
    DEBUGGER.enable()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If GPU is available use it - otherwise use the CPU
# SEEDS = [862559, 454354, 737532, 275105, 523498]
SEEDS = [275105]
LOG_DIR, LOG_PATH = make_log_dir(ARGS, SEEDS)

def main():
    for seed in SEEDS:
        V = defaultdict(float)
        usage_log = []
        recon_loss_list, vq_loss_list = [], []
        eval_reward_list, episode_reward_list = [], []
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR)
        env, action_space, _, _  = create_env(ARGS.env_name, seed, video=False) # env, action_space, s, info
        model = VQVAE().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

        warmup(env, memory, seed, DEVICE, LOG_DIR, ARGS.warmup)

        for epoch in range(ARGS.epoch):
            st = time.time()
            if epoch % ARGS.VQVAE_cycle == 0:
                lrec, lvq = train_model_and_plot(model, memory, optimizer, ARGS, epoch, seed, LOG_DIR, usage_log)
                recon_loss_list += lrec
                vq_loss_list += lvq

            if epoch % 100 == 0:
                model.quantizer.reinitialize_unused_code(min_usage=5)

            if epoch % ARGS.MDP_cycle == 0:
                mdp = create_mdp(model, actions=range(action_space.n), transitions=memory.get_all(), log_dir=LOG_DIR, M=ARGS.min_visits)

            pi, V = VI(mdp['P'], mdp['R'], mdp['states'], mdp['actions'], LOG_DIR, V=V, s_max=mdp['s_max'], R_max=mdp['R_max'])
            transitions, episode_rewards = collect_transitions(model, ARGS.env_name, pi, memory, ARGS.transitions, DEVICE, LOG_DIR)
            episode_reward_list += episode_rewards
            update_mdp(mdp, model, transitions, LOG_DIR, M=ARGS.min_visits)
            
            if (epoch+1) % ARGS.MDP_cycle == 0: # if mdp gets recreated next cycle make histogram of N_sa before this 
                plot_N_sa_histogram(mdp['N_sa'], LOG_DIR, epoch, seed)

            if epoch % ARGS.eval_cycle == 0:
                eval_reward = eval_planner(model, pi, ARGS, video, seed, DEVICE, epoch, LOG_DIR)
                eval_reward_list.append(eval_reward)
            log(LOG_DIR, f"Epoch: {epoch}, Duration: {time.time() - st:.4f}", console_log=True, show_steps=True, show_eps=True, show_codebook_usage=True, show_transition_percentage=True)

        plot_model_loss(recon_loss_list, vq_loss_list, seed, LOG_DIR)
        plot_planner_reward(episode_reward_list, eval_reward_list, seed, LOG_DIR)
        plot_usage_log(usage_log)

if __name__ == "__main__":
    main()

    # TODO: Drastically reduce min_visits 100 -> 5                          (CURRENTLY TRYING THIS)

    # TODO: Collect more frames before updating the mdp try collecting 5k   (CURRENTLY TRYING THIS)

    # TODO: Lower latent_dim 16 or 8 for better generalization but less expressiveness
    
    # TODO: Lower commitment loss 0.25 -> 0.1 or 0.05 encourage exploration
    




    # NOTE: Seem like it did some cool things around epoch 100: python main.py --warmup 10000 --eval-cycle 10 --VQVAE-cycle 50 --MDP-cycle 50 --min-visit 5 --epoch 2500 --debug --transitions 5000 --max-iterations 5000

    # NOTE: previous: python main.py --warmup 10000 --eval-cycle 5 --VQVAE-cycle 10 --MDP-cycle 10 --min-visit 5 --epoch 2500 --debug --transitions 5000 --max-iterations 3000

    # NOTE: Try the following: python main.py --warmup 10000 --eval-cycle 5 --VQVAE-cycle 10 --MDP-cycle 30 --min-visit 3 --epoch 2500 --debug --transitions 5000 --max-iterations 3000

    # NOTE: Try this next python main.py --warmup 20000 --transitions 10000 --epoch 2500 --VQVAE-cycle 10 --MDP-cycle 20 --eval-cycle 10 --min-visits 3 --max-iterations 3000 --initial-iterations 10000 --batch-size 32 --lr 2e-4 --debug
            # If planner is unstable increase --MDP-cycle 20 -> 50
            # If transitions are sparse lower --min_visits 5 -> 3 (Percentage of Transitions Known <20%), and increase --transitions 5000 -> 10000