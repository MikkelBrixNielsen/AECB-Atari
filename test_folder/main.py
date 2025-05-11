import torch
from model import VQVAE
from torch import optim
from utils import MemoryBuffer, VideoRecorder, DEBUGGER
from utils import make_log_dir, create_argparser, create_env, warmup, train_model_and_plot, eval_planner, create_mdp, update_mdp, VI, collect_transitions, plot_model_loss, plot_planner_reward, log
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
        recon_loss_list, vq_loss_list = [], []
        eval_reward_list, episode_reward_list = [], []
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR)
        env, action_space, _, _  = create_env(ARGS.env_name, seed, video=False) # env, action_space, s, info
        model = VQVAE().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=ARGS.lr)

        warmup(env, memory, seed, DEVICE, LOG_DIR, ARGS.warmup)

        for epoch in range(ARGS.epoch):
            st = time.time()
            if epoch % ARGS.retrain_cycle == 0:
                lrec, lvq = train_model_and_plot(model, memory, optimizer, ARGS, epoch, seed, LOG_DIR)
                mdp = create_mdp(model, actions=range(action_space.n), transitions=memory.get_all(), log_dir=LOG_DIR, M=ARGS.min_visits)
                recon_loss_list += lrec
                vq_loss_list += lvq

            pi, V = VI(mdp['P'], mdp['R'], mdp['states'], mdp['actions'], LOG_DIR, V=V)
            transitions, episode_rewards = collect_transitions(model, env, pi, memory, ARGS.transitions, DEVICE, LOG_DIR)
            episode_reward_list += episode_rewards
            update_mdp(mdp, model, transitions, LOG_DIR, M=ARGS.min_visits)

            if epoch % ARGS.eval_cycle == 0:
                eval_reward = eval_planner(model, pi, ARGS, video, seed, DEVICE, epoch, LOG_DIR)
                eval_reward_list.append(eval_reward)
            
            log(LOG_DIR, f"Epoch: {epoch}, Duration: {time.time() - st}", console_log=True, show_steps=True, show_eps=True)

        plot_model_loss(recon_loss_list, vq_loss_list, seed, LOG_DIR)
        plot_planner_reward(episode_reward_list, eval_reward_list, seed, LOG_DIR)
    
if __name__ == "__main__":
    main()

    # NOTE: Try and increase the number of training cycles the VQVAE is allowed to do each epoch