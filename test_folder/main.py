import torch
import time
from collections import defaultdict
from utils import MemoryBuffer, VideoRecorder, VC
from utils import create_log_dir, create_argparser, create_env, log
from training import warmup, initial_model_training, additional_model_training, eval_planner, collect_transitions
from plotting import plot_model_loss, plot_planner_reward, plot_N_sa_histogram, plot_usage_log
from mdp import create_mdp, update_mdp, VI

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
        V = defaultdict(float)
        usage_log, recon_loss_list, vq_loss_list, eval_reward_list, episode_reward_list = [], [], [], [], []
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR)
        env, action_space, _, _  = create_env(ARGS.env_name, seed, video=False) # env, action_space, s, info
        warmup(env, memory, DEVICE, LOG_DIR, ARGS.warmup)
        model, optimizer, loss = initial_model_training(memory, ARGS, 0, seed, LOG_DIR, DEVICE, usage_log)
        append_loss([recon_loss_list, vq_loss_list, eval_reward_list, episode_reward_list], loss)
        mdp = create_mdp(model, range(action_space.n), memory.get_all(), LOG_DIR, M=1)

        for epoch in range(ARGS.epoch):
            st = time.time()
            pi, V = VI(mdp['P'], mdp['R'], mdp['states'], mdp['actions'], LOG_DIR, V=V, s_max=mdp['s_max'], R_max=mdp['R_max'])
            plot_N_sa_histogram(mdp['N_sa'], LOG_DIR, epoch, seed)
            if epoch % ARGS.eval_cycle == 0:
                eval_reward_list.append(eval_planner(model, pi, ARGS, video, seed, DEVICE, epoch, LOG_DIR))
            update_mdp(mdp, model, collect_transitions(model, ARGS.env_name, pi, memory, ARGS.transitions, DEVICE, LOG_DIR, episode_reward_list), LOG_DIR, M=ARGS.min_visits)
            log(LOG_DIR, f"Epoch: {epoch}, Duration: {time.time() - st:.4f}", console_log=True, show_steps=True, show_eps=True, show_codebook_usage=True, show_transition_percentage=True)

        plot_model_loss(recon_loss_list, vq_loss_list, seed, LOG_DIR)
        plot_planner_reward(episode_reward_list, eval_reward_list, seed, LOG_DIR)
        plot_usage_log(usage_log)

if __name__ == "__main__":
    main()

# if epoch % ARGS.VQVAE_cycle == 0:
    # additional_model_training(model, optimizer, memory, ARGS, epoch, seed, LOG_DIR, usage_log)	
# if epoch % ARGS.MDP_cycle == 0:
    # mdp = create_mdp(model, actions=range(action_space.n), transitions=memory.get_all(), log_dir=LOG_DIR, M=ARGS.min_visits)
# update_mdp(mdp, model, transitions, LOG_DIR, M=ARGS.min_visits)
# pi, V = VI(mdp['P'], mdp['R'], mdp['states'], mdp['actions'], LOG_DIR, V=V, s_max=mdp['s_max'], R_max=mdp['R_max'])


    # TODO: Drastically reduce min_visits 100 -> 5                          (CURRENTLY TRYING THIS)

    # TODO: Collect more frames before updating the mdp try collecting 5k   (CURRENTLY TRYING THIS)

    # TODO: Lower latent_dim 16 or 8 for better generalization but less expressiveness
    
    # TODO: Lower commitment loss 0.25 -> 0.1 or 0.05 encourage exploration
    

    # NOTE: python main.py --warmup 10000 --eval-cycle 10 --VQVAE-cycle 50 --MDP-cycle 50 --min-visit 5 --epoch 2500 --debug --transitions 5000 --max-iterations 5000
    # NOTE: python main.py --warmup 10000 --eval-cycle 5 --VQVAE-cycle 10 --MDP-cycle 10 --min-visit 5 --epoch 2500 --debug --transitions 5000 --max-iterations 3000
    # NOTE: python main.py --warmup 10000 --eval-cycle 5 --VQVAE-cycle 10 --MDP-cycle 30 --min-visit 3 --epoch 2500 --debug --transitions 5000 --max-iterations 3000

    # NOTE: Try this next python main.py --warmup 20000 --transitions 10000 --epoch 2500 --VQVAE-cycle 10 --MDP-cycle 20 --eval-cycle 10 --min-visits 3 --max-iterations 3000 --initial-iterations 10000 --batch-size 32 --lr 2e-4 --debug