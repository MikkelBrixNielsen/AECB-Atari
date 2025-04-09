import torch
from model import VQVAE
from torch import optim
from utils import MDPBuilder, MemoryBuffer, VideoRecorder, make_log_dir, create_argparser, create_env, warmup, train_VQ_VAE, eval_planner, value_iteration, plot_runs

args = create_argparser()

# some hyperparameters
GAMMA = 0.99
THETA = 1e-6
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50000
WARMUP = 1000

# global variables 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If GPU is available use it - otherwise use the CPU
LOG_DIR, LOG_PATH = make_log_dir(args)


def main():
    # seeds = [834920, 174635, 908172, 562349, 310786]
    seeds = [834920]


    # runs = [] # list of runs 
    for i in range(len(seeds)):
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR) 
        model = VQVAE()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        env, n_action, _, _  = create_env(args.env_name, seeds[i], video=False)
        warmup(env, memory, seeds[i], DEVICE, WARMUP)

        # repeat FIXME: MAKE THIS A PART OF THE TRAINING LOOP
        train_VQ_VAE(model, memory, optimizer, args)
        mdp = MDPBuilder(model.encoder, model.quantizer).build(memory.get_all())
        _, pi = value_iteration(mdp, GAMMA, THETA)
        eval_planner(model, pi, args.env_name, n_action, seeds[i], memory, video, DEVICE)

        # collect data for run and add it to list of runs
    # plot data for list of runs
    # plot_runs(runs)
    
if __name__ == "__main__":
    main()


# TODO: logging of training for planner
# TODO: logging of traininig for VQ-VAE
# TODO: collect data to plot reward, loss, avg reward and avg loss for each seeded run
# TODO: test shit and see if it works
# TODO: sikkert rette en masse fejl :P