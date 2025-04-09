import torch
from model import VQVAE
from torch import optim
from utils import MDPBuilder, MemoryBuffer, VideoRecorder
from utils import make_log_dir, create_argparser, create_env, warmup, train_VQ_VAE, eval_planner, value_iteration, interact_with_env, plot_runs

args = create_argparser()

# some hyperparameters
GAMMA = 0.99
EPS_START = 1 # NOT USED 
EPS_END = 0.05 # NOT USED
EPS_DECAY = 50000 # NOT USED
WARMUP = 5000

# global variables 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If GPU is available use it - otherwise use the CPU
LOG_DIR, LOG_PATH = make_log_dir(args)


def main():
    seeds = [834920, 174635, 908172, 562349, 310786]

    # runs = [] # list of runs 
    for i in range(len(seeds)):
        total_steps = WARMUP
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR) 
        env, n_action, _, _  = create_env(args.env_name, seeds[i], video=False)
        warmup(env, memory, seeds[i], DEVICE, WARMUP)

        for epoch in range(args.epoch):
            model = VQVAE()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            print(f"epoch: {epoch}")
            train_VQ_VAE(model, memory, optimizer, args)
            mdp = MDPBuilder(model.encoder, model.quantizer).build(memory.get_all())
            _, pi = value_iteration(mdp, GAMMA)
            
            if epoch % args.eval_cycle == 0:
                eval_planner(model, pi, args.env_name, n_action, seeds[i], video, DEVICE, epoch, LOG_DIR)
            
            total_steps += interact_with_env(model, pi, env, n_action, memory, seeds[i], DEVICE)
            print(f"mem length: {len(memory)}")
            print(f"Steps taken: {total_steps}")

            print(f"MAYBE SOME OTHER TRACKING / LOGGING STUFF")

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
# TODO: check for effective codebook usage by assessing distribution of "indicies" (model.quantizer.forward()[2])