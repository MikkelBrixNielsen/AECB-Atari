import math
from matplotlib import pyplot as plt
import torch
import torchvision.utils as vutils
from model import VQVAE
from torch import optim
from _utils import MDPBuilder, MemoryBuffer, VideoRecorder
from _utils import make_log_dir, create_argparser, create_env, warmup, train_VQ_VAE, eval_planner, value_iteration, interact_with_env, plot_runs

args = create_argparser()

# some hyperparameters
GAMMA = 0.99
EPS_START = 1-.5
EPS_END = 0.05
EPS_DECAY = 50000
WARMUP = 5000

# global variables 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If GPU is available use it - otherwise use the CPU
LOG_DIR, LOG_PATH = make_log_dir(args)


def main():
    seeds = [834920, 174635, 908172, 562349, 310786]
    episodes = 50 # number of games the newly calculated policy should play before resuming training

    # runs = [] # list of runs 
    #for i in range(len(seeds)):
    for i in range(1):
        total_steps = WARMUP
        memory, video = MemoryBuffer(50000), VideoRecorder(LOG_DIR) 
        env, n_action, _, _  = create_env(args.env_name, seeds[i], video=False)
        warmup(env, memory, seeds[i], DEVICE, WARMUP)

        # DEBUGGING STUFF TO JUST SEE AN INPUT IMAGE
        #lol = torch.stack([state.squeeze(0) for state, *_ in memory.sample(1)]).unsqueeze(1).to(DEVICE)
        #print(lol)
        #lol = lol[:, :, 31:, 5:79] # initial playing area no borders no blocks only ball and paddle yay
        #grid_lol = vutils.make_grid(lol, nrow=2, normalize=True, scale_each=True)
        #plt.figure(figsize=(2,2))
        #plt.imshow(grid_lol.permute(1, 2, 0).cpu().numpy())
        #plt.title("xd")
        #plt.axis('off')
        #plt.show()
        #print("what the flip")
        
        model = VQVAE()
        model.to(DEVICE) # chatGPT said add this - i think it might matter when using GPU and CPU, but if only CPU it prolly makes no difference

        #for epoch in range(args.epoch):
        for epoch in range(100):
            #model = VQVAE()
            #model.to(DEVICE) # chatGPT said add this - i think it might matter when using GPU and CPU, but if only CPU it prolly makes no difference
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            print(f"epoch: {epoch}")
            train_VQ_VAE(model, memory, optimizer, args)
            mdp = MDPBuilder(model.encoder, model.quantizer).build(memory.get_all())
            _, pi = value_iteration(mdp, GAMMA)
            
            #if epoch % args.eval_cycle == 0:
            if epoch % 1 == 0:
                eval_planner(model, pi, args.env_name, n_action, seeds[i], video, DEVICE, epoch, LOG_DIR)
                with torch.no_grad():
                    #batch = torch.stack([state.squeeze(0) for state, *_ in memory.sample(args.batch_size)]).unsqueeze(1).to(DEVICE)
                    batch = torch.stack([state.squeeze(0) for state, *_ in memory.sample(16)]).unsqueeze(1).to(DEVICE)
                    recon, _ = model(batch)

                    # stack batch and reconstructed images (2x16) into a single 8x4 image grid
                    grid = vutils.make_grid(torch.cat([batch, recon], dim=0), nrow=8, normalize=True, scale_each=True)
                    plt.figure(figsize=(12, 6))
                    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                    plt.title("Input (top) vs. Reconstruction (bottom)")
                    plt.axis('off')
                    plt.savefig(f'log_reconstruction_images/epoch_{epoch}')

                    ## 16 batch images in an 4x4 image grid
                    #grid_batch = vutils.make_grid(batch, nrow=4, normalize=True, scale_each=True)
                    #plt.figure(figsize=(12, 6))
                    #plt.imshow(grid_batch.permute(1, 2, 0).cpu().numpy())
                    #plt.title("Input (top) vs. Reconstruction (bottom)")
                    #plt.axis('off')
                    #plt.savefig(f'log_reconstruction_images/epoch_{epoch}_input')

                    ## reconstructed versions of the 16 batch images in an 4x4 image grid
                    #grid_recon = vutils.make_grid(recon, nrow=4, normalize=True, scale_each=True)
                    #plt.figure(figsize=(12, 6))
                    #plt.imshow(grid_recon.permute(1, 2, 0).cpu().numpy())
                    #plt.title("Input (top) vs. Reconstruction (bottom)")
                    #plt.axis('off')
                    #plt.savefig(f'log_reconstruction_images/epoch_{epoch}_reconstructed')
                #model = VQVAE() # reset model after evaluation? idk just for lols
                
            # decaying epsilon threshold for eps-greedy action selection - this encourages early exploration quite heavily
            EPSILON = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY) 

            for _ in range(episodes):
                # total_steps += interact_with_env(model, pi, env, n_action, memory, seeds[i], DEVICE)

                # interact_with_env with epsilon threshold provided
                total_steps += interact_with_env(model, pi, env, n_action, memory, seeds[i], DEVICE, eps_threshold=EPSILON) 

            print(f"mem length: {len(memory)}")
            print(f"Steps taken: {total_steps}")
            print(f"epsilon: {EPSILON:.5f}")

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
# TODO: test with higher/lower num_embeddings
# TODO: normalize input to [-1,1] or [0,1] --> should help make reconstruction losses more consistent, apparently?