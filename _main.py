import math
from matplotlib import pyplot as plt
import torch
import torchvision.utils as vutils
from model import VQVAE
from torch import optim
from _utils import MDPBuilder, MemoryBuffer, VideoRecorder
from _utils import make_log_dir, create_argparser, create_env, warmup, train_VQ_VAE, eval_planner, value_iteration, interact_with_env, sample_memory, plot_runs
import os
from itertools import count
import random


args = create_argparser()

# some hyperparameters
GAMMA       = 0.99
EPS_START   = 1
EPS_END     = 0.1
EPS_DECAY   = 100000
WARMUP      = 1000
MEM_BUFF    = 25000

# global variables 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If GPU is available use it - otherwise use the CPU
LOG_DIR, LOG_PATH = make_log_dir(args)

def plot_input_vs_recon(batch, recon, epoch, dir):
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

    plt.tight_layout()
    plt.savefig(f'{dir}/img_{epoch}')
    plt.close()

def main():
    seeds = [834920, 174635, 908172, 562349, 310786]

    if args.load_model: # LOAD PREVIOUSLY TRAINED MODEL FOR GAMEPLAY (MAINLY FOR TESTING/DEBUGGING PURPOSES)
        model = VQVAE().to(DEVICE)
        model_path = args.load_model
        try:
            model.load_state_dict(torch.load(model_path), map_location=DEVICE)["model_state_dict"]
            print(f"Loaded model {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}\nStarting training from scratch.")

        tests_dir = os.path.join(LOG_DIR, "z_gameplay_results")
        os.makedirs(tests_dir, exist_ok=True)

        total_steps = WARMUP
        memory, video = MemoryBuffer(MEM_BUFF), VideoRecorder(tests_dir)
        seed = seeds[random.randint(0,len(seeds))]
        env, n_action, _, _  = create_env(args.env_name, seed, video=False)
        warmup(env, memory, seed, DEVICE, WARMUP)
        for epoch in count():
            mdp = MDPBuilder(model.encoder, model.quantizer).build(memory.get_all())
            _, pi = value_iteration(mdp, GAMMA)

            EPSILON = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY) 
            total_steps += interact_with_env(model, pi, env, n_action, memory, seed, DEVICE, eps_threshold=EPSILON) 
            
            if epoch % 100 == 0:
                eval_planner(model, pi, args.env_name, n_action, seed, video, DEVICE, epoch, tests_dir)
    else: # TRAIN VQ-VAE
        # runs = [] # list of runs 
        for i in range(len(seeds)):
            models_dir  = os.makedirs(os.path.join(LOG_DIR, f"seed_{seeds[i]}", "models"), exist_ok=True)
            images_dir  = os.makedirs(os.path.join(LOG_DIR, f"seed_{seeds[i]}", "images"), exist_ok=True)
            evals_dir   = os.makedirs(os.path.join(LOG_DIR, f"seed_{seeds[i]}", "evaluation"), exist_ok=True)
            total_steps = WARMUP
            memory, video = MemoryBuffer(MEM_BUFF), VideoRecorder(LOG_DIR) 
            env, n_action, _, _  = create_env(args.env_name, seeds[i], video=False)
            warmup(env, memory, seeds[i], DEVICE, WARMUP)

            model = VQVAE()
            model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(args.epoch):
                print(f"epoch: {epoch}")

                train_VQ_VAE(model, memory, optimizer, args, MAX_ITERATIONS=1000)
                mdp = MDPBuilder(model.encoder, model.quantizer).build(memory.get_all())
                _, pi = value_iteration(mdp, GAMMA)

                if epoch % args.eval_cycle == 0:
                    eval_planner(model, pi, args.env_name, n_action, seeds[i], video, DEVICE, epoch, evals_dir)
                    with torch.no_grad():
                        state_batch, _, next_state_batch, _ = sample_memory(memory, args)
                        batch = torch.cat([state_batch, next_state_batch], dim=0) # (bs*2, 4, 84, 84)
                        recon, _ = model(batch)

                    plot_input_vs_recon(batch, recon, epoch, images_dir)

                    # Save current model
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),  # Optional
                    }, os.path.join(models_dir, f"model_{epoch}.pth"))

                        # # Sample one stacked input
                        # sample = batch[0].detach().cpu()

                        # # Visualize each frame in the stack
                        # import matplotlib.pyplot as plt
                        # fig, axs = plt.subplots(1, 4, figsize=(10, 3))
                        # for q in range(4):
                        #     axs[q].imshow(sample[q], cmap='gray')
                        #     axs[q].set_title(f'Frame {q}')
                        #     axs[q].axis('off')
                        # plt.suptitle("Stacked Input to VQ-VAE")
                        # plt.show()

                #EPSILON = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * total_steps / EPS_DECAY) 
                EPSILON = .25

                for _ in range(args.episodes):
                    total_steps += interact_with_env(model, pi, env, n_action, memory, seeds[i], DEVICE, eps_threshold=EPSILON) 

                print(f"mem length: {len(memory)}",
                      f"Steps taken: {total_steps}", 
                      f"epsilon: {EPSILON:.5f}")

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