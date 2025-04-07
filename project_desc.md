# Project 2: Learning to play Atari games on a discrete latent space

Devise an algorithm that learns to play Atari games from raw pixel observations in two steps:

- i: Avector-quantized auto-encoder that maps observations to a discrete codebook space.

- ii: A planner that applies dynamic programming on a Markov Decision Process defined on the codebook space learned in the previous step.

Evaluate your algorithm on at least three Atari games supported by the Gymnasium API. For vector quantization, you can take [van den Oort et al.,  2017] as basisor design your alternative approach. Replicate your experiments multiple times and report the evolution of the  evaluation-time total reward as a function of the gradient-descent steps. Your report should contain:

- A detailed description of all your design choices including the auto-encoder architecture andits training regime, as well as a technical justification of all these choices.
- The resulting algorithm as a clean and descriptive pseudo-code.
- How well your resulting model is solving the task compared to Deep Q-Learning.
- What the limitations of your own algorithm are and how they can be improved further in future work.

# General Remarks
Train the models as long as your computation resources permit. If a run result is predicted to take unacceptably long, feel free to reduce the maximum number of episodes or the capacities of the used function approximators.
