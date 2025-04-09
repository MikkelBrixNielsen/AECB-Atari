
# Current implementation is straight up copy paste from asking chatGPT what an VQ-VAE is

import torch 
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),  # 42x42
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),           # 21x21
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=latent_dim, kernel_size=3, stride=1, padding=1)    # 21x21
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=64, kernel_size=4, stride=2, padding=1),  # 42x42
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),          # 84x84
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        # Reshape input
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat = x_perm.view(-1, self.embedding_dim) # Flatten to (BHW, C)

        # Distance computation
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.T
            + self.embeddings.weight.pow(2).sum(1)
        )

        # Vector quantization
        indices = torch.argmin(dist, dim=1)
        quantized = self.embeddings(indices).view(x_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]

        # Computing quantization losses (∥sg[z_e(x)]−e∥^2 + β∥z_e(x)−sg[e]∥^2) 
        # implemented w.r.t the training objective of equation (3) from van den Oord: log p(x∣z_q(x)) + ∥sg[z_e(x)]−e∥^2 + β∥z_e(x)−sg[e]∥^2
        # NOTE this is in the quantization layer, so the reconstruction loss, log p(x∣z_q(x)), is added before backpropagation in train_VQ_VAE
        # NOTE - ".detach()" implements the use of stop-gradient "sg[]" (I THINK, MAYBE VERIFY - FIXME) 
        # NOTE - reduction='sum' makes mse_loss calculate the sum of squared differences (which follows the learning objective) 
        # NOTE - using reduction='mean' (default) provides additional normalization (calculates the mean squared error instead) 
        e_loss = F.mse_loss(quantized.detach(), x, reduction='mean') # e_loss = ∥ z_e(x) − sg[e] ∥^2 
        #e_loss = F.mse_loss(quantized.detach(), x) # uses reduction='mean'
        q_loss = F.mse_loss(quantized, x.detach(), reduction='mean') # q_loss = ∥ sg[z_e(x)] − e ∥^2 
        #q_loss = F.mse_loss(quantized, x.detach()) # uses reduction='mean'
        loss = q_loss + self.commitment_cost * e_loss # q_loss + β * e_loss
        
        # NOTE - apparently this lets gradients flow through the quantization step
        quantized = x + (quantized - x).detach() # straight-through estimator

        # quantized output z_q(x), quantization loss, indicies of chosen codebook vectors
        return quantized, loss, indices 

class VQVAE(nn.Module):
    def __init__(self, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss