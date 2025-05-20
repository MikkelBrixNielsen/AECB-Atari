import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=4, hidden_channels=64, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(                                                 # input: (bs, in_channels, 84, 84)
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1), # (bs, hidden_channels, 42, 42)
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1), # (bs, hidden_channels, 21, 21)
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1), # (bs, hidden_channels, 11, 11)
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim, 5, stride=1, padding=0), # (bs, latent_dim, 7, 7)
        )
        
    def forward(self, x):
        return self.net(x)  # shape: (bs, latent_dim, 7, 7)

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, hidden_channels=64, out_channels=4):
        super().__init__()
        self.net = nn.Sequential(                                                 # input: (bs, latent_dim, 7, 7)
            nn.ConvTranspose2d(latent_dim, hidden_channels, kernel_size=5, stride=1, padding=0), # (bs, hidden_channels, 11, 11)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1), # (bs, hidden_channels, 21, 21)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1), # (bs, hidden_channels, 42, 42)
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1), # (bs, out_channels, 84, 84)
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
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        flat = x_perm.view(-1, self.embedding_dim) # Flatten to (BHW, C)

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embeddings.weight.T
            + self.embeddings.weight.pow(2).sum(1)
        )

        indices = torch.argmin(dist, dim=1)
        quantized = self.embeddings(indices).view(x_perm.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # back to [B, C, H, W]

        e_loss = F.mse_loss(quantized.detach(), x, reduction='sum') # e_loss = ∥ z_e(x) − sg[e] ∥^2 
        q_loss = F.mse_loss(quantized, x.detach(), reduction='sum') # q_loss = ∥ sg[z_e(x)] − e ∥^2

        loss = q_loss + self.commitment_cost * e_loss # q_loss + β * e_loss
        
        quantized = x + (quantized - x).detach() # straight-through estimator

        return quantized, loss, indices.view(x.shape[0], x.shape[2], x.shape[3])

class VQVAE(nn.Module):
    def __init__(self, channels=4, latent_dim=8, num_embeddings=16, hidden_channels=64, commitment_cost=0.4):
        super().__init__()
        self.encoder = Encoder(channels, hidden_channels, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_channels, channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss