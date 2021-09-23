import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n):
        super(VectorQuantizer, self).__init__()
        self.n = n
        self.embedding = torch.eye(n, device=device)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()

        min_encoding_indices = self.find_closest_encodings_indices(z)

        min_encodings, z_q = self.get_quantized_embeddings(min_encoding_indices)
        z_q = z_q.view(z.shape)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices

    def find_closest_encodings_indices(self, z):
        z_flattened = z.view(-1, self.n)
        min_encoding_indices = z_flattened.argmax(dim=1).unsqueeze(1)
        return min_encoding_indices

    def get_quantized_embeddings(self, encoding_indices):
        min_encodings = torch.zeros(encoding_indices.shape[0], self.n).to(device)
        min_encodings.scatter_(1, encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding)

        return min_encodings, z_q
