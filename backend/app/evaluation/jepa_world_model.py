"""
PyTorch I-JEPA Latent World Model Architecture.
Provides non-generative semantic latent prediction in Z-space (384-dim).
Eliminates pixel/token reconstruction overhead for high-speed alignment.
"""
import torch
import torch.nn as nn
from typing import Tuple

class IJEPALatentWorldModel(nn.Module):
    def __init__(self, embed_dim: int = 384, latent_dim: int = 128):
        super().__init__()
        # Context Encoder LayerNorm + GELU
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, latent_dim)
        )
        
        # Predictor LayerNorm + GELU
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        predicted = self.predictor(latent)
        return latent, predicted

    def predict_latent_similarity(self, query_vec: torch.Tensor, chunk_vec: torch.Tensor) -> float:
        """
        Computes L2 Latent Energy Distance between Query context and Target chunk.
        """
        self.eval()
        with torch.no_grad():
            _, pred_query = self.forward(query_vec)
            dist = torch.dist(pred_query, chunk_vec, p=2).item()
            # Convert L2 distance into similarity metric bounded [0, 1]
            sim = 1.0 / (1.0 + dist)
            return float(sim)