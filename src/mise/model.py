"""Two-tower retrieval model: separate MLP encoders for users and items,
trained so that a user's embedding sits close (cosine) to embeddings of
recipes they'd actually cook, and far from everything else."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, embed_dim: int = 32):
        super().__init__()
        self.user_tower = Tower(user_dim, embed_dim)
        self.item_tower = Tower(item_dim, embed_dim)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor):
        return self.user_tower(user_x), self.item_tower(item_x)

    def in_batch_softmax_loss(self, user_x: torch.Tensor, item_x: torch.Tensor, temperature: float = 0.07):
        """Sampled softmax over in-batch negatives (standard two-tower
        training recipe — every other item in the batch acts as a free
        negative for a given user's positive item)."""
        user_emb, item_emb = self(user_x, item_x)
        logits = user_emb @ item_emb.T / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)
