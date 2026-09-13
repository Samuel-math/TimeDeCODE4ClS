"""Standard vector quantization and bidirectional masked-token classification."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PatchVQVAE(nn.Module):
    """One shared, single-level VQ codebook for independent scalar-channel patches."""
    def __init__(self, patch_size=8, dim=64, codes=64):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(nn.Linear(patch_size, 128), nn.GELU(), nn.Linear(128, dim))
        self.decoder = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, patch_size))
        self.codebook = nn.Embedding(codes, dim)
        nn.init.uniform_(self.codebook.weight, -1 / codes, 1 / codes)

    def patches(self, x):
        b, t, c = x.shape
        if t % self.patch_size:
            raise ValueError('Input length must be divisible by patch_size')
        return x.transpose(1, 2).reshape(b, c, t // self.patch_size, self.patch_size)

    def quantize(self, patches):
        z = self.encoder(patches)
        flat = z.reshape(-1, z.shape[-1])
        e = self.codebook.weight
        distance = flat.square().sum(-1, keepdim=True) + e.square().sum(-1) - 2 * flat @ e.T
        ids = distance.argmin(-1).reshape(z.shape[:-1])
        q = self.codebook(ids)
        vq_loss = F.mse_loss(q, z.detach()) + .25 * F.mse_loss(z, q.detach())
        straight_through = z + (q - z).detach()
        return ids, straight_through, vq_loss

    def forward(self, patches):
        ids, q, vq_loss = self.quantize(patches)
        reconstruction = self.decoder(q)
        return reconstruction, vq_loss, ids

    @torch.no_grad()
    def tokens(self, x):
        return self.quantize(self.patches(x))[0]


class MaskedClassifier(nn.Module):
    """Bidirectional encoder; mask ID is never a target code."""
    def __init__(self, codes, dim, patches, channels, classes, layers=2, dropout=.1):
        super().__init__()
        self.codes = codes
        self.embedding = nn.Embedding(codes + 1, dim)
        positions = torch.arange(patches)[:, None]
        scales = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.) / dim))
        pos = torch.zeros(patches, dim)
        pos[:, 0::2] = torch.sin(positions * scales)
        pos[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer('position', pos)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 4, dropout,
                                           batch_first=True, activation='gelu', norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.token_head = nn.Linear(dim, codes)
        self.projection = nn.Linear(dim, 8)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 8, 128),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(128, classes))

    def encode(self, ids):
        b, c, p = ids.shape
        x = self.embedding(ids.reshape(b * c, p)) + self.position[:p]
        return self.norm(self.encoder(x)).reshape(b, c, p, -1)

    def masked_loss(self, ids, ratio, generator=None):
        if not 0 < ratio < 1:
            raise ValueError('mask_ratio must be between zero and one')
        b, c, p = ids.shape
        count = max(1, min(p - 1, round(p * ratio))) if p > 1 else 1
        order = torch.rand(b, c, p, device=ids.device, generator=generator).argsort(-1)
        mask = torch.zeros_like(ids, dtype=torch.bool).scatter_(-1, order[..., :count], True)
        hidden = self.encode(ids.masked_fill(mask, self.codes))
        return F.cross_entropy(self.token_head(hidden[mask]), ids[mask])

    def forward(self, ids):
        hidden = self.encode(ids)
        b, c, p, d = hidden.shape
        pooled = F.adaptive_avg_pool1d(hidden.reshape(b * c, p, d).transpose(1, 2), 4)
        pooled = pooled.transpose(1, 2).reshape(b, c, 4, d)
        return self.classifier(self.projection(pooled))
