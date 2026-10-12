"""EMA vector quantization with bounded search memory and resumable state."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from AutoEncoder.utils import shift_dim


class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False,
                 restart_thres=1.0, distance_chunk_size=1024):
        super().__init__()
        if n_codes <= 0 or embedding_dim <= 0 or distance_chunk_size <= 0:
            raise ValueError('Codebook sizes must be positive')
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.clone())
        self.register_buffer('_initialized', torch.tensor(False))
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.distance_chunk_size = distance_chunk_size

    @property
    def _need_init(self):
        return not bool(self._initialized.item())

    @_need_init.setter
    def _need_init(self, value):
        self._initialized.fill_(not value)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Older checkpoints contain EMA counts but no initialization marker.
        key = prefix + '_initialized'
        if key not in state_dict:
            counts = state_dict.get(prefix + 'N')
            initialized = counts is not None and bool(torch.any(counts > 0))
            state_dict[key] = self._initialized.new_tensor(initialized)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _tile(self, x):
        count, width = x.shape
        if count == 0:
            raise ValueError('Cannot initialize a codebook from an empty latent')
        if count < self.n_codes:
            repeats = (self.n_codes + count - 1) // count
            x = x.repeat(repeats, 1)
            x = x + torch.randn_like(x) * (0.01 / math.sqrt(width))
        return x

    @torch.no_grad()
    def _sample_codes(self, flat_inputs):
        candidates = self._tile(flat_inputs)
        indices = torch.randperm(len(candidates), device=candidates.device)[:self.n_codes]
        codes = candidates[indices].contiguous()
        if dist.is_initialized():
            dist.broadcast(codes, 0)
        return codes

    @torch.no_grad()
    def _init_embeddings(self, z):
        flat_inputs = shift_dim(z.float(), 1, -1).flatten(end_dim=-2)
        codes = self._sample_codes(flat_inputs)
        self.embeddings.copy_(codes)
        self.z_avg.copy_(codes)
        self.N.fill_(1)
        self._initialized.fill_(True)

    @torch.no_grad()
    def _nearest_indices(self, flat_inputs):
        # Exhaustive nearest-neighbour search; only the voxel dimension is split.
        code_norm = self.embeddings.square().sum(dim=1).unsqueeze(0)
        indices = []
        for start in range(0, len(flat_inputs), self.distance_chunk_size):
            block = flat_inputs[start:start + self.distance_chunk_size]
            distances = (block.square().sum(dim=1, keepdim=True)
                         - 2 * block @ self.embeddings.t() + code_norm)
            indices.append(distances.argmin(dim=1))
        return torch.cat(indices)

    @torch.no_grad()
    def _update_embeddings(self, flat_inputs, flat_indices):
        counts = torch.bincount(flat_indices, minlength=self.n_codes).to(self.N.dtype)
        sums = torch.zeros_like(self.z_avg)
        sums.index_add_(0, flat_indices, flat_inputs)
        if dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        self.N.mul_(0.99).add_(counts, alpha=0.01)
        self.z_avg.mul_(0.99).add_(sums, alpha=0.01)
        total = self.N.sum()
        weights = (self.N + 1e-7) / (total + self.n_codes * 1e-7) * total
        self.embeddings.copy_(self.z_avg / weights.unsqueeze(1))
        if not self.no_random_restart:
            replacement = self._sample_codes(flat_inputs)
            usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).to(self.embeddings.dtype)
            self.embeddings.mul_(usage).add_(replacement * (1 - usage))

    def forward(self, z, update_codebook=None):
        if z.ndim != 5 or z.shape[1] != self.embedding_dim or z.numel() == 0:
            raise ValueError('Quantizer expects nonempty [B, embedding_dim, D, H, W] latents')
        update_codebook = self.training if update_codebook is None else bool(update_codebook and self.training)
        # EMA and distance arithmetic stay in FP32 under mixed precision.
        with torch.autocast(device_type=z.device.type, enabled=False):
            if self._need_init and update_codebook:
                self._init_embeddings(z.detach())
            flat_inputs = shift_dim(z.detach().float(), 1, -1).flatten(end_dim=-2)
            flat_indices = self._nearest_indices(flat_inputs)
            encoding_indices = flat_indices.view(z.shape[0], *z.shape[2:])
            embeddings = shift_dim(F.embedding(encoding_indices, self.embeddings), -1, 1)
            commitment_loss = 0.25 * F.mse_loss(z.float(), embeddings.detach())
            counts = torch.bincount(flat_indices, minlength=self.n_codes).float()
            probabilities = counts / flat_indices.numel()
            perplexity = torch.exp(-(probabilities * torch.log(probabilities + 1e-7)).sum())
            if update_codebook:
                self._update_embeddings(flat_inputs, flat_indices)
        embeddings = embeddings.to(z.dtype)
        embeddings_st = (embeddings - z).detach() + z
        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity)

    def dictionary_lookup(self, encodings):
        return F.embedding(encodings, self.embeddings)
