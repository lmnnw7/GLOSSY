"""Shared checkpoint, device and latent-scaling helpers."""
import torch
from config_io import PROJECT_ROOT, project_path
from sampling_config import validate_decode_settings


def resolve_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select --device cpu or install the CUDA environment.')
    return torch.device(name)


def load_autoencoder(path, device='cpu'):
    from AutoEncoder.model.REV_VQVAE import REV_VQVAE
    return REV_VQVAE.load_from_checkpoint(
        path, map_location='cpu', initialize_perceptual=False
    ).to(device).eval().requires_grad_(False)


def normalize_latent(z, ae):
    lo, hi = ae.codebook.embeddings.amin(), ae.codebook.embeddings.amax()
    if not torch.isfinite(hi - lo) or hi <= lo:
        raise ValueError('The codebook has a zero or non-finite intensity range')
    return (z - lo) / (hi - lo) * 2 - 1


def denormalize_latent(z, ae):
    lo, hi = ae.codebook.embeddings.amin(), ae.codebook.embeddings.amax()
    return (z + 1) / 2 * (hi - lo) + lo


def decode_latent(ae, latent, mode='full', patch_size=256, sliding_window=1, overlap_ratio=0.5):
    """Global decoding is the manuscript path. Tiling is an explicit alternative."""
    if mode == 'full':
        return ae.decode(latent, quantize=True)
    ratios = tuple(int(v) for v in ae.cfg.model.downsample)
    validate_decode_settings(latent.shape, ratios, mode, patch_size, sliding_window, overlap_ratio)
    ratio = ratios[0]
    if mode == 'gaussian':
        return ae.decode_sliding_gaussian(latent, quantize=True, patch_size=patch_size,
                                          overlap_ratio=overlap_ratio, compress_ratio=ratio)
    return ae.decode_sliding(latent, quantize=True, patch_size=patch_size,
                             sliding_window=sliding_window, compress_ratio=ratio)
