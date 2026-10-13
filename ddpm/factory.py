"""Construct models consistently for training and generation."""
import hashlib
from ddpm.MSP_DiffNet import MSP_DiffNet, GaussianDiffusion


def codebook_digest(ae):
    return hashlib.sha256(ae.codebook.embeddings.detach().float().cpu().numpy().tobytes()).hexdigest()


def model_config(args):
    config = dict(dim=args.model_dim, dim_mults=tuple(args.dim_mults),
                channels=args.volume_channels, init_kernel_size=3,
                cond_classes=args.num_classes, learn_sigma=False,
                use_sparse_linear_attn=tuple(args.use_attn), vq_size=args.vq_size,
                num_mid_DiT=args.num_dit, patch_size=args.patch_size,
                sub_volume_size=tuple(args.sub_volume_size or (8, 8, 8)))
    for name in ('resnet_groups', 'DiT_num_heads', 'attn_heads',
                 'res_condition', 'cond_dim', 'mlp_ratio', 'init_dim', 'init_kernel_size'):
        if hasattr(args, name):
            config[name] = getattr(args, name)
    if config.get('cond_dim') is not None:
        config['cond_classes'] = None
    return config


def diffusion_config(args):
    return dict(channels=args.volume_channels, timesteps=args.timesteps, loss_type=args.loss_type)


def from_checkpoint(checkpoint, args, device):
    cfg = checkpoint.get('model_config')
    dcfg = checkpoint.get('diffusion_config')
    saved = checkpoint.get('args')
    saved = vars(saved) if hasattr(saved, '__dict__') else saved
    saved = saved or {}
    if cfg is None:
        if saved:
            # Old checkpoints used a fixed 8-voxel local latent sub-volume.
            for name in ('model_dim', 'dim_mults', 'volume_channels', 'num_classes',
                         'use_attn', 'vq_size', 'num_dit', 'patch_size', 'sub_volume_size',
                         'resnet_groups', 'DiT_num_heads', 'attn_heads',
                         'res_condition', 'cond_dim', 'mlp_ratio', 'init_dim', 'init_kernel_size'):
                if name in saved:
                    setattr(args, name, saved[name])
        cfg = model_config(args)
    if dcfg is None:
        for name in ('timesteps', 'loss_type'):
            if name in saved:
                setattr(args, name, saved[name])
        dcfg = diffusion_config(args)
        dcfg['channels'] = cfg['channels']
    model = MSP_DiffNet(**cfg).to(device)
    weights = checkpoint.get('ema') or checkpoint.get('model')
    if weights is None:
        raise KeyError('Diffusion checkpoint must contain ema or model weights')
    weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=True)
    return model.eval(), GaussianDiffusion(**dcfg).to(device), cfg


def add_model_arguments(parser):
    parser.add_argument('--model-dim', type=int, default=72)
    parser.add_argument('--dim-mults', nargs='+', type=int, default=[1, 1, 2, 4, 8])
    parser.add_argument('--use-attn', nargs='+', type=int, default=[0, 0, 0, 1, 1])
    parser.add_argument('--volume-channels', type=int, default=8)
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=1)
    parser.add_argument('--sub-volume-size', nargs=3, type=int, default=[8, 8, 8])
    parser.add_argument('--vq-size', type=int, default=64)
    parser.add_argument('--num-dit', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--loss-type', choices=['l1', 'l2'], default='l1')
