"""MSP-DiffNet training on one GPU or torchrun/DDP, without changing REV-VQVAE."""
import argparse
import copy
from datetime import timedelta
import logging
import os
from pathlib import Path
import random
import sys
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset.Singleres_dataset import Singleres_dataset
from ddpm.MSP_DiffNet import MSP_DiffNet, GaussianDiffusion
from ddpm.factory import add_model_arguments, model_config, diffusion_config, codebook_digest
from runtime import load_autoencoder, decode_latent, denormalize_latent
from evaluation.volume_output import save_volume
from sampling_config import (add_sampling_arguments, resolve_sampling_options,
                             decode_options, validate_decode_settings)
from config_io import validate_latent_metadata


@torch.no_grad()
def update_ema(ema, model, decay):
    params = dict(model.named_parameters())
    for name, target in ema.named_parameters():
        target.mul_(decay).add_(params[name], alpha=1-decay)
    buffers = dict(model.named_buffers())
    for name, target in ema.named_buffers():
        target.copy_(buffers[name])


def main(args):
    resolve_sampling_options(args, preview=True)
    if args.sample_sampling_strategy == 'ddim' and not 1 <= args.sample_ddim_steps <= args.timesteps:
        raise ValueError('sample-ddim-steps must lie in [1, timesteps]')
    if not torch.cuda.is_available():
        raise RuntimeError('Diffusion training requires a CUDA GPU')
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if distributed:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    try:
        _train(args, device, rank, world_size, distributed)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def _train(args, device, rank, world_size, distributed):
    seed = args.global_seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result_dir = Path(args.results_dir)
    checkpoint_dir = result_dir / 'checkpoints'
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / 'samples').mkdir(exist_ok=True)
        if not args.ckpt and (checkpoint_dir / 'last.pt').exists():
            raise FileExistsError('This results directory has a checkpoint; use --ckpt or a new directory')
    if distributed:
        dist.barrier()
    logger = logging.getLogger('GLOSSY')
    logger.setLevel(logging.INFO)
    if rank == 0:
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        for handler in (logging.StreamHandler(), logging.FileHandler(result_dir / 'log.txt')):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

    ae = load_autoencoder(args.AE_ckpt, device)
    if args.volume_channels != ae.embedding_dim:
        raise ValueError('volume-channels must match the VAE embedding dimension')
    ratios = tuple(int(v) for v in ae.cfg.model.downsample)
    preview_decoding = decode_options(args, preview=True)
    if not args.disable_ckpt_sampling:
        validate_decode_settings((1, ae.embedding_dim, *args.resolution), ratios, **preview_decoding)
    if args.sub_volume_size is None:
        if any(args.vq_size % r for r in ratios):
            raise ValueError('vq-size must be divisible by the VAE compression ratios')
        args.sub_volume_size = [args.vq_size // r for r in ratios]
    cfg, dcfg = model_config(args), diffusion_config(args)
    network = MSP_DiffNet(**cfg).to(device)
    if any(n % d for n, d in zip(args.resolution, network.spatial_divisor)):
        raise ValueError(f'resolution must be divisible by {network.spatial_divisor}')
    diffusion = GaussianDiffusion(**dcfg).to(device)
    ema = copy.deepcopy(network).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.enable_amp)
    start_epoch, skip_batches, train_steps = 0, 0, 0
    digest = codebook_digest(ae)
    checkpoint = None
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        if checkpoint.get('ae_codebook_sha256', digest) != digest:
            raise ValueError('Resume requires the same VAE codebook')
        if checkpoint.get('model_config', cfg) != cfg or checkpoint.get('diffusion_config', dcfg) != dcfg:
            raise ValueError('Resume model/diffusion arguments differ from the checkpoint')
        weights = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
        network.load_state_dict(weights, strict=True)
        ema.load_state_dict(checkpoint['ema'], strict=True)
        optimizer.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0)
        skip_batches = checkpoint.get('batch_in_epoch', -1) + 1
        train_steps = checkpoint.get('train_steps', 0)
        if not train_steps and Path(args.ckpt).stem.isdigit():
            train_steps = int(Path(args.ckpt).stem)
        if checkpoint.get('world_size', world_size) != world_size:
            raise ValueError('Resume requires the same world size for sampler and RNG restoration')

    dataset = Singleres_dataset(args.data_path, resolution=args.resolution, class_id=getattr(args, 'class_id', 0))
    if max(label for label, _ in dataset.all_files) >= args.num_classes:
        raise ValueError('Dataset class IDs exceed num-classes')
    records = [entry for entry in dataset.roots.values() if isinstance(entry, dict)]
    validate_latent_metadata(records + list(dataset.latent_metadata.values()), digest, ratios)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True)
    model = DDP(network, device_ids=[device.index]) if distributed else network
    model.train()
    if checkpoint and 'rng_states' in checkpoint:
        rng = checkpoint['rng_states'][rank]
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        torch.cuda.set_rng_state(rng['cuda'], device)
    logger.info('Training %d volumes; batch_size=%d per GPU, world_size=%d', len(dataset), args.batch_size, world_size)
    running, log_steps, started = 0., 0, time()

    def save_checkpoint(epoch, batch):
        rng = dict(python=random.getstate(), numpy=np.random.get_state(),
                   torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
        rngs = [None] * world_size
        if distributed:
            dist.all_gather_object(rngs, rng)
        else:
            rngs[0] = rng
        if rank == 0:
            state = dict(model=network.state_dict(), ema=ema.state_dict(), opt=optimizer.state_dict(),
                         scaler=scaler.state_dict(), args=vars(args), epoch=epoch,
                         batch_in_epoch=batch, train_steps=train_steps,
                         model_config=cfg, diffusion_config=dcfg, ae_codebook_sha256=digest,
                         world_size=world_size, rng_states=rngs, downsample=ratios)
            target = checkpoint_dir / f'{train_steps:07d}.pt'
            torch.save(state, target)
            temporary = checkpoint_dir / 'last.tmp'
            torch.save(state, temporary)
            temporary.replace(checkpoint_dir / 'last.pt')
            logger.info('Saved %s', target)
        if distributed:
            dist.barrier()

    last_epoch, last_batch = start_epoch, skip_batches - 1
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for batch_index, (z, y, res) in enumerate(loader):
            if epoch == start_epoch and batch_index < skip_batches:
                continue
            last_epoch, last_batch = epoch, batch_index
            if z.shape[1] != args.volume_channels:
                raise ValueError('Latent channels do not match volume-channels')
            z, y, res = z.to(device), y.to(device), res.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args.enable_amp):
                t = torch.randint(0, diffusion.num_timesteps, (z.shape[0],), device=device)
                loss = diffusion.p_losses(model, z, t, y=y, res=res)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_steps += 1
            if train_steps % 10 == 0:
                update_ema(ema, network, 0. if train_steps < 2000 else 0.995)
            running += loss.detach().item()
            log_steps += 1
            if train_steps % args.log_every == 0:
                average = torch.tensor(running / log_steps, device=device)
                if distributed:
                    dist.all_reduce(average)
                    average /= world_size
                logger.info('step=%d loss=%.6f steps/s=%.2f', train_steps, average.item(), log_steps / (time()-started))
                running, log_steps, started = 0., 0, time()
            if train_steps % args.ckpt_every == 0:
                # fork_rng keeps optional sampling from changing subsequent training noise.
                if rank == 0 and not args.disable_ckpt_sampling:
                    with torch.no_grad(), torch.random.fork_rng(devices=[device.index]):
                        z = torch.randn(1, args.volume_channels, *args.resolution, device=device)
                        sample = diffusion.sample(ema, z, y=torch.zeros(1, device=device, dtype=torch.long),
                                                  res=z.new_tensor(args.resolution)/64,
                                                  strategy=args.sample_sampling_strategy,
                                                  ddim_steps=args.sample_ddim_steps, eta=args.sample_eta)
                        volume = decode_latent(ae, denormalize_latent(sample, ae), **preview_decoding)
                        save_volume(volume[0], result_dir / 'samples' / f'{train_steps:07d}',
                                    args.sample_save_format, args.sample_raw_dtype,
                                    args.sample_raw_byte_order, args.spacing)
                save_checkpoint(epoch, batch_index)
        skip_batches = 0
    if last_batch >= 0:
        save_checkpoint(last_epoch, last_batch)
    logger.info('Training finished at step %d', train_steps)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument('--data-path', required=True, help='Actual latent directory/file or optional JSON mapping')
    parser.add_argument('--class-id', type=int, default=0, help='Class label for a direct latent directory/file')
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--AE-ckpt', required=True)
    parser.add_argument('--resolution', nargs=3, type=int, default=[16, 16, 16], help='Latent D H W, not output voxel size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--enable-amp', '--enable_amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--ckpt-every', type=int, default=500)
    parser.add_argument('--ckpt')
    parser.add_argument('--model', default='MSP-DiffNet', help='Compatibility label')
    parser.add_argument('--disable-ckpt-sampling', action='store_true')
    add_sampling_arguments(parser, preview=True)
    parser.add_argument('--sample-save-format', choices=['raw', 'nii.gz', 'both'], default='raw')
    parser.add_argument('--sample-raw-dtype', choices=['uint16', 'u16', 'float32', 'f32'], default='uint16')
    parser.add_argument('--sample-raw-byte-order', choices=['little', 'big'], default='little')
    parser.add_argument('--spacing', nargs=3, type=float, default=[1.03, 1.03, 1.03], help='Storage X Y Z spacing in micrometres')
    return parser


def cli():
    args = build_parser().parse_args()
    for name in ('batch_size', 'epochs', 'log_every', 'ckpt_every', 'timesteps'):
        if getattr(args, name) <= 0:
            raise ValueError(f'{name} must be positive')
    if args.num_workers < 0 or args.lr <= 0 or min(args.resolution) <= 0:
        raise ValueError('Check num-workers, learning rate and resolution')
    main(args)


if __name__ == '__main__':
    cli()
