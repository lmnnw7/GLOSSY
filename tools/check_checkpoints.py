"""Strictly load VAE/diffusion checkpoints and optionally test tiny forward paths."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIFFUSION_RESUME_FIELDS = {
    'model', 'ema', 'opt', 'scaler', 'args', 'epoch', 'batch_in_epoch',
    'train_steps', 'world_size', 'rng_states', 'model_config', 'diffusion_config',
    'ae_codebook_sha256',
}


def check(args, report):
    import torch
    from runtime import load_autoencoder, resolve_device, decode_latent
    from ddpm.factory import codebook_digest, from_checkpoint
    device = resolve_device(args.device)
    saved_vae = torch.load(args.ae_ckpt, map_location='cpu', weights_only=False)
    if not isinstance(saved_vae, dict) or 'state_dict' not in saved_vae:
        raise ValueError('VAE must be a Lightning checkpoint with state_dict and model configuration')
    ae = load_autoencoder(args.ae_ckpt, device)
    ratios = tuple(int(value) for value in ae.cfg.model.downsample)
    if min(args.volume_size) <= 0 or any(size % ratio for size, ratio in zip(args.volume_size, ratios)):
        raise ValueError('volume-size must be positive and divisible by VAE compression ratios')
    shape = tuple(size // ratio for size, ratio in zip(args.volume_size, ratios))
    digest = codebook_digest(ae)
    report['vae'] = dict(
        path=str(Path(args.ae_ckpt).resolve()), strict_load='passed', stage=ae.stage,
        downsample=list(ratios), embedding_dim=ae.embedding_dim, patch_size=ae.patch_size,
        codebook_sha256=digest, codebook_initialized=not ae.codebook._need_init,
        legacy_codebook_marker_migrated='codebook._initialized' not in saved_vae['state_dict'],
        training_state_present=all(key in saved_vae for key in ('optimizer_states', 'loops')),
    )
    del saved_vae
    if not args.skip_forward:
        with torch.inference_mode():
            image = torch.zeros(1, int(ae.cfg.dataset.image_channels), *args.volume_size, device=device)
            embeddings, _ = ae.encode(image, include_embeddings=True, quantize=True)
            reconstruction = decode_latent(ae, embeddings, mode='full')
            if reconstruction.shape != image.shape or not torch.isfinite(reconstruction).all():
                raise ValueError('VAE encode/decode returned an invalid shape or non-finite values')
            report['vae']['forward_shape'] = list(reconstruction.shape)
        del image, embeddings, reconstruction
    if args.diff_ckpt:
        checkpoint = torch.load(args.diff_ckpt, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError('Expected diffusion checkpoint containing model/ema state dictionaries')
        recorded_digest = checkpoint.get('ae_codebook_sha256')
        if recorded_digest is not None and recorded_digest != digest:
            raise ValueError('The diffusion checkpoint records a different VAE codebook')
        # Release VAE GPU tensors before constructing the diffusion network.
        channels = ae.embedding_dim
        del ae
        model, diffusion, config = from_checkpoint(checkpoint, args, device)
        if config['channels'] != channels:
            raise ValueError('Diffusion latent channels differ from the selected VAE')
        if any(size % divisor for size, divisor in zip(shape, model.spatial_divisor)):
            raise ValueError(f'Test latent dimensions {shape} must be divisible by {model.spatial_divisor}')
        missing = sorted(DIFFUSION_RESUME_FIELDS - set(checkpoint))
        report['diffusion'] = dict(
            path=str(Path(args.diff_ckpt).resolve()), strict_load='passed', model_config=config,
            diffusion_config=dict(channels=diffusion.channels, timesteps=diffusion.num_timesteps,
                                  loss_type=diffusion.loss_type),
            vae_pairing='verified_digest' if recorded_digest is not None else 'not_recorded',
            complete_resume_fields_present=not missing, missing_resume_fields=missing,
        )
        if recorded_digest is None:
            report['notes'].append('Old diffusion checkpoint has no VAE digest; verify the original VAE pairing manually.')
        if missing:
            report['notes'].append('Diffusion weights may be usable for inference, but exact training resume is not established.')
        if not args.skip_forward:
            with torch.inference_mode():
                latent = torch.zeros(1, channels, *shape, device=device)
                if config.get('cond_dim') is not None:
                    condition = latent.new_zeros(1, config['cond_dim'])
                elif config.get('cond_classes') is not None:
                    condition = torch.zeros(1, dtype=torch.long, device=device)
                else:
                    condition = None
                output = model(latent, torch.zeros(1, dtype=torch.long, device=device),
                               y=condition, res=latent.new_tensor(shape) / 64)
                if output.shape != latent.shape or not torch.isfinite(output).all():
                    raise ValueError('Diffusion denoiser returned an invalid shape or non-finite values')
                report['diffusion']['forward_shape'] = list(output.shape)
    report['passed'] = True
    report['forward_tested'] = not args.skip_forward
    report['notes'].append('Strict loading and tiny forward tests do not establish previous accuracy or convergence.')


def build_parser():
    from ddpm.factory import add_model_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ae-ckpt', required=True)
    parser.add_argument('--diff-ckpt')
    parser.add_argument('--volume-size', nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    parser.add_argument('--skip-forward', action='store_true', help='Only load weights and inspect metadata')
    parser.add_argument('--output', help='Optional new JSON report path; input JSON is not needed')
    add_model_arguments(parser)
    return parser


def main(args):
    if args.output and Path(args.output).exists():
        raise FileExistsError(f'Report already exists: {args.output}')
    report = dict(passed=False, notes=[])
    try:
        check(args, report)
    except Exception as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        traceback.print_exc()
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main(build_parser().parse_args())
