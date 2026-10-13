"""In-code sampling defaults and command-line options for VAE decoding."""
import math

SAMPLING_DEFAULTS = dict(strategy='ddpm', ddim_steps=100, eta=0.0)
DECODE_DEFAULTS = dict(mode='full', patch_size=256, sliding_window=1, overlap_ratio=0.5)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def add_sampling_arguments(parser, preview=False, include_sampling=True, allow_auto=False):
    prefix = 'sample-' if preview else ''
    if include_sampling:
        parser.add_argument(f'--{prefix}sampling-strategy', choices=['ddpm', 'ddim'],
                            default=SAMPLING_DEFAULTS['strategy'])
        flags = [f'--{prefix}ddim-steps']
        if not preview:
            flags.append('--ddim_steps')
        parser.add_argument(*flags, type=int, default=SAMPLING_DEFAULTS['ddim_steps'])
        parser.add_argument(f'--{prefix}eta', type=float, default=SAMPLING_DEFAULTS['eta'])
    modes = ['full', 'sliding', 'gaussian'] + (['auto'] if allow_auto else [])
    parser.add_argument(f'--{prefix}decode-mode', choices=modes, default=DECODE_DEFAULTS['mode'],
                        help='VAE decoding mode (default: full)')
    parser.add_argument(f'--{prefix}decode-patch-size', type=int, default=DECODE_DEFAULTS['patch_size'],
                        help='Decoded voxel patch size; Gaussian patches shrink to fit small inputs')
    parser.add_argument(f'--{prefix}decode-sliding-window', type=int, default=DECODE_DEFAULTS['sliding_window'],
                        help='Patch batches per input volume for ordinary sliding decoding')
    parser.add_argument(f'--{prefix}decode-overlap-ratio', type=float, default=DECODE_DEFAULTS['overlap_ratio'],
                        help='Gaussian patch overlap in [0,1)')


def resolve_sampling_options(args, preview=False, include_sampling=True, allow_auto=False):
    """Validate CLI values; also fill in-code defaults for direct Python callers."""
    prefix = 'sample_' if preview else ''
    defaults = {f'decode_{name}': value for name, value in DECODE_DEFAULTS.items()}
    if include_sampling:
        defaults.update(sampling_strategy=SAMPLING_DEFAULTS['strategy'],
                        ddim_steps=SAMPLING_DEFAULTS['ddim_steps'], eta=SAMPLING_DEFAULTS['eta'])
    for name, default in defaults.items():
        attribute = prefix + name
        value = getattr(args, attribute, None)
        if value is None:
            value = default
        setattr(args, attribute, value)
    modes = ('full', 'sliding', 'gaussian', 'auto') if allow_auto else ('full', 'sliding', 'gaussian')
    if getattr(args, prefix + 'decode_mode') not in modes:
        raise ValueError(f'decode_mode must be one of {modes}')
    for name in ('decode_patch_size', 'decode_sliding_window'):
        _positive_int(getattr(args, prefix + name), name)
    overlap = getattr(args, prefix + 'decode_overlap_ratio')
    _finite_number(overlap, 'decode_overlap_ratio')
    if not 0 <= overlap < 1:
        raise ValueError('decode_overlap_ratio must be in [0,1)')
    if include_sampling:
        if getattr(args, prefix + 'sampling_strategy') not in ('ddpm', 'ddim'):
            raise ValueError('sampling_strategy must be ddpm or ddim')
        _positive_int(getattr(args, prefix + 'ddim_steps'), 'ddim_steps')
        eta = getattr(args, prefix + 'eta')
        _finite_number(eta, 'eta')
        if eta < 0:
            raise ValueError('eta must be nonnegative')
    return args


def decode_options(args, preview=False, mode=None):
    prefix = 'sample_' if preview else ''
    return dict(mode=mode or getattr(args, prefix + 'decode_mode'),
                patch_size=getattr(args, prefix + 'decode_patch_size'),
                sliding_window=getattr(args, prefix + 'decode_sliding_window'),
                overlap_ratio=getattr(args, prefix + 'decode_overlap_ratio'))


def validate_decode_settings(shape, ratios, mode='full', patch_size=256,
                             sliding_window=1, overlap_ratio=0.5):
    """Check shape-dependent constraints without allocating a volume."""
    if mode not in ('full', 'sliding', 'gaussian'):
        raise ValueError('Decode mode must be full, sliding or gaussian')
    if len(shape) != 5 or any(size <= 0 for size in shape):
        raise ValueError('Decoding expects nonempty [B,C,D,H,W] latents')
    if mode == 'full':
        return None
    ratios = tuple(int(value) for value in ratios)
    if len(ratios) != 3 or min(ratios) <= 0 or len(set(ratios)) != 1:
        raise ValueError('Sliding decoding requires three equal, positive compression ratios')
    _positive_int(patch_size, 'decode.patch_size')
    _positive_int(sliding_window, 'decode.sliding_window')
    if patch_size % ratios[0]:
        raise ValueError('Decode patch size must be a positive multiple of the compression ratio')
    side = patch_size // ratios[0]
    if mode == 'sliding':
        if any(size < side or size % side for size in shape[-3:]):
            raise ValueError('Sliding decode requires each latent dimension to be divisible by decode_patch_size / compression; reduce patch_size or use gaussian')
    else:
        _finite_number(overlap_ratio, 'decode.overlap_ratio')
        if not 0 <= overlap_ratio < 1:
            raise ValueError('Gaussian overlap_ratio must be in [0,1)')
        side = min(side, *shape[-3:])
    return side * ratios[0]
