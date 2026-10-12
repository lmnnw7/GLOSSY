"""Strict JSON loading and data/RAW configuration validation without ML dependencies."""
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DTYPES = {'uint16': 'uint16', 'u16': 'uint16', 'int16': 'int16',
              'i16': 'int16', 'float32': 'float32', 'f32': 'float32'}
RAW_BYTE_ORDERS = {'little': 'little', 'le': 'little', 'l': 'little',
                   'big': 'big', 'be': 'big', 'b': 'big'}


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite JSON number: {value}')
    return result


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def load_json_object(path):
    """Read UTF-8 (with or without BOM), rejecting ambiguous or non-standard JSON."""
    path = project_path(path)
    try:
        with path.open(encoding='utf-8-sig') as stream:
            value = json.load(stream, object_pairs_hook=_unique_object,
                              parse_float=_finite_float, parse_constant=_reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid JSON in {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'JSON must contain an object: {path}')
    return value


def _data_paths(value, label):
    paths = value if isinstance(value, list) else [value]
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError(f'{label} must be a nonempty path or list of paths')
    unique = {}
    for path in paths:
        resolved = str(project_path(path))
        unique.setdefault(os.path.normcase(resolved), resolved)
    return list(unique.values()) if isinstance(value, list) else next(iter(unique.values()))


def read_data_roots(path, numeric_labels=False, direct_kind=None, class_id=0):
    """Keep legacy path mappings and support explicit images/latents entries.

    Relative data paths always refer to the project root, independent of cwd or
    the manifest's location. Exported manifest metadata is retained unchanged.
    """
    path = project_path(path)
    if path.suffix.lower() != '.json':
        if direct_kind is not None:
            if direct_kind not in ('images', 'latents'):
                raise ValueError('Direct data kind must be images or latents')
            if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
                raise ValueError('class-id must be a nonnegative integer')
            return {str(class_id): {direct_kind: str(path)}}
        return {'0': str(path)}
    roots = load_json_object(path)
    if not roots:
        raise ValueError(f'Data JSON must contain a nonempty dataset mapping: {path}')
    result = {}
    for label, entry in roots.items():
        if not label.strip():
            raise ValueError(f'Dataset names must not be empty: {path}')
        if numeric_labels:
            try:
                class_id = int(label)
            except ValueError as exc:
                raise ValueError('Diffusion class IDs must be nonnegative integers') from exc
            if class_id < 0:
                raise ValueError('Diffusion class IDs must be nonnegative integers')
            label = str(class_id)
        if label in result:
            raise ValueError(f'Duplicate class ID after normalization: {label}')
        if isinstance(entry, dict):
            entry = dict(entry)
            fields = [field for field in ('images', 'latents') if field in entry]
            if not fields:
                raise ValueError(f'Dataset {label} must define images or latents')
            for field in fields:
                entry[field] = _data_paths(entry[field], f'Dataset {label}.{field}')
        else:
            entry = _data_paths(entry, f'Dataset {label}')
        result[label] = entry
    return result


def image_roots(path):
    roots = []
    for label, entry in read_data_roots(path).items():
        source = entry.get('images') if isinstance(entry, dict) else entry
        if source is None:
            raise ValueError(f'Dataset {label} has no images for VAE training/reconstruction')
        roots.extend(source if isinstance(source, list) else [source])
    return list(dict.fromkeys(roots))


def _numbers(value, length, name, positive=False):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must contain {length} numbers')
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or
           not math.isfinite(item) or (positive and item <= 0) for item in value):
        raise ValueError(f'{name} must contain finite {"positive " if positive else ""}numbers')
    return list(value)


def validate_raw_meta(meta, require_shape=True):
    """Validate merged metadata; preserve supplied intensity normalization."""
    if not isinstance(meta, dict):
        raise ValueError('RAW metadata must be a JSON object')
    result = dict(meta)
    if require_shape and 'shape' not in result:
        raise ValueError('RAW metadata requires shape in storage X,Y,Z order')
    if 'shape' in result:
        shape = result['shape']
        if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
            raise ValueError('RAW shape must contain three positive integers in X,Y,Z order')
        result['shape'] = list(shape)
    for field, default, aliases in (
            ('dtype', 'uint16', RAW_DTYPES), ('byte_order', 'little', RAW_BYTE_ORDERS),
            ('axis_order', 'XYZ', {'xyz': 'XYZ'}), ('spacing_unit', 'mm', {'mm': 'mm', 'um': 'um'})):
        value = result.get(field, default)
        if not isinstance(value, str) or value.lower() not in aliases:
            raise ValueError(f'Invalid RAW {field}: {value!r}; accepted values: {sorted(aliases)}')
        result[field] = aliases[value.lower()]
    if 'spacing' in result:
        result['spacing'] = _numbers(result['spacing'], 3, 'RAW spacing', positive=True)
    if 'value_range' in result:
        low, high = _numbers(result['value_range'], 2, 'RAW value_range')
        if high <= low:
            raise ValueError('RAW value_range maximum must exceed its minimum')
        result['value_range'] = [low, high]
    if 'scale' in result:
        _numbers([result['scale']], 1, 'RAW scale', positive=True)
    return result


def add_raw_arguments(parser):
    parser.add_argument('--raw-shape', nargs=3, type=int, help='RAW storage X Y Z dimensions; bypasses JSON sidecars')
    parser.add_argument('--raw-dtype', choices=['uint16', 'int16', 'float32'], help='RAW input dtype (default: uint16)')
    parser.add_argument('--raw-byte-order', choices=['little', 'big'], help='RAW input byte order (default: little)')
    parser.add_argument('--raw-spacing', nargs=3, type=float, help='RAW input X Y Z spacing')
    parser.add_argument('--raw-spacing-unit', choices=['mm', 'um'], help='RAW spacing unit (default: mm)')
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument('--raw-value-range', nargs=2, type=float, help='Fixed RAW intensity minimum and maximum')
    normalization.add_argument('--raw-scale', type=float, help='Divide RAW intensities by this positive scale')


def raw_options(args):
    fields = ('shape', 'dtype', 'byte_order', 'spacing', 'spacing_unit', 'value_range', 'scale')
    result = {field: getattr(args, 'raw_' + field) for field in fields
              if getattr(args, 'raw_' + field, None) is not None}
    if not result:
        return None
    validate_raw_meta(result, require_shape=False)
    return result


def add_vae_input_arguments(parser):
    parser.add_argument('--data-path', help='Source volume directory/file, or optional legacy JSON mapping')
    parser.add_argument('--results-dir', help='Override model.default_root_dir')
    parser.add_argument('--ckpt', help='Override the initialization/resume checkpoint')
    add_raw_arguments(parser)


def vae_input_options(args):
    return dict(data_path=getattr(args, 'data_path', None), results_dir=getattr(args, 'results_dir', None),
                checkpoint_path=getattr(args, 'ckpt', None), raw_meta=raw_options(args))


def validate_latent_metadata(records, digest, ratios):
    """Keep VAE/latent pairing checks for JSON manifests and directory inputs."""
    quantized = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Latent provenance must be a dictionary')
        if record.get('ae_codebook_sha256', digest) != digest:
            raise ValueError('Latents were encoded with a different VAE codebook')
        if 'downsample' in record and list(record['downsample']) != list(ratios):
            raise ValueError('Latent compression does not match the loaded VAE')
        if record.get('latent_axis_order', 'CDHW') != 'CDHW':
            raise ValueError('Latents must use model C,D,H,W axes')
        if record.get('latent_normalization', 'codebook_minmax') != 'codebook_minmax':
            raise ValueError('Latents must use codebook_minmax normalization')
        if 'quantized' in record:
            if not isinstance(record['quantized'], bool):
                raise ValueError('Latent quantized provenance must be boolean')
            quantized.add(record['quantized'])
    if len(quantized) > 1:
        raise ValueError('Do not mix quantized and continuous latent exports')
