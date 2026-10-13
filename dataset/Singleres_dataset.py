"""Single-resolution images or latent volumes with explicit shape checks."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from dataset.io_utils import collect_volume_files, load_scalar_image, to_model_tensor
from utils.config_io import project_path, read_data_roots


class Singleres_dataset(Dataset):
    def __init__(self, root_dir, resolution=(32, 32, 32), generate_latents=False, class_id=0, raw_meta=None):
        self.resolution = tuple(int(v) for v in resolution)
        if len(self.resolution) != 3 or min(self.resolution) <= 0:
            raise ValueError('resolution must contain three positive latent dimensions')
        self.generate_latents = generate_latents
        self.raw_meta = raw_meta
        self.latent_metadata = {}
        self.all_files = []
        self.roots = read_data_roots(root_dir, numeric_labels=True,
                                    direct_kind='images' if generate_latents else 'latents', class_id=class_id)
        for label, entry in self.roots.items():
            try:
                class_id = int(label)
            except ValueError as exc:
                raise ValueError('Diffusion class IDs must be nonnegative integers') from exc
            if class_id < 0:
                raise ValueError('Diffusion class IDs must be nonnegative integers')
            explicit = isinstance(entry, dict)
            source = entry.get('images' if generate_latents else 'latents') if explicit else entry
            if source is None:
                raise ValueError(f'Missing image/latent root for class {label}')
            sources = source if isinstance(source, list) else [source]
            for source in sources:
                folder = project_path(source)
                if not generate_latents and not explicit:
                    folder = Path(str(folder) + '_latents')
                paths = collect_volume_files([folder])
                if not paths:
                    raise FileNotFoundError(f'No .raw/.nii.gz files found in {folder}')
                if not generate_latents and any(p.lower().endswith('.raw') for p in paths):
                    raise ValueError('Use multichannel .nii.gz for latent volumes; RAW is for source images')
                self.all_files.extend((class_id, path) for path in paths)
                if not generate_latents:
                    for path in paths:
                        metadata_path = Path(str(path) + '.meta.pt')
                        if metadata_path.is_file():
                            metadata = torch.load(metadata_path, map_location='cpu', weights_only=True)
                            if not isinstance(metadata, dict):
                                raise ValueError(f'Invalid latent provenance: {metadata_path}')
                            self.latent_metadata[path] = metadata
        self.file_num = len(self.all_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, index):
        label, path = self.all_files[index]
        img = load_scalar_image(path, raw_meta=self.raw_meta)
        if self.generate_latents:
            return to_model_tensor(img), path
        # Latents are stored directly as [C, D, H, W]; do not transpose again.
        latent = img.data.to(torch.float32)
        if tuple(latent.shape[-3:]) != self.resolution:
            raise ValueError(f'{path}: latent shape {tuple(latent.shape[-3:])} != configured {self.resolution}')
        if not torch.isfinite(latent).all():
            raise ValueError(f'{path}: latent contains non-finite values')
        return latent, torch.tensor(label, dtype=torch.long), torch.tensor(self.resolution, dtype=torch.float32) / 64.0
