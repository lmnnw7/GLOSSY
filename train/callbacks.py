"""Save reconstruction previews without changing persistent training state."""
from pathlib import Path
import numpy as np
import torch
import torchio as tio
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from dataset.io_utils import to_storage_tensor


class VolumeLogger(Callback):
    def __init__(self, batch_frequency, max_volumes, clamp=True, increase_log_steps=True):
        super().__init__()
        if batch_frequency <= 0:
            raise ValueError('batch_frequency must be positive')
        self.batch_freq = batch_frequency
        self.max_volumes = max_volumes
        self.log_steps = [2 ** n for n in range(int(np.log2(batch_frequency)) + 1)]
        if not increase_log_steps:
            self.log_steps = [batch_frequency]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, volumes, global_step, current_epoch, batch_idx):
        root = Path(save_dir) / 'volumes' / split
        root.mkdir(parents=True, exist_ok=True)
        for name, tensors in volumes.items():
            for index, tensor in enumerate(tensors):
                tensor = (tensor + 1) / 2
                if self.clamp:
                    tensor = tensor.clamp(0, 1)
                tensor = to_storage_tensor(tensor)
                filename = f'{name}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}-{index}.nii.gz'
                tio.ScalarImage(tensor=tensor).save(str(root / filename))

    @rank_zero_only
    def log_vid(self, pl_module, batch, batch_idx, split='train'):
        if self.max_volumes <= 0 or not self.check_frequency(batch_idx):
            return
        if not callable(getattr(pl_module, 'log_volumes', None)):
            return
        was_training = pl_module.training
        # Previews must not consume the random sequence used by training.
        devices = [pl_module.device.index] if pl_module.device.type == 'cuda' else []
        try:
            pl_module.eval()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                volumes = pl_module.log_volumes(batch, split=split, batch_idx=batch_idx)
            volumes = {name: tensor[:self.max_volumes].detach().cpu()
                       for name, tensor in volumes.items()}
            self.log_local(pl_module.logger.save_dir, split, volumes,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)
        finally:
            pl_module.train(was_training)

    def check_frequency(self, batch_idx):
        return batch_idx % self.batch_freq == 0 or batch_idx in self.log_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_vid(pl_module, batch, batch_idx, split='train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_vid(pl_module, batch, batch_idx, split='val')
