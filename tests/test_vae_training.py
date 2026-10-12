"""Small CPU VAE training tests; perceptual features are replaced only in tests."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from AutoEncoder.model.REV_VQVAE import REV_VQVAE
from AutoEncoder.model.codebook import Codebook
from AutoEncoder.model.VolumetricPerceptualLoss import volume_standardize
from dataset.core_volume import CoreVolumeDataset, split_volume_files
from train.callbacks import VolumeLogger


class TestPerceptualLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.perceptual_model = nn.Sequential(nn.Conv3d(1, 1, 1), nn.Dropout3d())

    def forward(self, source, reconstruction):
        return F.l1_loss(source, reconstruction)


def configuration(stage=1, ratio=4):
    return OmegaConf.create(dict(
        model=dict(seed=7, stage=stage, embedding_dim=2, n_codes=16, n_hiddens=8,
                   downsample=[ratio] * 3, norm_type='group', num_groups=4,
                   discriminator_iter_start=0, disc_channels=4, disc_layers=1,
                   disc_loss_type='hinge', volume_gan_weight=1, gan_feat_weight=1,
                   perceptual_weight=1, perceptual_3d=False, l1_weight=4,
                   no_random_restart=True, restart_thres=1, lr=3e-4),
        dataset=dict(patch_size=16, image_channels=1)))


def make_model(stage=1, ratio=4):
    with patch('AutoEncoder.model.REV_VQVAE.Perceptual_Loss', TestPerceptualLoss):
        model = REV_VQVAE(configuration(stage, ratio), initialize_perceptual=False)
    if stage == 2:
        model.codebook._init_embeddings(torch.randn(1, 2, 4, 4, 4))
    return model


class TrainingProbe(pl.Callback):
    def __init__(self, case):
        self.case = case
        self.steps = []

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.codes = {name: value.clone() for name, value in model.codebook.state_dict().items()}
        self.encoder = [value.detach().clone() for value in model.encoder.parameters()]
        self.discriminator = [value.detach().clone() for value in model.volume_discriminator.parameters()]
        self.decoder = [value.detach().clone() for value in model.decoder.parameters()]

    def on_train_batch_end(self, trainer, model, output, batch, batch_idx):
        loss = output['loss'] if isinstance(output, dict) else output
        self.case.assertTrue(torch.isfinite(loss))
        self.case.assertFalse(model.perceptual_loss.training)
        is_discriminator = batch_idx % 2 == 1 and model.volume_gan_weight > 0
        if is_discriminator or model.stage == 2:
            for name, value in model.codebook.state_dict().items():
                self.case.assertTrue(torch.equal(self.codes[name], value), name)
        if is_discriminator or model.stage == 2:
            for before, after in zip(self.encoder, model.encoder.parameters()):
                self.case.assertTrue(torch.equal(before, after))
        if is_discriminator:
            self.case.assertTrue(all(torch.equal(a, b) for a, b in zip(self.decoder, model.decoder.parameters())))
            self.case.assertTrue(any(not torch.equal(a, b) for a, b in zip(self.discriminator, model.volume_discriminator.parameters())))
        else:
            self.case.assertTrue(all(torch.equal(a, b) for a, b in zip(self.discriminator, model.volume_discriminator.parameters())))
            self.case.assertTrue(any(not torch.equal(a, b) for a, b in zip(self.decoder, model.decoder.parameters())))
        if model.stage == 2:
            self.case.assertFalse(model.encoder.training)
            self.case.assertFalse(model.codebook.training)
            self.case.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.steps.append(is_discriminator)


class VAETrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def loader(self):
        return DataLoader([{'data': torch.rand(1, 16, 16, 16) * 2 - 1} for _ in range(4)], batch_size=1)

    def trainer(self, root, steps, callbacks=None, checkpointing=False):
        return pl.Trainer(accelerator='cpu', devices=1, precision='32-true',
                          max_steps=steps, max_epochs=3, limit_train_batches=4,
                          num_sanity_val_steps=0, logger=False, enable_progress_bar=False,
                          enable_model_summary=False, enable_checkpointing=checkpointing,
                          default_root_dir=root, callbacks=callbacks or [])

    def test_quantizer_matches_dense_formula_and_straight_through_gradient(self):
        book = Codebook(11, 3, no_random_restart=True, distance_chunk_size=5).eval()
        z = torch.randn(2, 3, 3, 2, 2, requires_grad=True)
        flat = z.detach().permute(0, 2, 3, 4, 1).reshape(-1, 3)
        distances = flat.square().sum(1, keepdim=True) - 2 * flat @ book.embeddings.t() + book.embeddings.square().sum(1)
        indices = distances.argmin(1).reshape(2, 3, 2, 2)
        expected = F.embedding(indices, book.embeddings).permute(0, 4, 1, 2, 3)
        output = book(z)
        self.assertTrue(torch.equal(output['encodings'], indices))
        torch.testing.assert_close(output['embeddings'], expected)
        torch.testing.assert_close(output['commitment_loss'], .25 * F.mse_loss(z, expected))
        output['embeddings'].sum().backward()
        torch.testing.assert_close(z.grad, torch.ones_like(z))

    def test_ema_matches_one_hot_reference_without_restart(self):
        book = Codebook(11, 3, no_random_restart=True, distance_chunk_size=5).train()
        book._init_embeddings(torch.randn(1, 3, 3, 3, 3))
        z = torch.randn(2, 3, 3, 2, 2)
        flat = z.permute(0, 2, 3, 4, 1).reshape(-1, 3)
        indices = (flat.square().sum(1, keepdim=True) - 2 * flat @ book.embeddings.t() + book.embeddings.square().sum(1)).argmin(1)
        one_hot = F.one_hot(indices, book.n_codes).float()
        expected_counts = .99 * book.N + .01 * one_hot.sum(0)
        expected_sums = .99 * book.z_avg + .01 * (flat.t() @ one_hot).t()
        total = expected_counts.sum()
        weights = (expected_counts + 1e-7) / (total + book.n_codes * 1e-7) * total
        book(z)
        torch.testing.assert_close(book.N, expected_counts)
        torch.testing.assert_close(book.z_avg, expected_sums)
        torch.testing.assert_close(book.embeddings, expected_sums / weights.unsqueeze(1))

    def test_loaded_and_legacy_codebooks_do_not_reinitialize(self):
        original = Codebook(11, 3, no_random_restart=True)
        original(torch.randn(1, 3, 3, 3, 3))
        for legacy in (False, True):
            state = {name: value.clone() for name, value in original.state_dict().items()}
            if legacy:
                del state['_initialized']
            loaded = Codebook(11, 3, no_random_restart=True)
            loaded.load_state_dict(state, strict=True)
            self.assertFalse(loaded._need_init)
            with patch.object(loaded, '_init_embeddings', side_effect=AssertionError('Unexpected reinitialization')):
                loaded(torch.randn(1, 3, 3, 3, 3))

    def test_mixed_precision_quantizer_is_finite_and_can_skip_updates(self):
        book = Codebook(11, 3, no_random_restart=True)
        book._init_embeddings(torch.randn(1, 3, 3, 3, 3))
        before = {name: value.clone() for name, value in book.state_dict().items()}
        z = torch.randn(1, 3, 3, 3, 3, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = book(z, update_codebook=False)
        self.assertEqual(output['embeddings'].dtype, z.dtype)
        self.assertEqual(output['commitment_loss'].dtype, torch.float32)
        (output['embeddings'].float().square().mean() + output['commitment_loss']).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        for name, value in book.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_both_stages_train_with_isolated_optimizer_and_codebook_updates(self):
        with tempfile.TemporaryDirectory() as root:
            for stage in (1, 2):
                for ratio in (4, 8):
                    model = make_model(stage, ratio)
                    probe = TrainingProbe(self)
                    trainer = self.trainer(root, 4, [probe])
                    trainer.fit(model, self.loader())
                    self.assertEqual(probe.steps, [False, True, False, True])

    def test_disabled_gan_updates_only_generator(self):
        model = make_model()
        model.volume_gan_weight = 0
        with tempfile.TemporaryDirectory() as root:
            probe = TrainingProbe(self)
            trainer = self.trainer(root, 4, [probe])
            trainer.fit(model, self.loader())
            self.assertEqual(probe.steps, [False] * 4)
            self.assertFalse(trainer.optimizers[1].state)

    def test_complete_training_checkpoint_resume_for_both_stages(self):
        with tempfile.TemporaryDirectory() as root:
            for stage in (1, 2):
                model = make_model(stage)
                trainer = self.trainer(root, 2)
                trainer.fit(model, self.loader())
                checkpoint = str(Path(root) / f'stage{stage}.ckpt')
                trainer.save_checkpoint(checkpoint)
                fresh = make_model(stage)
                restored = self.trainer(root, 3)
                with patch.object(fresh.codebook, '_init_embeddings', side_effect=AssertionError('Resume reinitialized codes')):
                    restored.fit(fresh, self.loader(), ckpt_path=checkpoint)
                self.assertEqual(restored.global_step, 3)
                self.assertTrue(restored.optimizers[0].state)
                self.assertTrue(restored.optimizers[1].state)

    def test_encoder_freeze_survives_preview_failure_and_mode_changes(self):
        model = make_model(stage=2).train()
        model.log_volumes = Mock(side_effect=RuntimeError('preview failed'))
        logger = VolumeLogger(1, 1)
        with self.assertRaisesRegex(RuntimeError, 'preview failed'):
            logger.log_vid(model, {}, 0)
        self.assertTrue(model.training)
        self.assertFalse(model.codebook.training)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.perceptual_loss.training)

    def test_validation_shape_rejects_truncation(self):
        model = make_model().eval()
        with self.assertRaisesRegex(ValueError, 'patch_size'):
            model(torch.randn(1, 1, 20, 16, 16), log_volume=True, val=True)

    def test_validation_runs_and_logs_finite_checkpoint_metric(self):
        with tempfile.TemporaryDirectory() as root:
            for stage in (1, 2):
                model = make_model(stage)
                trainer = self.trainer(root, 2)
                trainer.fit(model, self.loader())
                trainer.validate(model, self.loader())
                self.assertTrue(torch.isfinite(trainer.callback_metrics['val/recon_loss']))

    def test_checkpoint_loading_skips_pretrained_initialization(self):
        with tempfile.TemporaryDirectory() as root:
            model = make_model()
            trainer = self.trainer(root, 2)
            trainer.fit(model, self.loader())
            destination = str(Path(root) / 'weights.ckpt')
            trainer.save_checkpoint(destination)
            with patch('AutoEncoder.model.REV_VQVAE.Perceptual_Loss', TestPerceptualLoss):
                loaded = REV_VQVAE.load_from_checkpoint(destination, initialize_perceptual=False).eval()
            source = torch.randn(1, 1, 16, 16, 16)
            model.eval()
            with torch.no_grad():
                _, expected = model(source, log_volume=True, val=True)
                _, actual = loaded(source, log_volume=True, val=True)
            torch.testing.assert_close(actual, expected)

    def test_constant_volume_standardization_is_finite(self):
        result = volume_standardize(torch.ones(1, 1, 4, 4, 4))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.abs().sum().item(), 0)

    def test_split_is_disjoint_repeatable_and_does_not_reset_global_rng(self):
        for count in (2, 10, 40, 41, 100):
            files = [f'{index:03}.nii.gz' for index in range(count)]
            random.seed(123)
            before = random.getstate()
            train = split_volume_files(files, 'train')
            validation = split_volume_files(files, 'val')
            self.assertEqual(before, random.getstate())
            self.assertFalse(set(train) & set(validation))
            self.assertEqual(set(train) | set(validation), set(files))
            self.assertEqual(train, split_volume_files(files, 'train'))
        with self.assertRaises(ValueError):
            split_volume_files(['one.nii.gz'], 'train')

    def test_small_dataset_and_zero_patch_sampling_terminate(self):
        with tempfile.TemporaryDirectory() as root:
            for index in range(2):
                tio.ScalarImage(tensor=torch.zeros(1, 20, 16, 18)).save(str(Path(root) / f'{index}.nii.gz'))
            train = CoreVolumeDataset(root, split='train', patch_size=16, max_patch_attempts=2)
            self.assertEqual(tuple(train[0]['data'].shape), (1, 16, 16, 16))
            validation = CoreVolumeDataset(root, split='val', patch_size=16)
            self.assertTrue(torch.equal(validation[0]['data'], validation[0]['data']))
            self.assertEqual(tuple(validation[0]['data'].shape), (1, 16, 16, 16))

    def test_gaussian_decode_covers_edges_and_small_inputs(self):
        for ratio in (4, 8):
            book = Codebook(11, 2).eval()
            class FirstChannel(nn.Module):
                def forward(self, features):
                    return F.interpolate(features[:, :1], scale_factor=ratio, mode='nearest')
            model = SimpleNamespace(
                cfg=SimpleNamespace(model=SimpleNamespace(downsample=[ratio] * 3),
                                    dataset=SimpleNamespace(image_channels=1)),
                codebook=book, post_vq_conv=nn.Identity(), decoder=FirstChannel())
            for shape in ((2, 2, 5, 4, 3), (1, 2, 1, 2, 1)):
                latent = torch.randn(*shape, requires_grad=True)
                for quantize in (False, True):
                    for overlap in (0, .5, .99):
                        result = REV_VQVAE.decode_sliding_gaussian(
                            model, latent, quantize=quantize, patch_size=256, overlap_ratio=overlap)
                        features = book(latent, update_codebook=False)['embeddings'] if quantize else latent
                        expected = model.decoder(features)
                        self.assertEqual(tuple(result.shape), tuple(expected.shape))
                        self.assertEqual(result.device.type, 'cpu')
                        self.assertFalse(result.requires_grad)
                        self.assertTrue(torch.isfinite(result).all())
                        # Retain the supplied weight / (weight + 1e-8) normalization.
                        torch.testing.assert_close(result, expected.detach(), rtol=2e-4, atol=2e-4)
            with self.assertRaisesRegex(ValueError, 'compress_ratio'):
                REV_VQVAE.decode_sliding_gaussian(model, latent, compress_ratio=ratio * 2)


if __name__ == '__main__':
    unittest.main()
