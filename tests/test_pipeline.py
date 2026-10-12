"""CPU regression tests; no dataset, trained weights or downloads required."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torchio as tio
from ddpm.MSP_DiffNet import MSP_DiffNet, GaussianDiffusion
from ddpm.factory import from_checkpoint
from dataset.Singleres_dataset import Singleres_dataset
from dataset.io_utils import load_scalar_image, save_raw_volume, to_model_tensor, to_storage_tensor
from evaluation.volume_output import save_volume
from runtime import decode_latent, resolve_device
from train.generate_training_latent import encode_volume


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def small_model(self, **kwargs):
        cfg = dict(dim=24, dim_mults=(1, 1, 2), sub_volume_size=(4, 4, 4),
                   channels=2, cond_classes=1, use_sparse_linear_attn=(0, 0, 0),
                   resnet_groups=6, DiT_num_heads=4, attn_heads=2)
        cfg.update(kwargs)
        return MSP_DiffNet(**cfg)

    def test_forward_backward_and_batched_resolution(self):
        model = self.small_model()
        x = torch.randn(2, 2, 8, 8, 8)
        result = model(x, torch.tensor([1, 2]), y=torch.zeros(2, dtype=torch.long), res=torch.tensor([.125]*3))
        self.assertEqual(result.shape, x.shape)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(model.final_conv[-1].weight.grad).all())

    def test_four_and_eight_compression_layouts(self):
        for sub in ((8, 8, 8), (16, 16, 16)):
            model = self.small_model(sub_volume_size=sub).eval()
            with torch.no_grad():
                x = torch.randn(1, 2, 16, 16, 16)
                self.assertEqual(model(x, torch.tensor([1]), y=torch.tensor([0])).shape, x.shape)

    def test_continuous_condition_from_key_code(self):
        model = self.small_model(cond_classes=None, cond_dim=2)
        x = torch.randn(2, 2, 8, 8, 8)
        self.assertEqual(model(x, torch.tensor([1, 1]), y=torch.tensor([[.1,.2],[.2,.3]])).shape, x.shape)

    def test_disabled_resolution_condition(self):
        model = self.small_model(res_condition=False)
        x = torch.randn(1, 2, 8, 8, 8)
        self.assertEqual(model(x, torch.tensor([1]), y=torch.tensor([0])).shape, x.shape)

    def test_invalid_shape_is_rejected_before_patch_truncation(self):
        model = self.small_model()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 7, 8, 8), torch.tensor([1]), y=torch.tensor([0]))

    def test_ddim_reaches_clean_endpoint_for_batch_two(self):
        diffusion = GaussianDiffusion(channels=2, timesteps=10)
        clean = torch.full((2, 2, 4, 4, 4), .2)
        def oracle(x, t, **kwargs):
            alpha = diffusion.alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
            return (x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        result = diffusion.sample(oracle, torch.randn_like(clean), strategy='ddim', ddim_steps=5)
        torch.testing.assert_close(result, clean, atol=1e-4, rtol=1e-4)
        with self.assertRaises(ValueError):
            diffusion.sample(oracle, clean, strategy='ddim', ddim_steps=11)

    def test_stochastic_ddim_direction_includes_sigma(self):
        diffusion = GaussianDiffusion(channels=2, timesteps=10)
        x = torch.randn(2, 2, 3, 3, 3)
        noise = torch.full_like(x, .1)
        t, previous = torch.tensor([8, 8]), torch.tensor([4, 4])
        alpha, alpha_prev = diffusion.alphas_cumprod[8], diffusion.alphas_cumprod[4]
        sigma = .5 * ((1-alpha_prev)/(1-alpha)*(1-alpha/alpha_prev)).sqrt()
        x0 = diffusion.predict_start_from_noise(x, t, noise)
        expected = alpha_prev.sqrt()*x0 + (1-alpha_prev-sigma.square()).sqrt()*noise
        with patch('torch.randn_like', side_effect=lambda value: torch.zeros_like(value)):
            actual = diffusion.p_sample_ddim(lambda *a, **k: noise, x, t, previous, eta=.5, clip_denoised=False)
        torch.testing.assert_close(actual, expected)

    def test_raw_roundtrip_uint16_big_endian_and_axis_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'core.raw'
            data = np.linspace(0, 1, 120, dtype=np.float32).reshape(4, 5, 6)
            save_raw_volume(data, str(path), dict(dtype='uint16', byte_order='big', spacing=[1.03]*3, spacing_unit='um'))
            image = load_scalar_image(str(path))
            np.testing.assert_allclose(image.data[0].numpy(), data, atol=1/65535)
            torch.testing.assert_close(to_storage_tensor(to_model_tensor(image)), image.data*2-1)
            np.testing.assert_allclose(np.diag(image.affine)[:3], [.00103]*3)
            with self.assertRaises(ValueError):
                save_raw_volume(np.full((2,2,2), np.nan), str(path), {})

    def test_raw_sidecar_overrides_directory_default(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'raw_meta.json').write_text(json.dumps(dict(shape=[2,2,2], dtype='uint16', value_range=[0,100])))
            path = directory/'core.raw'
            np.arange(8, dtype=np.uint16).tofile(path)
            Path(str(path)+'.json').write_text(json.dumps(dict(value_range=[0,7])))
            self.assertEqual(load_scalar_image(str(path)).data.max().item(), 1.)

    def test_raw_loading_without_json_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'core.raw'
            data = np.arange(120, dtype=np.uint16).reshape(4, 5, 6)
            data.astype('>u2').tofile(path)
            metadata = dict(shape=[4, 5, 6], dtype='uint16', byte_order='big',
                            value_range=[0, 119], spacing=[1.03] * 3, spacing_unit='um')
            image = load_scalar_image(str(path), raw_meta=metadata)
            np.testing.assert_allclose(image.data[0].numpy(), data / 119., atol=1e-7)
            np.testing.assert_allclose(np.diag(image.affine)[:3], [.00103] * 3)
            self.assertFalse(list(Path(directory).glob('*.json')))

    def test_direct_latent_directory_and_pt_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / 'latent.nii.gz'
            expected = torch.randn(2, 4, 6, 8)
            tio.ScalarImage(tensor=expected).save(str(path))
            torch.save(dict(ae_codebook_sha256='matching', downsample=[8] * 3), str(path) + '.meta.pt')
            dataset = Singleres_dataset(str(directory), resolution=(4, 6, 8), class_id=1)
            latent, label, _ = dataset[0]
            torch.testing.assert_close(latent, expected)
            self.assertEqual(label.item(), 1)
            self.assertEqual(dataset.latent_metadata[str(path)]['ae_codebook_sha256'], 'matching')
            self.assertFalse(list(directory.glob('*.json')))

    def test_latent_manifest_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tio.ScalarImage(tensor=torch.randn(2, 4, 6, 8)).save(str(directory/'latent.nii.gz'))
            manifest = directory/'data.json'
            manifest.write_text(json.dumps({'0': {'latents': str(directory)}}))
            dataset = Singleres_dataset(str(manifest), resolution=(4,6,8))
            self.assertEqual(dataset[0][0].shape, (2,4,6,8))
            wrong = Singleres_dataset(str(manifest), resolution=(8,8,8))
            with self.assertRaises(ValueError):
                wrong[0]

    def test_patch_export_preserves_spatial_assembly(self):
        # A toy encoder isolates export plumbing without changing or loading the VAE.
        class AE:
            embedding_dim = 1
            cfg = argparse.Namespace(model=argparse.Namespace(downsample=[2,2,2]))
            codebook = argparse.Namespace(embeddings=torch.tensor([[-1.], [1.]]))
            def encode(self, x, **kwargs):
                return torch.nn.functional.avg_pool3d(x, 2)
        sample = torch.arange(512).float().reshape(1,8,8,8)/511
        latent = encode_volume(sample, AE(), 4, 2, 'cpu', False)
        torch.testing.assert_close(latent, torch.nn.functional.avg_pool3d(sample.unsqueeze(0),2)[0])

    def test_tiled_decoder_uses_checkpoint_compression_and_rejects_truncation(self):
        class AE:
            cfg = argparse.Namespace(model=argparse.Namespace(downsample=[4,4,4]))
            def decode_sliding(self, latent, **kwargs):
                return kwargs
        ae = AE()
        latent = torch.empty(1,2,16,16,16)
        result = decode_latent(ae, latent, mode='sliding', patch_size=64)
        self.assertEqual(result['compress_ratio'], 4)
        with self.assertRaises(ValueError):
            decode_latent(ae, latent, mode='sliding', patch_size=128)

    def test_gaussian_decode_dispatch_and_sliding_batch_options(self):
        class AE:
            cfg = argparse.Namespace(model=argparse.Namespace(downsample=[4,4,4]))
            def decode_sliding_gaussian(self, latent, **kwargs):
                return kwargs
            def decode_sliding(self, latent, **kwargs):
                return kwargs
        latent = torch.empty(1, 2, 7, 4, 5)
        result = decode_latent(AE(), latent, mode='gaussian', patch_size=256, overlap_ratio=.25)
        self.assertEqual(result['compress_ratio'], 4)
        self.assertEqual(result['overlap_ratio'], .25)
        self.assertTrue(result['quantize'])
        result = decode_latent(AE(), torch.empty(1,2,16,16,16), mode='sliding', patch_size=64, sliding_window=3)
        self.assertEqual(result['sliding_window'], 3)
        with self.assertRaises(ValueError):
            decode_latent(AE(), latent, mode='other')

    def test_generation_raw_and_nifti_have_same_intensities_and_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)/'sample'
            model_volume = torch.linspace(-1,1,120).reshape(1,6,4,5)
            save_volume(model_volume, base, save_format='both')
            raw = load_scalar_image(str(base)+'.raw')
            nii = load_scalar_image(str(base)+'.nii.gz')
            torch.testing.assert_close(raw.data, nii.data, atol=1/65535, rtol=0)
            np.testing.assert_allclose(raw.affine, nii.affine, atol=1e-9)

    def test_checkpoint_reconstructs_saved_architecture(self):
        cfg = dict(dim=24, dim_mults=(1,1,2), channels=2, cond_classes=1,
                   sub_volume_size=(4,4,4), use_sparse_linear_attn=(0,0,0),
                   resnet_groups=6, DiT_num_heads=4)
        network = MSP_DiffNet(**cfg)
        checkpoint = dict(ema=network.state_dict(), model_config=cfg,
                          diffusion_config=dict(channels=2, timesteps=10, loss_type='l1'))
        restored, diffusion, restored_cfg = from_checkpoint(checkpoint, argparse.Namespace(), 'cpu')
        self.assertFalse(restored.training)
        self.assertEqual(diffusion.num_timesteps, 10)
        self.assertEqual(restored_cfg, cfg)

    def test_auto_device_has_cpu_fallback(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device().type, 'cpu')

    def test_legacy_checkpoint_restores_saved_local_window_and_patch(self):
        cfg = dict(dim=24, dim_mults=(1, 1, 2), channels=2, cond_classes=1,
                   sub_volume_size=(4, 4, 4), patch_size=1, use_sparse_linear_attn=(0, 0, 0),
                   resnet_groups=6, DiT_num_heads=4)
        network = MSP_DiffNet(**cfg)
        saved = argparse.Namespace(model_dim=24, dim_mults=[1, 1, 2], volume_channels=2,
                                   num_classes=1, use_attn=[0, 0, 0], vq_size=64,
                                   num_dit=1, patch_size=1, sub_volume_size=[4, 4, 4],
                                   resnet_groups=6, DiT_num_heads=4, timesteps=10, loss_type='l1')
        caller = argparse.Namespace(sub_volume_size=None)
        checkpoint = dict(ema={'module.' + key: value for key, value in network.state_dict().items()}, args=saved)
        restored, diffusion, restored_cfg = from_checkpoint(checkpoint, caller, 'cpu')
        self.assertEqual(restored_cfg['patch_size'], 1)
        self.assertEqual(restored_cfg['sub_volume_size'], (4, 4, 4))
        self.assertEqual(diffusion.num_timesteps, 10)
        for key, value in network.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_explicit_reconstruction_raw_and_nifti_have_same_intensities(self):
        from evaluation import ae_encode_decode_recon as reconstruction

        class AE:
            def eval(self):
                return self

            def patch_encode_sliding(self, sample, **kwargs):
                return sample

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'core.nii.gz'
            affine = np.diag([.00103, .002, .003, 1.])
            tio.ScalarImage(tensor=torch.full((1, 4, 6, 8), .5), affine=affine).save(str(source))
            predicted = torch.linspace(-1.2, 1.2, 192).reshape(1, 1, 8, 4, 6)
            args = argparse.Namespace(
                ae_ckpt='test-fixture', data_path=str(source), output_dir=str(directory / 'output'),
                device='cpu', max_files=0, patch_size_encode=2, decode_mode='full',
                save_format='both', sample_raw_dtype='uint16', sample_raw_byte_order='big', print_stats=False)
            with patch.object(reconstruction, 'load_autoencoder', return_value=AE()), \
                 patch.object(reconstruction, 'decode_latent', return_value=predicted):
                reconstruction.main(args)
            base = directory / 'output' / 'core_recon'
            raw = load_scalar_image(str(base) + '.raw')
            nii = load_scalar_image(str(base) + '.nii.gz')
            expected = to_storage_tensor(((predicted[0] + 1) / 2).clamp(0, 1))
            torch.testing.assert_close(nii.data, expected, atol=1e-7, rtol=0)
            torch.testing.assert_close(raw.data, nii.data, atol=1/65535, rtol=0)
            np.testing.assert_allclose(raw.affine, nii.affine, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
