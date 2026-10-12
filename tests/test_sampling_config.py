"""Sampling CLI tests runnable without PyTorch or GPU dependencies."""
import argparse
import unittest

from sampling_config import (add_sampling_arguments,
                             resolve_sampling_options, decode_options, validate_decode_settings)


class SamplingCommandTests(unittest.TestCase):
    def parse(self, flags=None, preview=False):
        parser = argparse.ArgumentParser()
        add_sampling_arguments(parser, preview=preview)
        return resolve_sampling_options(parser.parse_args(flags or []), preview=preview)

    def test_parser_exposes_full_defaults_without_resolving(self):
        for preview in (False, True):
            parser = argparse.ArgumentParser()
            add_sampling_arguments(parser, preview=preview)
            options = parser.parse_args([])
            prefix = 'sample_' if preview else ''
            self.assertEqual(getattr(options, prefix + 'decode_mode'), 'full')
            self.assertEqual(getattr(options, prefix + 'decode_patch_size'), 256)
            self.assertEqual(getattr(options, prefix + 'decode_sliding_window'), 1)
            self.assertEqual(getattr(options, prefix + 'decode_overlap_ratio'), .5)
        generation, preview = self.parse(), self.parse(preview=True)
        self.assertEqual(generation.sampling_strategy, 'ddpm')
        self.assertEqual(preview.sample_sampling_strategy, 'ddim')
        self.assertEqual(generation.ddim_steps, 100)

    def test_cli_selects_each_mode_and_sampling_settings(self):
        for mode in ('full', 'sliding', 'gaussian'):
            with self.subTest(mode=mode):
                options = self.parse(['--decode-mode', mode, '--decode-patch-size', '64',
                                      '--decode-sliding-window', '2', '--decode-overlap-ratio', '.25',
                                      '--sampling-strategy', 'ddim', '--ddim_steps', '25', '--eta', '.2'])
                self.assertEqual(decode_options(options), dict(mode=mode, patch_size=64,
                                                               sliding_window=2, overlap_ratio=.25))
                self.assertEqual(options.sampling_strategy, 'ddim')
                self.assertEqual(options.ddim_steps, 25)
                self.assertEqual(options.eta, .2)

    def test_preview_flags_switch_modes_independently(self):
        options = self.parse(['--sample-decode-mode', 'gaussian', '--sample-eta', '.2',
                              '--sample-decode-overlap-ratio', '.75'], preview=True)
        self.assertEqual(decode_options(options, preview=True)['mode'], 'gaussian')
        self.assertEqual(options.sample_eta, .2)
        self.assertEqual(options.sample_decode_overlap_ratio, .75)
        self.assertEqual(self.parse().decode_mode, 'full')

    def test_direct_callers_use_in_code_defaults(self):
        for preview in (False, True):
            args = resolve_sampling_options(argparse.Namespace(), preview=preview)
            self.assertEqual(decode_options(args, preview=preview),
                             dict(mode='full', patch_size=256, sliding_window=1, overlap_ratio=.5))
        args = resolve_sampling_options(argparse.Namespace(decode_mode='gaussian', decode_patch_size=None))
        self.assertEqual(args.decode_mode, 'gaussian')
        self.assertEqual(args.decode_patch_size, 256)

    def test_invalid_values_fail_before_sampling(self):
        for name, value in (
            ('decode_mode', 'other'), ('decode_patch_size', 0), ('decode_patch_size', True),
            ('decode_sliding_window', -1), ('decode_overlap_ratio', 1), ('decode_overlap_ratio', -.1),
            ('decode_overlap_ratio', float('nan')), ('sampling_strategy', 'other'),
            ('ddim_steps', 1.5), ('eta', -1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                resolve_sampling_options(argparse.Namespace(**{name: value}))

    def test_sliding_rejects_truncation_and_uses_checkpoint_ratio(self):
        self.assertEqual(validate_decode_settings((1, 2, 16, 16, 16), [4] * 3, 'sliding', 64), 64)
        for shape, ratios, patch in (
            ((1, 2, 16, 16, 16), [4] * 3, 128),
            ((1, 2, 17, 16, 16), [4] * 3, 64),
            ((1, 2, 16, 16, 16), [4, 8, 4], 64)):
            with self.assertRaises(ValueError):
                validate_decode_settings(shape, ratios, 'sliding', patch)

    def test_gaussian_fits_small_and_anisotropic_inputs(self):
        for ratio in (4, 8):
            patch = validate_decode_settings((2, 8, 7, 4, 5), [ratio] * 3, 'gaussian', 256)
            self.assertEqual(patch, 4 * ratio)
            self.assertEqual(validate_decode_settings((1, 8, 1, 2, 1), [ratio] * 3, 'gaussian', 256, overlap_ratio=.99), ratio)
        with self.assertRaises(ValueError):
            validate_decode_settings((1, 8, 7, 4, 5), [8] * 3, 'gaussian', 255)

    def test_reconstruction_accepts_auto_and_decode_only_options(self):
        parser = argparse.ArgumentParser()
        add_sampling_arguments(parser, include_sampling=False, allow_auto=True)
        self.assertEqual(parser.parse_args([]).decode_mode, 'full')
        args = resolve_sampling_options(parser.parse_args(['--decode-mode', 'auto']), include_sampling=False, allow_auto=True)
        self.assertEqual(decode_options(args, mode='gaussian')['mode'], 'gaussian')
        self.assertFalse(hasattr(args, 'sampling_strategy'))


if __name__ == '__main__':
    unittest.main()
