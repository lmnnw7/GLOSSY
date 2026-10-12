"""Directory inputs, RAW command-line metadata and latent pairing without JSON files."""
import argparse
from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest.mock import patch

from config_io import (add_raw_arguments, add_vae_input_arguments, image_roots, project_path,
                       raw_options, read_data_roots, vae_input_options, validate_latent_metadata)


class DirectInputTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        add_raw_arguments(parser)
        return parser

    def test_directory_and_volume_inputs_do_not_read_json(self):
        with patch('config_io.load_json_object', side_effect=AssertionError('JSON must not be read')):
            self.assertEqual(image_roots('data/core_images'), [str(project_path('data/core_images'))])
            for path in ('data/core_images_latents', 'data/latent.nii.gz'):
                roots = read_data_roots(path, numeric_labels=True, direct_kind='latents', class_id=2)
                self.assertEqual(roots, {'2': {'latents': str(project_path(path))}})

    def test_direct_class_ids_are_nonnegative_integers(self):
        for label in (-1, True, 1.5, '1'):
            with self.subTest(label=label), self.assertRaises(ValueError):
                read_data_roots('data', direct_kind='images', class_id=label)

    def test_legacy_json_mapping_stays_optional(self):
        with patch('config_io.load_json_object', return_value={'0': 'data/core_images'}):
            roots = read_data_roots('config/legacy.json', numeric_labels=True, direct_kind='latents')
        self.assertEqual(roots, {'0': str(project_path('data/core_images'))})

    def test_raw_cli_metadata_and_no_input_defaults(self):
        parser = self.parser()
        self.assertIsNone(raw_options(parser.parse_args([])))
        options = raw_options(parser.parse_args(['--raw-shape', '128', '64', '192', '--raw-dtype', 'uint16',
                                                '--raw-byte-order', 'big', '--raw-value-range', '0', '4095',
                                                '--raw-spacing', '1.03', '1.03', '1.1', '--raw-spacing-unit', 'um']))
        self.assertEqual(options['shape'], [128, 64, 192])
        self.assertEqual(options['value_range'], [0, 4095])
        self.assertEqual(options['byte_order'], 'big')
        self.assertEqual(options['spacing_unit'], 'um')
        self.assertEqual(raw_options(parser.parse_args(['--raw-dtype', 'float32'])), {'dtype': 'float32'})

    def test_invalid_raw_cli_values_fail_before_loading_models(self):
        parser = self.parser()
        for flags in (['--raw-shape', '0', '64', '64'], ['--raw-value-range', '1', '0'],
                      ['--raw-scale', '-1'], ['--raw-spacing', '1', 'nan', '1']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                raw_options(parser.parse_args(flags))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--raw-value-range', '0', '1', '--raw-scale', '1'])

    def test_vae_options_override_paths_and_checkpoint(self):
        parser = argparse.ArgumentParser()
        add_vae_input_arguments(parser)
        args = parser.parse_args(['--data-path', 'data/cores', '--results-dir', 'results/vae',
                                  '--ckpt', 'checkpoints/stage1.ckpt', '--raw-shape', '128', '128', '128'])
        options = vae_input_options(args)
        self.assertEqual(options['data_path'], 'data/cores')
        self.assertEqual(options['results_dir'], 'results/vae')
        self.assertEqual(options['checkpoint_path'], 'checkpoints/stage1.ckpt')
        self.assertEqual(options['raw_meta']['shape'], [128] * 3)

    def test_directory_provenance_uses_the_same_pairing_rules(self):
        record = dict(ae_codebook_sha256='matching', downsample=[8] * 3,
                      latent_axis_order='CDHW', latent_normalization='codebook_minmax', quantized=True)
        validate_latent_metadata([record, {'latents': 'data/latents'}], 'matching', [8] * 3)
        validate_latent_metadata([], 'matching', [8] * 3)

    def test_incompatible_or_mixed_latent_provenance_is_rejected(self):
        for record in (dict(ae_codebook_sha256='other'), dict(downsample=[4] * 3),
                       dict(latent_axis_order='CXYZ'), dict(latent_normalization='other'),
                       dict(quantized=1)):
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_latent_metadata([record], 'matching', [8] * 3)
        with self.assertRaises(ValueError):
            validate_latent_metadata([{'quantized': True}, {'quantized': False}], 'matching', [8] * 3)


if __name__ == '__main__':
    unittest.main()
