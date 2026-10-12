"""Real JSON parsing, path resolution and RAW validation without ML dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config_io import (PROJECT_ROOT, image_roots, load_json_object, project_path,
                       read_data_roots, validate_raw_meta)


class JSONConfigTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures in the known workspace rather than an external temp folder.
        self.directory = tempfile.TemporaryDirectory(prefix='.json-config-test-', dir=PROJECT_ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        if not self.root.is_relative_to(PROJECT_ROOT):
            raise RuntimeError('JSON test fixtures must remain inside the project')
        self.path = self.root / 'settings.json'

    def write(self, value):
        self.path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
        return self.path

    def test_utf8_bom_and_unicode_paths(self):
        self.path.write_text('{"sample": "data/cor\u00e9"}', encoding='utf-8-sig')
        self.assertEqual(load_json_object(self.path), {'sample': 'data/cor\u00e9'})

    def test_duplicate_keys_and_nonfinite_numbers_are_rejected(self):
        for content in ('{"0": "a", "0": "b"}', '{"0": {"images":"a", "images":"b"}}',
                        '{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}',
                        '{"value": 1e999}', '[]', '{bad json}'):
            with self.subTest(content=content):
                self.path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_json_object(self.path)

    def test_legacy_and_explicit_data_mappings_are_equivalent(self):
        expected = str(PROJECT_ROOT / 'data' / 'core_images')
        for value in ('data/core_images', ['data/core_images'], {'images': ['data/core_images']}):
            self.assertEqual(image_roots(self.write({'sandstone': value})), [expected])

    def test_relative_paths_use_project_root_not_manifest_directory(self):
        path = self.write({'0': {'images': ['data/core_images', './data/core_images'],
                                  'latents': 'data/core_images_latents', 'quantized': True}})
        roots = read_data_roots(path, numeric_labels=True)
        self.assertEqual(roots['0']['images'], [str(project_path('data/core_images'))])
        self.assertEqual(roots['0']['latents'], str(project_path('data/core_images_latents')))
        self.assertTrue(roots['0']['quantized'])
        with patch('config_io.PROJECT_ROOT', self.root):
            self.assertEqual(project_path('data'), self.root / 'data')

    def test_expanded_absolute_paths_and_single_directory(self):
        self.assertEqual(read_data_roots(self.root), {'0': str(self.root)})
        self.assertEqual(project_path('~/glossy-test'), Path.home() / 'glossy-test')
        self.assertEqual(image_roots(self.write({'sample': str(self.root)})), [str(self.root)])

    def test_empty_or_malformed_sources_are_rejected(self):
        for content in ({}, {'': 'data'}, {'0': ''}, {'0': []}, {'0': ['data', 1]},
                        {'0': {'images': None}}, {'0': {'image': 'data'}}):
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_data_roots(self.write(content))
        with self.assertRaises(ValueError):
            image_roots(self.write({'0': {'latents': 'data/latents'}}))

    def test_numeric_class_ids_are_checked_and_normalized(self):
        roots = read_data_roots(self.write({'01': {'images': 'data'}}), numeric_labels=True)
        self.assertEqual(list(roots), ['1'])
        for content in ({'sandstone': 'data'}, {'-1': 'data'}, {'1.5': 'data'},
                        {'0': 'data', '00': 'other'}):
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_data_roots(self.write(content), numeric_labels=True)

    def test_latent_manifest_provenance_is_retained(self):
        metadata = dict(latents=['data/latents'], downsample=[8, 8, 8], quantized=True,
                        ae_codebook_sha256='abc', latent_normalization='codebook_minmax',
                        latent_axis_order='CDHW')
        loaded = read_data_roots(self.write({'0': metadata}), numeric_labels=True)['0']
        for key in set(metadata) - {'latents'}:
            self.assertEqual(loaded[key], metadata[key])

    def test_raw_defaults_and_aliases_preserve_values(self):
        original = dict(shape=[2, 3, 4], dtype='u16', byte_order='be', axis_order='xyz',
                        value_range=[0, 65535], spacing=[1.03] * 3, spacing_unit='um')
        result = validate_raw_meta(original)
        self.assertEqual(result['dtype'], 'uint16')
        self.assertEqual(result['byte_order'], 'big')
        self.assertEqual(result['axis_order'], 'XYZ')
        self.assertEqual(result['value_range'], original['value_range'])
        self.assertEqual(result['spacing'], original['spacing'])
        self.assertEqual(original['dtype'], 'u16')
        self.assertEqual(validate_raw_meta({'shape': [1, 2, 3]})['spacing_unit'], 'mm')

    def test_invalid_raw_fields_fail_before_data_read(self):
        for field, value in (
                ('shape', [1.5, 2, 3]), ('shape', [True, 2, 3]), ('shape', [0, 2, 3]),
                ('dtype', 'uint32'), ('byte_order', 'typo'), ('axis_order', 'ZYX'),
                ('spacing', [1, -1, 1]), ('spacing', [1, float('inf'), 1]),
                ('spacing_unit', 'cm'), ('value_range', [2, 1]), ('value_range', [0, float('nan')]),
                ('scale', 0), ('scale', True)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_raw_meta({'shape': [1, 2, 3], field: value})
        with self.assertRaises(ValueError):
            validate_raw_meta({})

    def test_shared_raw_defaults_and_per_file_overrides(self):
        defaults = validate_raw_meta({'shape': [2, 2, 2], 'value_range': [0, 100]})
        self.path.write_text('{"value_range": [0, 7]}', encoding='utf-8-sig')
        merged = validate_raw_meta({**defaults, **load_json_object(self.path)})
        self.assertEqual(merged['shape'], [2, 2, 2])
        self.assertEqual(merged['value_range'], [0, 7])

    def test_all_supplied_json_schemas(self):
        config = PROJECT_ROOT / 'config'
        self.assertTrue(image_roots(config / 'REV_VQVAE_data.json'))
        self.assertEqual(list(read_data_roots(config / 'SingleRes_dataset.json', numeric_labels=True)), ['0'])
        raw = validate_raw_meta(load_json_object(config / 'raw_meta.example.json'))
        self.assertEqual(raw['shape'], [128] * 3)
        self.assertEqual(raw['value_range'], [0, 65535])


if __name__ == '__main__':
    unittest.main()
