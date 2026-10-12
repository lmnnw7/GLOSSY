"""RAW slicing/cropping regressions for the consolidated notebook implementation."""
import argparse
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
from preprocessing.core_subvolume_extraction import extract_subvolumes, extract_crop
from preprocessing.prepare_core_blocks import prepare


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / ('_preprocessing_test_' + uuid.uuid4().hex)
        self.root.mkdir()

    def tearDown(self):
        workspace = Path(__file__).resolve().parents[1]
        if self.root.resolve().parent != workspace:
            raise RuntimeError('Unexpected test cleanup path')
        shutil.rmtree(self.root)

    def source(self, shape=(8, 12, 16), order='little'):
        array = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        path = self.root / ('source_' + order + '.raw')
        array.astype('<u2' if order == 'little' else '>u2').tofile(path)
        return path, array

    def test_slab_blocks_match_original_array_slicing(self):
        source, array = self.source()
        rows = extract_subvolumes(source, self.root / 'blocks', array.shape, (4, 4, 4))
        self.assertEqual(len(rows), 24)
        for row in rows:
            x, y, z = row['x'], row['y'], row['z']
            actual = np.fromfile(self.root / 'blocks' / row['file'], dtype='<u2').reshape(4, 4, 4)
            np.testing.assert_array_equal(actual, array[x:x + 4, y:y + 4, z:z + 4])

    def test_zyx_source_becomes_xyz_without_intensity_changes(self):
        source, array = self.source(order='big')
        rows = extract_subvolumes(source, self.root / 'blocks', array.shape, (4, 4, 4),
                                 storage_order='ZYX', byte_order='big')
        xyz = array.transpose(2, 1, 0)
        for row in rows:
            x, y, z = row['x'], row['y'], row['z']
            actual = np.fromfile(self.root / 'blocks' / row['file'], dtype='>u2').reshape(4, 4, 4)
            np.testing.assert_array_equal(actual, xyz[x:x + 4, y:y + 4, z:z + 4])

    def test_overlapping_stride_uses_complete_windows(self):
        source, array = self.source(shape=(8, 8, 8))
        rows = extract_subvolumes(source, self.root / 'blocks', array.shape, (4, 4, 4), stride=(2, 2, 2))
        self.assertEqual(len(rows), 27)
        self.assertEqual({row['x'] for row in rows}, {0, 2, 4})

    def test_crop_matches_original_bounds_and_xyz_conversion(self):
        source, array = self.source()
        output = self.root / 'crop.raw'
        shape = extract_crop(source, output, array.shape, (1, 2, 3), (2, 3, 4), storage_order='ZYX')
        self.assertEqual(shape, (4, 3, 2))
        actual = np.fromfile(output, dtype='<u2').reshape(shape)
        np.testing.assert_array_equal(actual, array[1:3, 2:5, 3:7].transpose(2, 1, 0))

    def test_truncated_input_and_out_of_bounds_crop_are_rejected(self):
        source, array = self.source()
        with self.assertRaises(ValueError):
            extract_subvolumes(source, self.root / 'blocks', (9, 12, 16))
        with self.assertRaises(ValueError):
            extract_crop(source, self.root / 'crop.raw', array.shape, (7, 0, 0), (2, 3, 4))

    def test_existing_output_is_protected(self):
        source, array = self.source()
        output = self.root / 'crop.raw'
        extract_crop(source, output, array.shape, (0, 0, 0), (2, 3, 4))
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            extract_crop(source, output, array.shape, (1, 0, 0), (2, 3, 4))
        self.assertEqual(original, output.read_bytes())

    def test_preparation_split_retains_all_voxels_and_disjoint_groups(self):
        source, array = self.source(shape=(16, 16, 16))
        output = self.root / 'prepared'
        args = argparse.Namespace(input=str(source), output_root=str(output), shape=list(array.shape),
                                  block_size=4, byte_order='little', validation_count=8, test_count=8, seed=216)
        rows = prepare(args)
        groups = [{path.name for path in (output / group).glob('*.raw')}
                  for group in ('train', 'validation', 'test')]
        self.assertEqual([len(group) for group in groups], [48, 8, 8])
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        reconstructed = np.empty_like(array)
        for row in rows:
            x, y, z = row['x'], row['y'], row['z']
            block = np.fromfile(output / 'all_blocks' / row['file'], dtype='<u2').reshape(4, 4, 4)
            reconstructed[x:x + 4, y:y + 4, z:z + 4] = block
        np.testing.assert_array_equal(reconstructed, array)


if __name__ == '__main__':
    unittest.main()
