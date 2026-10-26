import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from helpers.Reader import Reader
import Inference
from uger.data import load_interaction_stream


class Scores(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = torch.device('cpu')

    def computer(self):
        return torch.ones(3, 1), torch.tensor([[0.0], [4.0], [2.0], [3.0]])


class ProtocolTests(unittest.TestCase):
    def test_reader_matches_main_count_split_and_catalogue_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'Example' / 'interactions'
            source.mkdir(parents=True)
            path = source / 'freq.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream, delimiter='\t')
                writer.writerow(['user_id', 'item_id', 'timestamp'])
                for index in range(197):
                    writer.writerow([500 + index % 11, 900 + index % 29, 1500000000 + index / 1000])
            args = SimpleNamespace(path=directory, suffix='interactions', dataset='Example',
                                   train_ratio=0.6, batch_size=16, fname='freq', s_fname='split',
                                   random_seed=2021, n_snapshots=5, val_ratio=0.5, sep='\t',
                                   buffer_size=None)
            reader = Reader(args)
            main = load_interaction_stream(path, 0.6, 5, 0.1, 'count')
            self.assertEqual(reader.n_users, main.num_users)
            self.assertEqual(reader.n_items, main.num_items)
            self.assertEqual(reader.buffer_size, len(main.snapshots[1]))
            for index in range(6):
                actual = np.loadtxt(Path(reader.snapshots_path) / f'incre_block{index}', dtype=np.int64, ndmin=2)
                np.testing.assert_array_equal(actual, main.snapshots[index])
            for index in range(5):
                validation = np.loadtxt(Path(reader.snapshots_path) / f'val_block{index}', dtype=np.int64, ndmin=2)
                test = np.loadtxt(Path(reader.snapshots_path) / f'test_block{index}', dtype=np.int64, ndmin=2)
                np.testing.assert_array_equal(np.concatenate((validation, test)), main.snapshots[index + 1])
            self.assertFalse((Path(reader.snapshots_path) / 'test_block5').exists())

    def test_full_catalogue_and_shared_validation_test_history_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'hist_block0').write_text('0\t0\t0\n1\t1\t1\n')
            (directory / 'val_block0').write_text('0\t1\t2\n')
            (directory / 'test_block0').write_text('0\t1\t3\n0\t0\t4\n1\t1\t5\n2\t3\t6\n')
            args = SimpleNamespace(batch_size=2)
            corpus = SimpleNamespace(snapshots_path=str(directory), n_items=4)
            validation = Inference.Test(args, Scores(), corpus, 'val', 0)
            test = Inference.Test(args, Scores(), corpus, 'test', 0)
            self.assertEqual(validation[0][0], 1.0)
            self.assertEqual(test[0][0], 1.0)
            counts = Inference.EVALUATION_RECORDS[0, 'test']
            self.assertEqual(counts['evaluated_users'], 2)
            self.assertEqual(counts['excluded_users'], 1)
            self.assertEqual(counts['masked_targets'], 2)

    def test_ground_truth_duplicates_do_not_reduce_recall(self):
        values = Inference.computeTopNAccuracy([[1, 1]], torch.tensor([[1, 2]]), [1, 10], 3, 'cpu')
        self.assertEqual(values[0], [1.0, 1.0])
        self.assertEqual(values[1], [1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
