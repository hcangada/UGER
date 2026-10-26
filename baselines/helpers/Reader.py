import logging
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from utils import utils

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from uger.data import merge_edges, read_interactions


class Reader:
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', default='../data/', help='Dataset root relative to the working directory.')
        parser.add_argument('--suffix', default='interactions', help='Interaction subdirectory.')
        parser.add_argument('--dataset', default='Amazon-cds', help='Dataset directory name.')
        parser.add_argument('--sep', default='\t', help='Input field delimiter.')
        parser.add_argument('--train_ratio', type=float, default=0.6, help='Initial chronological training fraction.')
        parser.add_argument('--fname', default='freq', help='Input filename without .csv.')
        parser.add_argument('--s_fname', default='', help='Generated snapshot directory.')
        parser.add_argument('--n_snapshots', type=int, default=5, help='Number of subsequent evaluation windows.')
        parser.add_argument('--split_type', choices=['size'], default='size', help='Chronological partition by interaction count.')
        parser.add_argument('--val_ratio', type=float, default=0.5, help='Validation fraction of each subsequent evaluation block.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.suffix = args.suffix
        self.dataset = args.dataset
        self.train_ratio = args.train_ratio
        self.batch_size = args.batch_size
        self.fname = args.fname
        self.s_fname = args.s_fname
        self.random_seed = args.random_seed
        self.n_snapshots = args.n_snapshots
        self.val_ratio = args.val_ratio
        if not 0 < self.train_ratio < 1 or not 0 < self.val_ratio < 1 or self.n_snapshots < 1:
            raise ValueError('Invalid chronological split settings.')
        self._read_data()
        self.n_users = int(self.data_df['user_id'].max()) + 1
        self.n_items = int(self.data_df['item_id'].max()) + 1
        self.dataset_size = len(self.data_df)
        self.n_batches = math.ceil(self.dataset_size / self.batch_size)
        self.path = os.path.join(self.prefix, self.dataset, self.suffix, self.s_fname)
        os.makedirs(self.path, exist_ok=True)
        self._set_snap_boundaries()
        self.buffer_size = self.incre_size if args.buffer_size is None else args.buffer_size
        self._save_snapshot_files()
        self.user_list = self.data_df['user_id'].to_numpy()
        initial = self.data_df.iloc[:self.snap_boundaries[0]]
        self.user_clicked_set = initial.groupby('user_id')['item_id'].unique().to_dict()
        logging.info('Users: %d, items: %d, interactions: %d', self.n_users, self.n_items, self.dataset_size)

    def _set_snap_boundaries(self):
        self.n_train_batches = int(self.dataset_size * self.train_ratio)
        self.n_test_batches = self.dataset_size - self.n_train_batches
        blocks = np.array_split(np.arange(self.n_train_batches, self.dataset_size), self.n_snapshots)
        if self.n_train_batches < 1 or any(len(block) < 2 for block in blocks):
            raise ValueError('Each evaluation block requires at least two interactions.')
        self.incre_size = len(blocks[0])
        self.snap_boundaries = [self.n_train_batches] + [int(block[-1]) + 1 for block in blocks[:-1]]

    def _save_snapshot_files(self):
        self.snapshots_path = os.path.join(self.path, 'snapshots')
        os.makedirs(self.snapshots_path, exist_ok=True)
        for index, boundary in enumerate(self.snap_boundaries):
            previous = 0 if index == 0 else self.snap_boundaries[index - 1]
            end = self.dataset_size if index + 1 == self.n_snapshots else self.snap_boundaries[index + 1]
            cut = boundary + max(1, min(int((end - boundary) * self.val_ratio), end - boundary - 1))
            blocks = {
                'hist': self.data_df.iloc[:boundary],
                'incre': self.data_df.iloc[previous:boundary],
                'val': self.data_df.iloc[boundary:cut],
                'test': self.data_df.iloc[cut:end],
            }
            for kind, frame in blocks.items():
                utils.write_interactions_to_file(os.path.join(self.snapshots_path, f'{kind}_block{index}'), frame.to_numpy(dtype=np.int64))
        terminal = {
            'hist': self.data_df,
            'incre': self.data_df.iloc[self.snap_boundaries[-1]:],
        }
        for kind, frame in terminal.items():
            utils.write_interactions_to_file(os.path.join(self.snapshots_path, f'{kind}_block{self.n_snapshots}'), frame.to_numpy(dtype=np.int64))

    def _read_data(self):
        source = os.path.join(self.prefix, self.dataset, self.suffix, self.fname + '.csv')
        import hashlib

        edges, self.user_mapping, self.item_mapping = read_interactions(source, self.sep)
        self.data_df = pd.DataFrame(merge_edges(edges), columns=['user_id', 'item_id', 'timestamp'])
        self.fingerprint = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if self.data_df.empty or (self.data_df[['user_id', 'item_id']] < 0).any().any():
            raise ValueError('Input requires nonnegative user and item IDs and at least one interaction.')
