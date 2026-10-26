import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset as BaseDataset

from utils import utils


class Dataset(BaseDataset):
    def __init__(self, args, corpus, data_type, data_idx):
        self.corpus = corpus
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.time_idx = data_idx
        self.data_type = data_type
        self.base_seed = int(args.random_seed)
        if data_type in {'uniform', 'invdeg'} and data_idx > 0:
            filename = f'{data_type}_block_{data_idx}'
        elif 'infer' in data_type and data_idx > 0:
            filename = f'{args.dyn_method}_{args.random_seed}_block_{data_idx}'
        else:
            filename = f'{data_type}_block{data_idx}'
        self.train_file = os.path.join(corpus.snapshots_path, filename)
        self.train_data = np.asarray(utils.read_data_from_file_int(self.train_file), dtype=np.int64).reshape(-1, 3)
        self.trainUser, self.trainItem, self.trainTime = self.train_data.T
        self.UserItemNet = sp.csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)), shape=(corpus.n_users, corpus.n_items), dtype=np.float32)
        self.UserItemNet.data.fill(1)
        self.Graph = None
        self.user_set = set(self.trainUser.tolist())
        self.item_set = set(self.trainItem.tolist())
        self.num_users, self.num_items = corpus.n_users, corpus.n_items
        self.user_neigh = {user: self.UserItemNet[user].indices.tolist() for user in self.user_set}
        columns = self.UserItemNet.tocsc()
        self.item_neigh = {item: columns[:, item].indices.tolist() for item in self.item_set}
        observed_path = os.path.join(corpus.snapshots_path, f'hist_block{data_idx}')
        observed = np.asarray(utils.read_data_from_file_int(observed_path), dtype=np.int64).reshape(-1, 3)
        self.observed_items = np.unique(observed[:, 1])
        self.observed_by_user = {}
        for user, item, _ in observed:
            self.observed_by_user.setdefault(int(user), set()).add(int(item))
        self._sample_indices = np.asarray([index for index, (user, _, _) in enumerate(self.train_data) if len(self.observed_by_user.get(int(user), set())) < len(self.observed_items)], dtype=np.int64)

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, index):
        return self._get_feed_dict(int(self._sample_indices[index]))

    def get_batch(self, indices):
        if torch.is_tensor(indices):
            indices = indices.tolist()
        samples = [self[int(index)] for index in indices]
        if not samples:
            return {}
        return {key: torch.stack([torch.as_tensor(sample[key]) for sample in samples]) for key in samples[0]}

    def _get_feed_dict(self, index):
        user, item, timestamp = self.train_data[index]
        negatives = self._sample_neg_items(int(user)).reshape(-1)
        return {'user_id': torch.tensor([user], dtype=torch.long), 'item_id': torch.cat((torch.tensor([item], dtype=torch.long), negatives)), 'timestamp': int(timestamp)}

    def _sample_neg_items(self, user_id, rng=None):
        rng = np.random if rng is None else rng
        blocked = self.observed_by_user.get(int(user_id), set())
        available_count = len(self.observed_items) - len(blocked)
        count = int(self.args.num_neg)
        if count < 1 or available_count < 1:
            raise ValueError('Recommendation training requires a positive negative count and an eligible unobserved item.')
        replacement = available_count < count
        chosen, selected = [], set()
        budget = max(128, count * 16)
        while len(chosen) < count and budget > 0:
            draw = min(max(16, (count - len(chosen)) * 2), budget)
            proposals = rng.choice(self.observed_items, size=draw, replace=True)
            budget -= draw
            for value in proposals:
                item = int(value)
                if item in blocked or (not replacement and item in selected):
                    continue
                chosen.append(item)
                selected.add(item)
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            available = np.asarray([item for item in self.observed_items if int(item) not in blocked and (replacement or int(item) not in selected)], dtype=np.int64)
            chosen.extend(rng.choice(available, count - len(chosen), replace=replacement).tolist())
        return torch.tensor(chosen, dtype=torch.long).reshape(1, -1)

    def _sample_neg_items_er(self, user_id, index):
        seed = (self.base_seed * 1000003 + self.time_idx * 10007 + int(index) * 97 + int(user_id) * 13) % 2 ** 32
        return self._sample_neg_items(user_id, np.random.default_rng(seed))

    def _convert_sp_mat_to_sp_tensor(self, matrix):
        coo = matrix.tocoo().astype(np.float32)
        indices = torch.as_tensor(np.stack((coo.row, coo.col)), dtype=torch.long)
        values = torch.as_tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()

    def getSparseGraph(self):
        if self.Graph is None:
            adjacency = sp.bmat([[None, self.UserItemNet], [self.UserItemNet.T, None]], format='csr')
            degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
            inverse = np.zeros_like(degree)
            np.divide(1.0, np.sqrt(degree), out=inverse, where=degree > 0)
            scale = sp.diags(inverse)
            self.Graph = self._convert_sp_mat_to_sp_tensor(scale @ adjacency @ scale).to(self._device)
        return self.Graph
