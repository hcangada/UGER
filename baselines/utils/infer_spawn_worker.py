import os
import sys
import numpy as np
import torch
import pandas as pd
from time import perf_counter
from torch.utils.data import DataLoader
from models import Dataloader
from utils import utils

def stable_topk_by_score(samples_with_score: np.ndarray, k: int) -> np.ndarray:
    k = int(min(k, len(samples_with_score)))
    if k <= 0:
        return samples_with_score[:0, :3].astype(np.int64)
    u = samples_with_score[:, 0].astype(np.int64)
    i = samples_with_score[:, 1].astype(np.int64)
    t = samples_with_score[:, 2].astype(np.int64)
    score = samples_with_score[:, 3].astype(np.float64)
    order = np.lexsort((t, i, u, -score))
    top = samples_with_score[order[:k], :3].astype(np.int64)
    return top

def stable_sort_u_i_t(arr_u_i_t: np.ndarray) -> np.ndarray:
    if len(arr_u_i_t) == 0:
        return arr_u_i_t
    u = arr_u_i_t[:, 0].astype(np.int64)
    i = arr_u_i_t[:, 1].astype(np.int64)
    t = arr_u_i_t[:, 2].astype(np.int64)
    order = np.lexsort((t, i, u))
    return arr_u_i_t[order]

def dict_to_namespace(d: dict):

    class NS:
        pass
    ns = NS()
    for (k, v) in d.items():
        setattr(ns, k, v)
    return ns

def import_model_ctor(model_ctor: str):
    (mod_path, cls_name) = model_ctor.rsplit('.', 1)
    mod = __import__(mod_path, fromlist=[cls_name])
    return getattr(mod, cls_name)

def resolve_prev_data(args, corpus, dyn: str, idx: int):
    prev_data = Dataloader.Dataset(args, corpus, 'hist', idx - 1)
    if 'curr' in args.dyn_method and idx > 1:
        prev_data = Dataloader.Dataset(args, corpus, str(args.dyn_method) + '_' + str(args.random_seed), idx - 1)
    return prev_data

def run_sampling_worker(payload: dict):
    tmpdir = payload.get('tmpdir', '')
    if tmpdir:
        os.environ['TMPDIR'] = tmpdir
    cublas_ws = payload.get('cublas_ws', '')
    if cublas_ws:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = cublas_ws
    gpu = str(payload.get('gpu', '0'))
    if gpu in ['', '-1', 'cpu', 'CPU']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    torch.set_num_threads(1)
    seed = int(payload['seed'])
    utils.fix_seed(seed)
    corpus_path = payload['corpus_path']
    corpus = pd.read_pickle(corpus_path)
    args = dict_to_namespace(payload['args_dict'])
    idx = int(payload['idx'])
    dyn = payload['dyn']
    data_type = payload['data_type']
    data_dict = Dataloader.Dataset(args, corpus, data_type, idx)
    prev_data = resolve_prev_data(args, corpus, dyn, idx)
    sample_size = min(int(args.buffer_size), len(prev_data.train_data))
    prev_dl = DataLoader(prev_data, batch_size=int(args.batch_size), shuffle=False, num_workers=0, pin_memory=False)
    ModelCls = import_model_ctor(payload['model_ctor'])
    prev_model = ModelCls(args, corpus, prev_data, idx - 1)
    prev_model.to(prev_model._device)
    prev_model.load_model(prev_model.model_path + f'_snap{idx - 1}')
    if 'infer' in dyn:
        prev_train_data = prev_data.train_data
        start = perf_counter()
        infer_scores = utils.compute_infer_score(prev_model=prev_model, prev_dl=prev_dl)
        top_u_i_t = stable_topk_by_score(infer_scores, sample_size)
        pool = np.concatenate([data_dict.train_data, top_u_i_t], axis=0)
        pool = pool.astype(np.int64)
        pool = stable_sort_u_i_t(pool)
        selection_seconds = perf_counter() - start
        out_path = os.path.join(corpus.snapshots_path, str(args.dyn_method) + '_' + str(args.random_seed) + f'_block_{idx}')
        utils.write_interactions_to_file(out_path, pool)
        with open(out_path + '.seconds', 'w') as destination:
            destination.write(str(selection_seconds))
    else:
        raise ValueError(f'Unknown dyn: {dyn}')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return
