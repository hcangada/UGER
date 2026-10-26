import os
import torch
import datetime
import numpy as np
import random
from torch.utils.data import DataLoader

def read_data_from_file(filename, flag=0):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if flag == 0:
            data = [line.replace('\n', '').split() for line in lines]
        elif flag == 1:
            data = [line.replace('\n', '').split('::') for line in lines]
    return data

def read_data_from_file_int(filename, flag=0):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if flag == 0:
            data = [str_list_to_int(line.split()) for line in lines]
        elif flag == 1:
            data = [str_list_to_int(line.split('::')) for line in lines]
    return data

def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory - ' + directory)

def get_user_dil_from_edgelist(edges):
    dil = {}
    for edge in edges:
        if dil.get(edge[0]) is None:
            dil[edge[0]] = []
        dil[edge[0]].append(edge[1])
    return dil

def get_user_item_set(edges):
    user_set = set()
    item_set = set()
    for edge in edges:
        user_set.add(edge[0])
        item_set.add(edge[1])
    return (list(user_set), list(item_set))

def write_interactions_to_file(filename, data):
    with open(filename, 'w+') as f:
        for d in data:
            f.writelines('{}\t{}\t{}\n'.format(d[0], d[1], d[2]))

def batch_to_gpu(batch: dict, device) -> dict:
    for c in batch:
        if type(batch[c]) is torch.Tensor:
            batch[c] = batch[c].to(device)
    return batch

def squeeze_dict(batch, dim=0):
    return {key: value if torch.is_tensor(value) else torch.as_tensor(value) for (key, value) in batch.items()}

def check_dir(file_name: str):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)

def fix_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
import random, torch

def snapshot_rng():
    return {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(s):
    random.setstate(s['py'])
    np.random.set_state(s['np'])
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
import numpy as np

def topk(np_scores, top_k):
    top_k = min(top_k, len(np_scores))
    top_idx = np.argsort(-np_scores[:, 3])[:top_k]
    top_samples = np_scores[top_idx]
    top_user_item = top_samples[:, :3].astype(int)
    return top_user_item

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def flatten_grads(param_list):
    flat = []
    for p in param_list:
        if p.grad is None:
            flat.append(p.new_zeros(p.numel()))
        else:
            flat.append(p.grad.view(-1))
    return torch.cat(flat, dim=0)

def compute_infer_score(prev_model, prev_dl):
    was_training = prev_model.training
    prev_model.eval()
    parameters = tuple(parameter for parameter in prev_model.parameters() if parameter.requires_grad)
    aggregate = [torch.zeros_like(parameter) for parameter in parameters]
    cached = list(prev_dl)
    rows = prev_dl.dataset.train_data
    output = np.column_stack((rows, np.zeros(len(rows), dtype=np.float64)))
    index_by_edge = {tuple(int(value) for value in edge): index for index, edge in enumerate(rows)}
    try:
        for current in cached:
            batch = batch_to_gpu(squeeze_dict(current), prev_model._device)
            losses = prev_model.loss(batch, reduction='none', state='sampling')
            gradients = torch.autograd.grad(losses.sum(), parameters, allow_unused=True)
            for total, gradient in zip(aggregate, gradients):
                if gradient is not None:
                    total.add_(gradient.detach())
        for current in cached:
            batch = batch_to_gpu(squeeze_dict(current), prev_model._device)
            losses = prev_model.loss(batch, reduction='none', state='sampling')
            weights = torch.ones_like(losses, requires_grad=True)
            gradients = torch.autograd.grad((losses * weights).sum(), parameters, create_graph=True, allow_unused=True)
            products = [(gradient * total).sum() for gradient, total in zip(gradients, aggregate) if gradient is not None and gradient.requires_grad]
            scores = torch.autograd.grad(sum(products), weights)[0] if products else torch.zeros_like(losses)
            batch_rows = zip(batch['user_id'].reshape(-1).tolist(), batch['item_id'][:, 0].tolist(), batch['timestamp'].reshape(-1).tolist())
            for edge, score in zip(batch_rows, scores.detach().cpu().double().tolist()):
                output[index_by_edge[tuple(int(value) for value in edge)], 3] = score
    finally:
        prev_model.train(was_training)
    scores = output[:, 3]
    if scores.size:
        lower, upper = scores.min(), scores.max()
        output[:, 3] = (scores - lower) / (upper - lower) if upper > lower else 0.0
    if not np.isfinite(output).all():
        raise FloatingPointError('Influence scores must be finite.')
    return output
