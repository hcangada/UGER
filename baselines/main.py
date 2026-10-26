import argparse
import json
import logging
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from helpers.Reader import Reader
from helpers.Runner import Runner
from helpers.Runner_cons import Runner_cons
from helpers.Runner_PISA import Runner_PISA
from models import Dataloader
from models.general.LGN import LGN
from models.general.MF import MF
from models.general.Contrastive_LGN import Contrastive_LGN
from models.general.PISA_LGN import PISA_LGN
from utils import utils
from utils.spawn_sampler import spawn_sampling
import Inference


MODELS = {'LGN': LGN, 'MF': MF, 'Contrastive_LGN': Contrastive_LGN, 'PISA_LGN': PISA_LGN}
RUNNERS = {'Runner': Runner, 'Runner_cons': Runner_cons, 'Runner_PISA': Runner_PISA}
BASE_STRATEGIES = {'pretrain', 'finetune', 'fulltrain', 'finetune-uniform-curr', 'finetune-invdeg-curr', 'finetune-infer-curr'}
CUSTOM_TOKENS = {'finetune', 'contra', 'layerwise', 'piw', 'onlyuser', 'onlyitem', 'reinit', 'stdev', 'kd', 'userneigh', 'userneighcur', 'stability', 'plasticity'}


def parse_args():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--model_name', choices=MODELS, default='LGN')
    selection, _ = selector.parse_known_args()
    model_class = MODELS[selection.model_name]
    runner_class = RUNNERS[model_class.runner]
    parser = argparse.ArgumentParser(description='Legacy comparison baselines with chronological next-block evaluation.', parents=[selector])
    parser.add_argument('--dyn_method', default='pretrain', help='Training strategy; see README.md for supported comparison routes.')
    parser.add_argument('--gpu', default='0', help='CUDA device selection; use -1 for CPU.')
    parser.add_argument('--verbose', type=int, default=logging.INFO, help='Logging level.')
    parser.add_argument('--log_file', default='', help='Log file.')
    parser.add_argument('--result_file', default='', help='Optional result identifier.')
    parser.add_argument('--random_seed', type=int, default=2021, help='Random seed.')
    parser.add_argument('--buffer-size', '--buffer_size', type=int, default=None, help='Fixed replay capacity; defaults to the first incoming-window size.')
    parser.add_argument('--t_opt', type=int, default=5, help='Legacy snapshot directory identifier.')
    parser.add_argument('--force_train', action='store_true', help='Replace checkpoints instead of reusing them.')
    parser.add_argument('--pretrain_model_path', default='', help='Optional shared checkpoint prefix for the initial stage.')
    parser = Reader.parse_data_args(parser)
    parser = runner_class.parse_runner_args(parser)
    parser = model_class.parse_model_args(parser)
    args = parser.parse_args()
    custom = args.model_name in {'Contrastive_LGN', 'PISA_LGN'} and args.dyn_method.startswith('finetune') and set(args.dyn_method.split('-')).issubset(CUSTOM_TOKENS)
    if args.dyn_method not in BASE_STRATEGIES and not custom:
        parser.error('Unsupported comparison strategy.')
    if args.model_name in {'Contrastive_LGN', 'PISA_LGN'} and args.dyn_method in BASE_STRATEGIES - {'pretrain', 'finetune'}:
        parser.error('Replay comparisons use LGN or MF; custom distillation models use their own strategies.')
    if (args.buffer_size is not None and args.buffer_size < 0) or args.num_neg < 1 or args.epoch < 1 or args.tepoch < 1:
        parser.error('Invalid replay capacity, negative count, or training epoch count.')
    if not args.s_fname:
        budget = 'window' if args.buffer_size is None else str(args.buffer_size)
        args.s_fname = f'{args.split_type}_{args.train_ratio}_s{args.n_snapshots}_v{args.val_ratio}_{args.model_name}_{args.dyn_method}_k{budget}_seed{args.random_seed}'
    route = f'{args.model_name}_{args.dyn_method}_seed{args.random_seed}'
    artifact_root = Path(__file__).resolve().parent / 'artifacts' / args.dataset / route
    if not args.model_path:
        args.model_path = str(artifact_root / 'checkpoints' / 'model')
    if not args.test_result_file:
        args.test_result_file = str(artifact_root / 'results')
    if not args.log_file:
        args.log_file = str(artifact_root / 'run.log')
    Path(args.test_result_file).mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    return args, model_class, runner_class


def summarize(args, corpus):
    metrics = {}
    for index in range(corpus.n_snapshots):
        source = Path(args.test_result_file) / f'test_snap{index}.json'
        if source.exists():
            for name, value in json.loads(source.read_text()).items():
                metrics.setdefault(name, []).append(float(value))
    canonical = json.loads((Path(args.test_result_file) / 'metrics.json').read_text())
    names = ('Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20')
    timings = json.loads((Path(args.test_result_file) / 'checkpoints.json').read_text())
    incremental = [entry['training_seconds'] + entry['selection_seconds'] for entry in timings if entry['checkpoint'] > 0]
    summary = {'method': canonical_method(args), 'backbone': 'MF' if args.model_name == 'MF' else 'LightGCN', 'stage_mean': {name: float(np.mean([entry['metrics'][name] for entry in canonical])) for name in names}, 'evaluated_stages': len(canonical), 'mean_incremental_stage_seconds': float(np.mean(incremental)) if incremental else 0.0}
    (Path(args.test_result_file) / 'summary.json').write_text(json.dumps(summary, indent=2))


def canonical_method(args):
    if args.model_name == 'PISA_LGN':
        return 'PISA'
    if args.model_name == 'Contrastive_LGN':
        return 'LWC-KD-PIW' if 'piw' in args.dyn_method else 'LWC-KD'
    if 'uniform' in args.dyn_method:
        return 'Uniform'
    if 'invdeg' in args.dyn_method:
        return 'Invdeg'
    if 'infer' in args.dyn_method:
        return 'INFER'
    return {'fulltrain': 'Retraining', 'finetune': 'Fine-tuning', 'pretrain': 'Frozen'}[args.dyn_method]


def main(args, model_class, runner_class):
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.gpu in {'-1', 'cpu'} else args.gpu
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    utils.fix_seed(args.random_seed)
    corpus = Reader(args)
    args.buffer_size = corpus.buffer_size
    corpus_path = os.path.join(corpus.path, 'Reader.pkl')
    pd.to_pickle(corpus, corpus_path)
    (Path(args.test_result_file) / 'arguments.json').write_text(json.dumps(vars(args), indent=2))
    configuration = {'method': canonical_method(args), 'backbone': 'MF' if args.model_name == 'MF' else 'LightGCN', 'dataset': args.dataset, 'seed': args.random_seed, 'data_fingerprint': corpus.fingerprint, 'initial_fraction': args.train_ratio, 'num_snapshots': args.n_snapshots, 'split_mode': 'count', 'validation_fraction': None, 'window_validation_fraction': args.val_ratio, 'buffer_size': args.buffer_size, 'embedding_dim': args.emb_size, 'num_layers': getattr(args, 'n_layers', 0), 'keep_prob': max(args.keep_prob, 0.0) if getattr(args, 'keep_prob', -1) > 0 else 1.0, 'message_dropout': 0.0, 'delimiter': args.sep}
    (Path(args.test_result_file) / 'config.json').write_text(json.dumps(configuration, indent=2))
    runner = runner_class(args, corpus)
    data_type = 'hist' if args.dyn_method in {'pretrain', 'fulltrain'} else 'incre'
    records = []
    evaluations = []
    for index in range(corpus.n_snapshots + 1):
        selection_seconds = 0.0
        data = Dataloader.Dataset(args, corpus, data_type, index)
        previous = Dataloader.Dataset(args, corpus, 'hist', index - 1) if index else None
        if index and args.buffer_size > 0 and any(name in args.dyn_method for name in ('uniform', 'invdeg', 'infer')):
            desired = args.buffer_size
            if 'infer' in args.dyn_method:
                spawn_sampling(args=args, corpus_path=corpus_path, idx=index, dyn=args.dyn_method, data_type=data_type, model_cls=model_class)
                timing_path = Path(corpus.snapshots_path) / f'{args.dyn_method}_{args.random_seed}_block_{index}.seconds'
                selection_seconds = float(timing_path.read_text())
                data = Dataloader.Dataset(args, corpus, f'{args.dyn_method}_{args.random_seed}', index)
            else:
                replay_name = 'uniform' if 'uniform' in args.dyn_method else 'invdeg'
                pool = Dataloader.Dataset(args, corpus, replay_name, index - 1) if index > 1 else previous
                size = min(desired, len(pool.train_data))
                selection_start = perf_counter()
                probabilities = None
                if replay_name == 'invdeg':
                    users, counts = np.unique(pool.train_data[:, 0], return_counts=True)
                    degree = dict(zip(users.tolist(), counts.tolist()))
                    weights = np.asarray([1 / degree[int(user)] for user in pool.train_data[:, 0]])
                    probabilities = weights / weights.sum()
                selected = np.random.choice(len(pool.train_data), size=size, replace=False, p=probabilities)
                replay_data = np.concatenate((data.train_data, pool.train_data[selected]))
                selection_seconds = perf_counter() - selection_start
                utils.write_interactions_to_file(os.path.join(corpus.snapshots_path, f'{replay_name}_block_{index}'), replay_data)
                data = Dataloader.Dataset(args, corpus, replay_name, index)
        utils.fix_seed(args.random_seed)
        model = model_class(args, corpus, data, index)
        model.apply(model.init_weights)
        model.to(model._device)
        if index == 0 and args.pretrain_model_path and not args.force_train:
            initial_source = Path(args.pretrain_model_path)
            if not initial_source.is_file():
                initial_source = Path(args.pretrain_model_path + '_snap0')
            target = Path(args.model_path + '_snap0')
            if not initial_source.exists():
                raise FileNotFoundError('The shared initial checkpoint does not exist.')
            if initial_source.resolve() != target.resolve():
                checkpoint = torch.load(initial_source, map_location='cpu', weights_only=False)
                if checkpoint.get('data_fingerprint', corpus.fingerprint) != corpus.fingerprint:
                    raise ValueError('The shared checkpoint and interaction file differ.')
                if checkpoint.get('num_users', corpus.n_users) != corpus.n_users or checkpoint.get('num_items', corpus.n_items) != corpus.n_items:
                    raise ValueError('The shared checkpoint and catalogue dimensions differ.')
                if checkpoint.get('format_version') == 2:
                    expected = 'mf' if args.model_name == 'MF' else 'lightgcn'
                    if str(checkpoint.get('backbone', '')).lower() != expected:
                        raise ValueError('The shared checkpoint uses a different recommendation backbone.')
                    if checkpoint.get('stage') != 0 or checkpoint.get('checkpoint_kind') != 'pretrained':
                        raise ValueError('The shared checkpoint must be a stage-zero pretrained model.')
                    checkpoint_configuration = checkpoint['config']
                    settings = {'embedding_dim': args.emb_size, 'initial_fraction': args.train_ratio, 'num_snapshots': args.n_snapshots, 'seed': args.random_seed}
                    if args.model_name != 'MF':
                        settings['num_layers'] = args.n_layers
                    for key, value in settings.items():
                        if checkpoint_configuration[key] != value:
                            raise ValueError(f'The shared checkpoint differs for {key}.')
                    configuration['validation_fraction'] = checkpoint_configuration['validation_fraction']
                else:
                    configuration['validation_fraction'] = checkpoint.get('initial_validation_fraction')
                torch.save({'model_state_dict': checkpoint['model_state_dict'], 'optimizer_state_dict': None, 'num_users': corpus.n_users, 'num_items': corpus.n_items, 'data_fingerprint': corpus.fingerprint, 'initial_validation_fraction': configuration['validation_fraction']}, target)
                (Path(args.test_result_file) / 'config.json').write_text(json.dumps(configuration, indent=2))
        start = perf_counter()
        if args.dyn_method == 'pretrain' and index > 0:
            model.load_model(args.model_path + '_snap0')
            runner.write_results(model, args, corpus, index)
            outcome = 0
            training_seconds = 0.0
        elif 'plasticity' in args.dyn_method and model_class is PISA_LGN:
            runner.train(model, data, args, corpus, previous, index, args.force_train, 0)
            training_seconds = model._training_seconds
            outcome = runner.train(model, data, args, corpus, previous, index, args.force_train, 1)
            training_seconds += model._training_seconds
        else:
            outcome = runner.train(model, data, args, corpus, previous, index, args.force_train)
            training_seconds = model._training_seconds
        best_epoch = int(outcome[-1] if isinstance(outcome, tuple) else outcome)
        encoder_state = {name: value.detach().cpu() for name, value in model.state_dict().items() if name in {'u_embeddings.weight', 'i_embeddings.weight'}}
        snapshot = {'format_version': 'baseline-diagnostic-1', 'config': dict(configuration), 'stage': index, 'method': canonical_method(args), 'backbone': configuration['backbone'], 'model_state_dict': encoder_state, 'graph_edges': data.train_data.copy(), 'data_fingerprint': corpus.fingerprint, 'num_users': corpus.n_users, 'num_items': corpus.n_items, 'checkpoint_kind': 'continual'}
        torch.save(snapshot, Path(args.test_result_file) / f'stage_{index:03d}.pt')
        records.append({'checkpoint': index, 'best_epoch': best_epoch, 'end_to_end_seconds': perf_counter() - start, 'training_seconds': training_seconds, 'selection_seconds': selection_seconds})
        (Path(args.test_result_file) / 'checkpoints.json').write_text(json.dumps(records, indent=2))
        if index < corpus.n_snapshots:
            evaluations.append({**Inference.EVALUATION_RECORDS[index, 'test'], 'validation': Inference.EVALUATION_RECORDS[index, 'val'], 'trained_through_stage': index, 'evaluated_on_stage': index + 1})
            (Path(args.test_result_file) / 'metrics.json').write_text(json.dumps(evaluations, indent=2))
    summarize(args, corpus)


if __name__ == '__main__':
    main(*parse_args())
