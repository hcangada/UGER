import os
import gc
import copy
import torch
import logging
import numpy as np
import random
from time import time
from torch.utils.data import DataLoader
from models import Dataloader
from typing import Dict, List, NoReturn
from utils import utils
from models.Model import Model
import Inference
import json

class Runner_cons(object):

    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
        parser.add_argument('--tepoch', type=int, default=200, help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
        parser.add_argument('--l2', type=float, default=0.0001, help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=1024, help='Batch size during training.')
        parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1, help='pin_memory in DataLoader')
        parser.add_argument('--test_result_file', type=str, default='', help='')
        return parser

    def __init__(self, args, corpus):
        self.epoch = args.epoch
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.result_file = args.result_file
        self.dyn_method = args.dyn_method
        self.time = None
        self.snap_boundaries = corpus.snap_boundaries
        self.snapshots_path = corpus.snapshots_path
        self.test_result_file = args.test_result_file
        self.tepoch = args.tepoch

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError('Unknown Optimizer: ' + self.optimizer_name)
        return optimizer

    def write_results(self, model, args, corpus, snap_idx):
        if snap_idx >= corpus.n_snapshots:
            return
        v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
        t_results = Inference.Test(args, model, corpus, 'test', snap_idx)
        logging.info('Trained model testing')
        val_str = Inference.print_results(None, v_results, None)
        val_result_filename_ = os.path.join(self.test_result_file, 'val_snap{}.txt'.format(snap_idx))
        open(val_result_filename_, 'w+').write(val_str)
        test_str = Inference.print_results(None, None, t_results)
        result_filename_ = os.path.join(self.test_result_file, 'test_snap{}.txt'.format(snap_idx))
        open(result_filename_, 'w+').write(test_str)
        Ks = [10, 20, 50, 100]
        json_t = {f'{metric}@{k}': v for (metric, v) in zip(['Recall', 'NDCG', 'MRR', 'Precision'], t_results) for (k, v) in zip(Ks, v)}
        with open(os.path.join(self.test_result_file, 'test_snap{}.json'.format(snap_idx)), 'w') as f:
            json.dump(json_t, f, indent=4)

    def train(self, model, data_dict, args, corpus, prev_data, snap_idx, force_train=False):
        model._training_seconds = 0.0
        logging.info('Training time stage: {}'.format(snap_idx))
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        if os.path.exists(model.model_path + '_snap{}'.format(snap_idx)) and not force_train:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            model.load_model(model.model_path + '_snap{}'.format(snap_idx))
            self.write_results(model, args, corpus, snap_idx)
            return (0, 0)
        elif snap_idx > 0 and 'pretrain' in args.dyn_method:
            logging.info('Time_idx {} model already exists. Skip training and test directly.'.format(snap_idx))
            model.load_model(model.model_path + '_snap0')
            self.write_results(model, args, corpus, snap_idx)
            return (0, 0)
        else:
            logging.info('Time_idx {} model does not exist. Start training.'.format(snap_idx))
        prev_model = None
        hist_model = None
        if snap_idx > 0 and 'finetune' in args.dyn_method:
            model.load_model(model.model_path + '_snap{}'.format(snap_idx - 1))
            model.freeze_flag = 0
            prev_model = copy.deepcopy(model)
            prev_model.Graph = Dataloader.Dataset(args, corpus, 'incre', snap_idx - 1).getSparseGraph()
            prev_model.eval()
            prev_model.requires_grad_(False)
            if 'reinit' in args.dyn_method:
                model.add_drop.apply(model.init_weights)
                if 'stdev' in self.dyn_method:
                    torch.nn.init.normal_(model.W_prev, mean=0, std=1)
                    torch.nn.init.normal_(model.W, mean=0, std=1)
                else:
                    torch.nn.init.normal_(model.W_prev, mean=0, std=0.01)
                    torch.nn.init.normal_(model.W, mean=0, std=0.01)
        elif snap_idx > 0 and 'newtrain' in args.dyn_method:
            prev_model = copy.deepcopy(model)
            prev_model.load_model(model.model_path + '_snap{}'.format(snap_idx - 1))
            prev_model.Graph = Dataloader.Dataset(args, corpus, 'incre', snap_idx - 1).getSparseGraph()
            prev_model.eval()
            prev_model.requires_grad_(False)
        num_epoch = self.tepoch if 'finetune' in self.dyn_method or 'newtrain' in self.dyn_method else self.epoch
        if snap_idx == 0:
            num_epoch = self.epoch
        best_recall = float('-inf')
        best_epoch = 0
        patience = 20
        cnt = 0
        for epoch in range(num_epoch):
            model.epoch = epoch
            fit_start = time()
            losses = self.fit(model, data_dict, prev_data, snap_idx, True, prev_model)
            model._training_seconds += time() - fit_start
            logging.info(f'Epoch {epoch} total_loss={losses[0]:.4f} bpr_loss={losses[1]:.4f} kd_loss={losses[2]:.4f} kd_user={losses[3]:.4f} kd_item={losses[4]:.4f} kd_user_neigh={losses[5]:.4f} kd_item_neigh={losses[6]:.4f}')
            if np.isnan(losses[0]).any():
                logging.info('NaN loss, stop training')
                exit()
            if snap_idx >= corpus.n_snapshots:
                best_epoch = epoch + 1
                if epoch + 1 == num_epoch:
                    model.save_model(add_path=f'_snap{snap_idx}')
                continue
            if epoch >= 0 and ((epoch + 1) % 2 == 0 or epoch + 1 == num_epoch):
                v_results = Inference.Test(args, model, corpus, 'val', snap_idx)
                if v_results[0][0] > best_recall:
                    best_epoch = epoch + 1
                    best_recall = v_results[0][0]
                    save_path = f'_snap{snap_idx}'
                    model.save_model(add_path=save_path)
                if epoch + 1 > 20:
                    if v_results[0][0] < best_recall:
                        cnt += 1
                    else:
                        cnt = 0
                    if cnt >= patience:
                        break
        logging.info(f'Training complete. Best validation epoch: {best_epoch:03d}')
        model_path = f'{model.model_path}_snap{snap_idx}'
        model.load_model(model_path)
        self.write_results(model, args, corpus, snap_idx)
        return best_epoch

    def fit(self, model, data, prev_data, snap_idx, shuffle, prev_model):
        if not len(data):
            return [0.0] * 7
        if 'piw' in self.dyn_method and prev_model is not None:
            with torch.no_grad():
                model.update_kmeans(prev_model)
        gc.collect()
        torch.cuda.empty_cache()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)
        total_losses = []
        for current in dl:
            current = utils.batch_to_gpu(utils.squeeze_dict(current), model._device)
            current['batch_size'] = len(current['user_id'])
            losses = self.train_recommender_vanilla(data, model, current, prev_data, snap_idx, prev_model)
            total_losses.append(losses)
        return [np.mean(loss).item() for loss in zip(*total_losses)]

    def train_recommender_vanilla(self, data, model, current, prev_data, time_idx, prev_model):
        model.train()
        losses = model.loss(data, current, prev_data, time_idx, prev_model, reduction='mean')
        model.optimizer.zero_grad()
        losses[0].backward()
        model.optimizer.step()
        return [loss.cpu().data.numpy() for loss in losses]
