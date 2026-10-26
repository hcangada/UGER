import logging

import torch
from torch.nn import functional as F

from utils import utils


class Model(torch.nn.Module):
    reader = 'Reader'
    runner = 'Runner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='', help='Checkpoint prefix.')
        parser.add_argument('--num_neg', type=int, default=4, help='Recommendation negatives per positive interaction.')
        return parser

    @staticmethod
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def __init__(self, args, corpus):
        super().__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = args.model_path
        self.num_neg = args.num_neg
        self.item_num = corpus.n_items
        self.user_num = corpus.n_users
        self.dyn_method = args.dyn_method
        self.optimizer = None
        self.test_result_file = args.test_result_file
        self._define_params()
        self.total_parameters = self.count_variables()
        logging.info('Trainable parameters: %d', self.total_parameters)

    def loss(self, current_data, reduction='mean', state='training'):
        predictions = self.forward_scores(current_data)
        losses = F.softplus(predictions[:, 1:1 + self.num_neg] - predictions[:, :1]).mean(dim=1)
        if reduction == 'mean':
            return losses.mean()
        if reduction == 'sum':
            return losses.sum()
        if reduction == 'none':
            return losses
        raise ValueError(f'Unknown reduction: {reduction}')

    def forward(self, batch, mode='scores'):
        if mode == 'scores':
            return self.forward_scores(batch)
        if mode == 'loss_none':
            return self.loss(batch, reduction='none', state='sampling')
        raise ValueError(f'Unknown mode: {mode}')

    def forward_scores(self, batch):
        users, items = self.computer()
        user_ids = batch['user_id'].long().reshape(-1, 1)
        item_ids = batch['item_id'].long()
        return (users[user_ids] * items[item_ids]).sum(dim=-1)

    def save_model(self, model_path=None, add_path=None):
        target = (model_path or self.model_path) + (add_path or '')
        utils.check_dir(target)
        torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None}, target)

    def load_model(self, model_path=None, add_path=None, flag=0):
        target = (model_path or self.model_path) + (add_path or '')
        checkpoint = torch.load(target, map_location=self._device, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return checkpoint

    def count_variables(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
