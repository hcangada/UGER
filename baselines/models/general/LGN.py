import torch.nn as nn
from models.Model import Model
import torch
import torch.nn.functional as F

class LGN(Model):

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
        parser.add_argument('--keep_prob', type=float, default=-1)
        return Model.parse_model_args(parser)

    def __init__(self, args, corpus, data, time_idx):
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.keep_prob = args.keep_prob
        self.Graph = data.getSparseGraph()
        self.time_idx = time_idx
        self.data = data
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def emb_gate(self, x):
        gating_weight = F.normalize(torch.randn((self.emb_size, self.emb_size)).to(self.data._device))
        gating_bias = F.normalize(torch.randn((1, self.emb_size)).to(self.data._device))
        gate = torch.sigmoid(torch.matmul(x, gating_weight) + gating_bias)
        return torch.mul(x, gate)

    def __dropout(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values), device=values.device) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse_coo_tensor(index.t(), values, size, device=values.device).coalesce()
        return g

    def computer(self):
        users_emb = self.u_embeddings.weight
        items_emb = self.i_embeddings.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.training and self.keep_prob > 0:
            g_droped = self.__dropout(self.Graph, self.keep_prob)
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        (users, items) = torch.split(light_out, [self.user_num, self.item_num])
        return (users, items)

    def forward_scores(self, batch):
        (all_users, all_items) = self.computer()
        u_ids = batch['user_id']
        i_ids = batch['item_id']
        if u_ids.dim() == 2:
            u_ids = u_ids[:, 0]
        u_ids = u_ids.long()
        i_ids = i_ids.long()
        u_ids = u_ids.unsqueeze(-1).expand(-1, i_ids.size(1))
        cf_u_vectors = all_users[u_ids]
        cf_i_vectors = all_items[i_ids]
        scores = (cf_u_vectors * cf_i_vectors).sum(dim=-1)
        return scores

    def infer_user_scores(self, u_ids, i_ids):
        (all_users, all_items) = self.computer()
        cf_u_vectors = all_users[u_ids]
        cf_i_vectors = all_items[i_ids]
        scores = torch.matmul(cf_u_vectors, cf_i_vectors.t())
        return scores

    def get_all_layer_embs(self):
        with torch.no_grad():
            users_emb = self.u_embeddings.weight
            items_emb = self.i_embeddings.weight
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            if self.training and self.keep_prob > 0:
                g_droped = self.__dropout(self.Graph, self.keep_prob)
            else:
                g_droped = self.Graph
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            return embs
