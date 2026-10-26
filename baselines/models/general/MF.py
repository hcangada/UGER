import torch.nn as nn
from models.Model import Model
import torch

class MF(Model):

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        return Model.parse_model_args(parser)

    def __init__(self, args, corpus, data, time_idx=0):
        self.emb_size = args.emb_size
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def computer(self):
        return self.u_embeddings.weight, self.i_embeddings.weight

    def infer_user_scores(self, u_ids, i_ids):
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        scores = torch.matmul(cf_u_vectors, cf_i_vectors.t())
        return scores
