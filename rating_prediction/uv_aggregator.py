import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.u2e = u2e
        self.r2e = r2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, history_uv, history_r):

        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            tmp_label = history_r[i]

            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                # uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                # uv_rep = self.v2e.weight[nodes[i]]
            # e_uv_mean = torch.mean(e_uv,dim=0)#mean by columns
            e_r = self.r2e.weight[[i - 1 for i in tmp_label]]
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))
            o_history = torch.mean(o_history, dim=0)
            embed_matrix[i] = o_history
        to_feats = embed_matrix
        return to_feats


