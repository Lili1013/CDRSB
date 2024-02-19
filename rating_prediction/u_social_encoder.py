import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class U_Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, base_model=None, cuda="cpu"):
        super(U_Social_Encoder, self).__init__()

        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        # nn.init.xavier_uniform_(self.linear1.weight)#
        self.bn_1 = nn.BatchNorm1d(self.embed_dim)

    def forward(self, nodes):

        to_neighs = []
        for node in nodes:
            try:
                to_neighs.append(self.social_adj_lists[int(node)])
            except:
                to_neighs.append([int(node)])#no sicial network, add ownself
        neigh_feats = self.aggregator.forward(nodes, to_neighs)  # user-user network

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()

        # self-connection could be considered.
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.bn_1(self.linear1(combined)))

        return combined