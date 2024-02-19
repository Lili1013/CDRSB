import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, embed_dim,dec_embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.mlp_1 = nn.Linear((self.embed_dim + self.dec_embed_dim), self.embed_dim)
        nn.init.xavier_uniform_(self.mlp_1.weight)
        self.mlp_2 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        nn.init.xavier_uniform_(self.mlp_2.weight)
        self.mlp_3 = nn.Linear(self.embed_dim // 2, 1)
        nn.init.xavier_uniform_(self.mlp_3.weight)
        self.bn_1 = nn.BatchNorm1d(self.embed_dim)
        self.bn_2 = nn.BatchNorm1d(self.embed_dim // 2)

    def forward_1(self,embed_z,embed_c,embed):
        reps = embed.repeat(2, 1)
        embed_z = torch.reshape(embed_z,(1,embed_z.shape[0]))
        embed_c = torch.reshape(embed_c,(1,embed_c.shape[0]))
        nei_reps = torch.cat((embed_z,embed_c),dim=0)
        u_cat_z = torch.cat((nei_reps, reps), dim=1)
        # u_cat_z = torch.cat((embed_u_z, embed_u), dim=1)
        u_1 = F.relu(self.bn_1(self.mlp_1(u_cat_z)))
        u_2 = F.relu(self.bn_2(self.mlp_2(u_1)))
        u_3 = F.softmax(self.mlp_3(u_2))

        return u_3
    def forward(self,embed_z,embed_c,embed):
        # reps = embed.repeat(2, 1)
        reps = embed.unsqueeze(1)
        reps = torch.cat([reps,reps],dim=1)
        embed_z = embed_z.unsqueeze(1)
        embed_c = embed_c.unsqueeze(1)
        nei_reps = torch.cat((embed_z,embed_c),dim=1)
        u_cat_z = torch.cat((nei_reps, reps), dim=2)
        # u_cat_z = torch.cat((embed_u_z, embed_u), dim=1)
        u_1 = self.bn_1(F.relu(self.mlp_1(u_cat_z)).view(-1,self.embed_dim)).view(reps.shape)
        u_2 = self.bn_2(F.relu(self.mlp_2(u_1)).view(-1,self.embed_dim//2)).view(nei_reps.shape)
        u_3 = F.softmax(self.mlp_3(u_2),dim=0)

        return u_3
