import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Decomposing_Network(nn.Module):

    def __init__(self, enc_u, enc_v, dec_embed_dim,cuda="cpu",u=True):
        super(Decomposing_Network, self).__init__()
        self.enc_u = enc_u
        self.enc_v = enc_v
        self.embed_dim = enc_u.embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.u = u
        self.device = cuda
        self.w_u_z1 = nn.Linear(self.embed_dim, self.embed_dim)
        # nn.init.xavier_uniform_(self.w_u_z1.weight)
        self.bn_u_z1 = nn.BatchNorm1d(self.embed_dim)
        self.w_u_z2 = nn.Linear(self.embed_dim, self.dec_embed_dim)
        # nn.init.xavier_uniform_(self.w_u_z2.weight)
        self.bn_u_z2 = nn.BatchNorm1d(self.dec_embed_dim)
        self.w_u_c1 = nn.Linear(self.embed_dim, self.embed_dim)
        # nn.init.xavier_uniform_(self.w_u_c1.weight)
        self.bn_u_c1 = nn.BatchNorm1d(self.embed_dim)
        self.w_u_c2 = nn.Linear(self.embed_dim, self.dec_embed_dim)
        # nn.init.xavier_uniform_(self.w_u_c2.weight)
        self.bn_u_c2 = nn.BatchNorm1d(self.dec_embed_dim)


        self.w_v_z1 = nn.Linear(self.embed_dim, self.embed_dim)
        # nn.init.xavier_uniform_(self.w_v_z1.weight)
        self.bn_v_z1 = nn.BatchNorm1d(self.embed_dim)
        self.w_v_z2 = nn.Linear(self.embed_dim, self.dec_embed_dim)
        # nn.init.xavier_uniform_(self.w_v_z2.weight)
        self.bn_v_z2 = nn.BatchNorm1d(self.dec_embed_dim)
        self.w_v_c1 = nn.Linear(self.embed_dim, self.embed_dim)
        # nn.init.xavier_uniform_(self.w_v_c1.weight)
        self.bn_v_c1 = nn.BatchNorm1d(self.embed_dim)
        self.w_v_c2 = nn.Linear(self.embed_dim, self.dec_embed_dim)
        # nn.init.xavier_uniform_(self.w_v_c2.weight)
        self.bn_v_c2 = nn.BatchNorm1d(self.dec_embed_dim)


    def forward(self,nodes):
        if self.u:
            embed = self.enc_u(nodes)
            output_emb_z = self.representation_net_u_z(embed)
            output_emb_c = self.representation_net_u_c(embed)
        else:
            embed = self.enc_v(nodes)
            output_emb_z = self.representation_net_v_z(embed)
            output_emb_c = self.representation_net_v_c(embed)
        return output_emb_z,output_emb_c,embed

    def representation_net_u_z(self,embed):
        '''
        decomposing user/item representation to  variable z
        :param embed:
        :return:
        '''
        x_1 = F.relu(self.bn_u_z1(self.w_u_z1(embed)))
        x_2 = F.relu(self.bn_u_z2(self.w_u_z2(x_1)))
        x_3 = F.dropout(x_2, p=0.1,training=self.training)
        return x_3
    def representation_net_u_c(self,embed):
        '''
        decomposing user/item representation to  variable c
        :param embed:
        :return:
        '''
        x_1 = F.relu(self.bn_u_c1(self.w_u_c1(embed)))
        x_2 = F.relu(self.bn_u_c2(self.w_u_c2(x_1)))
        x_3 = F.dropout(x_2, p=0.1,training=self.training)
        # x_4 = F.dropout(x_2, p=0.1, training=self.training)
        return x_3
    def representation_net_v_z(self,embed):
        '''
        decomposing item representation to  variable z
        :param embed:
        :return:
        '''
        x_1 = F.relu(self.bn_v_z1(self.w_v_z1(embed)))
        x_2 = F.relu(self.bn_v_z2(self.w_v_z2(x_1)))
        x_3 = F.dropout(x_2, p=0.1,training=self.training)
        return x_3
    def representation_net_v_c(self,embed):
        '''
        decomposing item representation to  variable c
        :param embed:
        :return:
        '''
        x_1 = F.relu(self.bn_v_c1(self.w_v_c1(embed)))
        x_2 = F.relu(self.bn_v_c2(self.w_v_c2(x_1)))
        x_2 = F.dropout(x_2, p=0.1,training=self.training)
        return x_2

