import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Mi_Net(nn.Module):
    def __init__(self, embed_dim, rep_u,rep_v,num,u=True):
        super(Mi_Net, self).__init__()
        self.embed_dim = embed_dim
        self.rep_u = rep_u
        self.rep_v = rep_v
        self.u = u
        self.num = num

        self.fc_mu_zc_1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.fc_mu_zc_2 = nn.Linear(self.embed_dim // 2, self.embed_dim)
        self.fc_var_zc_1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.fc_var_zc_2 = nn.Linear(self.embed_dim // 2, self.embed_dim)

        self.fc_mu_za_1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.fc_mu_za_2 = nn.Linear(self.embed_dim // 2, self.embed_dim)
        self.fc_var_za_1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.fc_var_za_2 = nn.Linear(self.embed_dim // 2, self.embed_dim)

    def forward(self,nodes,name,u):
        if u:
            output_emb_z, output_emb_c,_ = self.rep_u(nodes)
        else:
            output_emb_z, output_emb_c,_ = self.rep_v(nodes)
        if name == 'zc':
            h_zc_mu = F.elu(self.fc_mu_zc_1(output_emb_z))
            zc_mu = self.fc_mu_zc_2(h_zc_mu)
            h_zc_var = F.elu(self.fc_var_zc_1(output_emb_z))
            zc_logvar = F.tanh(self.fc_var_zc_2(h_zc_var))
            # print(len(nodes))
            new_order = torch.randperm(len(nodes))
            output_emb_c_rand = output_emb_c[new_order]
            loglikeli_zc, bound_zc = self.mi_loss(zc_mu, zc_logvar, output_emb_c, output_emb_c_rand)
            return loglikeli_zc, bound_zc

    def mi_loss(self,mu,logvar,outp,outp_rand):
        pos = -(mu - outp) ** 2 /torch.exp(logvar)
        neg = -(mu - outp_rand) ** 2 / torch.exp(logvar)
        # lld = -torch.mean(torch.sum(pos,dim=-1))
        lld = (-(mu-outp)**2/logvar.exp()-logvar).sum(dim=1).mean(dim=0)
        bound = torch.mean(torch.sum(pos,dim=-1)-torch.sum(neg,dim=-1))/2
        return -lld, bound




