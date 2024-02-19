import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from attention import Attention

class CDR_SB(nn.Module):
    def __init__(self, rep_u, rep_v, mi_net, device, u2e, v2e, history_uv):
        super(CDR_SB, self).__init__()
        self.rep_u = rep_u
        self.rep_v = rep_v
        self.embed_dim = rep_u.dec_embed_dim
        self.mi_net = mi_net
        self.device = device
        self.user_emb = u2e
        self.item_emd = v2e
        self.history_uv = history_uv
        self.sim_threshold = 0.5
        # self.tau = tau
        self.att = Attention(rep_u.embed_dim, self.embed_dim)

        #z and c agg
        self.feat_fuse_layer_u = nn.Linear(self.embed_dim*2,self.embed_dim)
        self.feat_fuse_norm_u = nn.BatchNorm1d(self.embed_dim)
        self.feat_fuse_layer_v = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.feat_fuse_norm_v = nn.BatchNorm1d(self.embed_dim)

        self.w_uv_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        nn.init.xavier_uniform_(self.w_uv_1.weight)
        self.uv_bn_1 = nn.BatchNorm1d(self.embed_dim)
        self.w_uv_2 = nn.Linear(self.embed_dim, self.embed_dim//2)
        self.uv_bn_2 = nn.BatchNorm1d(self.embed_dim//2)
        self.w_uv_3 = nn.Linear(self.embed_dim//2, self.embed_dim // 4)
        self.uv_bn_3 = nn.BatchNorm1d(self.embed_dim // 4)
        self.w_uv_final = nn.Linear(self.embed_dim//4, 1)

        self.criterion = nn.MSELoss()


    def forward(self, nodes_u, nodes_v,red_by_friends):
        embed_u_matrix = torch.empty(len(nodes_u), self.embed_dim, dtype=torch.float).to(self.device)
        embed_v_matrix = torch.empty(len(nodes_v), self.embed_dim, dtype=torch.float).to(self.device)
        embed_u_z, embed_u_c, embed_u = self.rep_u(nodes_u)
        embed_v_z, embed_v_c, embed_v = self.rep_v(nodes_v)

        for each_index in range(len(nodes_u)):
            user_id = int(nodes_u[each_index])
            item_id = int(nodes_v[each_index])
            red_by_friend = int(red_by_friends[each_index])
            purd_items = [each_item for each_item in self.history_uv[user_id] if each_item != item_id]
            if len(purd_items) > 0:
                if red_by_friend:
                    sims = self.calculate_sim(item_id, purd_items).tolist()
                    if max(sims) > self.sim_threshold:
                        u_feat = embed_u_z[each_index] + embed_u_c[each_index]
                        v_feat = embed_v_z[each_index] + embed_v_c[each_index]
                    else:
                        u_feat = embed_u_z[each_index]
                        v_feat = embed_v_z[each_index]
                else:
                    u_feat = embed_u_z[each_index]
                    v_feat = embed_v_z[each_index]
            else:
                u_feat = embed_u_z[each_index] + embed_u_c[each_index]
                v_feat = embed_v_z[each_index] + embed_u_c[each_index]

            embed_u_matrix[each_index] = u_feat
            embed_v_matrix[each_index] = v_feat

        uv = torch.cat([embed_u_matrix, embed_v_matrix], dim=1)
        # uv = F.dropout(F.relu(self.uv_bn(self.w_uv_1(uv))), training=self.training)
        uv = self.uv_bn_1(F.relu(self.w_uv_1(uv)))
        uv = self.uv_bn_2(F.relu(self.w_uv_2(uv)))
        uv = self.uv_bn_3(F.relu(self.w_uv_3(uv)))
        uv_score = self.w_uv_final(uv)
        # uv_score = F.sigmoid(uv_score)
        return uv_score.squeeze()

    def calculate_sim(self, item_id, purd_items):
        anchor_item = self.item_emd.weight[item_id]
        com_items = self.item_emd.weight[purd_items]
        anchor_item = anchor_item.unsqueeze(0).repeat(com_items.shape[0], 1)

        sim = F.cosine_similarity(anchor_item, com_items, dim=1)
        return sim

    def loss(self, nodes_u, nodes_v, scores, labels_list):
        # scores = self.forward(nodes_u, nodes_v)
        # cl_total_loss = self.calc_ssl_loss_strategy()
        mi_total_loss, lld_loss, mi_loss = self.calculate_mi_loss(nodes_u, nodes_v)

        prediction_loss = self.criterion(scores, labels_list)

        return prediction_loss, mi_total_loss, lld_loss, mi_loss

    def calculate_mi_loss(self, nodes_u, nodes_v):
        loglikeli_zc_loss_u, bound_zc_loss_u = self.mi_net(nodes_u, name='zc', u=True)
        # loglikeli_za_loss_u, bound_za_loss_u = self.mi_net(nodes_u, name='za',u=True)
        # loglikeli_ca_loss_u, bound_ca_loss_u = self.mi_net(nodes_u, name='ca',u=True)
        loglikeli_zc_loss_v, bound_zc_loss_v = self.mi_net(nodes_v, name='zc', u=False)
        # loglikeli_za_loss_v, bound_za_loss_v = self.mi_net(nodes_v, name='za',u=False)
        # loglikeli_ca_loss_v, bound_ca_loss_v = self.mi_net(nodes_v, name='ca',u=False)
        mi_total_loss = loglikeli_zc_loss_u + loglikeli_zc_loss_v \
                        + bound_zc_loss_u + \
                        bound_zc_loss_v
        return mi_total_loss, loglikeli_zc_loss_u + loglikeli_zc_loss_v, bound_zc_loss_u + \
               bound_zc_loss_v




