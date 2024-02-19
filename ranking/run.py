
import os
import sys
curPath = os.path.abspath(os.path.dirname((__file__)))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

import warnings
warnings.filterwarnings('ignore')

import pickle
from loguru import logger

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from para_parser import parse
from uv_aggregator import UV_Aggregator
from uv_encoder import UV_Encoder
from u_social_aggregator import U_Social_Aggregator
from u_social_encoder import U_Social_Encoder
from decomposing_network import Decomposing_Network
from cdr_sb import CDR_SB
from mi_net import Mi_Net
from sklearn.manifold import TSNE
from get_data import Data
from evaluation import calculate_hr_ndcg


def train(model, device, train_loader, optimizer, epoch, beta,data_model):
    model.train()

    total_loss = []

    for i, data in enumerate(train_loader, 0):

        batch_nodes_u, batch_nodes_v, labels_list, f_list = data
        batch_nodes_u, batch_nodes_v, labels_list,f_list = data_model.generate_train_instances(batch_nodes_u.tolist(),
                                                                                               batch_nodes_v.tolist(),f_list.tolist())
        batch_nodes_u, batch_nodes_v, labels_list,f_list = torch.tensor(batch_nodes_u),torch.tensor(batch_nodes_v),\
                                                           torch.tensor(labels_list),torch.tensor(f_list)
        optimizer.zero_grad()
        scores = model.forward(batch_nodes_u.to(device), batch_nodes_v.to(device),f_list.to(device))
        pred_loss,cl_loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device),scores,labels_list.to(device))
        loss = pred_loss+beta*cl_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss = loss.item()
        total_loss.append(running_loss)
        if i % 100 == 0:
            logger.info(f'Epoch: {epoch}, i: {i}, loss :{sum(total_loss)/len(total_loss)}')
    return total_loss

def test(model, device, test_pos_samples,test_neg_samples,beta,data_model,k):
    logger.info('start test')
    model.eval()
    with torch.no_grad():
        hits,ndcgs = calculate_hr_ndcg(model,test_pos_samples,test_neg_samples,k,device)
        hr = sum(hits) / len(hits)
        ndcg = sum(ndcgs) / len(ndcgs)
        return hr, ndcg


def main():
    args = parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dec_embed_dim = args.dec_embed_dim
    beta = args.beta
    train_neg_num = args.train_neg
    test_neg_num = args.test_neg
    top_k = args.top_k
    save_model_path = args.save_model_path
    train_model_weights_path = args.train_model_weights_path
    # dir_data = '../../datasets/epinions/processed_data/'
    dir_data = '../datasets/dianping/'
    user_pur_path = open(dir_data + "user_purd_items.pickle", 'rb')
    history_u_lists = pickle.load(user_pur_path)
    item_purd_path = open(dir_data + "item_purd_users.pickle", 'rb')
    history_v_lists = pickle.load(item_purd_path)
    user_rating_path = open(dir_data + "user_item_rating.pickle", 'rb')
    history_ur_lists = pickle.load(user_rating_path)
    item_rating_path = open(dir_data + "item_user_rating.pickle", 'rb')
    history_vr_lists = pickle.load(item_rating_path)
    data = pd.read_csv(dir_data+'dianping_inter_new.csv')
    data_train = data[data['x_label']==0]

    train_u, train_v, train_r,train_f = data_train['user_id'].values.tolist(),data_train['item_id'].values.tolist(),\
                                        data_train['rating'].values.tolist(),data_train['red_by_friends'].values.tolist()

    ratings_list = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    user_social_path = open(dir_data + 'user_social.pickle', 'rb')
    user_social_adj_lists = pickle.load(user_social_path)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r),torch.FloatTensor(train_f))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()
    u2e = nn.Embedding(num_users, embed_dim).to(device)
    nn.init.xavier_uniform_(u2e.weight)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    nn.init.xavier_uniform_(v2e.weight)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)
    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e,u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists,agg_u_history, cuda=device, uv=True)

    # # neighobrs
    agg_u_social = U_Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = U_Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, user_social_adj_lists, agg_u_social,
                             base_model=enc_u_history, cuda=device)
    # item feature
    # features: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e,u2e, embed_dim, cuda=device, uv=False)
    enc_v = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists,agg_v_history, cuda=device, uv=False)

    rep_u = Decomposing_Network(enc_u, enc_v, dec_embed_dim, cuda=device, u=True)
    rep_v = Decomposing_Network(enc_u, enc_v, dec_embed_dim, cuda=device, u=False)

    mi_net = Mi_Net(dec_embed_dim, rep_u, rep_v, args.batch_size, u=True)

    # source
    der_srec = CDR_SB(rep_u, rep_v, mi_net, device,u2e,v2e,history_u_lists).to(device)
    optimizer = torch.optim.RMSprop(der_srec.parameters(), lr=args.lr, alpha=0.9)
    # optimizer = torch.optim.Adam(der_srec.parameters(), lr=args.lr)
    data_model = Data(data,train_num_negs=train_neg_num,test_num_negs=test_neg_num,batch_size=args.batch_size)
    test_pos_samples,test_neg_samples = data_model.read_test_samples(path=dir_data+'dianping_test.txt')
    best_hr = 0.0
    best_ndcg = 0.0
    endure_count = 0
    logger.info('start train')
    for epoch in range(1, args.epochs + 1):
        total_loss = train(der_srec, device, train_loader, optimizer, epoch,beta,data_model)

        hr,ndcg = test(der_srec, device, test_pos_samples,test_neg_samples,beta,data_model,top_k)
        logger.info(f'epoch {epoch}, train loss: {sum(total_loss) / len(total_loss)}, hr: {hr}, ndcg: {ndcg}')
        # please add the validation set to tune the hyper-parameters based on your datasets.
        torch.save(der_srec.state_dict(),'{}/weights_epoch_{}.pth'.format(train_model_weights_path,epoch))
        # early stopping (no validation set in toy dataset)
        if hr > best_hr:
            best_hr = hr
            best_ndcg = ndcg
            endure_count = 0
            torch.save(der_srec.state_dict(),'{}.pth'.format(save_model_path))
        else:
            endure_count += 1
        logger.info(f'best hr: {best_hr}, best ndcg: {best_ndcg}')
        if endure_count > 5:
            break



if __name__ == '__main__':
    main()

