
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from para_parser import parse
from uv_aggregator import UV_Aggregator
from uv_encoder import UV_Encoder
from u_social_aggregator import U_Social_Aggregator
from u_social_encoder import U_Social_Encoder
from decomposing_network import Decomposing_Network
from cdr_sb import CDR_SB
from mi_net import Mi_Net



def data_to_list(path):
    '''
    :param path:
    :return: list(users),list(items).list(ratings)
    '''
    df = pd.read_csv(path)
    data_u = []
    data_v = []
    data_r = []
    data_f = []
    for x in df.groupby(by='user_id'):
        users = list(x[1]['user_id'])
        items = list(x[1]['item_id'])
        ratings = list(x[1]['rating'])
        red_by_f = list(x[1]['red_by_friends'])
        data_u.extend(users)
        data_v.extend(items)
        data_r.extend(ratings)
        data_f.extend(red_by_f)
    return data_u,data_v,data_r,data_f

def train(model, device, train_loader, optimizer, epoch, beta):
    model.train()
    running_loss = 0.0
    pred_total_loss = 0.0
    cl_total_loss = 0.0
    lld_total_loss = 0.0
    mi_total_loss = 0.0
    total_loss = []
    tmp_pred = []
    target = []
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list,f_list = data
        optimizer.zero_grad()
        scores = model.forward(batch_nodes_u.to(device), batch_nodes_v.to(device),f_list.to(device))
        pred_loss,cl_loss,lld_loss,mi_loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device),scores,labels_list.to(device))
        loss = pred_loss+beta*cl_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        pred_total_loss += pred_loss.item()
        cl_total_loss += cl_loss.item()
        lld_total_loss += lld_loss.item()
        mi_total_loss += mi_loss.item()
        tmp_pred.append(list(scores.data.cpu().numpy()))
        target.append(list(labels_list.numpy()))
        if i % 100 == 0:
            expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
            mae = mean_absolute_error(tmp_pred, target)
            tmp_pred = []
            target = []
            logger.info('[%d, %5d] loss: %.3f, pred_loss: %.3f, total mi loss: %.3f, lld loss:%.3f,, mi loss: %.3f, expected_rmse:%.3f,mae:%.3f'
                        % (
                epoch, i, running_loss / 100, pred_loss/100,cl_total_loss/100, lld_loss/100,mi_total_loss/100,expected_rmse,mae
                ))
            total_loss.append(running_loss / 100)
            running_loss = 0.0
            pred_total_loss = 0.0
            cl_total_loss = 0.0
    logger.info('train loss:.%3f'%(sum(total_loss)/len(total_loss)))
    return total_loss

def test(model, device, test_loader,beta):
    model.eval()
    tmp_pred = []
    target = []
    test_total_loss = []
    with torch.no_grad():
        for test_u, test_v, tmp_target, test_f in test_loader:
            test_u, test_v, tmp_target,test_f = test_u.to(device), test_v.to(device), tmp_target.to(device),test_f.to(device)
            output = model.forward(test_u, test_v,test_f)
            test_loss,cl_loss,_,_ = model.loss(test_u, test_v,output,tmp_target)
            loss = test_loss+beta * cl_loss
            tmp_pred.append(list(output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
            test_total_loss.append(loss.item())

    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    logger.info('loss: %.3f, test rmse: %.3f, test mae:%.3f' % ((sum(test_total_loss) / len(test_total_loss)),expected_rmse,mae))
    return expected_rmse, mae,test_total_loss


def main():
    # Training settings
    args = parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    embed_dim = args.embed_dim
    dec_embed_dim = args.dec_embed_dim
    beta = args.beta
    save_model_path = args.save_model_path

    # dir_data = '../../datasets/epinions/processed_data/'
    # dir_data = '/data/lwang9/DCE-SR/datasets/epinions/processed_data/'
    # dir_data = '/data/lwang9/DCE-SR/datasets/ciao/processed_data/'
    dir_data = '../datasets/dianping/'
    user_pur_path = open(dir_data + "user_purd_items.pickle", 'rb')
    history_u_lists = pickle.load(user_pur_path)
    item_purd_path = open(dir_data + "item_purd_users.pickle", 'rb')
    history_v_lists = pickle.load(item_purd_path)
    user_rating_path = open(dir_data + "user_item_rating.pickle", 'rb')
    history_ur_lists = pickle.load(user_rating_path)
    item_rating_path = open(dir_data + "item_user_rating.pickle", 'rb')
    history_vr_lists = pickle.load(item_rating_path)
    train_u, train_v, train_r, train_f = data_to_list(path=dir_data+'train.csv')
    test_u, test_v, test_r, test_f = data_to_list(path=dir_data+'test.csv')
    ratings_list = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    user_social_path = open(dir_data + 'user_social.pickle', 'rb')
    user_social_adj_lists = pickle.load(user_social_path)
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r),torch.FloatTensor(train_f))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r),torch.FloatTensor(test_f))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
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

    der_srec = CDR_SB(rep_u, rep_v, mi_net, device,u2e,v2e,history_u_lists).to(device)
    optimizer = torch.optim.RMSprop(der_srec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0
    train_loss_list = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        total_loss = train(der_srec, device, train_loader, optimizer, epoch,beta)
        expected_rmse, mae, test_total_loss = test(der_srec, device, test_loader,beta)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
            torch.save(der_srec.state_dict(),'{}.pth'.format(save_model_path))
        else:
            endure_count += 1
        train_loss = sum(total_loss) / len(total_loss)
        train_loss_list.append(train_loss)
        test_loss = sum(test_total_loss) / len(test_total_loss)
        test_loss_list.append(test_loss)
        logger.info("best rmse: %.4f, best mae:%.4f" % (best_rmse, best_mae))
        torch.save(der_srec.state_dict(),'{}/weights_epoch_{}.pth'.format(args.train_model_weights_path,epoch))
        if endure_count > 5:
            break




if __name__ == '__main__':
    main()

