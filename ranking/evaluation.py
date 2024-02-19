import math
import heapq
import numpy as np
import torch
import heapq
from loguru import logger

_model = None
_test_ratings = None
_test_negatives = None
_k = None
_device = None
def calculate_hr_ndcg(model,test_ratings,test_negatives,k,device):
    global _model
    global _test_negatives
    global _test_ratings
    global _k
    global _device
    _model = model
    _test_ratings = test_ratings
    _test_negatives = test_negatives
    _k = k
    _device = device

    hits,ndcgs = [],[]
    for idx in range(len(_test_ratings)):
        (hr,ndcg) = eval_one_user(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return hits,ndcgs
def eval_one_user(idx):
    rating = _test_ratings[idx]
    items = _test_negatives[idx]
    u_id = rating[0]
    i_id = rating[1]
    red_by_f = rating[2]
    items.append(i_id)
    user_id_input,item_id_input,red_by_f_input = [],[],[]
    for i in range(len(items)):
        user_id_input.append(u_id)
        item_id_input.append(items[i])
        if i == len(items)-1:
            red_by_f_input.append(red_by_f)
        else:
            red_by_f_input.append(0)

    user_id_input = torch.tensor(user_id_input).to(_device)
    item_id_input = torch.tensor(item_id_input).to(_device)
    red_by_f_input = torch.tensor(red_by_f_input).to(_device)
    predictions = _model.forward(user_id_input,item_id_input,red_by_f_input)
    map_item_score = {}
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    ranklist = heapq.nlargest(_k,map_item_score,key=map_item_score.get)
    hr = get_hit_ratio(ranklist,i_id)
    ndcg = get_ndcg(ranklist,i_id)
    return (hr,ndcg)

def get_hit_ratio(ranklist,item):
    for each_item in ranklist:
        if each_item == item:
            return 1
    return 0

def get_ndcg(ranklist,item):
    for i in range(len(ranklist)):
        each_item = ranklist[i]
        if each_item == item:
            return math.log(2)/math.log(i+2)
    return 0





def calculate_precision(pred_logits,true_lables):
    # _,pred_leables = torch.max(pred_logits,dim=1)
    pred_lists = pred_logits.tolist()
    pred_labels = []
    for each in pred_lists:
        if each > 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    pred_labels = torch.tensor(pred_labels)
    true_lables = true_lables.int()
    correct_pred = (pred_labels==true_lables).sum().item()
    total_pred = len(true_lables)
    precison = correct_pred/total_pred
    return precison

