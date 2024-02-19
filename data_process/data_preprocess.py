import os.path
import pickle
import random

import pandas as pd
import numpy as np
import scipy.io as scio
from loguru import logger

def id_map(df,path_to,name):
    '''
    map user and item ids
    :param df: original datasets
    :param path_to: save path
    :param name:
    :return:
    '''
    df.sort_values(by=name,inplace=True)
    user_ids = []
    for x in df.groupby(by=name):
        user_ids.append(x[0])
    map = {}
    for i in range(len(user_ids)):
        source = user_ids[i]
        map[source] = i
    save_file = open(path_to, 'wb')
    logger.info('start store')
    pickle.dump(map, save_file)
    save_file.close()

def process_rating(df,path,path_to):

    rating = {}
    if 'user_purd_items' in path:
        for x in df.groupby(by='user_id'):
            rating[x[0]] = list(x[1]['rating'])
    else:
        for x in df.groupby(by='item_id'):
            rating[x[0]] = list(x[1]['rating'])
    save_file = open(path_to, 'wb')
    logger.info('start store')
    pickle.dump(rating, save_file)
    save_file.close()
def split_datasets_rating(df,train_path,test_path):
    '''
    split train data and test data
    regard each user's 90% data as train data
    regard each user's 10% data as test data
    :param df:
    :return:
    '''
    train = []
    test = []
    for x in df.groupby(by='user_id'):
        each_train = x[1].sample(frac=0.9,replace=False,random_state=1)[['user_id','item_id','rating']]
        train.append(each_train)
        train_items = list(each_train['item_id'])
        items = list(x[1]['item_id'])
        test_items= list(set(items).difference(set(train_items)))
        each_test = x[1][x[1]['item_id'].isin(test_items)][['user_id','item_id','rating']]
        test.append(each_test)

    # train_path = '../datasets/epinions/processed_data/train.csv'
    train_df = pd.concat(train,axis=0,ignore_index=True)
    train_df.to_csv(train_path,index=False)
    logger.info('store train data, number is {}'.format(len(train_df)))

    # test_path = '../datasets/epinions/processed_data/test.csv'
    test_df = pd.concat(test, axis=0, ignore_index=True)
    test_df.to_csv(test_path, index=False)
    logger.info('store test data, number is {}'.format(len(test_df)))
def split_datasets_ranking(source_path,to_path):
    df = pd.read_csv(source_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df.sort_values(by=['user_id'], inplace=True)
    uid_field, iid_field = 'user_id', 'item_id'

    uid_freq = df.groupby(uid_field)[iid_field]
    u_i_dict = {}
    for u, u_ls in uid_freq:
        u_i_dict[u] = list(u_ls)
    new_label = []
    u_ids_sorted = sorted(u_i_dict.keys())

    for u in u_ids_sorted:
        items = u_i_dict[u]
        n_items = len(items)
        tmp_ls = [0] * (n_items - 1) + [1]
        # if n_items < 10:
        #     tmp_ls = [0] * (n_items - 1) + [1]
        # else:
        #     test_len = int(n_items * 0.1)
        #     train_len = n_items - test_len
        #     tmp_ls = [0] * train_len + [1] * test_len
        new_label.extend(tmp_ls)
    df['x_label'] = new_label
    df[['user_id','item_id','rating','x_label']].to_csv(to_path,index=False)

def process_user_social(path_source,path_map,to_path_1,to_path_2):
    '''
    process user's social network data
    :return:
    '''
    data_social_path = path_source
    data = scio.loadmat(data_social_path)
    df = pd.DataFrame(data=data['trust'],
                      columns=['user_id', 'follow_user_id'])
    map = open(path_map, 'rb')
    map = pickle.load(map)
    df.rename(columns={'user_id': 'original_user_id','follow_user_id':'original_follow_user_id'}, inplace=True)
    df['user_id'] = None
    df['follow_user_id'] = None
    logger.info('start transform')
    for key, value in map.items():
        logger.info(key)
        df.loc[df['original_user_id'].isin([key]), 'user_id'] = int(value)
        df.loc[df['original_follow_user_id'].isin([key]), 'follow_user_id'] = int(value)
    df.drop_duplicates(subset=['user_id', 'follow_user_id'], keep='last', inplace=True)
    df.dropna(axis=0,subset=['user_id','follow_user_id'],inplace=True)
    df.to_csv(to_path_1,index=False)
    user_social = {}
    for x in df.groupby(by='user_id'):
        user_social[x[0]] = list(x[1]['follow_user_id'])
    save_file = open(to_path_2, 'wb')
    print('start store')
    pickle.dump(user_social, save_file)
    save_file.close()
    # process_item_user_interactions(df)

def process_item_user_interactions(df,to_path):
    '''
    user set (in training set) who have interacted with the item
    :return:
    '''
    iu_interaction = {}
    for x in df.groupby(by='item_id'):
        iu_interaction[x[0]] = list(x[1]['user_id'])
    save_file = open(to_path, 'wb')
    print('start store')
    pickle.dump(iu_interaction, save_file)
    save_file.close()
def process_user_item_interactions(df,to_path):
    '''
    user's purchased history
    :param df:
    :return:
    '''
    ui_interaction = {}
    for x in df.groupby(by='user_id'):
        ui_interaction[x[0]] = list(x[1]['item_id'])
    save_file = open(to_path, 'wb')
    print('start store')
    pickle.dump(ui_interaction, save_file)
    save_file.close()

def filter_g_k_one(data,k=10,u_name='user_id',i_name='item_id',y_name='rating'):
    '''
    delete the records that user and item interactions lower than k
    '''
    item_group = data.groupby(i_name).agg({y_name:'count'}) #every item has the number of ratings
    item_g10 = item_group[item_group[y_name]>=k].index
    data_new = data[data[i_name].isin(item_g10)]

    user_group = data_new.groupby(u_name).agg({y_name: 'count'})  # every item has the number of ratings
    user_g10 = user_group[user_group[y_name] >= k].index
    data_new_1 = data_new[data_new[u_name].isin(user_g10)]
    return data_new_1

def map_user_item_id(data,user_path,item_path):
    '''
    map original user id and item id into consequent number
    '''
    data = data[['user_id','item_id','rating','timestamp']]
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_to_id = dict(zip(list(user),list(np.arange(user.shape[0]))))
    save_file = open(user_path, 'wb')
    print('start store')
    pickle.dump(user_to_id, save_file)
    save_file.close()
    item_to_id = dict(zip(list(item), list(np.arange(item.shape[0]))))
    save_file = open(item_path, 'wb')
    print('start store')
    pickle.dump(item_to_id, save_file)
    save_file.close()
    print('user num:',user.shape)
    print('item num',item.shape)
    data.rename(columns={'user_id': 'original_user_id', 'item_id': 'original_item_id'}, inplace=True)
    data['user_id'] = data['original_user_id'].map(user_to_id)
    data['item_id'] = data['original_item_id'].map(item_to_id)
    return data



if __name__ == '__main__':
    processed_data_path = '../datasets/epinions/'
    # #--------first step------------
    # data_path = '../datasets/epinions/rating_with_timestamp.mat'
    # data = scio.loadmat(data_path)
    # df = pd.DataFrame(data=data['rating_with_timestamp'],
    #                   columns=['user_id', 'item_id', 'category_id', 'rating', 'helpfulness', 'timestamp'])
    # data = filter_g_k_one(df,k=5)
    # if not os.path.exists(processed_data_path):
    #     os.makedirs(processed_data_path)
    # data = map_user_item_id(data,user_path=processed_data_path+'user_id_map.pickle',item_path=processed_data_path+'item_id_map.pickle')
    # data.to_csv(processed_data_path+'epinions_rating_with_timestamp.csv', index=False)
    # #------------second step: to save time, directly read file-----------
    # df = pd.read_csv(processed_data_path+'epinions_rating_with_timestamp.csv')
    # process_user_item_interactions(df,to_path=processed_data_path+'user_purd_items.pickle')
    # process_item_user_interactions(df,to_path=processed_data_path+'item_purd_users.pickle')
    # process_user_social(path_source='../datasets/epinions/trust.mat',path_map=processed_data_path+'user_id_map.pickle',to_path_1=processed_data_path+'user_trust.csv',to_path_2=processed_data_path+'user_social.pickle')
    # process_rating(df,path=processed_data_path+'user_purd_items.pickle',path_to=processed_data_path+'user_item_rating.pickle')
    # process_rating(df,path=processed_data_path+'item_purd_users.pickle',path_to = processed_data_path+'item_user_rating.pickle')
    # split_datasets_rating(df,train_path=processed_data_path+'train.csv',test_path=processed_data_path+'test.csv')
    # ranking split datasets
    split_datasets_ranking(source_path='../datasets/epinions/epinions_rating_with_timestamp.csv',to_path='../datasets/epinions/epinions_inter.csv')




