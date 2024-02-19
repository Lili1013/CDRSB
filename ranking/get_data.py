import torch
import torch.nn as nn

import multiprocessing
from functools import partial
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class Data(nn.Module):
    def __init__(self,data,train_num_negs,test_num_negs,batch_size):
        super(Data, self).__init__()
        self.df_all = data
        self.df_train = data[data['x_label']==0]
        self.df_test = data[data['x_label']==1]
        self.train_num_negs = train_num_negs
        self.test_num_negs = test_num_negs
        self.batch_size = batch_size

    def generate_train_instances_1(self,index_range, num_items):
        user_ids, item_ids, labels = [], [], []
        i = index_range[0]
        while i < index_range[1]:
            row = self.df_train.iloc[i]
            uid = row['user_id']
            iid = row['item_id']
            user_ids.append(int(uid))
            item_ids.append(int(iid))
            labels.append(1)
            for t in range(self.num_negs):
                j = np.random.randint(num_items)
                while len(self.df_all[(self.df_all['user_id'] == uid) & (self.df_all['item_id'] == j)]) > 0:
                    j = np.random.randint(num_items)
                user_ids.append(int(uid))
                item_ids.append(j)
                labels.append(0)
            i += 1
        return user_ids, item_ids, labels

    def get_train_instances(self):
        num_items = len(self.df_all['item_id'].unique())
        num_samples = len(self.df_train)
        chunk_size = num_samples // multiprocessing.cpu_count()
        index_ranges = [(i * chunk_size, min((i + 1) * chunk_size, num_samples)) for i in
                        range(multiprocessing.cpu_count())]
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        func = partial(self.generate_train_instances, num_items=num_items
                       )
        results = pool.map(func, index_ranges)
        pool.close()
        pool.join()
        if index_ranges[-1][1] < len(self.df_train):
            index_ranges = (index_ranges[-1][1], len(self.df_train))
        user_id_list, item_id_list, label_list = self.generate_train_instances(index_ranges,num_items)
        user_ids, item_ids, labels = [], [], []
        for result in results:
            user_ids.extend(result[0])
            item_ids.extend(result[1])
            labels.extend(result[2])
        user_ids.extend(user_id_list)
        item_ids.extend(item_id_list)
        labels.extend(label_list)

        return user_ids, item_ids, labels
    def generate_train_instances(self,batch_u,batch_v,batch_f):
        num_items = len(self.df_all['item_id'].unique())
        uids,iids,ratings,f = [],[],[],[]
        for i in range(len(batch_u)):
            u_id = batch_u[i]
            pos_i_id = batch_v[i]
            pos_f = batch_f[i]
            uids.append(u_id)
            iids.append(pos_i_id)
            f.append(pos_f)
            ratings.append(1)
            for t in range(self.train_num_negs):
                j = np.random.randint(num_items)
                while len(self.df_all[(self.df_all['user_id'] == u_id) & (self.df_all['item_id'] == j)]) > 0:
                    j = np.random.randint(num_items)
                uids.append(u_id)
                iids.append(j)
                ratings.append(0)
                f.append(0)
        return uids,iids,ratings,f
    def generate_test_instances(self,batch_u,batch_v):
        num_items = len(self.df_all['item_id'].unique())
        pos_samples,neg_samples = [],[]
        for i in range(len(batch_u)):
            u_id = batch_u[i]
            pos_i_id = batch_v[i]
            pos_samples.append([u_id,pos_i_id])
            each_neg_samples = []
            for t in range(self.test_num_negs):
                j = np.random.randint(num_items)
                while len(self.df_all[(self.df_all['user_id'] == u_id) & (self.df_all['item_id'] == j)]) > 0:
                    j = np.random.randint(num_items)
                each_neg_samples.append(j)
            neg_samples.append(each_neg_samples)
        return pos_samples,neg_samples
    def read_test_samples(self,path):
        pos_samples = []
        neg_samples = []
        with open(path) as f:
            for line in f:
                # each_neg_samples = []
                line = line.replace('\n','')
                each_content = line.split(' ')[:101]
                each_content = list(map(int,each_content))
                pos_samples.append([each_content[0],each_content[1],each_content[2]])
                neg_samples.append(each_content[3:101])
            # neg_samples.append(each_neg_samples)
        return pos_samples,neg_samples



if __name__ == '__main__':
    df = pd.read_csv('../datasets/epinions/epinion_inter.csv')
    data = Data(data=df,train_num_negs=1,test_num_negs=99,batch_size=128)
    # user_ids, item_ids, labels = data.get_train_instances()
    data.read_test_samples(path='../datasets/epinions/epinion_test.txt')
    print('gg')
