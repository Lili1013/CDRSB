import pandas as pd
import pickle
from loguru import logger

def item_statistic(df_data,friends_dict):
    df_data['red_by_friends'] = 0
    for index,row in df_data.iterrows():
        if index % 10000 == 0:
            print(index)
        user_id = row['user_id']
        item_id = row['item_id']
        if user_id in friends_dict:
            u_friends = friends_dict[user_id]
            friends_items = list(df_data[df_data['user_id'].isin(u_friends)]['item_id'].values)
            if item_id in friends_items:
                df_data.loc[(df_data['user_id'] == user_id) & (df_data['item_id'] == item_id), 'red_by_friends'] = 1
    return df_data
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
        each_train = x[1].sample(frac=0.9,replace=False,random_state=1)[['user_id','item_id','rating','red_by_friends']]
        train.append(each_train)
        train_items = list(each_train['item_id'])
        items = list(x[1]['item_id'])
        test_items= list(set(items).difference(set(train_items)))
        each_test = x[1][x[1]['item_id'].isin(test_items)][['user_id','item_id','rating','red_by_friends']]
        test.append(each_test)

    # train_path = '../datasets/epinions/processed_data/train.csv'
    train_df = pd.concat(train,axis=0,ignore_index=True)
    train_df.to_csv(train_path,index=False)
    logger.info('store train data, number is {}'.format(len(train_df)))

    # test_path = '../datasets/epinions/processed_data/test.csv'
    test_df = pd.concat(test, axis=0, ignore_index=True)
    test_df.to_csv(test_path, index=False)
    logger.info('store test data, number is {}'.format(len(test_df)))
if __name__ == '__main__':
    df_data = pd.read_csv('../datasets/douban_movie/douban_movie_inter.csv')
    friends_dict = pickle.load(open('../datasets/douban_movie/user_social.pickle','rb'))
    df_data_new = item_statistic(df_data,friends_dict)
    df_data_new.to_csv('../datasets/douban_movie/douban_movie_inter_new.csv',index=False)
    df = pd.read_csv('../datasets/douban_movie/douban_movie_inter_new.csv')
    split_datasets_rating(df, '../datasets/douban_movie/train.csv', '../datasets/douban_movie/test.csv')





