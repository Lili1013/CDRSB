import matplotlib.pyplot as plt
import numpy as np
import random

import pandas as pd
import seaborn as sns


def draw_emb_size():
    # fig,ax = plt.subplots()
    # ax.spines['right'].set_visible(False)
    # plt.figure(figsize=(12,12))
    fig,axs = plt.subplots(2,3,figsize=(15,10))
    x= np.arange(4)
    bar_width=0.5
    name_list = ['8','16','32','64']
    value_rmse_ciao = [0.8728,0.8634,0.8538,0.8587]
    value_mae_ciao = [0.6593,0.6545,0.6406,0.6417]
    value_rmse_epinions = [0.9604,0.9596,0.9453,0.9647]
    value_mae_epinions = [0.7389,0.7302,0.7253,0.7735]
    value_rmse_dianping = [0.6993,0.6967,0.6893,0.6918]
    value_mae_dianping = [0.5460,0.5414,0.5311,0.5616]
    # plt.subplot(2,3,1)
    for a,b in zip(x,value_rmse_ciao):
        axs[0,0].text(a,b,b,ha='center',va='bottom',rotation=45)
    axs[0,0].bar(x,value_rmse_ciao,bar_width,label='RMSE',color='red')
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(name_list)
    axs[0, 0].set_ylabel('RMSE')
    axs[0,0].spines['top'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)

    # plt.subplot(2,3,4)
    for a,b in zip(x,value_mae_ciao):
        axs[1,0].text(a,b,b,ha='center',va='bottom',rotation=45)
    axs[1,0].bar(x,value_mae_ciao,bar_width,label='MAE',color='red')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(name_list)

    # axs[0,0].xticks(x+bar_width/2,name_list)
    axs[1, 0].spines['top'].set_visible(False)
    axs[1, 0].spines['right'].set_visible(False)
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].set_xlabel('Ciao')
    # axs[1,0].xticks(x+bar_width/2,name_list)

    # plt.subplot(2,3,2)
    for a,b in zip(x,value_rmse_epinions):
        axs[0,1].text(a,b,b,ha='center',va='bottom',rotation=45)
    axs[0,1].bar(x,value_rmse_epinions,bar_width,label='RMSE')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(name_list)
    axs[0, 1].set_ylabel('RMSE')
    axs[0, 1].spines['top'].set_visible(False)
    axs[0, 1].spines['right'].set_visible(False)


    # plt.subplot(2,3,5)
    for a,b in zip(x,value_mae_epinions):
        axs[1,1].text(a,b,b,ha='center',va='bottom',rotation=45)
    axs[1,1].bar(x,value_mae_epinions,bar_width,label='MAE')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(name_list)
    axs[1, 1].set_ylabel('MAE')
    axs[1, 1].set_xlabel('Epinions')
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)


    # plt.subplot(2,3,3)
    for a,b in zip(x,value_rmse_dianping):
        axs[0,2].text(a,b,b,ha='center',va='bottom',rotation=45)
    axs[0,2].bar(x,value_rmse_dianping,bar_width,label='RMSE',color='green')
    axs[0, 2].set_xticks(x)
    axs[0, 2].set_xticklabels(name_list)
    axs[0, 2].set_ylabel('RMSE')
    axs[0, 2].spines['top'].set_visible(False)
    axs[0, 2].spines['right'].set_visible(False)
    # axs[0,2].xticks(x+bar_width/2,name_list)

    # plt.subplot(2,3,6)
    for a,b in zip(x,value_mae_dianping):
        axs[1,2].text(a,b,b,ha='center',va='bottom',rotation=45)
    axs[1,2].bar(x,value_mae_dianping,bar_width,label='MAE',color='green')
    axs[1, 2].set_xticks(x)
    axs[1, 2].set_xticklabels(name_list)
    axs[1, 2].set_ylabel('MAE')
    axs[1, 2].set_xlabel('Dianping')
    # axs[0,0].xticks(x+bar_width/2,name_list)
    axs[1, 2].spines['top'].set_visible(False)
    axs[1, 2].spines['right'].set_visible(False)

    # axs[1,2].xticks(x+bar_width/2,name_list)
    plt.subplots_adjust(hspace=0.3,wspace=0.5)
    plt.show()

def draw_beta():
    x = np.linspace(0,12,12)
    y_ciao_rmse = [0.8603,0.8578,0.8539,0.8564,0.8586,0.8587,0.8538,0.8522,0.8591,0.8586,0.8588,0.8537]
    y_ciao_mae = [0.6569,0.6528,0.6480,0.6501,0.6550,0.6537,0.6406,0.6482,0.6466,0.6515,0.6502,0.6476]
    y_epinions_rmse = [0.9448,0.9493,0.9482,0.9463,0.9484,0.9499,0.9453,0.9469,0.9480,0.9496,0.9490,0.9502]
    y_epinions_mae = [0.7278,0.7350,0.7283,0.7264,0.7316,0.7367,0.7253,0.7346,0.7373,0.7308,0.7289,0.7358]
    y_dianping_rmse = [0.6898,0.6972,0.6993,0.6917,0.6957,0.6904,0.6893,0.6904,0.7023,0.7012,0.6978,0.6947]
    y_dianping_mae = [0.5345,0.5395,0.5406,0.5389,0.5418,0.5423,0.5311,0.5401,0.5567,0.5525,0.5439,0.5412]
    fig,ax = plt.subplots(2,3,figsize=(15,30))
    ax[0,0].plot(x,y_ciao_rmse,label='line1',color='green')
    ax[0,0].set_xticks(np.linspace(0,12,12))
    ax[0,0].set_xticklabels(['0.001','0.01','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],rotation=45)
    ax[0, 0].set_ylabel('Ciao-RMSE')
    ax[0,0].tick_params(axis='both',which='major',labelsize=11)


    ax[0,1].plot(x,y_epinions_rmse,label='line2',color = 'purple')
    ax[0,1].set_xticks(np.linspace(0, 12, 12))
    ax[0,1].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[0, 1].set_ylabel('Epinions-RMSE')
    ax[0, 1].tick_params(axis='both', which='major', labelsize=11)

    ax[0,2].plot(x, y_dianping_rmse, label='line2', color='blue')
    ax[0,2].set_xticks(np.linspace(0, 12, 12))
    ax[0,2].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[0, 2].set_ylabel('Dianping-RMSE')
    ax[0, 2].tick_params(axis='both', which='major', labelsize=11)

    ax[1,0].plot(x, y_ciao_mae, label='line2', color='green')
    ax[1,0].set_xticks(np.linspace(0, 12, 12))
    ax[1,0].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[1, 0].set_ylabel('Ciao-MAE')
    # ax[1, 0].set_xlabel('Ciao')
    ax[1, 0].tick_params(axis='both', which='major', labelsize=11)

    ax[1,1].plot(x, y_epinions_mae, label='line2', color='purple')
    ax[1,1].set_xticks(np.linspace(0, 12, 12))
    ax[1,1].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[1, 1].set_ylabel('Epinions-MAE')
    # ax[1, 1].set_xlabel('Epinions')
    ax[1, 1].tick_params(axis='both', which='major', labelsize=11)

    ax[1,2].plot(x, y_dianping_mae, label='line2', color='blue')
    ax[1,2].set_xticks(np.linspace(0, 12, 12))
    ax[1,2].set_xticklabels(['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],rotation=45)
    ax[1, 2].set_ylabel('Dianping-MAE')
    # ax[1, 2].set_xlabel('Dianping',fontsize=11)
    ax[1, 2].tick_params(axis='both', which='major', labelsize=11)

    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    # plt.tight_layout()

    plt.show()
    pass

def draw_beta1():

    y_ciao_rmse = [0.8603, 0.8578, 0.8539, 0.8564, 0.8586, 0.8587, 0.8538, 0.8522, 0.8591, 0.8586, 0.8588, 0.8537]
    y_ciao_mae = [0.6569, 0.6528, 0.6480, 0.6501, 0.6550, 0.6537, 0.6406, 0.6482, 0.6466, 0.6515, 0.6502, 0.6476]
    y_epinions_rmse = [0.9448, 0.9493, 0.9482, 0.9463, 0.9484, 0.9499, 0.9453, 0.9469, 0.9480, 0.9496, 0.9490, 0.9502]
    y_epinions_mae = [0.7278, 0.7350, 0.7283, 0.7264, 0.7316, 0.7367, 0.7253, 0.7346, 0.7373, 0.7308, 0.7289, 0.7358]
    y_dianping_rmse = [0.6898, 0.6972, 0.6993, 0.6917, 0.6957, 0.6904, 0.6893, 0.6904, 0.7023, 0.7012, 0.6978, 0.6947]
    y_dianping_mae = [0.5345, 0.5395, 0.5406, 0.5389, 0.5418, 0.5423, 0.5311, 0.5401, 0.5567, 0.5525, 0.5439, 0.5412]
    df = pd.DataFrame({
        'x': ['0.001', '0.01', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
        'y_ciao_rmse':y_ciao_rmse,
        'y_ciao_mae':y_ciao_mae,
        'y_epinions_rmse':y_epinions_rmse,
        'y_epinions_mae':y_epinions_mae,
        'y_dianping_rmse':y_dianping_rmse,
        'y_dianping_mae':y_dianping_mae
    })
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    sns.lineplot(data=df,x='x',y='y_ciao_rmse',ax = ax[0,0])
    sns.lineplot(data=df, x='x', y='y_ciao_rmse', ax=ax[0, 1])
    sns.lineplot(data=df, x='x', y='y_ciao_rmse', ax=ax[0, 2])
    sns.lineplot(data=df, x='x', y='y_ciao_rmse', ax=ax[1, 0])
    sns.lineplot(data=df, x='x', y='y_ciao_rmse', ax=ax[1, 1])
    sns.lineplot(data=df, x='x', y='y_ciao_rmse', ax=ax[1, 2])

    plt.subplots_adjust(wspace=0.5,hspace=0.3)
    plt.xticks(rotation=45)
    plt.show()
if __name__ == '__main__':
    # draw_emb_size()
    # draw_beta()
    # draw_beta1()
    print(((1.2905-0.9453)+(1.1309-0.8538)+(1.0260-0.6880))/3)
    print(((1.0203-0.7253)+(0.9107-0.6406)+(0.8530-0.5264))/3)

