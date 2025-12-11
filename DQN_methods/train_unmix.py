'''
Author: Yuren
Date: 2024-07-25 16:39:16
LastEditors: Yuren
LastEditTime: 2024-07-26 15:21:21
FilePath: /Wavenumber_select/DQN_unmix/train_unmix.py
Description: 

Copyright (c) 2024 by Yuren, All Rights Reserved. 
'''

import torch
import numpy as np
from RL_utils import DQN_unmix,set_seed
from model_base.LightNet_location import LightNet_loc
import argparse

# load data
def load_data(file_path):
    data = np.load(file_path)['data']
    # n x width x height x channels 100 100 100 50
    label = np.load(file_path)['label']
    # n x width x height 100 100 100
    return [data, label]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_size', default=50, type=int, help='input action size')
    parser.add_argument('--state_size', default=100, type=int, help='input state size')
    parser.add_argument('--action_nums', default=5, type=int, help='input action nums')
    parser.add_argument('--batch_size', default=24, type=int, help='input batch size')
    parser.add_argument('--max_bands', default=40, type=int, help='input max selected bands')

    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--discount', type=float, default=0.6, help='discount_factor factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='epsilon factor')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='epsilon decay')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='epsilon min')
    parser.add_argument('--memory_size', type=int, default=10000, help='max memory_size')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 80], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=15000, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='1', help='gpu id')


    parser.add_argument('--data_path', default = '../data/data_0619_sy/data_0619_rein_clean.npz',type=str, help='dataset path')
    parser.add_argument('--standard_path', default = '../data/data_0619_sy/endmember.npz',type=str, help='standard path')
    parser.add_argument('--test_size', default=10, type=int, help='input test size')
    parser.add_argument('--test_mse', default=None, type=str, help='input test data mse')

    parser.add_argument('--pretrain_path', default='', help='pre-trained model .prh file path')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1024, type=int, help='random seed')
    
    parser.add_argument('--name', default='0726', type=str, help='name')
    params = parser.parse_args()
    

    set_seed(params.seed)

    params.device = torch.device(f"cuda:{params.pgu}" if torch.cuda.is_available() else "cpu")
    
    params.sdata = load_data(params.data_path)
    params.standard = np.load(params.standard_path)['noisy']
    if params.test_mse:
        #'../data/data_0619_sy/testdataset_full_mse.npz'
        params.test_data_mse = np.load(params.test_mse)['data']
    else:
        params.test_data_mse = 'NAN'


    params.model =  LightNet_loc(params.action_nums,params.action_size)
    # agent = DQNAgent(state_size=state_size,action_size=50,learning_rate=0.1,batch_size=batch_size,
    #                 action_nums=action_nums,max_selected_bands=15,sdata=data,
    #                 test_size=10, standard=standard, dqn_model=dqn_model,device=device,
    #                 test_data_mse=test_data_mse, model_name='0622_data0619_15')
    
    agent = DQN_unmix(params)

    agent.run(episodes=params.epoch)