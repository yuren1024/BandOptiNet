'''
Author: Yuren
Date: 2024-07-25 16:39:27
LastEditors: Yuren
LastEditTime: 2024-07-26 15:23:45
FilePath: /Wavenumber_select/DQN_unmix/train_segment.py
Description: 

Copyright (c) 2024 by Yuren, All Rights Reserved. 
'''
import torch
import numpy as np
from model_base.LightNet_location import LightNet_loc
from RL_utils import DQN_segment,set_seed
import argparse

# load data
def load_data(file_path):
    data = np.load(file_path)['data_nm']
    # n x width x height x channels  84 200 200 51
    label = np.load(file_path)['label']
    # n x width x height  84 200 200
    name = np.load(file_path)['name']
    # n x 1 
    return [data, label, name]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_size', default=51, type=int, help='input action size')
    parser.add_argument('--state_size', default=200, type=int, help='input state size')
    parser.add_argument('--action_nums', default=5, type=int, help='input action nums')
    parser.add_argument('--batch_size', default=12, type=int, help='input batch size')
    parser.add_argument('--max_bands', default=40, type=int, help='input max selected bands')

    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='learning rate decay factor')
    parser.add_argument('--discount', type=float, default=0.6, help='discount_factor factor')
    parser.add_argument('--epsilon', type=float, default=0.5, help='epsilon factor')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help='epsilon decay')
    parser.add_argument('--epsilon_min', type=float, default=0.05, help='epsilon min')
    parser.add_argument('--memory_size', type=int, default=10000, help='max memory_size')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 80], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=3100, type=int, help='Stopping epoch')
    parser.add_argument('--gpu', default='1', help='gpu id')

    parser.add_argument('--data_path', default = '../data/micro_data_0620/micro0620_200.npz',type=str, help='dataset path')
    parser.add_argument('--standard_path', default = '../data/micro_data_0620/micro_standard.npz',type=str, help='standard path')
    parser.add_argument('--test_size', default=10, type=int, help='input test size')

    parser.add_argument('--pretrain_path', default='', help='pre-trained model .prh file path')
    parser.add_argument('--save_freq', default=50, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1024, type=int, help='random seed')
    
    parser.add_argument('--name', default='0726', type=str, help='name')
    params = parser.parse_args()
    
    set_seed(params.seed)

    params.device = torch.device(f"cuda:{params.pgu}" if torch.cuda.is_available() else "cpu")
  
    params.sdata = load_data(params.data_path)
    params.standard = np.load(params.standard_path)['data']
    params.model =  LightNet_loc(params.action_nums,params.action_size)
    # agent = DQNAgent(state_size=state_size, action_size=action_size,learning_rate=0.1,batch_size=batch_size,
    #                 action_nums=action_nums,max_selected_bands=35,sdata=data,
    #                 device=device,test_size = 20, dqn_model=dqn_model, 
    #                 standard=standard, model_name='0704_datamicro0620_35')

    agent = DQN_segment(params)
    agent.run(episodes=params.epoch)