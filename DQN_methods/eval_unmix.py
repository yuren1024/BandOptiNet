'''
Author: Yuren
Date: 2024-07-25 16:38:52
LastEditors: Yuren
LastEditTime: 2024-07-26 17:39:36
FilePath: /Wavenumber_select/DQN_methods/eval_unmix.py
Description: 

Copyright (c) 2024 by Yuren, All Rights Reserved. 
'''
import numpy as np
import torch
from RL_utils import mcr_amend,DQN_unmix_pred
from sklearn.metrics import mean_squared_error as mse
from model_base.LightNet_location import LightNet_loc
import argparse

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
    parser.add_argument('--max_bands', default=40, type=int, help='input max selected bands')
    parser.add_argument('--prior',default=[12,14,15,16,17],type=list, help = 'prior')
    parser.add_argument('--data_path', default = '../data/data_0619_sy/data_0619_rein_clean.npz',type=str, help='dataset path')
    parser.add_argument('--standard_path', default = '../data/data_0619_sy/endmember.npz',type=str, help='standard path')
    parser.add_argument('--gpu', default='1', help='gpu id')
    parser.add_argument('--name', default='0726', type=str, help='name')
    params = parser.parse_args()


    params.device = torch.device(f"cuda:{params.pgu}" if torch.cuda.is_available() else "cpu")
    params.sdata = load_data(params.data_path)
    params.standard = np.load(params.standard_path)['noisy']
    params.model =  LightNet_loc(params.action_nums,params.action_size)


    agent = DQN_unmix_pred(params)

    agent.pred()
