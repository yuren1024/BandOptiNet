'''
                                                    __----~~~~~~~~~~~------___
                                   .  .   ~~//====......          __--~ ~~
                   -.            \_|//     |||\\  ~~~~~~::::... /~
                ___-==_       _-~o~  \/    |||  \\            _/~~-
        __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~
    _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /
  .~       .~       |   \\ -_    /  /-   /   ||      \   /
 /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /
 |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\
          '         ~-|      /|    |-~\~~       __--~~
                      |-~~-_/ |    |   ~\_   _-~            /\
                           /  \     \__   \/~                \__
                       _--~ _/ | .-~~____--~-/                  ~~==.
                      ((->/~   '.|||' -_|    ~~-/ ,              . _||
                                 -_     ~\      ~~---l__i__i__i--~~_/
                                 _-~-__   ~)  \--______________--~~
                               //.-~~~-~_--~- |-------~~~~~~~~
                                      //.-~~~--\
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                              神兽保佑            永无BUG
'''

import numpy as np
import torch
import random
import os
from collections import namedtuple
import datetime
import time

Transition = namedtuple('Transition',('state','action','next_state','reward','band_mask','history_selection'
                                      ,'old_state_loc','next_state_history',))

class ReplayMemory:
    def __init__(self,memory_size):
        self.memory_size = memory_size
        self.memory = []
        self.index = 0
    def push(self,state,action,next_state,reward,band_mask,history_selection,old_state_loc,next_state_history):
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.index] = Transition(state,action,next_state,reward,band_mask,history_selection,
                                             old_state_loc,next_state_history)
        self.index  = (self.index+1) % self.memory_size

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)
    def __len__(self):
        return len(self.memory)

class DQNFather:
    def __init__(self, params):
        """
        state_size=100, action_size=100, learning_rate=0.2, discount_factor=0.6, epsilon=0.5,
                 epsilon_decay=0.995, epsilon_min=0.05, memory_size=10000,batch_size = 16
                 , max_selected_bands=40, device = 'cpu', sdata=None, dqn_model=None
                 , model_name='0612'
        """
        self.state_size = params.state_size
        self.action_size = params.action_size
        self.memory_size = params.memory_size
        self.memory = ReplayMemory(self.memory_size)

        self.learning_rate = params.lr
        self.discount_factor = params.discount
        self.epsilon = params.epsilon
        self.epsilon_decay = params.epsilon_decay
        self.epsilon_min = params.epsilon_min
        self.device = params.device
        self.model = params.model.to(self.device)
 
        self.batch_size = params.batch_size

        self.max_selected_bands = params.max_bands
        self.action_nums = params.action_nums

        # 数据集分配
        self.sdata_train = params.sdata
        self.gt_train = params.sdata
        self.sdata_test = params.sdata
        self.gt_test = params.sdata
        self.model_name = params.name
        self._action_check()


    def _action_check(self):
        try:
            assert self.max_selected_bands > self.action_nums
            assert self.max_selected_bands % self.action_nums == 0
        except:
            print("Your action is illegal!!!")
            raise NotImplementedError

    def _replay(self):
        pass

    def update_q_function(self):
        self._replay()


    def choose_action(self):
        pass

    def state_print(self, phase='训练'):
        print(datetime.datetime.now().strftime("%F:%H:%M:%S"))
        print(f"=============================={phase} INIT =====================================")
        print(f"训练数据大小{self.sdata_train.shape} label：{self.gt_train.shape}")
        print(f"测试数据大小{self.sdata_test.shape} label：{self.gt_test.shape}")
        print(f"==============================={phase}START======================================")

    def save_model(self, model_name, epoch, reward):
        dir_path = 'checkpoint/' + model_name
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        torch.save({'epoch': epoch + 1,
                    'reward': reward, 
                    'state_dict': self.model.state_dict(),
                    }, dir_path + '/model_best_reward.pth')

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load('checkpoint/' + model_name + '/model_best_reward.pth')['state_dict'])
        return self.model

    def reset(self):
        pass

    def get_reward(self):
        pass

    def update_bands(self,selected_bands,action):
        for ac in action:
            selected_bands.append(ac)
        
        selected_bands.sort()
        return selected_bands

    def bands_mask(self,selected_bands):
        mask = torch.zeros(self.action_size)
        for band in selected_bands:
            mask[band] = float('-inf')
        return mask.clone().detach()
    
    def historical_encoding(self,selected_positions): #针对没有batch的
        selected_positions = torch.tensor(selected_positions).type(torch.int64)
        one_hot = torch.zeros([1, self.action_size])
        one_hot.scatter_(1, selected_positions.unsqueeze(0), 1)
        return one_hot
    
    # 统一排列
    def train(self):
        self.state_print()
        pass

    def test_prior(self):
        pass
    def test_no_prior(self): 
        pass

    def pred(self):
        pass

    def run(self,episodes = 10000):
        print(datetime.datetime.now().strftime("%F:%H:%M:%S"))
        time0 = time.time()
        self.train(episodes=episodes)
        time1 = time.time()
        print("训练耗时{:.2f} hours".format((time1-time0)/60/60))
        self.pred()
        print("预测耗时{:.2f} seconds".format(time.time()-time1))
        print(datetime.datetime.now().strftime("%F:%H:%M:%S"))

        