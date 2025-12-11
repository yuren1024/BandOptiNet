import torch
import torch.optim as optim
import random
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from RL_utils import mcr_amend,DQNFather,Transition,ReplayMemory
import torch.nn.functional as F
import datetime

class DQN_unmix(DQNFather):
    def __init__(self,params):
        '''
        state_size=100, action_size=100, learning_rate=0.1, discount_factor=0.6, epsilon=0.5,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000,batch_size = 24,action_nums = 5,
                 max_selected_bands=40,device='cpu',  sdata=None, test_size = 10, dqn_model=None, standard=None, test_data_mse=None,
                 model_name='0612'
        '''
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.Explr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params.milestones, gamma=params.gamma, last_epoch=-1)
        self.batch_size = params.batch_size

        self.max_selected_bands = params.max_bands
        self.action_nums = params.action_nums

        self.test_size = params.test_size
        self.data_n = len(params.sdata[0])

        self.sdata_train = params.sdata[0][:self.data_n-self.test_size]
        self.gt_train = params.sdata[1][:self.data_n-self.test_size]
        self.sdata_test = params.sdata[0][self.data_n-self.test_size:]
        self.gt_test = params.sdata[1][self.data_n-self.test_size:]
        self.pure_spectra = params.standard
        self.test_data_mse = params.test_data_mse
        self.train_index = 0  # 0-80  data_indexa ->train_index
        self.test_index = 0  # 0-4   data_indexc ->test_index
        self.model_name = params.name
        self._action_check()


    def _replay(self):
        if len(self.memory)<self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)
        minibatch = Transition(*zip(*minibatch))
        

        batch_state = torch.cat(minibatch.state).float().to(self.device)
        batch_action = torch.cat(minibatch.action).to(self.device)   #当前选择的动作，也是下一步的位置
        batch_reward = torch.cat(minibatch.reward).float().to(self.device)

        batch_band_mask = torch.cat(minibatch.band_mask).to(self.device)  #是一个布尔值掩码  用于赋值输出
        batch_history_selection = torch.cat(minibatch.history_selection).to(self.device) #当前选择历史
        batch_old_state_loc = torch.cat(minibatch.old_state_loc).to(self.device)    #当前位置
        batch_next_state_history = torch.cat(minibatch.next_state_history).to(self.device)  #下一步的选择历史

        non_final_next_state_batch = torch.cat(
            [torch.from_numpy(s).unsqueeze(0) for s in minibatch.next_state if s is not None]).float().to(self.device)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s:s is not None,minibatch.next_state))).bool()
        non_final_band_mask = batch_band_mask[non_final_mask]
        non_final_next_state_history = batch_next_state_history[non_final_mask]       
        non_final_new_state_loc = batch_action[non_final_mask]
        
        next_state_values = torch.zeros([self.batch_size,self.action_nums]).to(self.device)
        self.model.eval()
        action_temp = self.model(non_final_next_state_batch,non_final_new_state_loc,
                                 non_final_next_state_history)   #shape [batch-n,100]

        action_temp = action_temp+non_final_band_mask
        values_temp = torch.topk(action_temp,self.action_nums).values
        next_state_values[non_final_mask]  = values_temp
        expected_state_action_values = batch_reward+self.discount_factor*next_state_values
        state_action_values =  self.model(batch_state,batch_old_state_loc,batch_history_selection).gather(dim=1,index = batch_action)
        
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values,expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.optimizer.param_groups[0]['lr'] >= 1e-3:
            self.Explr.step()


    def choose_action(self,state,band_mask,old_loc,history_selection,episode=None):
        eps = self.epsilon
        if episode is not None:
            eps = 0
        if random.random()<eps:

            search_space = torch.randn(self.action_size)
            search_space = search_space + band_mask
            action = torch.topk(search_space,self.action_nums).indices
            action = action.tolist()

        else:
            self.model.eval()
            with torch.no_grad():
                old_loc = old_loc.to(self.device)
                history_selection = history_selection.to(self.device)
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.model(state,old_loc,history_selection)[0].detach().cpu()
                action = action + band_mask
                action = torch.topk(action,self.action_nums).indices
                action = action.tolist()

        action.sort()
        return action


    def reset(self):
        self.train_index = np.random.randint(0,self.data_n-self.test_size)

    def get_reward(self, selected_bands, T_flag=True):

        if T_flag:
            data = self.sdata_train[self.train_index]
            c_gt = self.gt_train[self.train_index]
        else:
            data = self.sdata_test[self.test_index]
            c_gt = self.gt_test[self.test_index]

        selected_bands.sort()
        input = data.reshape(self.state_size * self.state_size, -1)
        input = input[:, selected_bands]
        temp_pure = self.pure_spectra[:, selected_bands]
        est = mcr_amend(input, temp_pure)
        c_gt = c_gt.reshape(self.state_size * self.state_size, -1)
        est[np.isnan(est)] = 0
        loss = mse(c_gt, est)
        assert loss != 0
        reward = (0.01 / loss)
        return reward


    def train(self, episodes=15000):
        self.state_print()
        history_reward = -1
        for episode in range(episodes):
            self.reset()
            # selected_bands = [27, 34, 43, 47, 89]  # 默认经验假设
            
            selected_bands = random.sample(list(range(self.action_size)),self.action_nums)  ##随机先验
            selected_bands.sort()
        
            # print(selected_bands)
            # print(self.sdata_train[self.data_indexa].shape)
            state = self.sdata_train[self.train_index, :, :, selected_bands]  # 随机选择初始状态

            done = False
            final_reward = -100
            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            # history_selection = self.historical_encoding(selected_bands) #torch # 1*100
            for _ in range(self.max_selected_bands // self.action_nums-1):
                history_selection = self.historical_encoding(selected_bands)# 旧的位置历史
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)  #旧的位置
                action = self.choose_action(state,band_mask,old_state_loc,history_selection)  #band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action    #新的位置
                
                # print("ACTION 选择的情况！检测{}".format(action))

                selected_bands = self.update_bands(selected_bands,action) #新的历史记录  
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat!")
                    exit()

                next_state = self.sdata_train[self.train_index, :, :, action]
                reward = self.get_reward(selected_bands)
                if len(selected_bands) == self.max_selected_bands:
                    done = True
                    final_reward = reward
                    next_state = None
                state = torch.from_numpy(state).unsqueeze(0)
                action = torch.from_numpy(np.array(action)).unsqueeze(0)
                reward = torch.tensor([reward for _ in range(self.action_nums)]).unsqueeze(0)
                band_mask =  band_mask.unsqueeze(0)
                # 这里有长度变化的问题，应该传入位置编码
                next_state_history = self.historical_encoding(selected_bands)

                # 补None操作
                self.memory.push(state, action,next_state, reward,band_mask,history_selection,old_state_loc,next_state_history)
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask
                state = next_state

            print(f"Train Episode {episode}: Selected Bands {selected_bands} with reward {final_reward}")
            
            self.update_q_function()

            # 更新epsilon
            if (episode + 1) % 10 == 0:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay


            if (episode + 1) % 1000== 0:
                print("=====================================测试开始=====================================")
                temp_reward = self.test_prior(history_reward, episode)  # 就10个，取平均
                if temp_reward > history_reward:
                    history_reward = temp_reward
                self.test_no_prior()
                print("=====================================测试结束=====================================")



    def test_prior(self, history_reward, epoch):
        record_reward = []
        done = False
        for episode in range(10):
            self.test_index = episode
            # self.reset()
            # selected_bands = [27, 34, 43, 47, 89]  # 默认经验假设
            selected_bands = [11,19,24,34,37]  # 默认经验假设
            state = self.sdata_test[self.test_index, :, :, selected_bands]  # 随机选择初始状态

            done = False
            final_reward = -100

            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands
            for _ in range(self.max_selected_bands // self.action_nums-1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)

                action = self.choose_action(state,band_mask,old_state_loc,history_selection)  #band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action
                selected_bands = self.update_bands(selected_bands,action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Test A Epoch!")
                    exit()

                next_state = self.sdata_test[self.test_index, :, :, action]
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask

                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    reward = self.get_reward(selected_bands, T_flag=False)
                    done = True
                    assert reward > final_reward
                    final_reward = reward
                    print("当前第{}组的选择为:".format(episode+1))
                    print(selected_bands)
            record_reward.append(0.01/final_reward)

        print("Test Mean Mse:{:.6f}  STD:{:.6f}".format(np.mean(record_reward), np.std(record_reward)))
        if 0.01/np.mean(record_reward) > history_reward:
            self.save_model(self.model_name, epoch, np.mean(record_reward))
            print(f"SAVE！ Now Mse:{np.mean(record_reward)}")
        return 0.01/np.mean(record_reward)

    def test_no_prior(self):  #写一个无先验的随即版本
        record_reward = []
        done = False
        for episode in range(10):
            self.test_index = episode

            selected_bands = random.sample(list(range(self.action_size)),self.action_nums)
            selected_bands.sort()
            print("当前无先验的第{}组，初始选择为".format(episode+1))
            print(selected_bands)
            state = self.sdata_test[self.test_index, :, :, selected_bands]  # 随机选择初始状态

            done = False
            final_reward = -100

            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            for _ in range(self.max_selected_bands // self.action_nums-1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)
                
                action = self.choose_action(state,band_mask,old_state_loc,history_selection)  #band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action
                selected_bands = self.update_bands(selected_bands,action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Test B Epoch!")
                    exit()

                next_state = self.sdata_test[self.test_index, :, :, action]
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask

                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    reward = self.get_reward(selected_bands, T_flag=False)
                    done = True
                    assert reward > final_reward
                    final_reward = reward
                    print("当前无先验第{}组的最终选择为:".format(episode+1))
                    print(selected_bands)
            record_reward.append(0.01/final_reward)

        print("NO prior Test Mean Mse:{:.6f}  STD:{:.6f}".format(np.mean(record_reward), np.std(record_reward)))


    def pred(self):
        # 加一组的那个前环境下全性能的光谱的比较。
        self.epsilon = 0
        self.state_print('预测')
        self.model = self.load_model(self.model_name)
        record_reward = []
        record_selection = np.zeros([10, self.max_selected_bands])
        done = False

        for episode in range(10):
            self.test_index = episode
            selected_bands = [11,19,24,34,37]  # 默认经验假设
            state = self.sdata_test[self.test_index, :, :, selected_bands]  # 随机选择初始状态
            done = False
            final_reward = -100
            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            for _ in range(self.max_selected_bands // self.action_nums-1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)
                action = self.choose_action(state,band_mask,old_state_loc,history_selection)  # band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action
                selected_bands = self.update_bands(selected_bands,action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Pred Epoch!")
                    exit()
                next_state = self.sdata_test[self.test_index, :, :, action]
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask
                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    
                    reward = self.get_reward(selected_bands, T_flag=False)
                    selected_bands.sort()
                    record_selection[episode] = selected_bands
                    done = True
                    final_reward = reward
                    print(selected_bands)
            record_reward.append(0.01 / final_reward)
            if type(self.test_data_mse) is not str:
                if record_reward[episode] < self.test_data_mse[episode]:
                    print("Best!!!{:.6f} < {:.6f}".format(record_reward[episode], self.test_data_mse[episode]))
                else:
                    print("Bad!!!{:.2f} times".format(record_reward[episode] / self.test_data_mse[episode]))
        np.savez('checkpoint/' + self.model_name + '/selection_bands.npz', data=record_selection)
        print("Pred Mean MSE Loss:{:.6f}  STD:{:.6f}".format(np.mean(record_reward), np.std(record_reward)))


class DQN_unmix_pred(DQNFather):
    def __init__(self,params):
        self.prior = params.prior
        self.state_size = params.state_size
        self.action_size = params.action_size

        self.device = params.device
        self.model = params.model.to(self.device)

        self.max_selected_bands = params.max_bands
        self.action_nums = params.action_nums


        self.sdata_test = params.sdata[0]
        self.gt_test = params.sdata[1]
        self.pure_spectra = params.standard
        self.test_index = 0
        self.model_name = params.name
        self._action_check()

    def choose_action(self,state,band_mask,old_loc,history_selection,episode=None):
        eps = self.epsilon
        if episode is not None:
            eps = 0
        if random.random()<eps:

            search_space = torch.randn(self.action_size)
            search_space = search_space + band_mask
            action = torch.topk(search_space,self.action_nums).indices
            action = action.tolist()

        else:
            self.model.eval()
            with torch.no_grad():
                old_loc = old_loc.to(self.device)
                history_selection = history_selection.to(self.device)
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.model(state,old_loc,history_selection)[0].detach().cpu()
                action = action + band_mask
                action = torch.topk(action,self.action_nums).indices
                action = action.tolist()

        action.sort()
        return action
    
    def get_loss(self, selected_bands):
        data = self.sdata_test[self.test_index]
        c_gt = self.gt_test[self.test_index]
        input = data.reshape(self.state_size * self.state_size, -1)
        input = input[:, selected_bands]
        temp_pure = self.pure_spectra[:, selected_bands]
        est = mcr_amend(input, temp_pure)
        c_gt = c_gt.reshape(self.state_size * self.state_size, -1)
        est[np.isnan(est)] = 0
        loss = mse(c_gt, est)
        return loss
    

    def prior_check(self):
        if type(self.prior) is list:
            prior = self.prior
            prior.sort()
            laws_list = list(range(self.action_size))
            if len(prior) == self.action_nums and len(set(print))==len(prior) and np.sum(w in laws_list for w in prior) == len(prior):
                return True, prior
            else:
                raise ValueError
        else:#None
            return False, prior

    def state_print(self, phase='训练'):
        print(datetime.datetime.now().strftime("%F:%H:%M:%S"))
        print(f"=============================={phase} INIT =====================================")
        print(f"数据大小{self.sdata_test.shape} label：{self.gt_test.shape}")
        print(f"==============================={phase}START======================================")


    def unmix_report(self,selections,rewards):
        # selections : 场景数量 * 最大波数上限
        # rewards : 场景数量 *


        l  = len(self.sdata_test)



        pass

    def pred(self):
        self.epsilon = 0
        self.state_print('指导')
        self.model = self.load_model(self.model_name)
        record_reward = []
        record_selection = np.zeros([len(self.sdata_test), self.max_selected_bands])

        flag, prior = self.prior_check(self.prior)

        for episode in range(len(self.sdata_test)):
            self.test_index = episode
            if flag:
                selected_bands = prior
            else:
                selected_bands = random.sample(list(range(self.action_size)),self.action_nums)
                selected_bands.sort()
            state = self.sdata_test[self.test_index, :, :, selected_bands]
            band_mask = self.bands_mask(selected_bands)
            loc = selected_bands
            for _ in range(self.max_selected_bands // self.action_nums-1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)
                action = self.choose_action(state,band_mask,old_state_loc,history_selection)
                loc = action
                selected_bands = self.update_bands(selected_bands,action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Pred Epoch!")
                    exit()
                next_state = self.sdata_test[self.test_index, :, :, action]
                band_mask = self.bands_mask(selected_bands)
                state = next_state

            selected_bands.sort()
            loss = self.get_loss(selected_bands)
            record_selection[episode] = selected_bands
            record_reward.append(loss)
        
        self.unmix_report(record_selection,record_reward)
        # np.savez('checkpoint/' + self.model_name + '/selection_bands.npz', data=record_selection)
        # print("Pred Mean MSE Loss:{:.6f}  STD:{:.6f}".format(np.mean(record_reward), np.std(record_reward)))
