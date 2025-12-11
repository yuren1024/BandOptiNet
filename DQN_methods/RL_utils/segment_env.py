import torch
import torch.optim as optim
import random
import numpy as np
from RL_utils import DQNFather,Transition,ReplayMemory,get_pearsonr
import torch.nn.functional as F

class DQN_segment(DQNFather):
    def __init__(self,params):
        """
        state_size=200, action_size=51, learning_rate=0.1, discount_factor=0.6, epsilon=0.5,
                 epsilon_decay=0.99, epsilon_min=0.05, memory_size=10000,batch_size = 12,action_nums = 5,
                 max_selected_bands=40, device = 'cpu',sdata=None, test_size = 20, dqn_model=None, standard=None,
                 model_name='0622'
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.Explr = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params.milestones, gamma=params.gamma, last_epoch=-1)
        self.batch_size = params.batch_size
        self.max_selected_bands = params.max_bands
        self.action_nums = params.action_nums

        self.test_size = params.test_size
        self.data_n = len(params.sdata[0])

        self.sdata_train = params.sdata[0][:self.data_n-self.test_size]  # 116 200*200*81
        self.gt_train = params.sdata[1][:self.data_n-self.test_size]  # 116 200*200
        self.sdata_test = params.sdata[0][self.data_n-self.test_size:]
        self.gt_test = params.data[1][self.data_n-self.test_size:]
        self.pure_spectra = params.standard
        self.train_index = 0  # 0-80  data_indexa ->train_index
        self.test_index = 0  # 0-4   data_indexc ->test_index
        self.model_name = params.name
        self._action_check()

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = self.memory.sample(self.batch_size)
        minibatch = Transition(*zip(*minibatch))

        batch_state = torch.cat(minibatch.state).float().to(self.device)
        batch_action = torch.cat(minibatch.action).to(self.device)  # 当前选择的动作，也是下一步的位置
        batch_reward = torch.cat(minibatch.reward).float().to(self.device)

        batch_band_mask = torch.cat(minibatch.band_mask).to(self.device)  # 是一个布尔值掩码  用于赋值输出
        batch_history_selection = torch.cat(minibatch.history_selection).to(self.device)  # 当前选择历史
        batch_old_state_loc = torch.cat(minibatch.old_state_loc).to(self.device)  # 当前位置
        batch_next_state_history = torch.cat(minibatch.next_state_history).to(self.device)  # 下一步的选择历史

        non_final_next_state_batch = torch.cat(
            [torch.from_numpy(s).unsqueeze(0) for s in minibatch.next_state if s is not None]).float().to(self.device)
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, minibatch.next_state))).bool()
        non_final_band_mask = batch_band_mask[non_final_mask]
        non_final_next_state_history = batch_next_state_history[non_final_mask]
        non_final_new_state_loc = batch_action[non_final_mask]

        next_state_values = torch.zeros([self.batch_size, self.action_nums]).to(self.device)
        self.model.eval()
        action_temp = self.model(non_final_next_state_batch, non_final_new_state_loc,
                       non_final_next_state_history)  # shape [batch-n,100]
        action_temp = action_temp + non_final_band_mask
        values_temp = torch.topk(action_temp,self.action_nums).values
        next_state_values[non_final_mask]  = values_temp
        expected_state_action_values = batch_reward + self.discount_factor * next_state_values
        state_action_values = self.model(batch_state, batch_old_state_loc, batch_history_selection).gather(dim=1, index=batch_action)
        
        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.optimizer.param_groups[0]['lr'] >= 1e-3:
            # print(f"lr:{self.optimizer.param_groups[0]['lr']}")
            self.Explr.step()



    def choose_action(self, state, band_mask, old_loc, history_selection, episode=None):
        eps = self.epsilon
        if episode is not None:
            eps = 0
        if random.random() < eps:

            search_space = torch.randn(self.action_size)
            search_space = search_space + band_mask
            action = torch.topk(search_space, self.action_nums).indices
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
        self.train_index = np.random.randint(0, self.data_n-self.test_size)

    def get_reward(self, selected_bands, T_flag=True):
        A_flag = False  # 用于识别是否要全属性精度
        if T_flag:
            data = self.sdata_train[self.train_index]  # 200*200*51
            c_gt = self.gt_train[self.train_index]  # 200*200
        else:
            data = self.sdata_test[self.test_index]
            c_gt = self.gt_test[self.test_index]

            if len(selected_bands) == self.max_selected_bands:
                A_flag = True #如果是最后一个，自动设置为true，并且在测试集中才启用，因为在其它条件下是不需要kappa的，我需要简化一下代码。

        selected_bands.sort()
        input = data.reshape(self.state_size * self.state_size, self.action_size)

        input = input[:, selected_bands]

        c_gt = c_gt.reshape(self.state_size * self.state_size)
        c_gt = c_gt.astype(np.int32)

        loss, OA, AA, kappa = get_pearsonr(input, c_gt, selected_bands, self.pure_spectra, self.state_size, A_flag)
        # 我不知道为什么，把get_pearsonr写到类里面，速度非常慢，似乎是跑到gpu上面了，但我不知道如何避免

        assert loss != 0
        assert loss > 0
        reward = (loss ** 2) * 100
        if A_flag:
            return reward, OA, AA, kappa
        return reward


    def train(self, episodes=10000):
        self.state_print()
        history_reward = -1
        for episode in range(episodes):
            self.reset()

            selected_bands = random.sample(list(range(self.action_size)), self.action_nums)  ##随机先验
            selected_bands.sort()
            state = self.sdata_train[self.train_index, :, :, selected_bands]  # 随机选择初始状态

            done = False
            final_reward = -100
            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            for _ in range(self.max_selected_bands // self.action_nums - 1):

                history_selection = self.historical_encoding(selected_bands)  # 旧的位置历史
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)  # 旧的位置
                action = self.choose_action(state, band_mask, old_state_loc, history_selection)  # band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action  # 新的位置

                selected_bands = self.update_bands(selected_bands, action)
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
            if (episode + 1) % 8 == 0:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

            if (episode + 1) % 500 == 0:
                print("=====================================测试开始=====================================")
                temp_reward = self.test_prior(history_reward, episode)
                if temp_reward > history_reward:
                    history_reward = temp_reward
                self.test_no_prior()  # 无先验计划
                print("=====================================测试结束=====================================")

    def test_prior(self, history_reward, epoch):
        record_reward = []
        record_acc = []
        done = False
        # 我们必须要统计AA OA 和kappa的变化
        OA = []
        AA = []
        KAPPA = []

        selected_bands_record = np.zeros([self.test_size,self.max_selected_bands])
        for episode in range(self.test_size):
            self.test_index = episode
            # self.reset()
            selected_bands = [15,19,22,24,36] # 默认经验假设
            state = self.sdata_test[self.test_index, :, :, selected_bands]  # 随机选择初始状态

            done = False
            final_reward = -100

            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands
            for _ in range(self.max_selected_bands // self.action_nums - 1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)

                action = self.choose_action(state, band_mask, old_state_loc, history_selection)  # band mask是用于动作选择掩码的
                loc = action

                selected_bands = self.update_bands(selected_bands, action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Test A Epoch!")
                    exit()

                next_state = self.sdata_test[self.test_index, :, :, action]
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask

                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    reward, oa, aa, kappa = self.get_reward(selected_bands, T_flag=False)
                    OA.append(oa)
                    AA.append(aa)
                    KAPPA.append(kappa)
                    done = True
                    assert reward > final_reward
                    final_reward = reward
                    # print("当前第{}组的选择为:".format(episode + 1))
                    # print(selected_bands)
                    selected_bands_record[episode]=selected_bands
            record_reward.append(final_reward)
            now_acc = np.sqrt(final_reward / 100)
            record_acc.append(now_acc)
        print("=====================================================================")
        for ind,record in enumerate(selected_bands_record):
            print(f"{ind} :Prior selected bands: {record}")
        print("Test Acc:{:.6f}  STD:{:.6f}".format(np.mean(record_acc), np.std(record_acc)))
        print("=====================================================================")
        print("Test OA:{:.6f}  STD:{:.6f}".format(np.mean(OA), np.std(OA)))
        print("Test AA:{:.6f}  STD:{:.6f}".format(np.mean(AA), np.std(AA)))
        print("Test Kappa:{:.6f}  STD:{:.6f}".format(np.mean(KAPPA), np.std(KAPPA)))
        if np.mean(record_reward) > history_reward:
            self.save_model(self.model_name, epoch, np.mean(record_reward))
            print(f"SAVE！ Now Acc:{np.mean(record_acc)}")

        return np.mean(record_reward)

    def test_no_prior(self):  # 写一个无先验的随即版本
        record_acc = []
        done = False

        OA = []
        AA = []
        KAPPA = []

        selected_bands_record = np.zeros([self.test_size,self.max_selected_bands])
        for episode in range(self.test_size):
            self.test_index = episode

            selected_bands = random.sample(list(range(self.action_size)), self.action_nums)
            selected_bands.sort()
            print("当前无先验的第{}组，初始选择为".format(episode + 1))
            print(selected_bands)
            state = self.sdata_test[self.test_index, :, :, selected_bands]  # 随机选择初始状态

            done = False
            final_reward = -100

            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            for _ in range(self.max_selected_bands // self.action_nums - 1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)

                action = self.choose_action(state, band_mask, old_state_loc, history_selection)  # band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action
                selected_bands = self.update_bands(selected_bands, action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Test B Epoch!")
                    exit()

                next_state = self.sdata_test[self.test_index, :, :, action]

                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask

                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    reward, oa, aa, kappa = self.get_reward(selected_bands, T_flag=False)
                    OA.append(oa)
                    AA.append(aa)
                    KAPPA.append(kappa)
                    done = True
                    assert reward > final_reward
                    final_reward = reward
                    # print("当前无先验第{}组的最终选择为:".format(episode + 1))
                    # print(selected_bands)
                    selected_bands_record[episode]=selected_bands
            record_acc.append(np.sqrt(final_reward / 100))
        print("=====================================================================")
        for ind,record in enumerate(selected_bands_record):
            print(f"{ind} :No prior selected bands: {record}")
        print("NO prior Test Mean Loss:{:.6f}  STD:{:.6f}".format(np.mean(record_acc), np.std(record_acc)))
        print("=====================================================================")
        print("Test OA:{:.6f}  STD:{:.6f}".format(np.mean(OA), np.std(OA)))
        print("Test AA:{:.6f}  STD:{:.6f}".format(np.mean(AA), np.std(AA)))
        print("Test Kappa:{:.6f}  STD:{:.6f}".format(np.mean(KAPPA), np.std(KAPPA)))

    def pred(self):
        # 加一组的那个前环境下全性能的光谱的比较。
        self.epsilon = 0
        self.state_print('预测')
        self.model = self.load_model(self.model_name)
        record_acc = []
        record_selection = np.zeros([20, self.max_selected_bands])
        done = False
        OA = []
        AA = []
        KAPPA = []
        for episode in range(self.test_size):
            self.test_index = episode
            selected_bands = [15,19,22,24,36]  # 默认经验假设
            state = self.sdata_test[self.test_index, :, :, selected_bands]  # 随机选择初始状态
            done = False
            final_reward = -100
            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            for _ in range(self.max_selected_bands // self.action_nums - 1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)
                action = self.choose_action(state, band_mask, old_state_loc, history_selection)  # band mask是用于动作选择掩码的
                # action = self.top5_action(all_action,selected_bands)  #action是一个长度为5的列表
                loc = action

                selected_bands = self.update_bands(selected_bands, action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Pred Epoch!")
                    exit()
                next_state = self.sdata_test[self.test_index, :, :, action]
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask
                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    reward, oa, aa, kappa = self.get_reward(selected_bands, T_flag=False)
                    OA.append(oa)
                    AA.append(aa)
                    KAPPA.append(kappa)
                    selected_bands.sort()
                    record_selection[episode] = selected_bands
                    done = True
                    final_reward = reward
                    print(selected_bands)
            record_acc.append(np.sqrt(final_reward / 100))
            print("===================================================================")
        np.savez('checkpoint/' + self.model_name + '/selection_bands.npz', data=record_selection)
        print("Pred Mean Acc:{:.6f}  STD:{:.6f}".format(np.mean(record_acc), np.std(record_acc)))
        print("=====================================================================")
        print("Test OA:{:.6f}  STD:{:.6f}".format(np.mean(OA), np.std(OA)))
        print("Test AA:{:.6f}  STD:{:.6f}".format(np.mean(AA), np.std(AA)))
        print("Test Kappa:{:.6f}  STD:{:.6f}".format(np.mean(KAPPA), np.std(KAPPA)))
        print("OVER!")
