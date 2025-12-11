import torch
import torch.optim as optim
import random
import numpy as np
from model_base.LightNet_location import LightNet_loc

from DQN_unmix.RL_utils.base_env import DQNFather, Transition, ReplayMemory
import torch.nn.functional as F


class DQNAgent(DQNFather):
    def __init__(self, state_size=200, action_size=51, epsilon=0.05
                 , max_selected_bands=40, device='cpu', sdata=None,dqn_model=None
                 , model_name='0622'):

        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = epsilon
        self.device = device
        self.model = dqn_model.to(device)

        self.max_selected_bands = max_selected_bands

        self.data_n = len(sdata[0])
        self.test_size = self.data_n

        self.sdata_train = sdata[0]
        self.gt_train = sdata[1]
        self.sdata_test = sdata[0]
        self.gt_test = sdata[1]
        self.data_indexc = 0  # 88-116  28个
        self.model_name = model_name

    def choose_action(self, state, band_mask, old_loc, history_selection, episode=None):
        """
        old_loc:torch 当前位置
        history_selection  1*100
        """
        eps = self.epsilon
        if episode is not None:
            eps = 0
        if random.random() < eps:

            search_space = torch.randn(self.action_size)
            search_space = search_space + band_mask
            action = torch.topk(search_space, 5).indices
            action = action.tolist()

        else:
            self.model.eval()
            with torch.no_grad():
                old_loc = old_loc.to(self.device)
                history_selection = history_selection.to(self.device)
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

                action = self.model(state, old_loc, history_selection)[0].detach().cpu()
                action = action + band_mask
                action = torch.topk(action, 5).indices
                action = action.tolist()
        action.sort()

        return action


    def pred(self):
        self.epsilon = 0
        self.state_print('预测')
        self.model = self.load_model(self.model_name)
        for episode in range(self.test_size):
            self.data_indexc = episode
            # selected_bands = [15, 19, 22, 24, 36]  # 默认经验假设
            # selected_bands = [20, 24, 25, 26, 34]  # PLA PMMA
            selected_bands = [18, 22, 23, 26, 39]  # PVC PBAT
            state = self.sdata_test[self.data_indexc, :, :, selected_bands]  # 随机选择初始状态
            band_mask = self.bands_mask(selected_bands)  # 第一个bandmask生成
            loc = selected_bands  # 第一次动作
            for _ in range(self.max_selected_bands // 5 - 1):
                history_selection = self.historical_encoding(selected_bands)
                old_state_loc = torch.from_numpy(np.array(loc)).unsqueeze(0)
                action = self.choose_action(state, band_mask, old_state_loc, history_selection)  # band mask是用于动作选择掩码的
                loc = action

                selected_bands = self.update_bands(selected_bands, action)
                if len(selected_bands) != len(set(selected_bands)):
                    print("Bands Repeat In Pred Epoch!")
                    exit()
                next_state = self.sdata_test[self.data_indexc, :, :, action]
                band_mask = self.bands_mask(selected_bands)  # 根据已选的内容来生成之后的bandmask
                state = next_state

                if len(selected_bands) == self.max_selected_bands:
                    print(selected_bands)
            print("===================================================================")
        print("=====================================================================")
        print("OVER!")


# 加载数据的函数
def load_data(file_path):
    data = np.load(file_path)['data_nm']

    # n x width x height x channels  84 200 200 51
    label = np.load(file_path)['label']
    # n x width x height  84 200 200
    return [data, label]


if __name__ == '__main__':
    data_name = '0620'
    action_size = 51
    state_size = 400
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # data_path = '../data/micro_data_0620/PLA_PMMA.npz'
    data_path = '../data/micro_data_0620/PVC_PBAT.npz'
    print(data_path)
    data = load_data(data_path)



    dqn_model = LightNet_loc(5, action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, max_selected_bands=25,
                     sdata=data, device=device, dqn_model=dqn_model,
                     model_name='0626_datamicro0620_25')

    agent.pred()

