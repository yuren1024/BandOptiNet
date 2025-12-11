import torch
from torch import nn
# 实际上位置编码的引入是很重要的
class UnionModule(nn.Module):
    def __init__(self,in_ch,dim):
        super(UnionModule, self).__init__()
        self.block2  = nn.Sequential(
            nn.Conv2d(in_ch,dim,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim,dim,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.MaxPool2d(3,1,1)
        )
        self.block3  = nn.Sequential(
            nn.Conv2d(in_ch,dim,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim,dim,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.MaxPool2d(3,1,1)
        )
        self.block4  = nn.Sequential(
            nn.Conv2d(in_ch,dim,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.MaxPool2d(3,1,1)
        )
    def forward(self,x):
        # out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out = out2+out3+out4
        return out


class position_encode(nn.Module):
    def __init__(self,action_size):
        super(position_encode, self).__init__()
        self.action_size = action_size
        self.block = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(250),
            nn.Linear(5*self.action_size,16),
            nn.ReLU()
        )

    def forward(self,loc):
        loc = loc.type(torch.int64)
        b = len(loc)
        one_hot = torch.zeros([b,loc.shape[1],self.action_size]).to(loc.device)

        for ind in range(b):
            one_hot[ind].scatter_(1,loc[ind].unsqueeze(1), 1)
        out = self.block(one_hot)
        return out
        
class historical_encode(nn.Module):
    def __init__(self,action_size) -> None:
        super().__init__()
        self.action_size = action_size
        self.block = nn.Sequential(
            nn.Flatten(),
            # nn.BatchNorm1d(50),
            nn.Linear(self.action_size,4),
            nn.ReLU()
        )

    def forward(self,hist):
        # b = len(hist)
        # one_hot = torch.zeros([b, 1, self.action_size]).to(hist.device)
        # for ind in range(b):
        #     one_hot[ind].scatter_(1, hist[ind].unsqueeze(0), 1)
        # out = self.block(one_hot)
        out = self.block(hist)

        return out

# batch * 5 *50 + batch *1 *50    = batch *6 *50 = batch *300
def position_encoding(indices, depth):
    b = len(indices)
    one_hot = torch.zeros([b,indices.shape[1],depth])
    for ind in range(b):
        one_hot[ind].scatter_(1, indices[ind].unsqueeze(1), 1)
    return one_hot

# def historical_encoding(selected_positions, depth):
#     b = len(selected_positions)
#     one_hot = torch.zeros([b, 1, depth])
#     for ind in range(b):
#         one_hot[ind].scatter_(1, selected_positions[ind].unsqueeze(0), 1)
#     return one_hot

def historical_encoding(selected_positions,action_size): #针对没有batch的
    selected_positions = selected_positions.type(torch.int64)
    one_hot = torch.zeros([1, action_size])
    one_hot.scatter_(1, selected_positions.unsqueeze(0), 1)
    return one_hot

def one_hot_encoding(indices, depth):
    one_hot = torch.zeros((len(indices), depth))
    one_hot.scatter_(1, torch.tensor(indices).unsqueeze(1), 1)
    return one_hot

class LightNet_loc(nn.Module):
    def __init__(self,in_channel,action_size):
        super(LightNet_loc, self).__init__()
        self.in_channels=in_channel
        self.action_size = action_size
        self.UnionBlock1 = UnionModule(self.in_channels,32)
        self.UnionBlock2 = UnionModule(32,64)

        self.b1 = nn.Conv2d(32,128,1)
        self.b2 = nn.Conv2d(64,128,1)

        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3,1,1),
        )
        self.global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.soft = nn.Sequential(
            nn.Linear(256+20,self.action_size),
            nn.Softmax(dim = 1)
        )

        self.position_encode = position_encode(self.action_size)
        self.hist_encode = historical_encode(self.action_size)
        # 默认方法
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self,x,loc,hist):
        """
        :param x:   b*5*200*200
        :param loc: b*50    [0,0,0,1……,1,0,1,0,1]
        :return:    b*50    [0.9,0.1,0.2,1……,1.3,0.5,1.6,0.1,1.4]
        """
        loc = self.position_encode(loc)  #batch * 4
        hist = self.hist_encode(hist)   #batch * 4
 
        out1 = self.UnionBlock1(x)
        out2 = self.UnionBlock2(out1)
        out = self.b1(out1)+self.b2(out2)
        out = self.tail(out)
        out = self.global_average_pooling(out)
        out = torch.cat((out,loc,hist),dim=1)

        out = self.soft(out)
        return out
def bands_mask(selected_bands,action_size=100):
    mask = torch.zeros(action_size)
    for band in selected_bands:
        mask[band] = float('-inf')
    # return mask.clone().detach().unsqueeze(0)
    return mask.clone().detach()

if __name__ == '__main__':
    in_channel = 5
    action_size = 100
    net = LightNet_loc(in_channel,action_size)

    x = torch.randn([1,5,100,100])
    # loc = torch.tensor([[0, 10, 20, 30, 40],
    #                      [0, 10, 20, 30, 40],
    #                      [0, 10, 20, 30, 40],
    #                      [0, 10, 20, 30, 40]])
    # hist = torch.tensor([[0, 5, 11, 15, 20, 25, 30, 35, 40, 45],
    #                         [2, 6, 12, 15, 20, 25, 30, 35, 40, 45],
    #                         [3, 5, 15, 16, 20, 25, 30, 35, 40, 45],
    #                         [4, 7, 14, 17, 22, 25, 30, 35, 40, 45]])
    # loc:  当前位置选择的五个波数
    # hist：历史位置选择的五个波数 
    loc = torch.tensor([[0, 10, 20, 31, 41]])
    hist = torch.tensor([4, 7, 14, 17, 22, 25, 30, 35, 40, 99])
    hist = historical_encoding(hist,action_size)

    print(hist.shape)
    exit()
    #print(net(x,loc,hist).shape)
    action = net(x,loc,hist)[0].detach()
    print(action)
    mask = [0,4,7,10,14,17,20,22,25,30,31,35,40,41,45]
    mask = bands_mask(mask,action_size)
    print(mask)
    print(action+mask)
    action = action+mask
    action = torch.topk(action,5).indices
    print(action)
    action = action.tolist()
    print(action)
    action.sort()
    print(action)