import torch
from torch import nn

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

class LightNet(nn.Module):
    def __init__(self,img_size,action_size):
        super(LightNet, self).__init__()
        in_channels=5
        self.UnionBlock1 = UnionModule(in_channels,32)
        self.UnionBlock2 = UnionModule(32,64)
        # self.UnionBlock3 = UnionModule(64,128)
        self.b1 = nn.Conv2d(32,128,1)
        self.b2 = nn.Conv2d(64,128,1)
        # self.b3 = nn.Conv2d(128,128,1)
        self.tail = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3,1,1),
        )

        self.soft = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,action_size),
            nn.Softmax(dim = 1)
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # 默认方法
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self,x):
        out1 = self.UnionBlock1(x)
        out2 = self.UnionBlock2(out1)
        # out3 = self.UnionBlock3(out2)
        out = self.b1(out1)+self.b2(out2)
        out = self.tail(out)
        out = self.global_average_pooling(out)
        out = self.soft(out)
        return out


if __name__ == '__main__':
    from torchinfo import summary
    net = LightNet(100,100)

    summary(net, (1, 1, 100, 100))
    # x = torch.randn((1,3,256,256)).cuda()
    # print(net(x).shape)