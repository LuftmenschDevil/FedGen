import torch,os
import numpy as np
from itertools import chain
from torch import nn

class Classifier2(nn.Module):

    def __init__(self, input_nc,img_size=32, num_classes=10):
        super().__init__()
        self.image_size = img_size # 28
        # 特征提取器
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(input_nc, 32, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),


            # conv2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )
        # 分类器
        self.classifier = nn.Sequential(
            # fc3
            nn.Linear(64 * 8 * 8, 512),
            # nn.BatchNorm1d(512, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            # softmax
            nn.Linear(512, num_classes),
        )
        # 初始化
        for layer in chain(self.features, self.classifier):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        # x = x.view((x.size(0), 1, self.image_size, self.image_size)) # Input: (?,-1,32,32)
        # 普通的分类器
        x = self.features(x)
        # 接入分类器
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

net = Classifier2(1,32,10)

params = sum(p.numel() for p in list(net.parameters()))

print('#Params: %.1fM' % (params))