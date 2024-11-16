import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

### Lenet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(16, 32, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    

# CNN
class CNN28(nn.Module):
    def __init__(self):
        super(CNN28, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1,),   #(1，28，28）---> (16，28，28）
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2,),)              #（16,14,14）
        
        # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1

        # 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32 ,kernel_size=3,stride=1,padding=0,), # （16,14,14）---> (32,12,12）
            nn.BatchNorm2d(32),
            nn.ReLU(),  
            nn.AvgPool2d(kernel_size=2,stride=2,),               # （32,6,6）
        )

        # 定义第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=0,),  # （32,6 ,6 ）---> (32,4 ,4 ）
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2 ,stride=1,),              # （64,3,3）         
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    #  定义网络向前传播路径
    def forward(self, x):
        x = self.conv1(x)            # （batch_size,16,14,14）
        x = self.conv2(x)            # （batch_size,32,6,6）
        #x = F.dropout(x, p=0.2, training=self.training)  Drop out 防止过拟合
        x = self.conv3(x)            # （batch_size,64,6,6）
        x = x.view(x.size(0), -1)    #  (batch_size,64*6*6)
        output = self.classifier(x)  #  (batch_size,10)
        return output

