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
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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

#Resnet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  
        
        if use_1x1conv:
            self.conv3=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3=None
    
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.conv3:
            identity =self.conv3(x)
        out += identity
        
        out = self.relu(out)
        return out

# Define ResNet34
class resnet34(nn.Module):
    def __init__(self, num_classes=10):
        super(resnet34, self).__init__()
        self.in_channels = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual layer
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)

        # The average pool of dropout and the full connection layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 应用 Dropout
        x = self.dropout(x)

        x = self.fc(x)
        return x
