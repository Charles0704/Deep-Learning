# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' 

os.chdir(os.path.dirname(__file__))
 
class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(latent_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        x = F.relu(self.linear(x))
 
        x = torch.sigmoid(self.out(x))
        return x
 
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size,1)
 
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = torch.sigmoid(self.out(x))
        return x
 
loss_BCE = torch.nn.BCELoss(reduction='sum')
 
# 压缩后的特征维度
latent_size = 16
 
# encoder和decoder中间层的维度
hidden_size = 128
 
# 原始图片和生成图片的维度
input_size = output_size = 28*28
 
epochs = 1
batch_size = 32
learning_rate = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
modelname = ['gan-G.pth', 'gan-D.pth']
model_g = Generator(latent_size, hidden_size, output_size).to(device)
model_d = Discriminator(input_size, hidden_size).to(device)
 
optim_g = torch.optim.Adam(model_g.parameters(), lr=learning_rate)
optim_d = torch.optim.Adam(model_d.parameters(), lr=learning_rate)
 
try:
    model_g.load_state_dict(torch.load(modelname[0]))
    model_d.load_state_dict(torch.load(modelname[1]))
    print('[INFO] Load Model complete')
except:
    pass
 
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./MNIST", train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./MNIST", train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)
 
for epoch in range(epochs):
    Gen_loss = 0
    Dis_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f'[train]epoch:{epoch}'):
        bs = imgs.shape[0]
        T_imgs = imgs.view(bs, input_size).to(device)
        T_lbl = torch.ones(bs, 1).to(device)
        F_lbl = torch.zeros(bs, 1).to(device)
 
        sample = torch.randn(bs, latent_size).to(device)
        F_imgs = model_g(sample)
        F_Dis = model_d(F_imgs)
 
        loss_g = loss_BCE(F_Dis, T_lbl)
        loss_g.backward()
        optim_g.step()
        optim_g.zero_grad()
 
        # 训练判别器, 使用判别器分别判断真实图像和伪造图像
        T_Dis = model_d(T_imgs)
        F_Dis = model_d(F_imgs.detach())
 
        loss_d_T = loss_BCE(T_Dis, T_lbl)
        loss_d_F = loss_BCE(F_Dis, F_lbl)
        loss_d = loss_d_T + loss_d_F
        loss_d.backward()
        optim_d.step()
        optim_d.zero_grad()
 
        Gen_loss += loss_g.item()
        Dis_loss += loss_d.item()
    print(f'epoch:{epoch}|Train G Loss:', Gen_loss/len(train_loader.dataset),
          ' Train D Loss:', Dis_loss/len(train_loader.dataset))
 
    model_g.eval()
    model_d.eval()
    Gen_score = 0
    Dis_score = 0
    for imgs, lbls in tqdm(test_loader, desc=f'[eval]epoch:{epoch}'):
        bs = imgs.shape[0]
        T_imgs = imgs.view(bs, input_size).to(device)
        sample = torch.randn(bs, latent_size).to(device)
 
        F_imgs = model_g(sample)
 
        F_Dis = model_d(F_imgs)
        T_Dis = model_d(T_imgs)
 
        Gen_score += int(sum(F_Dis >= 0.5))
        Dis_score += int(sum(T_Dis >= 0.5)) + int(sum(F_Dis < 0.5))
 
    print(f'epoch:{epoch}|Test G Score:', Gen_score/len(test_loader.dataset),
          ' Test D Score:', Dis_score/len(test_loader.dataset)/2)
 
    model_g.train()
    model_d.train()
 
    model_g.eval()
    noise = torch.randn(1, latent_size).to(device)
    gen_imgs = model_g(noise)
    gen_imgs = gen_imgs[0].view(28, 28)
    model_g.train()
 
    torch.save(model_g.state_dict(), modelname[0])
    torch.save(model_d.state_dict(), modelname[1])
 
sample = torch.randn(1, latent_size).to(device)
gen_imgs = model_g(sample)
gen_imgs = gen_imgs[0].view(28, 28)
plt.matshow(gen_imgs.cpu().detach().numpy())
plt.show()
 
dataset = datasets.MNIST("./MNIST", train=False, transform=transforms.ToTensor())
index = 0
raw = dataset[index][0].view(28, 28)

plt.matshow(raw.cpu().detach().numpy())
plt.show()
raw = raw.view(1, 28*28)
result = model_d(raw.to(device))
print('The probability that the graph is true is: ', result.cpu().detach().numpy())