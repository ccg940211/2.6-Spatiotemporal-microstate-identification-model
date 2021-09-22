import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from scipy.io import loadmat, savemat
import os

'''数据初始化'''
num_epochs = 30
num_classes = 4
time_num = 100
batch_size = 100
display = 10
length = [13, 11]
learning_rate = 0.001
damp = 100

model_name = 'cnn_rnn_2D.ckpt'
data_file = 'PD去噪后二维数据/'
label_file = 'PD去噪后没有0的标签/'
result_name = 'result_of_2D_5'

'''调用GPU'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''定义卷积神经网络'''


class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN, self).__init__()

        self.con1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.BatchNorm3d(8),
            nn.ReLU())
        self.con2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=[1, 3, 3], padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU())
        self.con3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=[1, 13, 11], padding=[0, 0, 0]),
            nn.BatchNorm3d(32),
            nn.ReLU())
        # self.softmax = nn.Softmax(dim=-1)
        self.GRU = nn.GRU(32, 4, 2, batch_first=True)

    def forward(self, x):
        out = self.con1(x)
        out = self.con2(out)
        out = self.con3(out)
        out = out.permute(0,2,1,3,4)
        out = out.reshape(-1, time_num, 32)
        # out = self.softmax(out)
        out, h_n = self.GRU(out)
        return out


'''定义dataset'''


class mydataset(Dataset):
    def __init__(self, data, label):
        self.labels = label
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label


'''定义读取数据的函数'''


def matload(data_name, label_name):
    data_dic = loadmat(data_name)
    label_dic = loadmat(label_name)
    data = data_dic['data']
    label = label_dic['label']
    label = label - 1
    return data, label


'''主程序，读取数据'''
data_file_list = os.listdir(data_file)
label_file_list = os.listdir(label_file)
np.random.shuffle(data_file_list)
pre_len = len(data_file_list)
train_list = []
train_label_list =[]
test_list = []
test_label_list = []
test_name = []
for i, data_filename in enumerate(data_file_list):
    num1 = ''.join([x for x in data_filename if x.isdigit()])
    num1 = int(num1)
    for label_filename in label_file_list:
        num2 = ''.join([x for x in label_filename if x.isdigit()])
        num2 = int(num2)
        if num1 == num2:
            break
    data_name = data_file + data_filename
    label_name = label_file + label_filename

    if i < 0.8 * pre_len:
        train_list.append(data_name)
        train_label_list.append(label_name)
    else:
        test_list.append(data_name)
        test_label_list.append(label_name)
        print('The test dataset is : {}'.format(data_filename))
        test_name.append(data_filename)

'''模型基本参数初始化'''
model = CNN_RNN().to(device)
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_name))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

'''开始训练'''
print('start:')
for epoch in range(num_epochs):

    if epoch+1 == damp:
        learning_rate = learning_rate * 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    state = np.random.get_state()
    np.random.shuffle(train_list)
    np.random.set_state(state)
    np.random.shuffle(train_label_list)

    for i in range(len(train_list)):
        data, label = matload(train_list[i],train_label_list[i])
        cut = len(data) // time_num
        data = data[:cut*time_num, :].reshape(-1, time_num, length[0], length[1])
        label = label[:cut*time_num].reshape(-1, time_num)

        train_dataset = mydataset(data=data, label=label)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

        for j, (in_data, in_label) in enumerate(train_loader):
            in_data = in_data.to(device)
            in_label = in_label.to(device)

            in_data = in_data.float()
            in_data = in_data.view(-1, 1, time_num, length[0], length[1])

            # 前向传播+计算loss
            outputs = model(in_data)
            # outputs = outputs.reshape(-1, num_classes)
            # label1 = label1.reshape(-1)
            label1 = in_label.long()
            outputs = outputs.reshape(-1, num_classes)
            label1 = label1.view(-1)
            loss = criterion(outputs, label1)

            # 后向传播+调整参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每100个batch打印一次数据
        if (i + 1) % display == 0:
            print('Epoch [{}/{}], step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_list), loss.item()))

    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        torch.save(model.state_dict(), model_name)
        for j in range(len(test_list)):
            data, label = matload(test_list[j], test_label_list[j])
            cut = len(data) // time_num
            data = data[:cut * time_num, :].reshape(-1, time_num, length[0], length[1])
            label = label[:cut * time_num].reshape(-1, time_num)

            data = torch.Tensor(data)
            labels = torch.Tensor(label)

            data = data.to(device)
            label1 = labels.to(device)

            data1 = data.float()
            data1 = data1.view(-1, 1, time_num, length[0], length[1])

            # 前向传播+计算loss
            outputs = model(data1)
            outputs = outputs.reshape(-1, num_classes)
            label1 = label1.reshape(-1)
            label1 = label1.long()
            _, predicted = torch.max(outputs.data, 1)
            total += label1.size(0)
            correct += (predicted == label1).sum().item()
            print('acc: {:.4f}'.format(correct / total * 100))
    model.train()

# 保存模型参数
torch.save(model.state_dict(), model_name)
model = CNN_RNN().to(device)
model.load_state_dict(torch.load(model_name))
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

with torch.no_grad():
    total = 0
    correct = 0
    for j in range(len(test_list)):
        data, label = matload(test_list[j], test_label_list[j])
        cut = len(data) // time_num
        data = data[:cut * time_num, :].reshape(-1, time_num, length[0], length[1])
        label = label[:cut * time_num].reshape(-1, time_num)

        data = torch.Tensor(data)
        labels = torch.Tensor(label)

        data = data.to(device)
        label1 = labels.to(device)

        data1 = data.float()
        data1 = data1.view(-1, 1, time_num, length[0], length[1])

        # 前向传播+计算loss
        outputs = model(data1)
        outputs = outputs.reshape(-1, num_classes)
        label1 = label1.reshape(-1)
        label1 = label1.long()
        _, predicted = torch.max(outputs.data, 1)
        total += label1.size(0)
        correct += (predicted == label1).sum().item()
        print('acc: {:.4f}'.format(correct / total * 100))
        # savemat('PD RNN标签/' + test_name[j], {'label': predicted.numpy() + 1})
model.train()