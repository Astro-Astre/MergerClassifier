# -*- coding: utf-8-*-
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch import nn
from torchvision import transforms

transfer = transforms.Compose([
    transforms.ToTensor(),
])


def data_prepare(file, transform):
    img = pickle.load(file)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = transform(img)
    return img


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride=1, padding=3, bias=0.1),
            nn.MaxPool2d(20, stride=2, padding=9),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2, bias=0.1),
            nn.MaxPool2d(8, stride=2, padding=4),
            nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, bias=0.1),
            nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=0.1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(131072, 128, bias=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 128, bias=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 2, bias=0.1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        # print('第四层：',out.shape)
        out = out_conv4.view(batch_size, -1)
        out = self.fc(out)
        out_conv1 = out_conv1[0, 31, :, :]
        out_conv2 = out_conv2[0, 31, :, :]
        out_conv3 = out_conv3[0, 31, :, :]
        out_conv4 = out_conv4[0, 31, :, :]
        return out


def predict(path, model_name):
    net = torch.load(model_name)
    device = torch.device("cuda:0")
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        with open(path, 'rb') as f:
            img = data_prepare(f, transform=transfer)
            img = img.unsqueeze(0)
            img = img.to(device)
            outputs = net(img)
            _, predicted = torch.max(outputs, 1)
            return predicted[0]


# def verifyModel(data_path,txt_path,model_name):
#     pre = os.listdir(data_path)
#     f = open(txt_path, 'w')
#     galaxy = 0
#     merger = 0
#     for i in range(3):
#         percent = 1.0 * i / len(pre)  #用于显示进度
#         pred = predict(data_path+'\\'+pre[i],model_name)
#         if pred.item()==1:
#             line = pre[i] + ' 类别:Merger' + '\n'
#             merger += 1
#         if pred.item()==0:
#             line = pre[i] + ' 类别:Galaxy' + '\n'
#             galaxy += 1
#         f.write(line)
#         # 用于显示进度
#         sys.stdout.write("进度：%.4f"%(percent*100))
#         sys.stdout.write("%\r")
#         sys.stdout.flush()

#     print('merger_num:',merger)
#     print('galaxy_num:',galaxy)
#     f.close()


def verify_model(data_path, txt_path, model_name):
    pre = os.listdir(data_path)
    f = open(txt_path, 'w')
    galaxy = 0
    merger = 0
    for i in range(len(pre)):
        percent: float = 1.0 * i / len(pre)  # 用于显示进度
        pred = predict(data_path + '\\' + pre[i], model_name)
        if pred.item() == 1:
            line = pre[i].split('.')[0] + r'.jpg 1' + '\n'
            merger += 1
        if pred.item() == 0:
            line = pre[i].split('.')[0] + r'.jpg 0' + '\n'
            galaxy += 1
        f.write(line)
        # 用于显示进度
        sys.stdout.write("进度：%.4f" % (percent * 100))
        sys.stdout.write("%\r")
        sys.stdout.flush()

    print('merger_num:', merger)
    print('galaxy_num:', galaxy)
    f.close()


if __name__ == "__main__":
    start = time.time()
    # verifyModel(r'alfalfa_dat_normal',
    #     r'2020-01-04-alfalfa_model23.txt',
    #     r'model\model_normal-2021-01-03-085331\model_23.model')
    # verifyModel(r'alfalfa_dat_normal',
    #     r'2020-01-02-alfalfa_model48.txt',
    #     r'model\model_normal-2021-01-01-014012\model_48.model')
    # verifyModel(r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\verify_data\0',
    #     r'galaxy.txt',
    #     # r'model\model_normal-2021-01-01-014012\model_48.model'),
    #     r'D:\Code\MachineLearning\Model\2020.12.15_MergerClassifier\model_normal-2021-01-03-085331\model_23.model'),
    verify_model(r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\verify_data\1',
                 r'merger.txt',
                 # r'model\model_normal-2021-01-01-014012\model_48.model'),
                 r'D:\Code\MachineLearning\Model\2020.12.15_MergerClassifier\model_normal-2021-01-03-085331\model_23.model'),

    end_time = time.time()
    print('time cost:', end_time - start)
