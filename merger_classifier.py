# -*- coding: utf-8-*-
import datetime
import os
import pickle
import time
from multiprocessing import Pool, pool

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import fscore

# 配置参数
torch.manual_seed(1)    # 设置随机数种子，确保结果可重复
BATCH_SIZE = 64 # 批处理大小
LEARNING_RATE = 1e-3    # 学习率
EPOCHES = 50    # 训练次数
DATA_SOURCE_DIR = r"D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier"   # 总数据存放文件夹
MODEL_SOURCE_DIR = r"D:\Code\MachineLearning\Model\2020.12.15_MergerClassifier" # 总模型存放文件夹
TRAIN_DATA_TXT_PATH = r'train_data.txt'
TEST_DATA_TXT_PATH = r'test_data.txt'

def changeAxis(data):
    '''
    输入即将转换为张量之前的ndarray,要求为C×H×W
    由于ToTensor会将nadarrayHWC转为CHW
    要先把ndarray的CHW转为ndarray的HWC.
    '''
    img = np.swapaxes(data, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img

def dataPrepare(file, transform):
    img = pickle.load(file)
    img = changeAxis(img)
    img = transform(img)    # 数据标签转换为Tensor
    return img

def createPackage(model_package):
    if not os.path.isdir(model_package):
            os.mkdir(model_package)

def modelPackageWrite(info):
    '''
    返回文件夹名：model_info_datetime
    '''
    model_package = 'model_%s'%(info)
    if not model_package[-1] =='_':
        model_package+='_'
    model_package += str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    return model_package

class MerGalDataset(Dataset):
    '''
    重写Dataset以读取dat数据进行训练
    分别输入dat路径和label的txt文件及定义好的transform
    '''
    def __init__(self, txt_path, transform):
        super(MerGalDataset, self).__init__()
        with open(txt_path, 'r') as f:
            imgs = []
            for line in f:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = line.split()
                imgs.append((words[0], str(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, l = self.imgs[index]
        with open(fn, 'rb') as f:
            img = dataPrepare(f, transform=transfer)
            if l == '1': # merger
                label = np.array(1)
            else:
                label = np.array(0)
            return img, label
 
    def __len__(self):
        return len(self.imgs)

transfer = transforms.Compose([
            transforms.ToTensor(),
            ])
train_data = MerGalDataset(txt_path=TRAIN_DATA_TXT_PATH, transform=transfer)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

test_data = MerGalDataset(txt_path=TEST_DATA_TXT_PATH, transform=transfer)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride= 1, padding= 3, bias= 0.1),
            nn.MaxPool2d(20, stride=2, padding=9),
            nn.ReLU(True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride= 1, padding= 2, bias= 0.1),
            nn.MaxPool2d(8, stride=2, padding=4),
            nn.ReLU(True),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1,bias=0.1),
            nn.ReLU(True),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride= 1, padding= 1,bias= 0.1),
            nn.MaxPool2d(2, stride= 2, padding= 1),
            nn.ReLU(True),
            )
        self.fc = nn.Sequential(
            nn.Linear(131072, 128,bias= 0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 128,bias= 0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 2,bias= 0.1),
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
        return out, out_conv1, out_conv2, out_conv3, out_conv4


def predict(path,model_name):
    model = torch.load(model_name)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        with open(path, 'rb') as f:
            img = dataPrepare(f, transform=transfer)
            img = img.unsqueeze(0)
            img = img.to(device)
            outputs, conv1, conv2, conv3, conv4 = model(img)
            _, predicted = torch.max(outputs, 1)
            return predicted[0]

def trainModel(model_package,flag,last_epoch):
    '''
    训练
    '''
    createPackage('%s' %(model_package))

    writer = SummaryWriter(model_package)

    loss_func = nn.CrossEntropyLoss() #损失函数：交叉熵
    optimizer = optim.SGD(model.parameters(), lr= LEARNING_RATE, momentum= 0.9) # 优化器
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    start = -1
    
    # flag = True时进行断点续训
    if flag:
        path_checkpoint = '%s\checkpoint\ckpt_best_%d.pth' %(model_package, last_epoch)  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start = checkpoint['epoch']  # 读取上次的epoch
        print('start epoch: ', last_epoch + 1)

    # 保存训练记录
    train_log = open(model_package + '//train_log.txt','a')
    train_log.write(str(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
    train_log.write('\nBATCH_SIZE = %d\nLEARNING_RATE = %f\n\n' %(BATCH_SIZE, LEARNING_RATE))
    train_log.close()

    # 开始训练，带writer的都是tensorboard可视化的代码
    for echo in range(start + 1, EPOCHES):
        train_log = open(model_package + '//train_log.txt','a')
        train_loss = 0  # 训练损失
        train_acc = 0   # 训练准确度
        model.train()
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for i, (X, label) in enumerate(train_loader):   # 遍历train_loader
            label = torch.as_tensor(label, dtype = torch.long)
            X, label = X.to(device), label.to(device)
            out, conv1, conv2, conv3, conv4 = model(X)  # 正向传播
            lossvalue = loss_func(out, label)    # 求损失值
            optimizer.zero_grad()   # 优化器梯度归零
            lossvalue.backward()    # 反向转播，刷新梯度值
            optimizer.step()    # 优化器运行一步，注意optimizer搜集的是model的参数
            train_loss += float(lossvalue)  # 计算损失
            _, pred = out.max(1)
            num_correct = (pred == label).sum()
            for i in range(len(label)):
                if(label[i]==1 and pred[i]==1):
                    tp += 1
                if(label[i]==1 and pred[i]==0):
                    fn += 1
                if(label[i]==0 and pred[i]==0):
                    tn += 1
                if(label[i]==0 and pred[i]==1):
                    fp += 1
            acc = int(num_correct) / X.shape[0] # 计算精确度
            train_acc += acc
            writer.add_image('conv1_image', conv1, i, dataformats= 'HW')
            writer.add_image('conv2_image', conv2, i, dataformats= 'HW')
            writer.add_image('conv3_image', conv3, i, dataformats= 'HW')
            writer.add_image('conv4_image', conv4, i, dataformats= 'HW')

        fscore.caculate(tp,tn,fp,fn)
        writer.add_scalar('loss', train_loss, echo)
        losses.append(train_loss / len(train_loader))
        acces.append(100*train_acc / len(train_loader))
        print("echo:" + ' ' + str(echo))
        print("lose:" + ' ' + str(train_loss / len(train_loader)))
        print("accuracy:" + ' '+ str(train_acc / len(train_loader)))
        train_log.write("---------------------------echo---------------------------:" + ' ' + str(echo) + '\n')
        train_log.write("lose:" + ' ' + str(train_loss / len(train_loader)) + '\n')
        train_log.write("accuracy:" + ' '+ str(train_acc / len(train_loader)) + '\n')
        writer.add_scalar('train_loss',train_loss / len(train_loader),echo)
        writer.add_scalar('train_acc',train_acc / len(train_loader),echo)

        eval_loss = 0
        eval_acc = 0
        model.eval()    # 模型转化为评估模式
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for X, label in test_loader:
            label = torch.as_tensor(label, dtype = torch.long)
            X, label = X.to(device), label.to(device)
            testout, conv1, conv2, conv3, conv4 = model(X)
            testloss = loss_func(testout, label)
            eval_loss += float(testloss)
            _, pred = testout.max(1)
            for i in range(len(label)):
                if(label[i]==1 and pred[i]==1):
                    tp += 1
                if(label[i]==1 and pred[i]==0):
                    fn += 1
                if(label[i]==0 and pred[i]==0):
                    tn += 1
                if(label[i]==0 and pred[i]==1):
                    fp += 1
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            eval_acc += acc
 
        fscore.caculate(tp,tn,fp,fn)
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        print("testloss: " + str(eval_loss/len(test_loader)))
        print("testaccuracy:" + str(eval_acc/len(test_loader)) + '\n')
        train_log.write("testloss: " + str(eval_loss/len(test_loader)) + '\n')
        train_log.write("testaccuracy:" + str(eval_acc/len(test_loader)) + '\n' + '\n')
        writer.add_scalar('test_loss', eval_loss/len(test_loader),echo)
        writer.add_scalar('test_acc', eval_acc/len(test_loader),echo)
        checkpoint = {
            "net": model.state_dict(),
            'optimizer':optimizer.state_dict(),
            "epoch": echo
        }
        createPackage('%s/checkpoint' %(model_package))
        torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' %(model_package,str(echo)))
        torch.save(model, '%s/model_%d.model' %(model_package,echo))
        writer.close()

def verifyModel(path, label, model_name):
    pre = os.listdir(path)
    acc_num = 0
    for i in range(len(pre)):
        pred = predict(path + '\\' + pre[i], model_name)
        if(pred.item() == label):
            acc_num += 1
    if label == 1:
        print('Merger验证集准确率：', acc_num*100/len(pre), '%')
    if label == 0:
        print('Galaxy验证集准确率：', acc_num*100/len(pre), '%')



def verify(path,model):
    verifyModel(path+'\\0', 0, model)
    verifyModel(path+'\\1', 1, model)

if __name__ == "__main__":
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    start = time.time()
    model = Model()
    device = torch.device("cuda:0")
    model.to(device)
    model_package_name = '%s//'%(MODEL_SOURCE_DIR) + modelPackageWrite('normal_')
    # 模型保存文件夹（无需实现创建），是否断点续训，如果断点续寻，上次训练到第几个epoch了
    model_package_name = r'D:\Code\MachineLearning\Model\2020.12.15_MergerClassifier\model_normal-2021-01-03-085331'
    trainModel(model_package_name, flag=True, last_epoch= 24)

    # verifyModel(r'classifier_data\verify_data\\0', 0, r'model\model_normal-2021-01-01-014012\model_48.model')
    # verifyModel(r'classifier_data\verify_data\\1', 1, r'model\model_normal-2021-01-01-014012\model_48.model')
    endtime = time.time()
    print('time cost:',endtime-start)