# -*- coding: utf-8-*-

with open('alfalfa.txt','r') as f:
    data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
    f.close()             #关闭文件
acgnr_merger = []
acgnr_galaxy = []
name = []
label = []
jpg = []
acgnr = []

for i in range(len(data)):
    name = data[i].split(' ')[0]
    pred = data[i].split(' ')[1]
    label.append(pred[0])
    jpg = name.split('_')[1]
    acgnr = jpg.split('.')[0][3:]

for i in range(len(label)):
    if label[i] == '0':
        acgnr_galaxy.append(acgnr)
    else:
        acgnr_merger.append(acgnr)

print(len(acgnr_merger))
print(len(acgnr_galaxy))