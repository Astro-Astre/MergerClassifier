# -*- coding: utf-8-*-
import csv
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
    acgnr.append(jpg.split('.')[0][3:])

for i in range(len(label)):
    if label[i] == '0':
        acgnr_galaxy.append(acgnr[i])
    else:
        acgnr_merger.append(acgnr[i])

# print(len(acgnr_merger))
# print(len(acgnr_galaxy))
# print(acgnr_galaxy)
print(type(acgnr_merger[1]))
with open("ALFALFA_merger.csv","w",newline='') as f:
    for i in range(len(acgnr_merger)):
        f.write(acgnr_merger[i])
        f.write('\n')
with open("ALFALFA_galaxy.csv","w",newline='') as f:
    for i in range(len(acgnr_galaxy)):
        f.write(acgnr_galaxy[i])
        f.write('\n')