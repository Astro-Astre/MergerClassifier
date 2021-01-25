# -*- coding: utf-8-*-
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import identify as it

def readCSV(path):
    redshift = []
    with open(path) as f:
        header = csv.DictReader(f)
        for row in header:
            redshift.append(row['redshift'])
        return redshift

def delete(data):
    for i in range(len(data)):
        data[i] = data[i].split('.')[0]
    return data

if __name__ == "__main__":
    path = r'raw_data\jpg\merger_jpg'
    p = r'raw_data\fits\merger\z'
    jpg = delete(os.listdir(path))
    j = delete(os.listdir(p))
    length = len(jpg)
    ret = list(set(jpg).difference(set(j)))
    for i in range(length):
        jpg[i] = it.otsu(path+'//'+jpg[i]+'.jpg')
    for i in range(len(ret)):
        ret[i] = it.otsu(path+'//'+ret[i]+'.jpg')
    sns.set_style('darkgrid')
    # sns.histplot(ret,color='g',kde=True)
    # sns.histplot(jpg,color='r',kde=True)
    galaxy_redshift = readCSV(r'galaxy_redshift_radec.csv')
    sns.histplot(galaxy_redshift)
    plt.show()
