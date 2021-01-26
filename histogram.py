# -*- coding: utf-8-*-
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import identify as it
from astropy.io import fits
import os
class fitsDo:
    def __init__(self):
        pass
        
    # 打开filename的fits文件，后续使用直接调用hudl，主要用于drawPicture方法
    def openFits(self,filename):
        self.filename = filename
        self.hdul = fits.open(self.filename)

def getRedshiftFromCsv(galaxy_path):
    redshift = []
    i = 0
    with open(galaxy_path) as f:
        header = csv.DictReader(f)
        for row in header:
            if float(row['redshift'])<0.1 and i<30001:
                redshift.append(row['redshift'])
                i+=1
        return redshift

def delete(data):
    for i in range(len(data)):
        data[i] = data[i].split('.')[0]
    return data

if __name__ == "__main__":
    sns.set_style('darkgrid')

    # galaxy_path = r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\raw_data\jpg\redshift_galaxy_jpg'
    # galaxy_jpg = delete(os.listdir(galaxy_path))
    # galaxy_length = len(galaxy_jpg)
    # for i in range(galaxy_length):
    #     galaxy_jpg[i] = it.otsu(galaxy_path+'//'+galaxy_jpg[i]+'.jpg')
    # sns.histplot(galaxy_jpg)


    # merger_path = r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\raw_data\jpg\merger_jpg'
    # merger_delete_path = r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\raw_data\fits\merger\z'
    # merger_jpg = delete(os.listdir(merger_path))
    # # merger_delete = delete(os.listdir(merger_delete_path))
    # # merger_delete_ret = list(set(merger_jpg).difference(set(merger_delete)))
    # merger_length = len(merger_jpg)
    # for i in range(merger_length):
    #     merger_jpg[i] = it.otsu(merger_path+'//'+merger_jpg[i]+'.jpg')
    # # for i in range(len(merger_delete_ret)):
    # #     merger_delete_ret[i] = it.otsu(merger_path+'//'+merger_delete_ret[i]+'.jpg')
    # sns.histplot(merger_jpg)

    
    # readFits_notmg = fitsDo()
    # readFits_notmg.openFits(r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\catalog\darg_mergers.fits')
    # length_notmg = readFits_notmg.hdul[1].data.shape[0]
    # merger_redshift = np.zeros(length_notmg)
    # for i in range(length_notmg):
    #     merger_redshift[i] = readFits_notmg.hdul[1].data[i]['specz1']
    # sns.histplot(merger_redshift)
    

    galaxy_redshift = getRedshiftFromCsv(r'D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\catalog\galaxy_redshift_radec.csv')
    for i in range(len(galaxy_redshift)):
        galaxy_redshift[i] = float(galaxy_redshift[i])
    sns.histplot(galaxy_redshift)

    # # sns.histplot(ret,color='g',kde=True)
    # # sns.histplot(jpg,color='r')
    # # sns.histplot(jpg2,color='g')

    
    plt.show()