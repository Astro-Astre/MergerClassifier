# -*- coding: utf-8-*-
import os
import pickle
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from astropy.io import fits
from PIL import Image

# lable = 0： galaxy
PIXEL = 4
def normalization(data,max,min):
    '''
    0-1标准化
    data,max,min
    '''
    x = (data-min)/(max-min)
    return x


def display(save_package,signal,source_g,source_r,source_z,source):
    save_dir = save_package + '//' + source[:-5]
    hdu_g = fits.open(source_g+'//'+source)
    hdu_r = fits.open(source_r+'//'+source)
    hdu_z = fits.open(source_z+'//'+source)
    img_g, img_r, img_z = hdu_g[0].data, hdu_r[0].data, hdu_z[0].data
    max = np.array([np.max(img_g), np.max(img_r), np.max(img_z)])
    min = np.array([np.min(img_g), np.min(img_r), np.min(img_z)])
    # 异常处理，去除max = min的情况
    if not(max[0]==min[0] or max[1]==min[1] or max[2]==min[2]):
        # 归一化
        for j in range(256):
            img_g[j] = normalization(img_g[j],max[0],min[0])
            img_r[j] = normalization(img_r[j],max[1],min[1])
            img_z[j] = normalization(img_z[j],max[2],min[2])
        # 转成Image类型以便后续仿射变换
        img_g = np.array(Image.fromarray(img_g))
        img_r = np.array(Image.fromarray(img_r))
        img_z = np.array(Image.fromarray(img_z))
        img = np.array([img_g, img_r, img_z])
        saveImg(img, save_dir + '_raw.dat')
        if signal:
            flip(img_g, img_r, img_z, save_dir)
            zoom(img_g, img_r, img_z, save_dir)
            shift(img_g, img_r, img_z, save_dir)
            rotate(img_g, img_r, img_z, save_dir)

def display_grz(save_package,signal,source_g,source):
    save_dir = save_package + '//' + source[:-5]
    # hdu = fits.open(source_g+'//'+source)
    # img= hdu[0].data
    with open(source_g+'//'+source) as f:
        img = pickle.load(f)
    # 异常处理，去除max = min的情况
    max = np.array([np.max(img)])
    min = np.array([np.min(img)])
    if not(max[0]==min[0] or max[1]==min[1] or max[2]==min[2]):
        # 归一化
        for j in range(256):
            img[j] = normalization(img[j],max[0],min[0])
        # 转成Image类型以便后续仿射变换
        img = np.array(Image.fromarray(img))
        img = np.array([img])
        saveImg(img, save_dir + '_raw.dat')
        if signal:
            flip(img, save_dir)
            zoom(img, save_dir)
            shift(img, save_dir)
            rotate(img, save_dir)

def preProcessing(source_g, save_package, signal):
    p = Pool(2)
    source = os.listdir(source_g)    # 遍历给定g通道目录，需要保证文件目录三通道一一对应
    source_r = source_g[:-1] + 'r'
    source_z = source_g[:-1] + 'z'
    for i in range(len(source)):
        # 用于显示进度
        percent = 1.0 * i / len(source)  #用于显示进度
        if source_g.split('/') == 'g':
            p.apply_async(display,(save_package, signal, source_g, source_r, source_z, source[i],))
        else :
            p.apply_async(display_grz,(save_package, signal, source_g, source[i],))
        sys.stdout.write("转换进度：%.4f"%(percent*100))
        sys.stdout.write("%\r")
        sys.stdout.flush()
    p.close()
    p.join()

def saveImg(object,dir):
    with open(dir,'wb') as f:
        pickle.dump(object,f)

# cv2仿射变换，返回[g,r,z]的ndarray
def doWarpAffine(g,r,z,temp):
    g = cv2.warpAffine(g,temp,(g.shape[:2]))
    r = cv2.warpAffine(r,temp,(g.shape[:2]))
    z = cv2.warpAffine(z,temp,(g.shape[:2]))
    img = np.array([g, r, z])
    return img

# 随机垂直、水平、水平+垂直翻转图像
def flip(g,r,z,save_dir):
    seed = random.randint(-1,1)
    g = cv2.flip(g,seed)
    r = cv2.flip(r,seed)
    z = cv2.flip(z,seed)
    img = np.array([g, r, z])
    saveImg(img,save_dir+'_flipped.dat')

# 随机放大1.1 1.2 1.3倍
def zoom(g,r,z,save_dir):
    seed = round(random.uniform(1.1,1.3),1)
    height,width = g.shape[:2]
    temp = cv2.getRotationMatrix2D((height/2,width/2),0,seed)
    saveImg(doWarpAffine(g,r,z,temp),save_dir+'_zoomed.dat')

# 随机旋转0-90°
def rotate(g,r,z,save_dir):
    seed = random.randint(0,90)
    height,width = g.shape[:2]
    temp = cv2.getRotationMatrix2D((height/2,width/2),seed,1)
    saveImg(doWarpAffine(g,r,z,temp),save_dir+'_rotated.dat')

# 平移PIXEL个像素
def shift(g,r,z,save_dir):
    temp = np.float32([[1,0,PIXEL],[0,1,PIXEL]])
    saveImg(doWarpAffine(g,r,z,temp),save_dir+'_shifted.dat')

def dirExist(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
if __name__ == "__main__":
    source_dir = r'train'
    object_dir = r'train_s'
    dirExist(object_dir)
    preProcessing(source_dir, object_dir,signal=True)
