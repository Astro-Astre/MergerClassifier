#coding=utf-8 
import os
import sys
from os import close
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def otsu(img_dir):
    # with open(img_dir, 'rb') as f :
    #     img = pickle.load(f)
    #     img = np.swapaxes(img, 0, 2)
    #     img = np.swapaxes(img, 0, 1) # 因为ToTensor会将nadarrayHWC转为CHW，所以要先把ndarray的CHW转为ndarray的HWC
    # img = np.uint16(img)
    # img = Image.fromarray(img)
    # img = img.convert('I;16')
    # img = np.asarray(img)
    img = cv2.imread(img_dir,cv2.IMREAD_UNCHANGED)       #载入图像
    h, w = img.shape[:2]            #获取图像的高和宽 
    blured = cv2.blur(img,(5,5))    #进行滤波去掉噪声
    mask = np.zeros((h+2, w+2), np.uint8)  #掩码长和宽都比输入图像多两个像素点，满水填充不会超出掩码的非零边缘 
    #进行泛洪填充
    cv2.floodFill(blured, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
    #得到灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # 找到轮廓
    area = [cv2.contourArea(maxContours) for maxContours in contours]
    index = np.argmax(area)
    maxContours = contours[index]
    # print(area[index])
    output = area[index] / (h * w)*100
    # if (output > 80 and output<95) :
    #     print(">90:",img_dir)
    # if output < 5 :
    #     print("<5:",img_dir)

    # cv2.drawContours(img,maxContours,-1,(0,0,255),3)   # 绘制轮廓
    # cv2.imshow("result", img)
    # cv2.waitKey(0)

    return output
    # cv2.waitKey(300)
    # cv2.destroyAllWindows()


def show(path):
    img_dir = os.listdir(path)
    max = np.zeros(len(img_dir))
    x = range(len(img_dir))
    for i in range(len(img_dir)):
        # getBImg(img_dir)
        percent = 1.0 * i / len(img_dir)  #用于显示进度
        max[i] = otsu(path+'\\'+img_dir[i])
        sys.stdout.write("进度：%.4f"%(percent*100))
        sys.stdout.write("%\r")
        sys.stdout.flush()
    plt.scatter(x, max, marker='.', c='r')
    plt.show()


if __name__ == "__main__":
    # path = 'raw_data\\jpg\\merger_jpg'
    # show(path)
    path = r'raw_data/jpg/merger_zoom'
    jpg = os.listdir(path)
    # print(otsu('raw_data\\jpg\\alfalfa_jpg\\17674_agc227254.jpg'))
    # for i in range(len(jpg)):
    #     print(path+'\\'+jpg[i],end=' ')
    #     print('%.4f'%(otsu(path+'\\'+jpg[i])))
    print(otsu(r'raw_data/jpg/redshift_galaxy_jpg/26121.jpg'))
    # print(otsu(r'raw_data/jpg/merger_zoom\1_0.962000.jpg'))
    # print(otsu('raw_data\\jpg\\alfalfa_jpg\\21229_agc725773.jpg'))
    # print(otsu('raw_data\\jpg\\alfalfa_jpg\\1071_agc104543.jpg'))
    # path = 'raw_data\\jpg\\alfalfa_jpg'
    # show(path)

    '''
    现在的问题是，对于不铺满的来说很好预测，但是对于占比太大或者占比小于星点的，就不那么好用
    如果可以解决去除星点以及目标模式识别的问题，但是可能还是会用到额外的机器学习的问题，会增加复杂度
    但是如果画面占比尚可的话，目前的识别效果还算不错
    如果merger相隔较远的话，可能会识别成两个目标，但是也可以用过选取两个最大的连通域来进行拾取.
    还有个问题就是画面占比可能不会影响扁平状的星系，最好还需要可以测量出星系长和宽的，才更有效.
    '''
