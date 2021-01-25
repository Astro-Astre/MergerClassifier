# -*- coding: utf-8-*-
import os
import sys
import time
import identify as it
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start = time.time()
    f = open(r'galaxy.txt','r')
    data= f.readlines()
    f.close()
    for i in range(len(data)):
        percent = 1.0 * i / len(data)  #用于显示进度
        img_name, label = [str(j) for j in data[i].split()]
        img_name = img_name.split('_')[0]+'.jpg'
        img_dir =os.path.join(r'raw_data\jpg\redshift_galaxy_jpg',img_name)
        # img_dir =os.path.join('raw_data/jpg/merger_jpg/',img_name.split('.')[0]+'.jpg')
        try:
            y = it.otsu(img_dir)
        except AttributeError:
            print(img_dir)
        else:
            if label=='0':
                plt.scatter(i,y,c='r',marker='.')
            else :
                plt.scatter(i,y,c='b',marker='.')
            sys.stdout.write("进度：%.4f"%(percent*100))
            sys.stdout.write("%\r")
            sys.stdout.flush()
    endtime = time.time()
    print('time cost:',endtime-start)
    plt.show()