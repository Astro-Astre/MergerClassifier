# -*- coding: utf-8-*-
import os
import sys
import time
import identify as it
import matplotlib.pyplot as plt
import seaborn as sns


def show_redshift_hist(txt_path, jpg_dir, flag, figure_save_dir, right_label, bins):
    opened_file = open(txt_path, "r")
    txt_data = opened_file.readlines()
    opened_file.close()
    x = []
    for i in range(len(txt_data)):
        percent: float = 1.0 * i / len(txt_data)  # 用于显示进度

        img_name, label = [str(j) for j in txt_data[i].split()]
        if not img_name.split('.')[-1] == "jpg":
            img_name = img_name.split('_')[0] + '.jpg'
        img_dir = os.path.join(jpg_dir, img_name)

        try:
            percent_of_image = it.otsu(img_dir)
        except AttributeError:
            print("cannot open this image:", img_dir)
        else:
            if label != right_label:
                x.append(percent_of_image)

            sys.stdout.write(f"进度：{percent * 100:.4f}")
            sys.stdout.write("%\r")
            sys.stdout.flush()
    end_time = time.time()
    print('time cost:', end_time - start)
    sns.histplot(x, bins=bins)
    if flag:
        plt.savefig(figure_save_dir)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    show_redshift_hist(r"merger.txt",
                       r"D:\Code\MachineLearning\Data\2020.12.15_MergerClassifier\raw_data\jpg\merger_jpg",
                       bins=9, flag=True, figure_save_dir=r"Figure/Merger占比与错误图.png"
                       , right_label='1')
