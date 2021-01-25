# -*- coding: utf-8-*-
import os

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有dat的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('dat'):
                    continue
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + sub_dir + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':
    # txt保存路径，数据保存位置
    gen_txt(r'train_data.txt', r'classifier_data\train_data')
    gen_txt(r'test_data.txt', r'classifier_data\test_data')