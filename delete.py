# -*- coding: utf-8-*-
import os

def delete(source,object):
    object_list = os.listdir(object)
    source_list = os.listdir(source)
    for i in range(len(object_list)):
        object_list[i] = object_list[i][:-5]
    for i in range(len(source_list)):
        source_list[i] = source_list[i][:-5]
    ret = list(set(source_list).difference(set(object_list)))
    for i in range(len(ret)):
        os.remove(source+'/'+ret[i]+'.fits')


if __name__ == "__main__":
    # delete(r'raw_data\alfalfa\g',r'raw_data\alfalfa\r')
    # delete(r'raw_data\alfalfa\g',r'raw_data\alfalfa\z')
    # delete(r'raw_data\alfalfa\z',r'raw_data\alfalfa\g')
    # delete(r'raw_data\alfalfa\r',r'raw_data\alfalfa\g')
    # delete(r'raw_data\alfalfa\r',r'raw_data\alfalfa\z')
    # delete(r'raw_data\alfalfa\z',r'raw_data\alfalfa\r')
    dir1 = r'classifier_data\train_data\1'
    dir2 = r'classifier_data\test_data\1'
    dir_1 = os.listdir(dir1)
    dir_2 = os.listdir(dir2)
    for i in range(len(dir_1)):
        k = dir_1[i].split('_')[0]
        dir_1[i] = k
    for i in range(len(dir_2)):
        dir_2[i] = dir_2[i][:-4]
    t = dir_1 + dir_2
    ret = list(set(t))
    for i in range(len(ret)):
        os.remove('merger_dat_normal//'+ret[i]+'.dat')