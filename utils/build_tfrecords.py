'''
@Author  :   {AishuaiYao}
@License :   (C) Copyright 2020-, {None}
@Contact :   {aishuaiyao@163.com}
@Software:   ${utils}
@File    :   ${build_tfrecords}.py
@Time    :   ${v1:2020-06-26}
@Desc    :   practice
'''


import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np




class Builder():
    def __init__(self, dataset_path, output_path, classes, is_train=True):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.classes = classes
        self.is_train = is_train


    def transfrom(self):
        if self.is_train:
            writer = tf.io.TFRecordWriter(os.path.join(self.output_path, 'train.tfrecords'))
        else:
            writer = tf.io.TFRecordWriter(os.path.join(self.output_path , 'val.tfrecords'))

        for index, name in enumerate(self.classes):
            print('\nclasses', name)
            classes_path = os.path.join(self.dataset_path, name)
            img_list = os.listdir(classes_path)
            img_list = [os.path.join(classes_path, img) for img in img_list]
            for img in tqdm(img_list):
                img = Image.open(img)
                img = img.resize((224, 224))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())

        writer.close()



if __name__ == '__main__':
    is_train = True
    dataset = '/home/yas/下载/asl_dataset/train'
    classes = ['A', 'B', 'C', 'D', 'E']
    output = '/home/yas/下载/asl_dataset'

    builder = Builder(dataset, output, classes, is_train)
    builder.transfrom()









