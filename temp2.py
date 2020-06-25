import os
import tensorflow as tf
import mobilenet_v2
import numpy as np
import cv2

files = os.listdir('./images')
imgs = [cv2.imread(os.path.join('images',file)) for file in files]
imgs = [cv2.resize(img,(224,224)) for img in imgs]
imgs = [img / 255.0 for img in imgs]




tf.reset_default_graph()
# 构建计算图
images = tf.placeholder(tf.float32,(None,224,224,3))
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v2.mobilenet(images,depth_multiplier=1.0)


saver = tf.train.Saver()

with tf.Session() as sess:
    # latest_checkpoint检查checkpoint检查点文件，查找最新的模型
    # restore恢复图权重
    saver.restore(sess,r"./mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt")
    # get_tensor_by_name通过张量名称获取张量

    # get_tensor_by_name通过张量名称获取张量
    saver.save(sess,r'./new_model/new.ckpt')









