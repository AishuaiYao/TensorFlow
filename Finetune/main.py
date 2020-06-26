'''
@Author  :   {AishuaiYao}
@License :   (C) Copyright 2020-, {None}
@Contact :   {aishuaiyao@163.com}
@Software:   ${tensorflow finetune experience}
@File    :   ${main}.py
@Time    :   ${v1:2020-06-25}
@Desc    :   practice
'''


import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from mobilenet import mobilenet_v2
from utils  import build_tfrecords


class MyNet():
    def __init__(self, target_tensor, num_classes):
        self.target_tensor = target_tensor
        self.finetune_scope = "Finetune"
        self.num_classes = num_classes

    def _change_last_layers(self):
        tf.reset_default_graph()

        self.inputs = tf.placeholder(tf.float32, (None, 224, 224, 3), name="inputs")

        #fisrt build original network
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            logits, endpoints = mobilenet_v2.mobilenet(self.inputs, depth_multiplier=1.0)


        #second, set a new scope for your addiational
        with tf.variable_scope(self.finetune_scope):
            #use get_tensor_by_name to get a tensor
            mobilenet_tensor = tf.get_default_graph().get_tensor_by_name(self.target_tensor)
            #add your layers
            x = tf.layers.Conv2D(filters=5, kernel_size=1, name="Conv2d_1c_1x1")(mobilenet_tensor)
            self.output = tf.squeeze(x, axis=[1, 2])
            self.predictions = tf.nn.softmax(x, name="predictions")


    def net(self):
        self._change_last_layers()


# one-hot编码
def to_categorical(data, nums):
    multi = data/nums
    return np.repeat(np.eye(nums,nums), multi, axis=0)



def get_restore_vars(target_tensor=None):
    if target_tensor == None:
        print('tell me the target_tensor? please')
        return False

    target = target_tensor.split("/")[1]
    all_list = []
    # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
    vars = tf.global_variables()
    for var in vars:
        if var != []:
            print(var.name)
            all_list.append(var.name)

    try:
        all_list = list(map(lambda x: x.split("/")[1], all_list))
        idx = all_list.index(target)
        print(target_tensor, 'at', idx)
        return vars[:idx]
    except:
        print("target_tensor is not exist!")
        return False




if __name__ == "__main__":

    TARGET_TENSOR = "MobilenetV2/Logits/Dropout/Identity:0"
    ckpt_path = "../pretrained_models/mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt"
    my_model = "../models/my_model.ckpt"

    EPOCHS = 2
    BATCH_SIZE = 16
    DATASETS = 1000
    CLASSES = 5
    learning_rate = 1e-2


    x_train = np.random.random(size=(DATASETS, 224, 224, 3))
    y_train = to_categorical(DATASETS, CLASSES)


    mynet = MyNet(TARGET_TENSOR, CLASSES)
    mynet.net()


    restore_vars = get_restore_vars(TARGET_TENSOR)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mynet.finetune_scope)

    saver = tf.train.Saver(var_list=restore_vars)
    label = tf.placeholder(tf.float32, (None, CLASSES), name="label")
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=mynet.output)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=train_vars)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        writer = tf.compat.v1.summary.FileWriter(r"./logs", sess.graph)

        saver.restore(sess, ckpt_path)
        sess.run(tf.variables_initializer(var_list=train_vars))

        for i in tqdm(range(EPOCHS)):
            for j in range(DATASETS//BATCH_SIZE):
                start = (j * BATCH_SIZE) % x_train.shape[0]
                end = min(start + BATCH_SIZE, x_train.shape[0])
                _, losses = sess.run([train_step, loss], feed_dict={mynet.inputs: x_train[start:end], label: y_train[start:end]})
                print('iters:', j, '\t', 'losses:', losses.mean(axis = 0))

        saver.save(sess,my_model)
