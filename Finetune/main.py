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
from Finetune import mobilenet_v2



class MyNet():
    def __init__(self, target_tensor, num_classes):
        self.target_tensor = target_tensor
        self.finetune_scope = "Finetune"
        self.label = tf.placeholder(tf.float32, (None, num_classes))

    def _change_last_layers(self):
        # tf.reset_default_graph()

        #fisrt build original network
        self.inputs = tf.placeholder(tf.float32, (None, 224, 224, 3), name="inputs")
        with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            logits, endpoints = mobilenet_v2.mobilenet(self.inputs, depth_multiplier=1.0)


        #second, set a new scope for your addiational
        with tf.variable_scope(self.finetune_scope):
            #use get_tensor_by_name to get a tensor
            mobilenet_tensor = tf.get_default_graph().get_tensor_by_name(self.target_tensor)
            #add your layers
            x = tf.layers.Conv2D(filters=5, kernel_size=1, name="Conv2d_1c_1x1")(mobilenet_tensor)
            x = tf.squeeze(x, axis=[1, 2])
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
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    ## 观察新层权重是否更新
    # tf.compat.v1.summary.histogram("mobilenet_conv8", tf.get_default_graph().get_tensor_by_name(
    #     'MobilenetV2/expanded_conv_8/depthwise/depthwise_weights:0'))
    # tf.compat.v1.summary.histogram("mobilenet_conv9", tf.get_default_graph().get_tensor_by_name(
    #     'MobilenetV2/expanded_conv_9/depthwise/depthwise_weights:0'))

    ## 合并所有summary
    merge_all = tf.summary.merge_all()

    mynet = MyNet(TARGET_TENSOR, CLASSES)
    mynet.net()

    var_list = get_restore_vars(TARGET_TENSOR)
    saver = tf.train.Saver(var_list=var_list)

    train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mynet.finetune_scope)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=mynet.label, logits=mynet.predictions)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=train_var)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        writer = tf.compat.v1.summary.FileWriter(r"./logs", sess.graph)

        saver.restore(sess, ckpt_path)


        for i in tqdm(range(EPOCHS)):
            start = (i * BATCH_SIZE) % x_train.shape[0]
            end = min(start + BATCH_SIZE, x_train.shape[0])
            _, merge, losses = sess.run([train_step, merge_all, loss], feed_dict={mynet.inputs: x_train[start:end], mynet.label: y_train[start:end]})
            if i % 100 == 0:
                writer.add_summary(merge, i)
            print('i:', i, '\t', 'loss:', loss)

        saver.save(sess,my_model)



