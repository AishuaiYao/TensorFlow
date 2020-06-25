import os
import tensorflow as tf
import mobilenet_v2
import numpy as np
import tqdm


tf.reset_default_graph()
# 构建计算图
images = tf.placeholder(tf.float32,(None,224,224,3))
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v2.mobilenet(images,depth_multiplier=1.0)



# 获取目标张量，添加新层
with tf.variable_scope("finetune_layers"):
    # 获取目标张量，取出mobilenet中指定层的张量
    mobilenet_tensor = tf.get_default_graph().get_tensor_by_name("MobilenetV2/expanded_conv_14/output:0")
    # 将张量向新层传递
    x = tf.layers.Conv2D(filters=10,kernel_size=3,name="conv2d_1")(mobilenet_tensor)
    # 观察新层权重是否更新 tf.summary.histogram("conv2d_1",x)
    x = tf.nn.relu(x,name="relu_1")
    x = tf.layers.Conv2D(filters=256,kernel_size=3,name="conv2d_2")(x)
    x = tf.layers.Conv2D(10,3,name="conv2d_3")(x)
    predictions = tf.reshape(x, (-1,10))


# one-hot编码
def to_categorical(data, nums):
    return np.eye(data,nums)


# 随机生成数据
x_train = np.random.random(size=(141, 224, 224, 3))

y_train = to_categorical(141, 10)

# 训练条件配置
## label占位符
y_label = tf.placeholder(tf.int32, (None, 10))
## 收集变量作用域finetune_layers内的变量，仅更新添加层的权重
# train_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="finetune_layers")
## 定义loss
# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=predictions)
## 定义优化方法，用var_list指定需要更新的权重，此时仅更新train_var权重
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=train_var)
## 观察新层权重是否更新
tf.compat.v1.summary.histogram("mobilenet_conv8", tf.get_default_graph().get_tensor_by_name(
    'MobilenetV2/expanded_conv_8/depthwise/depthwise_weights:0'))
tf.compat.v1.summary.histogram("mobilenet_conv9", tf.get_default_graph().get_tensor_by_name(
    'MobilenetV2/expanded_conv_9/depthwise/depthwise_weights:0'))

## 合并所有summary
merge_all = tf.summary.merge_all()

## 设定迭代次数和批量大学
epochs = 10
batch_size = 16


# 获取指定变量列表var_list的函数
def get_var_list(target_tensor=None):
    '''获取指定变量列表var_list的函数'''
    if target_tensor == None:
        target_tensor = r"MobilenetV2/expanded_conv_14/output:0"
    target = target_tensor.split("/")[1]
    all_list = []
    all_var = []
    # 遍历所有变量，node.name得到变量名称
    # 不使用tf.trainable_variables()，因为batchnorm的moving_mean/variance不属于可训练变量
    for var in tf.global_variables():
        if var != []:
            print(var.name)
            all_list.append(var.name)
            all_var.append(var)
    try:
        all_list = list(map(lambda x: x.split("/")[1], all_list))
        # 查找对应变量作用域的索引
        ind = all_list[::-1].index(target)
        ind = len(all_list) - ind - 1
        print(ind)
        del all_list
        return all_var[:ind + 1]
    except:
        print("target_tensor is not exist!")


# 目标张量名称，要获取一个需要从文件中加载权重的变量列表var_list
target_tensor = "MobilenetV2/expanded_conv_14/output:0"
var_list = get_var_list(target_tensor)
saver = tf.train.Saver(var_list=var_list)

# 加载文件内的权重，并训练新层
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    writer = tf.compat.v1.summary.FileWriter(r"./logs", sess.graph)
    ## 初始化参数:从文件加载权重 train_var使用初始化函数
    sess.run(tf.variables_initializer(var_list=train_var))
    saver.restore(sess, "./mobilenet_v2_1.0_224/mobilenet_v2_1.0_224.ckpt")

    # for i in range(1):
    #     start = (i * batch_size) % x_train.shape[0]
    #     end = min(start + batch_size, x_train.shape[0])
    #     _, merge, losses = sess.run([train_step, merge_all, loss],feed_dict={images: x_train[start:end], y_label: y_train[start:end]})
    #     if i % 100 == 0:
    #         writer.add_summary(merge, i)
    # print('i:',i,'\t','loss:',loss)

    saver.save(sess,'./new_model/new.ckpt')



