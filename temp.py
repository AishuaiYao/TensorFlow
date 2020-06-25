# import tensorflow as tf






# # import_meta_graph可以直接从meta文件中加载图结构
# saver = tf.train.import_meta_graph(r"./new_model/new.ckpt.meta")
#
# # allow_soft_placement自动选择设备
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     # latest_checkpoint检查checkpoint检查点文件，查找最新的模型
#     # restore恢复图权重
#     saver.restore(sess,r'./new_model/new.ckpt')
#     graph = sess.graph
#     # get_tensor_by_name通过张量名称获取张量
#     writer = tf.compat.v1.summary.FileWriter(r'./logs',graph)
#     saver.save(sess,'./new_model/new2.ckpt')
#




import cv2
from PIL import Image

mask = Image.open("/home/yas/下载/PennFudanPed/PedMasks/FudanPed00001_mask.png")
mask.putpalette([
    0, 0, 0, # black background
    255, 0, 0, # index 1 is red
    255, 255, 0, # index 2 is yellow
    255, 153, 0, # index 3 is orange
])
mask
Image._show(mask)

print(' ')
