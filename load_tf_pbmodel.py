#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Date    : 2022/12/21 12:56
# @ Author  : paperClub
# @ Email   : paperclub@163.com
# @ Site    :


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow
if int(tensorflow.__version__[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf

from tensorflow import keras
import numpy as np


print(f"tf-version: {tf.__version__}")



# 图像预处理
dim = 224
imgfile = "./images/1.png"
image = keras.preprocessing.image.load_img(imgfile, target_size=(dim, dim))
image = keras.preprocessing.image.img_to_array(image)
image /= 255
img_input = np.asarray([image])


pb_model_file = "./frozen_models/mobilenet_v2_140_224.1.pb"


# 读取模型：
def load_model(pb_model_file):
    
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(open(pb_model_file, 'rb').read())
        tensors = tf.import_graph_def(graph_def, name="")

    session = tf.Session(graph=graph)
    with session.as_default():
        with graph.as_default():
            init = tf.global_variables_initializer()
            session.run(init)
            session.graph.get_operations()

    return session


tf_session = load_model(pb_model_file)
feed_input = tf_session.graph.get_tensor_by_name("keras_layer_input:0")
feches = tf_session.graph.get_tensor_by_name("Identity:0")
scores = tf_session.run(feches, feed_dict={feed_input: img_input})


scores = scores[0].tolist()
indx = np.argmax(scores)
categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
print(indx, categories[indx], scores[indx])



