#!/usr/bin/env python
# -*- coding: utf-8 -*-


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

# pb-model
pb_model_file = "./frozen_models/mobilenet_v2_140_224.1.pb"


# 读取模型方法1：
def load_model(pb_model_file):
    """ 重构，防止多次重载模型 """

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

print("scores: ", scores)
scores = scores[0].tolist()
indx = np.argmax(scores)
categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
print(indx, categories[indx], scores[indx])
"""
scores:  [[9.9013987e-06 1.6776613e-04 8.0864358e-04 2.6966727e-04 9.9874401e-01]]
4 sexy 0.998744010925293
"""


# 读取模型，方法2：
class NSWF:
    def __init__(self, pb_model_file):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(open(pb_model_file, 'rb').read())
            tensors = tf.import_graph_def(graph_def, name="")

        self.sess=tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                init = tf.global_variables_initializer()
                self.sess.run(init)
                self.sess.graph.get_operations()
                self.feed_input = self.sess.graph.get_tensor_by_name("keras_layer_input:0")
                self.feches = self.sess.graph.get_tensor_by_name("Identity:0")

    def predict(self, img_input):

        scores = self.sess.run(self.feches, feed_dict={self.feed_input: img_input})
        scores = scores[0].tolist()
        indx = np.argmax(scores)
        categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        print(scores)
        print(indx, categories[indx], scores[indx])



nswf = NSWF(pb_model_file)
nswf.predict(img_input)
"""
[9.901398698275443e-06, 0.000167766134836711, 0.0008086435846053064, 0.0002696672745514661, 0.998744010925293]
4 sexy 0.998744010925293
"""