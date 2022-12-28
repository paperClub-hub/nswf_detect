#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Date    : 2022/12/28 13:19
# @ Author  : paperClub
# @ Email   : paperclub@163.com
# @ Site    :


#############################################################
#                色情图像检测 -- nsfw (tf1.x / tf2.x )
############################################################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow
if int(tensorflow.__version__[0]) == 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf

from tensorflow import keras
from glob import glob
import numpy as np
import json




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

    def predict(self, img_path):
        image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = keras.preprocessing.image.img_to_array(image)/255
        # image /= 255
        img_input = np.asarray([image])

        scores = self.sess.run(self.feches, feed_dict={self.feed_input: img_input})
        scores = scores[0].tolist()
        indx = np.argmax(scores)
        # categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
        categories = ['画作', '变态的', '中立的', '色情的', '性感的']

        data = json.dumps({'class': categories[indx], 'score': scores[indx] }, ensure_ascii=False)

        return data


nsfw = NSWF('./frozen_models/mobilenet_v2_140_224.1.pb')

imagefiles = glob("./images/*.png")
for imgfile in imagefiles:
    res = nsfw.predict(imgfile)
    print(f"image: {imgfile} ==>>>>  {res}")