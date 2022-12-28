#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ Date    : 2022/12/28 12:54
# @ Author  : paperClub
# @ Email   : paperclub@163.com
# @ Site    :



import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

print(tf.__version__)

# tf = 2.1.0
# hub.KerasLayer 首次使用时需要翻墙访问 hub，
h5_path = 'mobilenet_v2_140_224.1/mobilenet_v2_140_224/saved_model.h5'
model = tf.keras.models.load_model(h5_path,
                                 custom_objects={'KerasLayer': hub.KerasLayer})

print(model.summary())

# full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(
#     x=tf.TensorSpec(model.inputs[0].shape,model.inputs[0].dtype))

# x可以修改为 =>>>>>>  keras_layer_input
full_model = tf.function(lambda keras_layer_input: model(keras_layer_input))
full_model = full_model.get_concrete_function(
    keras_layer_input=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))


# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]

print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("请保存如下打印结果 ")
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="mobilenet_v2_140_224.1.pb",
                  as_text=False)
