import tensorflow as tf
import numpy as np
from tensorflow.layers import Flatten
from preprocess import regularts
import pandas as pd
from pickles import pickling
from numpy import random
from load import train

optimizer = None

ts_shape = (56, 6)
output_size = 15

def build_net1(inputs,output_size=output_size):
    convolved1 = tf.layers.conv1d(
        inputs=inputs,
        filters=20,
        strides=1,
        kernel_size=5,
        padding="SAME",
        name="convolution-1")

    pooled1 = tf.layers.max_pooling1d(
        inputs=convolved1,
        pool_size=2,
        strides=2,
        name="max_pool-1")

    convolved2 = tf.layers.conv1d(
        inputs=pooled1,
        filters=30,
        strides=1,
        kernel_size=3,
        padding="SAME",
        name="convolution-2")

    pooled2 = tf.layers.max_pooling1d(
        inputs=convolved2,
        pool_size=2,
        strides=2,
        name="max_pool-2")

    convolved3 = tf.layers.conv1d(
        inputs=pooled2,
        filters=40,
        strides=1,
        kernel_size=3,
        padding="SAME",
        name="convolution-3")

    pooled3 = tf.layers.max_pooling1d(
        inputs=convolved3,
        pool_size=2,
        strides=2,
        name="max_pool-3")

    convolved4 = tf.layers.conv1d(
        inputs=pooled3,
        filters=50,
        strides=1,
        kernel_size=3,
        padding="SAME",
        name="convolution-4")

    pooled4 = tf.layers.max_pooling1d(
        inputs=convolved4,
        pool_size=4,
        strides=4,
        name="max_pool-4")

    flat_layer = Flatten()(pooled4)

    predict_logits = tf.layers.dense(
        inputs=flat_layer,
        units=output_size,
        name="predict_logits"
    )

    return predict_logits


