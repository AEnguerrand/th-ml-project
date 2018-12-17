import tensorflow as tf
import numpy as np
from tensorflow.layers import Flatten
from preprocess import regularts
import pandas as pd
from pickles import pickling
from numpy import random
from load import train

ts_shape = (56, 6)
output_size = 15


def build_net1(inputs, output_size=output_size):
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


def predict(inputs):
    return tf.sigmoid(build_net1(inputs), name="predict")


# def loss_function(ground_truth_labels, logits_predictions):
#     loss = tf.losses.sigmoid_cross_entropy(
#         multi_class_labels=ground_truth_labels,
#         logits=logits_predictions)
#     return loss
#
#
# def train_model(learning_rate=0.001):
#     # with tf.name_scope('Optimizer'):
#     global optimizer
#     if not optimizer:
#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#
#     # with tf.name_scope('Train'):
#     return optimizer.minimize(loss_function)

def build_graph():
    graph = tf.Graph()
    ts_shape = (56, 6)
    output_size = 15

    x_ts = tf.placeholder(tf.float32, [None, ts_shape[0], ts_shape[1]], name="x_ts_placeholder")
    y = tf.placeholder(tf.float32, [None, output_size], name="y_placeholder")

    logits_prediction = build_net1(inputs=x_ts)

    # OPTIMIZATION PART

    learning_rate = 0.001
    loss = tf.losses.sigmoid_cross_entropy(
    multi_class_labels=y,
    logits=logits_prediction)
# ,
#     weights=tf.constant(
#         [[ 151,  495,  924, 1193,  183,   30,  484,  102,  981,  208,  370,2313,  239,  175,0]])
# )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss,name="train")
    return graph



