from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _variable_with_weight_decay(shape, stddev, wd, name, reuse):
    var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
    if (wd is not None) and (not reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(data, prob, reuse=False):
    with tf.variable_scope("Hid1", reuse=reuse):
        W_hidden1 = _variable_with_weight_decay(shape=[9, 64], stddev=0.1, wd=5e-5, name='W_hidden1', reuse=reuse)
        b_hidden1 = tf.get_variable(name='b_hidden1', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h_hidden1 = tf.nn.relu(tf.matmul(data, W_hidden1) + b_hidden1)
    with tf.variable_scope("Dro1", reuse=reuse):
        h_hidden1_drop = tf.nn.dropout(h_hidden1, prob, name='drop1')

    with tf.variable_scope("Hid2", reuse=reuse):
        W_hidden2 = _variable_with_weight_decay(shape=[64, 64], stddev=0.1, wd=5e-5, name='W_hidden2', reuse=reuse)
        b_hidden2 = tf.get_variable(name='b_hidden2', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h_hidden2 = tf.nn.relu(tf.matmul(h_hidden1_drop, W_hidden2) + b_hidden2)
    with tf.variable_scope("Dro2", reuse=reuse):
        h_hidden2_drop = tf.nn.dropout(h_hidden2, prob, name='drop2')

    with tf.variable_scope("Hid3", reuse=reuse):
        W_hidden3 = _variable_with_weight_decay(shape=[64, 32], stddev=0.1, wd=5e-5, name='W_hidden3', reuse=reuse)
        b_hidden3 = tf.get_variable(name='b_hidden3', shape=[32], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h_hidden3 = tf.nn.relu(tf.matmul(h_hidden2_drop, W_hidden3) + b_hidden3)
    with tf.variable_scope("Dro3", reuse=reuse):
        h_hidden3_drop = tf.nn.dropout(h_hidden3, prob, name='drop3')

    with tf.variable_scope("Hid4", reuse=reuse):
        W_hidden4 = _variable_with_weight_decay(shape=[32, 3], stddev=0.1, wd=5e-5, name='W_hidden4', reuse=reuse)
        b_hidden4 = tf.get_variable(name='b_hidden4', shape=[3], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h_hidden4 = tf.add(tf.matmul(h_hidden3_drop, W_hidden4), b_hidden4)

    with tf.variable_scope("softmax", reuse=reuse):
        logits = tf.multiply(h_hidden4, 1, name='logits')

    return h_hidden4

def total_loss(logits, labels):
    labels = tf.reshape(labels, (-1, 3))
    logits = tf.reshape(logits, (-1, 3))
    _entropy = tf.sqrt(tf.reduce_sum(tf.square(logits - labels), 1))
    entropy = tf.reduce_mean(_entropy)
    return tf.add_n(tf.get_collection('losses')) + entropy

def cross_entropy_loss(logits, labels):
    labels = tf.reshape(labels, (-1, 3))
    logits = tf.reshape(logits, (-1, 3))
    _entropy = tf.sqrt(tf.reduce_sum(tf.square(logits - labels), 1))
    return tf.reduce_mean(_entropy)

def l2_loss():
    return tf.add_n(tf.get_collection('losses'))

def train(loss, learning_rate, batch):
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    return optimizer.minimize(loss, global_step=batch)


