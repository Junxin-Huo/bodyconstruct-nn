from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import numpy as np
import tensorflow as tf
from net import inference, total_loss, train, cross_entropy_loss, l2_loss
from loader import loadDataLabel


BATCH_SIZE = 1024

DATADIR = 'dataset_train'
NUM_EPOCHS = 5000
NETPATH = 'data/net.ckpt'
PBPATH = 'data/train.pb'
EVAL_FREQUENCY = 5000
VALID_GAP = 100
KEEP_PROB = 1.0

def main(argv=None):
    with tf.Graph().as_default():
        print('Start.')
        start_time = time.time()
        begin_time = start_time

        print('Loading images.')
        data, label = loadDataLabel(DATADIR, shuffle=True)
        validation_size = len(label) // 20
        validation_data = data[:validation_size, ...]
        validation_labels = label[:validation_size, ...]
        data = data[validation_size:, ...]
        label = label[validation_size:, ...]
        train_size = len(label)
        validation_size = len(validation_labels)
        print('Loaded %d images.' % (train_size + validation_size))
        print('Train size: %d' % train_size)
        print('Valid size: %d' % validation_size)

        elapsed_time = time.time() - start_time
        print('Loading images with label elapsed %.1f s' % elapsed_time)
        print('Building net......')
        start_time = time.time()

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 9], name='data')
        y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3])
        keep_prob = tf.placeholder(tf.float32, name='prob')
        x_valid = tf.placeholder(tf.float32, shape=[validation_size, 9])
        y_valid = tf.placeholder(tf.float32, shape=[validation_size, 3])

        # Train model.
        train_prediction = inference(x, keep_prob)
        train_prediction_valid = inference(x_valid, keep_prob, reuse=True)

        batch = tf.Variable(0, dtype=tf.float32)

        learning_rate = tf.train.exponential_decay(
            0.1,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size * 100,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learn', learning_rate)

        loss = total_loss(train_prediction, y)
        loss_valid = total_loss(train_prediction_valid, y_valid)
        loss_ce = cross_entropy_loss(train_prediction, y)
        loss_ce_valid = cross_entropy_loss(train_prediction_valid, y_valid)
        loss_l2 = l2_loss()
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss_valid', loss_valid)

        trainer = train(loss, learning_rate, batch)

        elapsed_time = time.time() - start_time
        print('Building net elapsed %.1f s' % elapsed_time)
        start_time = time.time()
        best_validation_loss = 100000.0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('graph/train', sess.graph)

            # Inital the whole net.
            tf.global_variables_initializer().run()
            print('Initialized!')
            for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)

                batch_data = data[offset:offset + BATCH_SIZE, ...]
                batch_labels = label[offset:offset + BATCH_SIZE, ...]

                # Train net.
                feed_dict = {x: batch_data,
                             y: batch_labels,
                             keep_prob: KEEP_PROB}
                sess.run(trainer, feed_dict=feed_dict)

                # Valid net.
                if (step % VALID_GAP == 0):
                    feed_dict = {x: batch_data,
                                 y: batch_labels,
                                 x_valid: validation_data,
                                 y_valid: validation_labels,
                                 keep_prob: 1.0}
                    summary, l, lr, l_valid, l_ce, l_ce_valid, l_l2 = sess.run(
                        [merged, loss, learning_rate, loss_valid, loss_ce, loss_ce_valid, loss_l2],
                        feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    if (step * BATCH_SIZE > NUM_EPOCHS * train_size * 0.9) & (l_valid < best_validation_loss):
                        best_validation_loss = l_valid
                        saver.save(sess, NETPATH)
                        print('Saving net at step %d' % step)
                        print('Learning rate: %f' % lr)
                        print('Train Data total loss:%f' % l)
                        print('Valid Data total loss:%f\n' % l_valid)
                        sys.stdout.flush()
                    if step % EVAL_FREQUENCY == 0:
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        print('Step %d (epoch %.2f), %.3f ms pre step' %
                              (step, step * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
                        print('Learning rate: %f' % lr)
                        print('L2 loss:%f' % l_l2)
                        print('Train Data cross entropy loss:%f' % l_ce)
                        print('Train Data total loss:%f' % l)
                        print('Valid Data cross entropy loss:%f' % l_ce_valid)
                        print('Valid Data total loss:%f\n' % l_valid)
                        sys.stdout.flush()

            train_writer.close()

        elapsed_time = time.time() - begin_time
        print('Total time: %.1f s' % elapsed_time)

if __name__ == '__main__':
    tf.app.run()