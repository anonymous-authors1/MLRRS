""" Utility functions. """

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


def l2_norm(v):
    return tf.reduce_sum(tf.square(v))


def binary_rating(ratings, bound=3):
    ratings[ratings[:, 2] < bound] = 0
    ratings[ratings[:, 2] >= bound] = 1

    # return ratings


# Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred - label))


def logistic_loss(pred, label):
    # print pred.eval()
    pred = tf.sigmoid(pred)
    return -tf.reduce_mean(label * tf.log(pred) + (1 - label) * tf.log(1 - pred))


def bpr_loss(pred, unused):
    return -tf.reduce_mean(tf.log(tf.sigmoid(pred)))


# Evaluation metrics
def hit_ratio_at_k(pred, label, k):
    _, indices = tf.math.top_k(pred, k=k)
    return tf.reduce_sum(tf.gather(label, indices))


def accuracy(pred, label):
    return tf.reduce_mean(tf.cast(tf.equal(tf.round(pred), label), dtype=tf.float32))


def rmse(pred, label):
    return tf.sqrt(mse(pred, label))


# numpy metrics
def np_hit_ratio_at_k(pred, label, k):
    top_k_indices = np.flip(np.argsort(pred), axis=-1)[:k]
    return np.sum(label[top_k_indices])


# evaluate on one data
def metrics_for_pos_neg_pairs(bool_outputbs):
    auc = np.mean(bool_outputbs)

    hr = [int(auc >= (101 - tmp) / 101.) for tmp in range(1, 101)]

    # ndcg when 1 positive and 100 negative examples
    pos_index = 101 - np.sum(bool_outputbs)
    ndcg = []
    for k in range(1, 101):
        if pos_index > k:
            ndcg.append(0)
        else:
            ndcg.append(1. / np.log(pos_index + 1))

    return auc, hr, ndcg


# evaluate metrics at 10 of all data
def metrics_by_pos_neg_pair(outputs):
    bool_outputbs = outputs > 0

    per_task_aucs = np.mean(bool_outputbs, axis=-1)
    test_auc = np.mean(per_task_aucs)

    # hit ratio at 10 when 1 positive and 100 negative examples
    test_hr_at_10 = np.mean(per_task_aucs >= 91. / 101)

    # ndcg when 1 positive and 100 negative examples
    pos_indices = 101 - np.sum(bool_outputbs, axis=-1)
    per_task_ndcgs = 1. / np.log(pos_indices + 1)
    ndcg = np.mean(per_task_ndcgs)

    return test_auc, test_hr_at_10, ndcg


def add_scalars_by_line(log_writer, name, data, iter, all=False):
    if all:
        num = len(data)
        for i in range(num):
            log_writer.add_scalar(name, data[i], iter * num + i)
    else:
        log_writer.add_scalar(name, data[-1], iter)
# if __name__ == "__main__":
#     # main()
#     import os
#     print os.listdir('./')
