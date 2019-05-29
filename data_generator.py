#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/26 16:44
# @Author  : Duocai Wu


""" Code for loading data. """
import numpy as np
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of rating data
    """

    def __init__(self, num_samples_per_class, batch_size, k_shot=1, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of sub-tasks)
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class

        if FLAGS.datasource == 'ml':
            self.generate = self.generate_rating_batch
            self.dim_input = 2
            self.dim_output = 1
            # 100k 943 1m 6040
            self.ratings_dir = config.get('ratings_dir', './data/movielen/ml-1m/u_based/{}_shot'.format(k_shot))
            # self.rating_file_num = config.get('rating_file_num', 6040)
            self.metatrain_rating_file_num = config.get('metatrain_rating_file_num', 895983)
            self.metatest_rating_file_num = config.get('metatest_rating_file_num', 85625)
        elif FLAGS.datasource == 'bpr':
            self.generate = self.generate_bpr_tuple_batch
            self.dim_output = None
            self.dim_input = 3

            if FLAGS.sub_source == 'ml-1m':
                self.ratings_dir = config.get('ratings_dir', './data/movielen/ml-1m/u_based/bpr/{}_shot'.format(k_shot))
                # 5, 673609, 70767
                # 10, 671180, 71603
                if k_shot == 1:
                    train_num, test_num = 672449, 71061
                elif k_shot == 5:
                    train_num, test_num = 673609, 70767
                elif k_shot == 10:
                    train_num, test_num = 671180, 71603
                else:
                    raise Exception('no data for {} shot'.format(k_shot))
            elif FLAGS.sub_source == 'netflix':
                self.ratings_dir = config.get('ratings_dir', './data/netflix/u_based/bpr/{}_shot'.format(k_shot))
                # 5, 673609, 70767
                # 10, 671180, 71603
                if k_shot == 1:
                    train_num, test_num = 625837, 95398
                elif k_shot == 5:
                    train_num, test_num = 625147, 94458
                elif k_shot == 10:
                    train_num, test_num = 624977, 94790
                else:
                    raise Exception('no data for {} shot'.format(k_shot))
            else:
                raise Exception('no data for {}'.format(FLAGS.sub_source))
            self.metatrain_rating_file_num = config.get('metatrain_rating_file_num', train_num)
            self.metatest_rating_file_num = config.get('metatest_rating_file_num', test_num)
        elif FLAGS.datasource == 'bpr_time':
            self.generate = self.generate_bpr_tuple_batch
            self.dim_output = None
            self.dim_input = 3

            if FLAGS.sub_source == 'ml-1m':
                self.ratings_dir = config.get('ratings_dir',
                                              './data/movielen/ml-1m/u_based/bpr_time/{}_shot'.format(k_shot))
                self.existing_user_ratings_dir = config.get('existing_user_ratings_dir',
                                                            './data/movielen/ml-1m/u_based/bpr_time/'
                                                            'existing_user_test/{}_shot'.format(k_shot))

                self.exist_data_num = config.get('exist_data_num', 76001)
                if k_shot == 1:
                    train_num, test_num = 664814, 69476
                elif k_shot == 5:
                    train_num, test_num = 643956, 67444
                elif k_shot == 10:
                    train_num, test_num = 619073, 66375
                else:
                    raise Exception('no data for {} shot'.format(k_shot))
            elif FLAGS.sub_source == 'netflix':
                self.ratings_dir = config.get('ratings_dir', './data/netflix/u_based/bpr_time/{}_shot'.format(k_shot))
                self.existing_user_ratings_dir = config.get('existing_user_ratings_dir',
                                                            './data/netflix/u_based/bpr_time/'
                                                            'existing_user_test_True/{}_shot'.format(k_shot))
                self.exist_data_num = config.get('exist_data_num', 18601)
                # 5, 673609, 70767
                # 10, 671180, 71603
                if k_shot == 1:
                    train_num, test_num = 616616, 94010
                elif k_shot == 5:
                    train_num, test_num = 582823, 89171
                elif k_shot == 10:
                    train_num, test_num = 541023, 83152
                else:
                    raise Exception('no data for {} shot'.format(k_shot))
            else:
                raise Exception('no data for {}'.format(FLAGS.sub_source))

            self.metatrain_rating_file_num = config.get('metatrain_rating_file_num', train_num)
            self.metatest_rating_file_num = config.get('metatest_rating_file_num', test_num)

        self.cur_train_batch_index = 0
        self.cur_test_batch_index = 0
        self.cur_exist_batch_index = 0

        # shuffle train and test sequence
        self.train_file_sequence = np.arange(self.metatrain_rating_file_num)
        np.random.shuffle(self.train_file_sequence)
        self.test_file_sequence = np.arange(self.metatest_rating_file_num)
        np.random.shuffle(self.test_file_sequence)

    def generate_rating_batch(self, train=True, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        inputs = np.zeros([batch_size, self.num_samples_per_class, self.dim_input], dtype=np.int32)
        outputs = np.zeros([batch_size, self.num_samples_per_class, self.dim_output], dtype=np.int32)

        for i in range(batch_size):
            if train:
                train_or_test = 'train'
                uid = self.train_file_sequence[self.cur_train_batch_index % self.metatrain_rating_file_num]
                self.cur_train_batch_index += 1
            else:
                train_or_test = 'test'
                uid = self.test_file_sequence[self.cur_test_batch_index % self.metatest_rating_file_num]
                self.cur_test_batch_index += 1

            path = '{}/{}/ratings_{}.csv'.format(self.ratings_dir, train_or_test, uid)
            cur_ratings = np.loadtxt(path, delimiter=',', dtype=np.int32)
            assert len(cur_ratings) == self.num_samples_per_class

            inputs[i, :, :] = cur_ratings[:self.num_samples_per_class, :2]
            outputs[i, :, :] = cur_ratings[:self.num_samples_per_class, 2].reshape(
                [self.num_samples_per_class, self.dim_output])

        return inputs, outputs

    def generate_bpr_tuple_batch(self, train=True, batch_size=None, shufffle=True, start_id=None):
        # only for test
        if start_id is not None and not FLAGS.train:
            self.cur_test_batch_index = start_id
        if batch_size is None:
            batch_size = self.batch_size
        inputs = np.zeros([batch_size, self.num_samples_per_class, self.dim_input], dtype=np.int32)

        for i in range(batch_size):
            if train:
                train_or_test = 'train'
                uid = self.cur_train_batch_index % self.metatrain_rating_file_num
                if shufffle:
                    uid = self.train_file_sequence[uid]
                self.cur_train_batch_index += 1
            else:
                train_or_test = 'test'
                uid = self.cur_test_batch_index % self.metatest_rating_file_num
                if shufffle:
                    uid = self.test_file_sequence[uid]
                self.cur_test_batch_index += 1

            path = '{}/{}/bpr_{}.csv'.format(self.ratings_dir, train_or_test, uid)
            cur_ratings = np.loadtxt(path, delimiter=',', dtype=np.int32)
            assert len(cur_ratings) == self.num_samples_per_class
            assert np.max(cur_ratings[:, 0]) < FLAGS.user_num
            assert np.max(cur_ratings[:, (1, 2)]) < FLAGS.item_num, 'file: {}'.format(uid)

            inputs[i, :, :] = cur_ratings

        return inputs

    def generate_exist_user_bpr_batch(self, batch_size=None, start_id=None):
        # only for test
        if start_id is not None:
            self.cur_exist_batch_index = start_id
        if batch_size is None:
            batch_size = self.batch_size
        inputs = np.zeros([batch_size, self.num_samples_per_class, self.dim_input], dtype=np.int32)

        for i in range(batch_size):
            uid = self.cur_exist_batch_index % self.exist_data_num
            self.cur_exist_batch_index += 1

            path = '{}/bpr_{}.csv'.format(self.existing_user_ratings_dir, uid)
            cur_ratings = np.loadtxt(path, delimiter=',', dtype=np.int32)
            assert len(cur_ratings) == self.num_samples_per_class
            assert np.max(cur_ratings[:, 0]) < FLAGS.user_num
            assert np.max(cur_ratings[:, (1, 2)]) < FLAGS.item_num, 'file: {}'.format(uid)

            inputs[i, :, :] = cur_ratings

        return inputs
