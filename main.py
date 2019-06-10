"""
Usage Instructions:
    mf:
        --datasource=bpr_time --sub_source=netflix --logdir=logs/netlfix/ --metatrain_iterations=20000
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from tensorflow.python.platform import flags

from utils import add_scalars_by_line, metrics_by_pos_neg_pair, metrics_for_pos_neg_pairs
from data_generator import DataGenerator
from maml import MAML

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

FLAGS = flags.FLAGS

# Data set / method options
flags.DEFINE_string('datasource', 'bpr_time', 'model type')
flags.DEFINE_string('sub_source', 'netflix', 'data source')
flags.DEFINE_integer('k_shot', 10, 'number of examples in the train set')

# Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 20000, 'number of meta-training iterations.')
flags.DEFINE_integer('meta_batch_size', 20, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', .007, 'the base learning rate of the meta learner')
flags.DEFINE_integer('update_batch_size', 300,
                     'number of examples(ratings) used for inner gradient update')
flags.DEFINE_integer('test_batch_size', 100,
                     'number of examples(ratings) used for meta gradient update')
flags.DEFINE_float('update_lr', .005, 'the learning rate for inner gradient update.')
flags.DEFINE_integer('num_updates', 10, 'number of inner gradient updates during training.')
flags.DEFINE_bool('stop_grad', False, 'stop grad when meta-update')
flags.DEFINE_bool('use_avg_init', True,
                  'init the features of new users using the average of features of existing users or not')
flags.DEFINE_float('lambda_lr', 1., 'rate of bpr loss comparing to meta loss as 1')

# MF parameters
# ml-1m: 'user_num', 6040, 'train_user_num', 5500, 'item_num', 3953,
# netflix, 10000,8700,13342
flags.DEFINE_integer('user_num', 10000, 'number of users')
flags.DEFINE_integer('train_user_num', 8700, 'number of users in meta train tasks')
flags.DEFINE_integer('item_num', 13342, 'number of items')
flags.DEFINE_integer('latent_d', 20, 'dimension of latent factors')
flags.DEFINE_float('sigma', .0001, 'sigma hyper-parameter for mf model')
flags.DEFINE_float('lamb_u', .0001, 'lamb u')
flags.DEFINE_float('lamb_v', .0001, 'lamb v')

# Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/netflix', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_bool('test_existing_user', False, 'True to train, False to test.')
flags.DEFINE_bool('test', False, 'True to train, False to test.')
flags.DEFINE_integer('summary_interval', 10, 'when to write log')
flags.DEFINE_bool('resume', False, 'whether resume model')
flags.DEFINE_integer('resume_iter', 20000, 'resume the model from this iter')
flags.DEFINE_string('load_dir',
                    '10_shot_mtype_bpr_time.mbs_20.ubs_300.meta_lr_0.007.update_step_10.'
                    'update_lr_0.005.lambda_lr_1.0.avg_f_True.time',
                    'where to load model')


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = FLAGS.summary_interval
    SAVE_INTERVAL = 500
    PRINT_INTERVAL = SUMMARY_INTERVAL * 2
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 10

    # write tensor
    logdir = os.path.join(FLAGS.logdir, 'mlRRS', 'logs',
                          '{}_shot_{}'.format(
                              FLAGS.k_shot, exp_string))
    save_dir = os.path.join(FLAGS.logdir, 'mlRRS', 'model',
                            '{}_shot_{}'.format(
                                FLAGS.k_shot, exp_string))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_writer = tf.summary.FileWriter(
        logdir=logdir,
        graph=sess.graph)
    # write scalar
    print('LOG:, dir: {}'.format(logdir))
    scalar_writer = SummaryWriter(log_dir=logdir + '.test')

    print('Done initializing, starting training.')
    prelosses, postlosses, postlossesa_b, outputas, outputbs = [], [], [], [], []

    # train_test_divider = FLAGS.update_batch_size - 1
    # train_divider = FLAGS.k_shot
    test_divider = FLAGS.update_batch_size - FLAGS.test_batch_size
    train_divider = test_divider
    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if FLAGS.datasource == 'ml':
            batch_x, batch_y = data_generator.generate()

            inputa = batch_x[:, :train_divider, :]
            labela = batch_y[:, :train_divider, :]
            inputb = batch_x[:, train_divider:, :]  # b used for testing
            labelb = batch_y[:, train_divider:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
        elif FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
            batch_x = data_generator.generate()
            # print('train data shape: {}'.format(batch_x.shape))
            inputa = batch_x[:, :train_divider, :]
            inputb = batch_x[:, train_divider:, :]  # b used for testing
            feed_dict = {model.inputa: inputa, model.inputb: inputb}

        input_tensors = []
        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1],
                                  model.total_lossesa_b[FLAGS.num_updates - 1]])
            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                input_tensors.extend([model.outputas, model.outputbs[FLAGS.num_updates - 1]])

        if itr < FLAGS.pretrain_iterations:
            input_tensors.append(model.pretrain_op)
        else:
            # input_tensors.append([model.pretrain_op, model.metatrain_op])
            input_tensors.append(model.metatrain_op)

        # run model
        result = sess.run(input_tensors, feed_dict)

        # write tensorboard summary
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[1])
            if FLAGS.log:
                train_writer.add_summary(result[0], itr)
            postlosses.append(result[2])
            postlossesa_b.append(result[3])
            outputas.append(result[4])
            outputbs.append(result[5])

        # print training process
        if itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'TRAIN: Pretrain Iteration ' + str(itr)
            else:
                print_str = 'TRAIN: Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ', pre update loss: {:.2f}, post update train loss: {:.2f}, ' \
                         'post update test loss: {:.2f}'.format(np.mean(prelosses),
                                                                np.mean(postlossesa_b),
                                                                np.mean(postlosses))
            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                outputas_np = np.array(outputas)
                # print('pre update train output example: {}'.format(outputas_np[0, 0, 0]))
                train_auc = np.mean(outputas_np > 0)
                outputbs_np = np.array(outputbs)

                test_auc, test_hr_at_10, test_ndcg = metrics_by_pos_neg_pair(outputbs_np)

                scalar_writer.add_scalar('META_TRAIN/pre update train AUC', train_auc, itr)
                scalar_writer.add_scalar('META_TRAIN/post update test AUC', test_auc, itr)
                scalar_writer.add_scalar('META_TRAIN/post update test hr_10', test_hr_at_10, itr)
                scalar_writer.add_scalar('META_TRAIN/post update test ndcg', test_ndcg, itr)
                print_str += '\n\t pre update train auc: {:.2f}, post update test hr@10: {:.2f}' \
                             ', post update test auc: {:.2f}'.format(train_auc, test_hr_at_10, test_auc)
            print(print_str)
            prelosses, postlosses, postlossesa_b, outputas, outputbs = [], [], [], [], []

        # test model for new users
        if itr % TEST_PRINT_INTERVAL == 0:
            if FLAGS.datasource == 'ml':
                batch_x, batch_y = data_generator.generate(train=False, batch_size=200)
                inputa = batch_x[:, :test_divider, :]
                inputb = batch_x[:, test_divider:, :]
                labela = batch_y[:, :test_divider, :]
                labelb = batch_y[:, test_divider:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
            elif FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                batch_x = data_generator.generate(train=False, batch_size=200)
                # print('test data shape: {}'.format(batch_x.shape))
                inputa = batch_x[:, :test_divider, :]
                inputb = batch_x[:, test_divider:, :]  # b used for testing
                feed_dict = {model.inputa: inputa, model.inputb: inputb}
            input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates - 1],
                             model.total_lossesa_b[FLAGS.num_updates - 1]]
            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                input_tensors.extend([model.outputas, model.outputbs[FLAGS.num_updates - 1]])

            if FLAGS.use_avg_init:
                model.assign_average_features_to_new_users(FLAGS.train_user_num)
            result = sess.run(input_tensors, feed_dict)

            scalar_writer.add_scalar('META_TEST/pre update train loss', result[0], itr)
            scalar_writer.add_scalar('META_TEST/post update train loss', result[2], itr)
            scalar_writer.add_scalar('META_TEST/post update test loss', result[1], itr)
            print_str = 'TEST: pre update loss: {:.2f}, post update train loss: {:.2f}, ' \
                        'post update test loss: {:.2f}'.format(result[0], result[2], result[1])

            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                outputas_np = np.array(result[3])
                train_auc = np.mean(outputas_np > 0)
                outputbs_np = np.array(result[4])

                test_auc, test_hr_at_10, test_ndcg = metrics_by_pos_neg_pair(outputbs_np)

                scalar_writer.add_scalar('META_TEST/pre update train auc', train_auc, itr)
                scalar_writer.add_scalar('META_TEST/post update test auc,', test_auc, itr)
                scalar_writer.add_scalar('META_TEST/post update test hr', test_hr_at_10, itr)
                scalar_writer.add_scalar('META_TEST/post update test ndcg', test_ndcg, itr)
                print_str += '\n\t pre update train auc: {:.2f}, post update test hr@10: {:.2f}' \
                             ', post update test auc: {:.2f}, post update test ndcg: {:.2f}' \
                             ''.format(train_auc, test_hr_at_10, test_auc, test_ndcg)
            print(print_str)

        # save model every SAVE_INTERVAL itrs
        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, '{}/model_{}'.format(save_dir, itr))

    # save model at last
    saver.save(sess, '{}/model_{}'.format(save_dir, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations))


def test(model, saver, sess, exp_string, data_generator, resume_itr):
    from baselines.utils import metrics_for_pos_neg_pairs
    # load test data
    print('loading test data....')
    test_divider = FLAGS.update_batch_size - FLAGS.test_batch_size
    test_batch_size = 10000
    if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
        batch_x = data_generator.generate(train=False, batch_size=test_batch_size, shufffle=False, start_id=0)
        # print('test data shape: {}'.format(batch_x.shape))
        inputa = batch_x[:, :test_divider, :]
        inputb = batch_x[:, test_divider:, :]  # b used for testing
        feed_dict = {model.inputa: inputa, model.inputb: inputb}
    input_tensors = []
    if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
        input_tensors.extend([model.outputas, model.outputbs[FLAGS.num_updates - 1]])

    if FLAGS.k_shot == 1:
        test_models = {
            '1_shot_mtype_bpr_time.mbs_20.ubs_300.meta_lr_0.007.update_step_10.'
            'update_lr_0.005.lambda_lr_1.0.avg_f_True': 15500
        }
    elif FLAGS.k_shot == 5:
        test_models = {
            '5_shot_mtype_bpr_time.mbs_20.ubs_300.meta_lr_0.01.'
            'update_step_10.update_lr_0.01.lambda_lr_1.0.avg_f_True': 20000
        }
    elif FLAGS.k_shot == 10:
        test_models = {
            '10_shot_mtype_bpr_time.mbs_20.ubs_300.meta_lr_0.01.'
            'update_step_10.update_lr_0.01.lambda_lr_1.0.avg_f_True': 20000
        }
    else:
        raise Exception('no models for {} shot'.format(FLAGS.k_shot))
    # start test
    print('| Model path | Train AUC(pre update) | Test AUC | TEST HR@10 | NDCG |')
    for model_dir in test_models.keys():
        recover_iter = test_models[model_dir]
        model_path = '{}/mlRRS/model/{}/model_{}'.format(FLAGS.logdir, model_dir, recover_iter)
        # model_path = './logs/bpr/{}_shot/{}/model_{}'.format(FLAGS.k_shot, model_dir, recover_iter)
        if os.path.exists(model_path + '.meta'):
            # print('loading model')
            saver.restore(sess=sess, save_path=model_path)
        else:
            raise Exception('No model saved at path {}'.format(model_path))

        logdir = os.path.join(FLAGS.logdir, 'mlRRS', 'logs',
                              'task_{}.{}.{}.test_size_{}'.format(FLAGS.k_shot, model_dir, str(datetime.now()),
                                                                  test_batch_size))
        scalar_writer = SummaryWriter(log_dir=logdir)

        # start test
        # print('testing...')
        if FLAGS.use_avg_init:
            model.assign_average_features_to_new_users(FLAGS.train_user_num)
        result = sess.run(input_tensors, feed_dict)

        print_str = ''
        if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
            outputas_np = np.array(result[-2])
            train_auc = np.mean(outputas_np > 0)
            outputbs_np = np.array(result[-1])
            bool_outputbs = outputbs_np > 0
            task_num = len(outputbs_np)
            # task_num = 500
            hrs = np.zeros(100)
            aucs = 0
            ndcgs = np.zeros(100)
            for i in range(task_num):
                auc, hr, ndcg = metrics_for_pos_neg_pairs(bool_outputbs[i])
                # print auc, hr, ndcg
                hrs += hr
                aucs += auc
                ndcgs += ndcg

            HR = 1. * hrs / task_num
            AUC = aucs / task_num
            NDCG = 1. * ndcgs / task_num

            scalar_writer.add_scalar('TEST/AUC/{}_shot'.format(FLAGS.k_shot), AUC, 0)
            add_scalars_by_line(scalar_writer, 'TEST/HR/{}_shot'.format(FLAGS.k_shot), HR, 0, all=True)
            add_scalars_by_line(scalar_writer, 'TEST/NDCG/{}_shot'.format(FLAGS.k_shot), NDCG, 0, all=True)
            print_str += '| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |'.format(
                model_dir[:2], train_auc, AUC, HR[9], NDCG[9])
        print(print_str)


def test_existing_user(model, saver, sess, exp_string, data_generator, resume_itr):
    SUMMARY_INTERVAL = FLAGS.summary_interval
    SAVE_INTERVAL = 500
    PRINT_INTERVAL = SUMMARY_INTERVAL * 2
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 10

    # write tensor
    logdir = os.path.join(FLAGS.logdir, 'mlRRS', 'logs',
                          '{}_shot_{}'.format(
                              FLAGS.k_shot, exp_string))
    save_dir = os.path.join(FLAGS.logdir, 'mlRRS', 'model',
                            '{}_shot_{}'.format(
                                FLAGS.k_shot, exp_string))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_writer = tf.summary.FileWriter(
        logdir=logdir,
        graph=sess.graph)
    # write scalar
    print('LOG:, dir: {}'.format(logdir))
    scalar_writer = SummaryWriter(log_dir=logdir + '.test')

    print('Done initializing, starting training online')
    prelosses, postlosses, postlossesa_b, outputas, outputbs = [], [], [], [], []

    test_divider = FLAGS.update_batch_size - FLAGS.test_batch_size
    train_divider = test_divider

    exist_hrs = np.zeros(100)
    exist_aucs = 0
    exist_ndcgs = np.zeros(100)
    for itr in range(0, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if FLAGS.datasource == 'ml':
            batch_x, batch_y = data_generator.generate()

            inputa = batch_x[:, :train_divider, :]
            labela = batch_y[:, :train_divider, :]
            inputb = batch_x[:, train_divider:, :]  # b used for testing
            labelb = batch_y[:, train_divider:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
        elif FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
            batch_x = data_generator.generate_exist_user_bpr_batch()
            # print('train data shape: {}'.format(batch_x.shape))
            inputa = batch_x[:, :train_divider, :]
            inputb = batch_x[:, train_divider:, :]  # b used for testing
            feed_dict = {model.inputa: inputa, model.inputb: inputb}

        # test exist user
        exist_preds = sess.run(model.outputbs[FLAGS.num_updates - 1], feed_dict=feed_dict)
        bool_outputs = exist_preds > 0
        task_num = len(bool_outputs)
        # task_num = 500
        hrs = np.zeros(100)
        aucs = 0
        ndcgs = np.zeros(100)
        for i in range(task_num):
            auc, hr, ndcg = metrics_for_pos_neg_pairs(bool_outputs)
            # print auc, hr, ndcg
            hrs += hr
            aucs += auc
            ndcgs += ndcg

        exist_hrs += 1. * hrs / task_num
        exist_aucs += aucs / task_num
        exist_ndcgs += 1. * ndcgs / task_num

        if itr != 0 and itr % SUMMARY_INTERVAL == 0:
            HR = 1. * exist_hrs / itr
            AUC = exist_aucs / itr
            NDCG = exist_ndcgs / itr
            scalar_writer.add_scalar('TEST_EXISTS/AUC_AVG'.format(FLAGS.k_shot), AUC, itr)
            add_scalars_by_line(scalar_writer, 'TEST_EXISTS/HR_AVG'.format(FLAGS.k_shot), HR, itr, all=True)
            add_scalars_by_line(scalar_writer, 'TEST_EXISTS/NDCG_AVG'.format(FLAGS.k_shot), NDCG, itr, all=True)

        # train model using new feedback
        input_tensors = []
        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1],
                                  model.total_lossesa_b[FLAGS.num_updates - 1]])
            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                input_tensors.extend([model.outputas, model.outputbs[FLAGS.num_updates - 1]])

        if itr < FLAGS.pretrain_iterations:
            input_tensors.append(model.pretrain_op)
        else:
            input_tensors.append(model.metatrain_op)

        # run model
        result = sess.run(input_tensors, feed_dict)

        # write tensorboard summary
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[1])
            if FLAGS.log:
                train_writer.add_summary(result[0], itr)
            postlosses.append(result[2])
            postlossesa_b.append(result[3])
            outputas.append(result[4])
            outputbs.append(result[5])

        # print training process
        if itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'TRAIN: Pretrain Iteration ' + str(itr)
            else:
                print_str = 'TRAIN: Iteration ' + str(itr - FLAGS.pretrain_iterations)

            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                outputas_np = np.array(outputas)
                # print('pre update train output example: {}'.format(outputas_np[0, 0, 0]))
                train_auc = np.mean(outputas_np > 0)
                outputbs_np = np.array(outputbs)

                test_auc, test_hr_at_10, test_ndcg = metrics_by_pos_neg_pair(outputbs_np)

                scalar_writer.add_scalar('META_TRAIN/pre update train AUC', train_auc, itr)
                scalar_writer.add_scalar('META_TRAIN/post update test AUC', test_auc, itr)
                scalar_writer.add_scalar('META_TRAIN/post update test hr_10', test_hr_at_10, itr)
                scalar_writer.add_scalar('META_TRAIN/post update test ndcg', test_ndcg, itr)
                print_str += '\n\t pre update train auc: {:.2f}, post update test hr@10: {:.2f}' \
                             ', post update test auc: {:.2f}'.format(train_auc, test_hr_at_10, test_auc)
            print(print_str)
            prelosses, postlosses, postlossesa_b, outputas, outputbs = [], [], [], [], []

        # test model for new users
        if itr % TEST_PRINT_INTERVAL == 0:
            if FLAGS.datasource == 'ml':
                batch_x, batch_y = data_generator.generate(train=False, batch_size=200)
                inputa = batch_x[:, :test_divider, :]
                inputb = batch_x[:, test_divider:, :]
                labela = batch_y[:, :test_divider, :]
                labelb = batch_y[:, test_divider:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}
            elif FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                batch_x = data_generator.generate(train=False, batch_size=200)
                # print('test data shape: {}'.format(batch_x.shape))
                inputa = batch_x[:, :test_divider, :]
                inputb = batch_x[:, test_divider:, :]  # b used for testing
                feed_dict = {model.inputa: inputa, model.inputb: inputb}
            input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates - 1],
                             model.total_lossesa_b[FLAGS.num_updates - 1]]
            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                input_tensors.extend([model.outputas, model.outputbs[FLAGS.num_updates - 1]])

            if FLAGS.use_avg_init:
                model.assign_average_features_to_new_users(FLAGS.train_user_num)
            result = sess.run(input_tensors, feed_dict)

            scalar_writer.add_scalar('META_TEST/pre update train loss', result[0], itr)
            scalar_writer.add_scalar('META_TEST/post update train loss', result[2], itr)
            scalar_writer.add_scalar('META_TEST/post update test loss', result[1], itr)
            print_str = 'TEST: Iteration: {}\n'.format(itr)

            if FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
                outputas_np = np.array(result[3])
                train_auc = np.mean(outputas_np > 0)
                outputbs_np = np.array(result[4])

                test_auc, test_hr_at_10, test_ndcg = metrics_by_pos_neg_pair(outputbs_np)

                scalar_writer.add_scalar('META_TEST/pre update train auc', train_auc, itr)
                scalar_writer.add_scalar('META_TEST/post update test auc,', test_auc, itr)
                scalar_writer.add_scalar('META_TEST/post update test hr', test_hr_at_10, itr)
                scalar_writer.add_scalar('META_TEST/post update test ndcg', test_ndcg, itr)
                print_str += '\n\t pre update train auc: {:.2f}, post update test hr@10: {:.2f}' \
                             ', post update test auc: {:.2f}, post update test ndcg: {:.2f}' \
                             ''.format(train_auc, test_hr_at_10, test_auc, test_ndcg)
            print(print_str)

        # save model every SAVE_INTERVAL itrs
        if (itr != 0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, '{}/model_{}'.format(save_dir, itr))

    # save model at last
    saver.save(sess, '{}/model_{}'.format(save_dir, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations))


def main():
    data_generator = DataGenerator(FLAGS.update_batch_size, FLAGS.meta_batch_size, k_shot=FLAGS.k_shot)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input

    if FLAGS.datasource == 'ml':
        input_tensors = {'inputa': tf.placeholder(tf.int32, shape=[None, None, 2]),
                         'inputb': tf.placeholder(tf.int32, shape=[None, None, 2]),
                         'labela': tf.placeholder(tf.float32, shape=[None, None, 1]),
                         'labelb': tf.placeholder(tf.float32, shape=[None, None, 1])}
    elif FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
        input_tensors = {
            'inputa': tf.placeholder(tf.int32, shape=[None, None, 3]),
            'inputb': tf.placeholder(tf.int32, shape=[None, None, 3]),
        }
    else:
        raise Exception('non-supported data source: {}'.format(FLAGS.datasource))

    model = MAML(dim_input, dim_output)
    if FLAGS.train or FLAGS.test_existing_user:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=input_tensors, prefix='META_TEST')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    exp_string = 'mtype_{}.mbs_{}.ubs_{}.meta_lr_{}.' \
                 'update_step_{}.update_lr_{}.' \
                 'lambda_lr_{}.avg_f_{}' \
                 '.time_{}'.format(FLAGS.datasource,
                                   FLAGS.meta_batch_size,
                                   FLAGS.update_batch_size,
                                   FLAGS.meta_lr, FLAGS.num_updates,
                                   FLAGS.update_lr,
                                   FLAGS.lambda_lr,
                                   FLAGS.use_avg_init,
                                   str(datetime.now()))

    resume_itr = 0
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    if FLAGS.resume:
        model_path = '{}/mlRRS/model/{}/model_{}'.format(FLAGS.logdir, FLAGS.load_dir, FLAGS.resume_iter)
        if os.path.exists(model_path + '.meta'):
            loader.restore(sess=sess, save_path=model_path)
            resume_itr = FLAGS.resume_iter
        else:
            raise Exception('No model saved at path {}'.format(model_path))
    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    if FLAGS.test_existing_user:
        test_existing_user(model, saver, sess, exp_string, data_generator, resume_itr)
    if FLAGS.test:
        test(model, saver, sess, exp_string, data_generator, resume_itr)


if __name__ == "__main__":
    main()
