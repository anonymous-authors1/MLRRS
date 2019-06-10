""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import flags

from utils import l2_norm, mse, bpr_loss

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, dim_input=2, dim_output=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.update_lr)
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())

        if FLAGS.datasource == 'ml':
            self.loss_func = mse
            # self.loss_func = logistic_loss
            self.forward = self.forward_mf
            self.construct_weights = self.construct_mf_weights
            self.weights = self.construct_weights()
        elif FLAGS.datasource == 'bpr' or FLAGS.datasource == 'bpr_time':
            self.loss_func = bpr_loss
            self.forward = self.forward_bpr
            self.construct_weights = self.construct_mf_weights
            self.weights = self.construct_weights()
        else:
            raise ValueError('Unrecognized data source.')

    def dense_gradients(self, loss, vars):
        grads = tf.gradients(loss, vars)
        for i in range(len(grads)):
            # convert to dense tensor
            # print(grads[i].dense_shape)
            if type(grads[i]) is tf.IndexedSlices:
                grads[i] = tf.convert_to_tensor(grads[i])
            # print(grads[i])
        # print(grads)
        return grads

    def assign_average_features_to_new_users(self, train_user_num):
        user_f = self.weights['users']
        user_f[train_user_num:].assign(tf.reduce_mean(user_f, axis=-1))
        assert isinstance(user_f, tf.Variable)

    def construct_model(self, input_tensors, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        if FLAGS.datasource == 'ml':
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None):
            weights = self.weights

            num_updates = FLAGS.num_updates

            def task_metalearn(inp):
                """ Perform gradient descent for one task in the meta-batch. """
                if FLAGS.datasource == 'ml':
                    inputa, inputb, labela, labelb = inp
                else:
                    inputa, inputb = inp
                    labela, labelb = None, None
                task_outputbs, task_lossesb = [], []
                task_lossesa_b = []
                task_reg_bs = []

                task_outputa, task_reg_a = self.forward(inputa, weights)
                task_lossa = self.loss_func(task_outputa, labela)
                grads = self.dense_gradients(task_lossa + task_reg_a, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(
                    zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
                output, reg_b = self.forward(inputb, fast_weights)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))
                task_lossesa_b.append(self.loss_func(self.forward(inputa, fast_weights)[0], labela))
                task_reg_bs.append(reg_b)

                for j in range(num_updates - 1):
                    output_a, task_reg_a = self.forward(inputa, fast_weights)
                    loss = self.loss_func(output_a, labela)
                    grads = self.dense_gradients(loss + task_reg_a, list(fast_weights.values()))

                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))
                    output, reg_b = self.forward(inputb, fast_weights)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))
                    task_lossesa_b.append(self.loss_func(self.forward(inputa, fast_weights)[0], labela))
                    task_reg_bs.append(reg_b)

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, task_lossesa_b, task_reg_a,
                               task_reg_bs]

                return task_output

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates,
                         [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if FLAGS.datasource == 'ml':
                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                                   dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            else:
                result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb),
                                   dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, lossesa_b, reg_as, reg_bs = result

        # Performance & Optimization
        # after the map_fn
        self.outputas, self.outputbs = outputas, outputbs
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_mean(lossesa) + tf.reduce_mean(reg_as)
            self.total_lossesa_b = total_lossesa_b = [tf.reduce_mean(lossesa_b[j]) for j in range(num_updates)]
            self.total_losses2 = total_losses2 = [tf.reduce_mean(lossesb[j]) + tf.reduce_mean(reg_bs[j]) for j in
                                                  range(num_updates)]

            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                # self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[num_updates - 1])
                meta_loss = self.total_losses2[num_updates - 1] + FLAGS.lambda_lr * total_loss1
                self.metatrain_op = optimizer.minimize(meta_loss)

            # Summaries
            tf.summary.scalar(prefix + '/Pre-update loss', total_loss1)
            # for j in range(num_updates):
            tf.summary.scalar(prefix + '/Post-update test loss', total_losses2[num_updates - 1])
            tf.summary.scalar(prefix + '/Post-update train loss', total_lossesa_b[num_updates - 1])

    # Network construction functions
    def construct_mf_weights(self):
        weights = {}
        weights['users'] = tf.Variable(
            FLAGS.sigma * tf.random_uniform([FLAGS.user_num, FLAGS.latent_d], dtype=tf.float32),
            name='user_embeddings')
        weights['items'] = tf.Variable(
                FLAGS.sigma * tf.random_uniform([FLAGS.item_num, FLAGS.latent_d], dtype=tf.float32),
                name='item_embeddings')

        return weights

    def forward_mf(self, inp, weights, name="mf_model"):
        """
        :param inp: batch of (user_id, item_id)
        :param weights:
        :param reuse:
        :param lamb_u:
        :param lamb_v:
        :param learning_r:
        :param name:
        :return:
        """
        U = weights['users']
        V = weights['items']
        with tf.name_scope(name):
            with tf.name_scope("batch_embeddings"):
                with tf.name_scope('batch_ids'):
                    u_indices = inp[:, 0]
                    v_indices = inp[:, 1]
                with tf.name_scope('batch_embeddings'):
                    u_batch = tf.nn.embedding_lookup(U, u_indices)
                    v_batch = tf.nn.embedding_lookup(V, v_indices)

            with tf.name_scope('preds'):
                predict_score = tf.reduce_sum(tf.multiply(u_batch, v_batch), axis=1, name='preds')

            # calculate loss
            with tf.name_scope("regularization"):
                reg = FLAGS.lamb_u * l2_norm(u_batch) + FLAGS.lamb_v * l2_norm(v_batch)

            return predict_score, reg

    def forward_bpr(self, inp, weights, name="bpr_model"):
        with tf.name_scope(name=name):
            u, i, j = inp[:, 0], inp[:, 1], inp[:, 2]
            U = weights['users']
            V = weights['items']
            u_batch = tf.nn.embedding_lookup(U, u)
            i_batch = tf.nn.embedding_lookup(V, i)
            j_batch = tf.nn.embedding_lookup(V, j)
            # print(u_batch.shape)
            ui = tf.reduce_sum(tf.multiply(u_batch, i_batch), axis=1)
            uj = tf.reduce_sum(tf.multiply(u_batch, j_batch), axis=1)
            # print(uj)

            uij = ui - uj
            # print(uij.shape)

            reg = FLAGS.lamb_u * l2_norm(u_batch) + FLAGS.lamb_v * (l2_norm(i_batch) + l2_norm(j_batch))

            return uij, reg
