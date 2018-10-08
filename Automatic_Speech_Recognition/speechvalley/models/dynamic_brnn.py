# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : dynamic_brnn.py
# Description  : Dynamic Bidirectional RNN model for Automatic Speech Recognition
# ******************************************************

import argparse
import time
import datetime
import os
from six.moves import cPickle
from functools import wraps

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

# from speechvalley.utils import load_batched_data, describe, setAttrs, list_to_sparse_tensor, dropout, get_edit_distance
# from speechvalley.utils import lnBasicRNNCell, lnGRUCell, lnBasicLSTMCell
from utils import load_batched_data, describe, setAttrs, list_to_sparse_tensor, dropout, get_edit_distance
from utils import lnBasicRNNCell, lnGRUCell, lnBasicLSTMCell, LayerNormBasicLSTMCell


def build_multi_dynamic_brnn(args,
                             maxTimeSteps,
                             inputX,
                             cell_fn,
                             seqLengths,
                             time_major=True):
    hid_input = inputX  # shape=(maxTimeSteps, args.batch_size, args.num_feature)
    for i in range(args.num_layer):
        scope = 'DBRNN_' + str(i + 1)
        forward_cell = cell_fn(args.num_hidden, activation=args.activation)
        backward_cell = cell_fn(args.num_hidden, activation=args.activation)
        # tensor of shape: [max_time, batch_size, input_size]
        outputs, output_states = bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                           inputs=hid_input,
                                                           dtype=tf.float32,
                                                           sequence_length=seqLengths,
                                                           time_major=True,
                                                           scope=scope)

        # forward output, backward output
        # tensor of shape: [max_time, batch_size, input_size]
        output_fw, output_bw = outputs
        # forward states, backward states
        output_state_fw, output_state_bw = output_states

        # output_fb = tf.concat(2, [output_fw, output_bw])
        output_fb = tf.concat([output_fw, output_bw], 2)  # 连接两个矩阵的操作 [max_time, batch_size, input_size*2]？？
        shape = output_fb.get_shape().as_list()
        output_fb = tf.reshape(output_fb, [shape[0], shape[1], 2, int(shape[2] / 2)])  # 第四维度表示取输出结果均值
        hidden = tf.reduce_sum(output_fb, 2)  # 得到第三维度上相加的值，代表了什么？？？？？
        hidden = dropout(hidden, args.keep_prob, (args.mode == 'train'))

        if i != (args.num_layer - 1):
            hid_input = hidden
        else:
            outputXrs = tf.reshape(hidden, [-1, args.num_hidden])  # reshape(tensor,shape,name=None)
            # -1代表把其他维度flatten成一维，应该是生成了什么？[ ?, num_hidden ] 只知道其中一个维度代表num_hidden

            # output_list = tf.split(0, maxTimeSteps, outputXrs)
            output_list = tf.split(outputXrs, maxTimeSteps, 0)  # 将outputXrs分成maxTmeSteps份，
            fbHrs = [tf.reshape(t, [args.batch_size, args.num_hidden]) for t in output_list]  # 把每一时刻的tensor分成
            # [batch_size, num_hidden]大小，并组成以时间为轴的列表
    return fbHrs


class DBiRNN(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps

        self.graph = None  # from def build_graph()
        self.inputX = None

        if args.layerNormalization is True:
            if args.rnncell == 'rnn':
                self.cell_fn = lnBasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = lnGRUCell
            elif args.rnncell == 'lstm':
                # self.cell_fn = lnBasicLSTMCell
                self.cell_fn = LayerNormBasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))
        else:
            if args.rnncell == 'rnn':
                self.cell_fn = tf.contrib.rnn.BasicRNNCell
            elif args.rnncell == 'gru':
                self.cell_fn = tf.contrib.rnn.GRUCell
            elif args.rnncell == 'lstm':
                self.cell_fn = tf.contrib.rnn.BasicLSTMCell
            else:
                raise Exception("rnncell type not supported: {}".format(args.rnncell))

        self.build_graph(args, maxTimeSteps)

    @describe
    def build_graph(self, args, maxTimeSteps):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(maxTimeSteps, args.batch_size, args.num_feature))  # [maxL,32,39]
            inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])  # convert inputX from [maxL,32,39]to[32*maxL,39]
            self.inputList = tf.split(inputXrs, maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]

            # 用于CTC，不知道为什么
            self.targetIxs = tf.placeholder(tf.int64)
            self.targetVals = tf.placeholder(tf.int32)
            self.targetShape = tf.placeholder(tf.int64)
            self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)  # 为什么要SparseTensor
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size))

            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'optimizer': args.optimizer,
                           'learning rate': args.learning_rate,
                           'keep prob': args.keep_prob,
                           'batch size': args.batch_size,
                           'board_dir': args.board_dir}

            fbHrs = build_multi_dynamic_brnn(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)

            with tf.name_scope('fc-layer'):  # 这应该是分类的层
                with tf.variable_scope('fc'):
                    weightsClasses = tf.Variable(
                        tf.truncated_normal([args.num_hidden, args.num_class], name='weightsClasses'))
                    # Outputs random values from a truncated normal distribution.The generated values follow a normal
                    # distribution with specified mean and standard deviation
                    biasesClasses = tf.Variable(tf.zeros([args.num_class]), name='biasesClasses')
                    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in fbHrs]
            logits3d = tf.stack(logits)  # 把logits中axis=0的维度打包成元组，其中axis=0维度表示的是每个batch的输出
            self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, logits3d, self.seqLengths))
            self.var_op = tf.global_variables()
            self.var_trainable_op = tf.trainable_variables()

            if args.grad_clip == -1:
                # not apply gradient clipping
                self.optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
            else:
                # apply gradient clipping
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.var_trainable_op), args.grad_clip)
                opti = tf.train.AdamOptimizer(args.learning_rate)
                self.optimizer = opti.apply_gradients(zip(grads, self.var_trainable_op))

            self.predictions = tf.to_int32(
                tf.nn.ctc_beam_search_decoder(logits3d, self.seqLengths, merge_repeated=False)[0][0])

            if args.level == 'cha':
                self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
                # tf.edit_distance  https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/edit_distance
                # tf.reduce_sum tensor各维求和

            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)

            # max_to_keep indicates the maximum number of recent checkpoint files to keep. As new files are created,
            # older files are deleted. If None or 0, all checkpoint files are kept. Defaults to 5 (that is, the 5 most
            # recent checkpoint files are kept.)

            # keep_checkpoint_every_n_hours: In addition to keeping the most recent max_to_keep checkpoint files, you
            # might want to keep one checkpoint file for every N hours of training. This can be useful if you want to
            # later analyze how a model progressed during a long training session. For example, passing keep_checkpoint
            # _every_n_hours=2 ensures that you keep one checkpoint file for every 2 hours of training. The default
            # value of 10,000 hours effectively disables the feature.

