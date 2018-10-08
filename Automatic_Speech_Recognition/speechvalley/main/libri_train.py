# encoding: utf-8
# ******************************************************
# Author       : 李鸿斌
# Last modified: 2018-03-15
# Email        : 652994327@qq.com
# Filename     : libri_train.py
# Description  : Training  models on LibriSpeech dataset for Automatic Speech Recognition
# ******************************************************

import time
import datetime
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

# from speechvalley.utils import load_batched_data, describe, describe, getAttrs, output_to_sequence, list_dirs, logging,\
#     count_params, target2phoneme, get_edit_distance, get_num_classes, check_path_exists, dotdict,\
#     activation_functions_dict, optimizer_functions_dict
from utils import load_batched_data, describe, describe, getAttrs, output_to_sequence, list_dirs, logging,\
    count_params, target2phoneme, get_edit_distance, get_num_classes, check_path_exists, dotdict,\
    activation_functions_dict, optimizer_functions_dict

from models import DBiRNN, DeepSpeech2


from tensorflow.python.platform import flags
from tensorflow.python.platform import app
# """Generic entry point script."""


# 第一个是参数名称，第二个参数是默认值，第三个是参数描述  分别为string， bool， integer类型
flags.DEFINE_string('task', 'libri', 'set task name of this program')
flags.DEFINE_string('train_dataset', 'train-clean-460', 'set the training dataset')
flags.DEFINE_string('dev_dataset', 'dev-clean', 'set the development dataset')
flags.DEFINE_string('test_dataset', 'test-clean', 'set the test dataset')

flags.DEFINE_string('mode', 'train', 'set whether to train, dev or test')

flags.DEFINE_boolean('keep', False, 'set whether to restore a model, when test mode, keep should be set to True')
flags.DEFINE_string('level', 'cha', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'DBiRNN', 'set the model to use, DBiRNN, BiRNN, ResNet..')  # ？？？？？？？？？
flags.DEFINE_string('rnncell', 'lstm', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 3, 'set the layers for rnn')
flags.DEFINE_string('activation', 'tanh', 'set the activation to use, sigmoid, tanh, relu, elu...')
flags.DEFINE_string('optimizer', 'adam', 'set the optimizer to use, sgd, adam...')
flags.DEFINE_boolean('layerNormalization', True, 'set whether to apply layer normalization to rnn cell')

flags.DEFINE_integer('batch_size', 32, 'set the batch size')
flags.DEFINE_integer('num_hidden', 256, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')     # ？？？为什么不是39的倍数而是60？？？
flags.DEFINE_integer('num_classes', 29, 'set the number of output classes')
flags.DEFINE_integer('num_steps', 1, 'set the number of steps for one subdir')
flags.DEFINE_integer('num_epochs', 0, 'set the number of epoch for the whole dataset')
flags.DEFINE_float('lr', 0.001, 'set the learning rate')
flags.DEFINE_float('dropout_prob', 0.1, 'set probability of dropout')
flags.DEFINE_float('grad_clip', 1, 'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_string('datadir', '/home/hongbin/data/libri_lhb', 'set the data root directory')
# flags.DEFINE_string('logdir', '/home/hongbin/data/libri_lhb/log', 'set the log directory')
flags.DEFINE_string('logdir', '/home/hongbin/data/libri_lhb/log_final', 'set the log directory')


FLAGS = flags.FLAGS
task = FLAGS.task

train_dataset = FLAGS.train_dataset
dev_dataset = FLAGS.dev_dataset
test_dataset = FLAGS.test_dataset

level = FLAGS.level
model_fn = DBiRNN     # 为什么不用FLAGS.model
rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

activation_fn = activation_functions_dict[FLAGS.activation]  # 从字典中选出对应的activation_function
optimizer_fn = optimizer_functions_dict[FLAGS.optimizer]     # 从字典中选出对应的optimizer

batch_size = FLAGS.batch_size
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature
num_classes = get_num_classes(level)   # 通过get_num_classes() 得到相应分类总数
num_steps = FLAGS.num_steps
num_epochs = FLAGS.num_epochs
lr = FLAGS.lr
grad_clip = FLAGS.grad_clip
datadir = FLAGS.datadir

logdir = FLAGS.logdir
savedir = os.path.join(logdir, level, 'save')
resultdir = os.path.join(logdir, level, 'result')
loggingdir = os.path.join(logdir, level, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

mode = FLAGS.mode
keep = FLAGS.keep
keep_prob = 1-FLAGS.dropout_prob

# 用于控制所用显存
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True

# 用于选择某个显卡进行训练
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   # 指定第一块GPU可用

print('%s mode...' % str(mode))   # %s代表打印mode
if mode == 'test' or mode == 'dev':
    batch_size = 10
    num_steps = 1


def get_data(datadir, level, train_dataset, dev_dataset, test_dataset, mode):
    """返回mode类型下的路径"""
    if mode == 'train':
        train_feature_dirs = [os.path.join(os.path.join(datadir, level, train_dataset),
                                           i, 'feature') for i in os.listdir(os.path.join(datadir, level, train_dataset))]

        train_label_dirs = [os.path.join(os.path.join(datadir, level, train_dataset),
                                         i, 'label') for i in os.listdir(os.path.join(datadir, level, train_dataset))]
        return train_feature_dirs, train_label_dirs

    if mode == 'dev':
        dev_feature_dirs = [os.path.join(os.path.join(datadir, level, dev_dataset),
                                         i, 'feature') for i in os.listdir(os.path.join(datadir, level, dev_dataset))]

        dev_label_dirs = [os.path.join(os.path.join(datadir, level, dev_dataset),
                                       i, 'label') for i in os.listdir(os.path.join(datadir, level, dev_dataset))]
        return dev_feature_dirs, dev_label_dirs

    if mode == 'test':
        test_feature_dirs = [os.path.join(os.path.join(datadir, level, test_dataset),
                                          i, 'feature') for i in os.listdir(os.path.join(datadir, level, test_dataset))]

        test_label_dirs = [os.path.join(os.path.join(datadir, level, test_dataset),
                                        i, 'label') for i in os.listdir(os.path.join(datadir, level, test_dataset))]
        return test_feature_dirs, test_label_dirs


logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(),
                                                                  '%Y-%m-%d %H:%M:%S') + str(num_epochs) + '.txt').replace(' ', '').replace('/', ''))


class Runner(object):  # python2.7的新式类，在3.5中可以省略
    def _default_configs(self):
        return {'level': level,
                'rnncell': rnncell,
                'batch_size': batch_size,
                'num_hidden': num_hidden,
                'num_feature': num_feature,
                'num_class': num_classes,
                'num_layer': num_layer,
                'activation': activation_fn,
                'optimizer': optimizer_fn,
                'learning_rate': lr,
                'keep_prob': keep_prob,
                'grad_clip': grad_clip,
                }

    @describe  # 修饰符号@，用于调用describe函数，其中用来显示加载过程中的信息
    def load_data(self, feature_dir, label_dir, mode, level):
        return load_batched_data(feature_dir, label_dir, batch_size, mode, level)
        # 用于把所有mini-batch的数据综合后放在batched data的list中，其中长度较短的语音信号用0填充至最大
        # batchedData为padding后的数据 maxTimeSteps是padding后每部分的矩阵长度, totalN
        #  load_batched_data（）： returns 3-element tuple: batched data (list), maxTimeLength (int), and total number
        # of samples (int)
        #  data_lists_to_batches（） ： padding the input list to a same dimension, integrate all data into batchInputs

    def run(self):
        # load data
        args_dict = self._default_configs()
        args = dotdict(args_dict)  # 创建dotdict类，类似创造自己的dict
        feature_dirs, label_dirs = get_data(datadir, level, train_dataset, dev_dataset, test_dataset, mode)

        # batchedData, maxTimeSteps, totalN = self.load_data(feature_dirs[0], label_dirs[0], mode, level)
        # model = model_fn(args, maxTimeSteps)
        # # 此两行作用不明白，删掉后不知道有什么影响

        # 记录每次epoch的
        # shuffle feature_dir and label_dir by same order
        FL_pair = list(zip(feature_dirs, label_dirs))  # zip()后返回特定zip数据？，list让其变成列表
        random.shuffle(FL_pair)  # 打乱列表中元素顺序
        feature_dirs, label_dirs = zip(*FL_pair)

        for feature_dir, label_dir in zip(feature_dirs, label_dirs):  # zip()返回结果可用于for, 展示时用list()展出
            id_dir = feature_dirs.index(feature_dir)
            print('dir id:{}'.format(id_dir))
            batchedData, maxTimeSteps, totalN = self.load_data(feature_dir, label_dir, mode, level)

            model = model_fn(args, maxTimeSteps)  # 建立神经网络的图

            num_params = count_params(model, mode='trainable')
            all_num_params = count_params(model, mode='all')
            model.config['trainable params'] = num_params
            model.config['all params'] = all_num_params
            print(model.config)

            with tf.Session(graph=model.graph, config=config) as sess:
                # restore from stored model
                if keep:  # 用于重新训练 keep == True
                    ckpt = tf.train.get_checkpoint_state(savedir)
                    # Returns CheckpointState proto from the "checkpoint" file.
                    if ckpt and ckpt.model_checkpoint_path:  # The checkpoint file
                        model.saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Model restored from:' + savedir)
                else:
                    print('Initializing')
                    sess.run(model.initial_op)

                for step in range(num_steps):
                    # training
                    start = time.time()
                    if mode == 'train':
                        print('step {} ...'.format(step + 1))

                    batchErrors = np.zeros(len(batchedData))
                    batchRandIxs = np.random.permutation(len(batchedData))
                    # 如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本

                    for batch, batchOrigI in enumerate(batchRandIxs):
                        # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，
                        # 利用它可以同时获得索引和值          这部分代码用于feed_Dict
                        batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                        batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                        feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs,
                                    model.targetVals: batchTargetVals, model.targetShape: batchTargetShape,
                                    model.seqLengths: batchSeqLengths}

                        if level == 'cha':
                            if mode == 'train':
                                _, l, pre, y, er = sess.run([model.optimizer, model.loss,
                                                             model.predictions, model.targetY, model.errorRate],
                                                            feed_dict=feedDict)

                                batchErrors[batch] = er  # batchError 207 batch 211

                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},step:{},train loss={:.3f},mean '
                                      'train CER={:.3f}, epoch: {}\n'.format(level, totalN, id_dir + 1, len(feature_dirs),
                                                                  batch + 1, len(batchRandIxs), step + 1, l,
                                                                             er / batch_size, num_epochs))

                            elif mode == 'dev':
                                l, pre, y, er = sess.run(
                                    [model.loss, model.predictions, model.targetY, model.errorRate],
                                    feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},dev loss={:.3f},'
                                      'mean dev CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs),
                                                                     batch + 1
                                                                     , len(batchRandIxs), l, er / batch_size))

                            elif mode == 'test':
                                l, pre, y, er = sess.run(
                                    [model.loss, model.predictions, model.targetY, model.errorRate],
                                    feed_dict=feedDict)
                                batchErrors[batch] = er
                                print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},test loss={:.3f},'
                                      'mean test CER={:.3f}\n'.format(level, totalN, id_dir + 1, len(feature_dirs),
                                                                      batch + 1, len(batchRandIxs), l, er / batch_size))
                        elif level == 'seq2seq':
                            raise ValueError('level %s is not supported now' % str(level))

                        # NOTE: ??????for what
                        # if er / batch_size == 1.0:
                        #     break

                        if batch % 20 == 0:
                            print('Truth:\n' + output_to_sequence(y, type=level))
                            print('Output:\n' + output_to_sequence(pre, type=level))

                        if mode == 'train' and ((step * len(batchRandIxs) + batch + 1) % 20 == 0 or (
                                step == num_steps - 1 and batch == len(batchRandIxs) - 1)):
                            # 每当算式结果是20倍数 或者 跑完一个 subdir的 batch后， 记录model
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=step)
                            print('Model has been saved in {}'.format(savedir))

                    end = time.time()
                    delta_time = end - start
                    print('subdir ' + str(id_dir + 1) + ' needs time:' + str(delta_time) + ' s')

                    if mode == 'train':
                        if (step + 1) % 1 == 0:
                            checkpoint_path = os.path.join(savedir, 'model.ckpt')
                            model.saver.save(sess, checkpoint_path, global_step=step)
                            print('Model has been saved in {}'.format(savedir))
                        epochER = batchErrors.sum() / totalN
                        print('subdir', id_dir + 1, 'mean train error rate:', epochER)  # 修改epoch成subdir
                        logging(model, logfile, epochER, id_dir, delta_time, mode='config')
                        logging(model, logfile, epochER, id_dir, delta_time, mode=mode)

                    if mode == 'test' or mode == 'dev':
                        with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
                            result.write(output_to_sequence(y, type=level) + '\n')
                            result.write(output_to_sequence(pre, type=level) + '\n')
                            result.write('\n')
                        epochER = batchErrors.sum() / totalN
                        print(' test error rate:', epochER)
                        logging(model, logfile, epochER, mode=mode)


if __name__ == '__main__':
    runner = Runner()
    runner.run()
