# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : lnRNNCell.py
# Description  : RNN Cell with Layer Normalization for Automatic Speech Recognition
# ******************************************************
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def ln(layer, gain, bias):
    self.dims = layer.get_shape().as_list()
    assert len(self.dims)==3, 'layer must be 3-D tensor'
    miu, sigma = tf.nn.moments(self.layer, axes=[2], keep_dims=True)
    ln_layer = tf.div(tf.subtract(self.layer, miu), tf.sqrt(sigma))
    ln_layer = ln_layer*gain + bias
    return ln_layer


class BasicRNNCell(tf.contrib.rnn.RNNCell):
  """The most basic RNN cell.
  Args:
    num_units: int, The number of units in the RNN cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units, activation=None, reuse=None):
    super(BasicRNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.nn.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
    with tf.variable_scope('layer_normalization'):
      gain = tf.get_variable('gain', shape=[self._num_units], initializer=tf.ones_initializer())
      bias = tf.get_variable('bias', shape=[self._num_units], initializer=tf.zeros_initializer())
    output = ln(_linear([inputs, state], self._num_units, True), gain, bias)
    return output, output


class GRUCell(tf.contrib.rnn.RNNCell):

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.nn.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope('layer_normalization'):
      gain1 = tf.get_variable('gain1', shape=[2*self._num_units], initializer=tf.ones_initializer())
      bias1 = tf.get_variable('bias1', shape=[2*self._num_units], initializer=tf.zeros_initializer())
      gain2 = tf.get_variable('gain2', shape=[self._num_units], initializer=tf.ones_initializer())
      bias2 = tf.get_variable('bias2', shape=[self._num_units], initializer=tf.zeros_initializer())

    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = tf.constant_initializer(1.0, dtype=dtype)
      value = tf.nn.sigmoid(ln(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer), gain1, bias1))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(ln(
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer), gain2, bias2))
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class BasicLSTMCell(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):
    super(BasicLSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    with tf.variable_scope('layer_normalization'):
      gain_h = tf.get_variable('gain_h', shape=[4*self._num_units], initializer=tf.ones_initializer())
      bias_h = tf.get_variable('bias_h', shape=[4*self._num_units], initializer=tf.zeros_initializer())
      gain_c = tf.get_variable('gain_c', shape=[self._num_units], initializer=tf.ones_initializer())
      bias_c = tf.get_variable('bias_c', shape=[self._num_units], initializer=tf.zeros_initializer())
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c, h = state
    else:
      c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

    concat = ln(_linear([inputs, h], 4 * self._num_units, True), gain_h, bias_h)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

    new_c = (
        c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
    new_h = self._activation(ln(new_c, gain_c, bias_c)) * sigmoid(o)

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    return new_h, new_state


class LayerNormBasicLSTMCell(rnn_cell_impl.RNNCell):
  """LSTM unit with layer normalization and recurrent dropout.
  This class adds layer normalization and recurrent dropout to a
  basic LSTM unit. Layer normalization implementation is based on:
    https://arxiv.org/abs/1607.06450.
  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  and is applied before the internal nonlinearities.
  Recurrent dropout is base on:
    https://arxiv.org/abs/1603.05118
  "Recurrent Dropout without Memory Loss"
  Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
  """

  def __init__(self,
               num_units,
               forget_bias=1.0,
               input_size=None,
               activation=math_ops.tanh,
               layer_norm=True,
               norm_gain=1.0,
               norm_shift=0.0,
               dropout_keep_prob=1.0,
               dropout_prob_seed=None,
               reuse=None):
    """Initializes the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      activation: Activation function of the inner states.
      layer_norm: If `True`, layer normalization will be applied.
      norm_gain: float, The layer normalization gain initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      norm_shift: float, The layer normalization shift initial value. If
        `layer_norm` has been set to `False`, this argument will be ignored.
      dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
        recurrent dropout probability value. If float and 1.0, no dropout will
        be applied.
      dropout_prob_seed: (optional) integer, the randomness seed.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
    """
    super(LayerNormBasicLSTMCell, self).__init__(_reuse=reuse)

    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)

    self._num_units = num_units
    self._activation = activation
    self._forget_bias = forget_bias
    self._keep_prob = dropout_keep_prob
    self._seed = dropout_prob_seed
    self._layer_norm = layer_norm
    self._norm_gain = norm_gain
    self._norm_shift = norm_shift
    self._reuse = reuse

  @property
  def state_size(self):
    return rnn_cell_impl.LSTMStateTuple(self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def _norm(self, inp, scope, dtype=dtypes.float32):
    shape = inp.get_shape()[-1:]
    gamma_init = init_ops.constant_initializer(self._norm_gain)
    beta_init = init_ops.constant_initializer(self._norm_shift)
    with vs.variable_scope(scope):
      # Initialize beta and gamma for use by layer_norm.
      vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
      vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
    normalized = layers.layer_norm(inp, reuse=True, scope=scope)
    return normalized

  def _linear(self, args):
    out_size = 4 * self._num_units
    proj_size = args.get_shape()[-1]
    dtype = args.dtype
    weights = vs.get_variable("kernel", [proj_size, out_size], dtype=dtype)
    out = math_ops.matmul(args, weights)
    if not self._layer_norm:
      bias = vs.get_variable("bias", [out_size], dtype=dtype)
      out = nn_ops.bias_add(out, bias)
    return out

  def call(self, inputs, state):
    """LSTM cell with layer normalization and recurrent dropout."""
    c, h = state
    args = array_ops.concat([inputs, h], 1)
    concat = self._linear(args)
    dtype = args.dtype

    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
    if self._layer_norm:
      i = self._norm(i, "input", dtype=dtype)
      j = self._norm(j, "transform", dtype=dtype)
      f = self._norm(f, "forget", dtype=dtype)
      o = self._norm(o, "output", dtype=dtype)

    g = self._activation(j)
    if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
      g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

    new_c = (
        c * math_ops.sigmoid(f + self._forget_bias) + math_ops.sigmoid(i) * g)
    if self._layer_norm:
      new_c = self._norm(new_c, "state", dtype=dtype)
    new_h = self._activation(new_c) * math_ops.sigmoid(o)

    new_state = rnn_cell_impl.LSTMStateTuple(new_c, new_h)
    return new_h, new_state


