# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

def pair_loss(y_true, y_pred):

    neg_value_mask = (1.0 - y_true) * y_pred
    neg_values = tf.boolean_mask(neg_value_mask, tf.not_equal(neg_value_mask, 0))

    pos_value_mask = y_true * y_pred
    pos_values = tf.boolean_mask(pos_value_mask, tf.not_equal(pos_value_mask, 0))

    len_neg = tf.shape(neg_values)
    len_pos = tf.shape(pos_values)

    min_length = tf.minimum(len_neg, len_pos)

    min_length_1d = tf.squeeze(min_length)

    def true_fn(neg_values, pos_values, min_length_1d):
        neg_values = tf.slice(neg_values, [0], [min_length_1d])
        pos_values = tf.slice(pos_values, [0], [min_length_1d])

        #pos_values_mesh, neg_values_mesh = tf.meshgrid(pos_values, neg_values)
        #pos_values = tf.reshape(pos_values_mesh, [-1])
        #neg_values = tf.reshape(neg_values_mesh, [-1])
        loss = tf.math.log(1.0 + tf.exp(- (pos_values - neg_values)))
        return K.mean(loss)

    def false_fn():
        return tf.zeros([1, ], tf.float32)

    return tf.cond(min_length_1d > 0, lambda: true_fn(neg_values, pos_values, min_length_1d), false_fn)

def point_loss(y_true, y_pred):
  return K.mean(K.binary_crossentropy(y_true, y_pred))

def listwise_loss(y_true, y_pred, context_index):
  y_true = tf.cast(y_true, tf.float32)
  batch_size = tf.shape(y_true)[0]
  context_index = tf.cast(context_index, tf.float32)
  print('context_index:%s' % context_index)
  tf.debugging.check_numerics(context_index, 'context_index nan')
  #context_index = tf.ones([batch_size, 1], tf.float32)
  mask = tf.equal(context_index, tf.transpose(context_index))
  print('mask:%s' % mask)

  y_pred = tf.tile(tf.expand_dims(y_pred, 1), [1, batch_size, 1])
  y_true = tf.tile(tf.expand_dims(y_true, 1), [1, batch_size, 1])
  mask = tf.cast(mask, tf.float32)
  y_pred = y_pred + (1-tf.expand_dims(mask, 2)) * -1e9  #* float('-inf')
  y_true = y_true * tf.expand_dims(mask, 2)
  true_neg, true_pos = y_true[:,:,0], y_true[:,:,1]
  pred_neg, pred_pos = y_pred[:,:,0], y_pred[:,:,1]

  #pos_softmax_logits = tf.nn.softmax(pred_pos, axis=0)
  #shift_pred_pos = pred_pos - tf.reduce_max(pred_pos)
  #log_pred_pos_shift = tf.math.log(tf.reduce_sum(tf.exp(shift_pred_pos)))
  #log_pos_softmax_output = shift_pred_pos - log_pred_pos_shift
  #neg_softmax_logits = tf.nn.softmax(pred_neg, axis=0)
  #shift_pred_neg = pred_neg - tf.reduce_max(pred_neg)
  #log_pred_neg_shift = tf.math.log(tf.reduce_sum(tf.exp(shift_pred_neg)))
  #log_neg_softmax_output = shift_pred_neg - log_pred_neg_shift

  loss_pos = -K.sum(true_pos * tf.math.log(tf.nn.softmax(pred_pos, axis=0) + 1e-9), axis=0)
  loss_neg = -K.sum(true_neg * tf.math.log(tf.nn.softmax(pred_neg, axis=0) + 1e-9), axis=0)
  ge_loss = K.mean((loss_pos+loss_neg)/(K.sum(mask, axis=0)))
  return ge_loss

def pointwise_pairwise_loss(y_true, y_pred):
  weight = 0.9
  pointwise_loss = K.mean(K.binary_crossentropy(y_true, y_pred))
  pairwise_loss = pair_loss(y_true, y_pred)
  return weight * pointwise_loss + (1-weight)*pairwise_loss

def two_logits_pointwise_listwise_loss(y_true, y_pred):
  #label[1] click state label[0] non-click state
  batch_size = tf.shape(y_true)[0]
  context_index = y_true[:,2:3] #tf.random.uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int64) #y_true[:,2:3]
  y_true = y_true[:,0:2]
  #context_index = tf.ones([batch_size, 1], tf.float32)
  weight = 0.9
  pointwise_loss = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=True))
  ll = listwise_loss(y_true, y_pred, context_index)
  return weight * pointwise_loss + (1-weight) * ll
