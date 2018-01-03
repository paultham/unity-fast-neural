#%%
import tensorflow as tf
import numpy as np
from models import VGG16
from pipeline import *

def content_loss(vggTrain, vggRef, weight):
    with tf.variable_scope('content_loss'):
        Y = vggRef.content_layer
        X = vggTrain.content_layer
        m, h, w, c = X.get_shape()
        return weight * tf.reduce_sum(tf.squared_difference(Y,X)) / tf.to_float(h*w*c)

def gram(X):
    with tf.variable_scope('gram'):
        m, h, w, c = X.get_shape().as_list()
        X = tf.reshape(X, tf.stack([m, -1, c]))
        return tf.matmul(X, X, transpose_a=True) / tf.to_float(w*h*c)

def style_loss(sess, vggTrain, style_grams, weight):
    with tf.variable_scope('style_loss'):
        loss = 0
        for i in range(len(vggTrain.style_layers)):
            with tf.variable_scope('style_loss_layer_'+str(i)):
                X = vggTrain.style_layers[i]
                Y = style_grams[i]
                loss += tf.reduce_sum(tf.squared_difference(Y, gram(X)))
    return loss * weight

def tv_loss(X, weight):
    with tf.variable_scope('tv_loss'):
        return weight * tf.reduce_sum(tf.image.total_variation(X))

def total_loss(sess, input_var, generator, vggTrain, vggRef, style_grams, params, global_step):
    J_content = content_loss(vggTrain, vggRef, params.content_weight)
    J_style = style_loss(sess, vggTrain, style_grams, params.style_weight)
    J_tv = tv_loss(generator.output, params.tv_weight)
    with tf.variable_scope('total_loss'):
        total_loss = J_content + J_style + J_tv
    with tf.variable_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(params.learn_rate).minimize(total_loss, global_step=global_step)
    return total_loss, train_step, J_content, J_style

def eval_style(params):
    with tf.Session() as sess:
        with tf.variable_scope('eval_style'):
            X = process_img(params.style_path, params.input_shape[0:2], crop=True)
            X = tf.expand_dims(X, 0)
            vggRef = VGG16(X, 'style_vgg')
            style_layers = [gram(l) for l in vggRef.style_layers]
            return sess.run(style_layers)