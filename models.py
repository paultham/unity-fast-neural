#%%
import tensorflow as tf
from keras.utils.data_utils import get_file
import h5py

class VGG16Weights:
    def __init__(self):
        WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.filename = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='6d6bbae143d832006294945121d1f1fc')

    def __enter__(self):
        self.open_file = h5py.File(self.filename, 'r')
        self.layer_names = self.open_file.attrs['layer_names']
        return self

    def __exit__(self, *args):
        if hasattr(self.open_file, 'close'):
            self.open_file.close()

    def get_weights(self, layer_id):
        # block1_conv1
        # block1_conv2
        # block1_pool
        # block2_conv1
        # block2_conv2
        # block2_pool
        # block3_conv1
        # block3_conv2
        # block3_conv3
        # block3_pool
        # block4_conv1
        # block4_conv2
        # block4_conv3
        # block4_pool
        # block5_conv1
        # block5_conv2
        # block5_conv3
        # block5_pool
        layer = self.open_file[self.layer_names[layer_id]]
        weight_names = layer.attrs['weight_names']
        W = layer[weight_names[0]].value
        B = layer[weight_names[1]].value
        return (W, B)

def conv(X, kernel=3, channel=3, stride=1, weights=None, bias=False, suffix=''):
    m, w, h, c = X.get_shape()
    if weights is None:
        # tf.contrib.layers.xavier_initializer
        # tf.truncated_normal_initializer(stddev=0.1)
        # tf.random_normal_initializer(stddev=0.1)
        W = tf.get_variable('conv_weight'+suffix, shape=[kernel, kernel, c, channel], initializer=tf.random_normal_initializer(stddev=0.1))
    else:
        W = tf.constant(weights[0], name='W')
        B = tf.constant(weights[1], name='B')
    pad = tf.to_int32((kernel-1) / 2)
    X = tf.pad(X, [[0,0], [pad, pad], [pad, pad], [0,0]], mode='REFLECT')
    layer = tf.nn.conv2d(X, filter=W, strides=[1, stride, stride, 1], padding='VALID')
    if bias and B is not None:
        layer = layer + B
    return layer

def relu(X):
    return tf.nn.relu(X)

def max_pool(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def instNorm(X):
    # epsilon = 1e-9
    # mean, var = tf.nn.moments(X, [1, 2], keep_dims=True)
    # return tf.div(tf.subtract(X, mean), tf.sqrt(tf.add(var, epsilon)))
    return tf.contrib.layers.instance_norm(X, scale=False)

def resBlock(X):
    i = X
    X = relu(instNorm(conv(X, 3, 128, suffix='1')))
    X = instNorm(conv(X, 3, 128, suffix='2'))
    return i + X

def deconv(X, kernel, channel, stride=2):
    m, prev_w, prev_h, prev_c = X.get_shape()
    output_shape = tf.constant([m, prev_w*stride, prev_h*stride, channel], dtype=tf.int32)
    W = tf.get_variable('deconv_weights', shape=(kernel,kernel,channel,prev_c), initializer=tf.random_normal_initializer(stddev=0.1))
    return tf.nn.conv2d_transpose(X, filter=W, strides=[1, stride, stride, 1], output_shape=output_shape, padding='SAME')

def upsample(X, kernel, channel, stride=2):
    m, h, w, c = X.get_shape()
    new_height = h * stride ** 2
    new_width = w * stride ** 2
    X = tf.image.resize_images(X, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return conv(X, kernel, channel, stride)

class VGG16:
    def __init__(self, X, name):
        self.style_layers = []
        with VGG16Weights() as w:
            with tf.name_scope(name):
                X = tf.reverse(X, [3])
                X = X - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
                with tf.name_scope('b1'):
                    X = relu(conv(X, 3, 64, weights=w.get_weights(0), bias=True))
                    X = relu(conv(X, 3, 64, weights=w.get_weights(1), bias=True))
                    self.style_layers.append(X)
                    X = max_pool(X)
                with tf.name_scope('b2'):
                    X = relu(conv(X, 3, 128, weights=w.get_weights(3), bias=True))
                    X = relu(conv(X, 3, 128, weights=w.get_weights(4), bias=True))
                    self.style_layers.append(X)
                    self.content_layer=X
                    X = max_pool(X)
                with tf.name_scope('b3'):
                    X = relu(conv(X, 3, 256, weights=w.get_weights(6), bias=True))
                    X = relu(conv(X, 3, 256, weights=w.get_weights(7), bias=True))
                    X = relu(conv(X, 3, 256, weights=w.get_weights(8), bias=True))
                    self.style_layers.append(X)
                    X = max_pool(X)
                with tf.name_scope('b4'):
                    X = relu(conv(X, 3, 512, weights=w.get_weights(10), bias=True))
                    X = relu(conv(X, 3, 512, weights=w.get_weights(11), bias=True))
                    X = relu(conv(X, 3, 512, weights=w.get_weights(12), bias=True))
                    self.style_layers.append(X)
                    # X = max_pool(X)
                # with tf.name_scope('b5'):
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(14), bias=True))
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(15), bias=True))
                #     X = relu(conv(X, 3, 512, weights=w.get_weights(16), bias=True))
                #     X = max_pool(X)

class SpriteGenerator:
    def __init__(self, X, name):
        with tf.variable_scope(name):
            # X = X/255.
            # X = tf.pad(X, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
            # X = X - tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')

            with tf.variable_scope('b1'):
                X = relu(instNorm(conv(X, 9, 32)))
            with tf.variable_scope('b2'):
                X = relu(instNorm(conv(X, 3, 64, stride=2)))
            with tf.variable_scope('b3'):
                X = relu(instNorm(conv(X, 3, 128, stride=2)))
            with tf.variable_scope('r4'):
                X = resBlock(X)
            with tf.variable_scope('r5'):
                X = resBlock(X)
            with tf.variable_scope('r6'):
                X = resBlock(X)
            with tf.variable_scope('r7'):
                X = resBlock(X)
            with tf.variable_scope('r8'):
                X = resBlock(X)
            with tf.variable_scope('d9'):
                X = relu(instNorm(upsample(X, 3, 64, stride=2)))
            with tf.variable_scope('d10'):
                X = relu(instNorm(upsample(X, 3, 32, stride=2)))
            with tf.variable_scope('d11'):
                X = instNorm(conv(X, 9, 3))
            with tf.variable_scope('output'):
                X = (tf.tanh(X) + 1.)*127.5

            # h = tf.shape(X)[1]
            # w = tf.shape(X)[2]
            # X = tf.slice(X, [0, 10, 10, 0], [-1, h - 20, w - 20, -1])
            self.output = X