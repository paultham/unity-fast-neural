from keras.utils.data_utils import get_file
from models import *
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