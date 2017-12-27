#%%

import tensorflow as tf
from models import VGG16, SpriteGenerator
from losses import *
from params import TrainingParams
import numpy as np


def summarize(test_fn):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    test_fn(sess)
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('summaries', sess.graph)
    sess.run(tf.global_variables_initializer())
    print('Done summarizing')

def test_model(sess):
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[2,256,256,3], name='input_images')
    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    vgg = VGG16(gen.output, 'train_vgg')
    vgg = VGG16(input_placeholder, 'train_ref')

def test_loss(sess):
    input_shape = [2,256,256,3]
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_images')
    input_style = np.zeros([2,256,256,3])
    params = TrainingParams()

    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')
    vggRef = VGG16(input_placeholder, 'train_ref')

    J = total_loss(sess, gen, vggTrain, vggRef, input_style, params)

# summarize(test_model)
# summarize(test_loss)
