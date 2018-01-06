#%%

import tensorflow as tf
from models import SpriteGenerator
from vgg import VGG16
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
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=[1,440,440,3], name='input_images')
    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    vgg = VGG16(gen.output, 'train_vgg')
    vgg = VGG16(input_placeholder, 'train_ref')

def test_loss(sess):

    params = TrainingParams()
    style_grams = eval_style(params)
    tf.reset_default_graph()    

    input_shape = [2,256,256,3]
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_images')
    input_style = np.zeros([2,256,256,3])
    params = TrainingParams()

    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')
    vggRef = VGG16(input_placeholder, 'train_ref')

    J = total_loss(sess, input_placeholder, gen, vggTrain, vggRef, style_grams, params)

def test_style_loss():
    params = TrainingParams()
    params.train_path = 'data/starry_night.jpg'    

    crop=False
    style_grams = eval_style(params, crop=crop)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    input_image = process_img(params.train_path, params.input_shape[0:2] if crop else None, crop=crop).eval()
    input_image = tf.expand_dims(input_image, 0)

    gen = SpriteGenerator(input_image, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')

    J_style = style_loss(vggTrain, style_grams, 1.0)

    cost = sess.run(J_style)
    print('%f' % (cost))

# test_style_loss()
summarize(test_model)
# summarize(test_loss)
