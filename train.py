#%%
import tensorflow as tf
from losses import *
from pipeline import *
from params import TrainingParams
from models import *

# Next steps
# - preproc the style image so that it's not mirrored pad, 
# but centrally cropped so that it maintains the aspect ratio and
# maximizes the size before bilinear downsize
# - futher add padding before and remove them after the generator net
# - switch to resize up instead of conv_transpose
def train(params, report_fn=None, start_new=False):

    print('Evaluating Target Style...')
    style_grams = eval_style(params)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    print('Defining Input Pipeline...')
    input_images = create_tf_pipeline('/Users/paul/Work/ai/images/tf/0.tfr', params)

    print('Building Model...')
    input_shape = [params.batch_size] + params.input_shape
    input_images.set_shape(input_shape)

    gen = SpriteGenerator(input_images, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')
    vggRef = VGG16(input_images, 'ref_vgg')

    print('Defining Losses...')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    J, train_step, J_content, J_style = total_loss(sess, input_images, gen, vggTrain, vggRef, style_grams, params, global_step)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', J)
        tf.summary.scalar('style_loss', J_style)
        tf.summary.scalar('content_loss', J_content)
    
    print('Starting...')
    if start_new:
        sess.run(tf.global_variables_initializer())

    with tf.train.MonitoredTrainingSession(checkpoint_dir='summaries', save_summaries_steps=2) as sess:
        while not sess.should_stop():
            _, total_cost, content_cost, style_cost = sess.run([train_step, J, J_content, J_style])
            if report_fn is not None:
                step = tf.train.global_step(sess, global_step)
                report_fn(params, step, 0, total_cost, content_cost, style_cost)

    print('Done...')

params = TrainingParams()
params.train_path = '/Users/paul/Work/ai/images/tf/*.tfr'
params.style_path='data/mosaic.jpg'
params.batch_size = 4
params.num_epoch = 1
params.learn_rate = 0.0001
params.total_train_sample = 1250
params.style_weight = 5.0
params.content_weight = 1.0
params.tv_weight=0.0
train(params)