#%%
import tensorflow as tf
from losses import *
from pipeline import *
from params import TrainingParams
from models import SpriteGenerator
from vgg import VGG16

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
    input_images = create_tf_pipeline(params)

    print('Building Model...')
    input_shape = [params.batch_size] + params.input_shape
    input_images.set_shape(input_shape)

    gen = SpriteGenerator(input_images, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')
    vggRef = VGG16(input_images, 'ref_vgg')

    print('Defining Losses...')
    J, train_step, J_content, J_style, global_step = total_loss(input_images, gen, vggTrain, vggRef, style_grams, params)

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', J)
        tf.summary.scalar('style_loss', J_style)
        tf.summary.scalar('content_loss', J_content)
    
    if start_new:
        print('Starting...')
        sess.run(tf.global_variables_initializer())
    else:
        print('Continuing...')

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=params.save_path, 
        log_step_count_steps=params.log_step,
        save_summaries_steps=params.summary_step
        ) as sess:
        while not sess.should_stop():
            _, total_cost, content_cost, style_cost = sess.run([train_step, J, J_content, J_style])
            if report_fn is not None:
                step = tf.train.global_step(sess, global_step)
                report_fn(params, step, total_cost, content_cost, style_cost)

    print('Done...')

def report_loss_simple(params, batch, total_cost, content_cost, style_cost):
    print('Batch ' + str(batch))

# params = TrainingParams()
# # aws
# # params.train_path='/home/ubuntu/work/data/unlabeled2017/*.jpg'
# # mbp
# # params.train_path='/Users/paul/Work/ai/images/val2017/*.jpg'
# # azure
# # params.train_path = '/home/paul/src/images/train2017/*.jpg'
# # tf
# params.train_path = ['/Users/paul/Work/ai/images/tf/%i.tfr' % (i) for i in range(2)]
# params.style_path='data/mosaic.jpg'
# params.batch_size = 4
# params.num_epoch = 1
# params.learn_rate = 0.0001
# params.total_train_sample = 4
# params.style_weight = 5.0
# params.content_weight = 1.0
# params.tv_weight=0.0
# params.summary_step=5
# params.save_path='summaries'

# params.last_time = None
# params.initial_batch = None
# train(params, report_fn=report_loss_simple, start_new=True)