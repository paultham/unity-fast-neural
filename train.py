#%%
import tensorflow as tf
from losses import *
from pipeline import *
from params import TrainingParams
from models import *

def train(params, report_fn=None):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    print('Building Model...')
    input_shape = [params.batch_size] + params.input_shape
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_images')

    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')
    vggRef = VGG16(input_placeholder, 'train_ref')

    print('Defining Losses...')
    input_style = process_img(params.style_path, params.input_shape[0:2]).eval()
    input_style = np.stack([input_style for n in range(params.batch_size)])
    J, train_step = total_loss(sess, gen, vggTrain, vggRef, input_style, params)

    print('Defining Input Pipeline...')
    files_iterator = create_pipeline(sess, params)
    next_files = files_iterator.get_next()

    print('Starting...')
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for epoch in range(params.num_epoch):
        sess.run(files_iterator.initializer)
        batch = 0
        while True:
            try:
                images = sess.run(next_files)
                _, total_cost = sess.run([train_step, J], feed_dict={input_placeholder:images})

                if report_fn is None:
                    print('Batch %i, Epoch %i, Cost %s' % (batch, epoch, str(total_cost)))
                else:
                    report_fn(batch, epoch, total_cost)
                batch += 1
            except tf.errors.OutOfRangeError:
                break
    
    print('Saving model to ' + params.save_path)
    saver.save(sess, params.save_path)

    print('Done...')

# params = TrainingParams()
# train(params)