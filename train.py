#%%
import tensorflow as tf
from losses import *
from pipeline import *
from params import TrainingParams
from models import *

def train(params, report_fn=None, restore_epoch=None):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    print('Building Model...')
    input_shape = [params.batch_size] + params.input_shape
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_images')

    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    vggTrain = VGG16(gen.output, 'train_vgg')
    vggRef = VGG16(input_placeholder, 'ref_vgg')

    print('Defining Losses...')
    input_style = process_img(params.style_path, params.input_shape[0:2], pad=True).eval()
    input_style = np.stack([input_style for n in range(params.batch_size)])
    J, train_step, J_content, J_style = total_loss(sess, input_placeholder, gen, vggTrain, vggRef, input_style, params)

    print('Defining Input Pipeline...')
    files_iterator = create_pipeline(sess, params)
    next_files = files_iterator.get_next()

    saver = tf.train.Saver()
    initial_epoch = 0
    if restore_epoch is not None:
        initial_epoch = restore_epoch+1
        saver.restore(sess, params.save_path + str(restore_epoch))
        print('Continuing...')
    else:
        print('Starting...')
        sess.run(tf.global_variables_initializer())

    for epoch in range(initial_epoch, initial_epoch+params.num_epoch):
        sess.run(files_iterator.initializer)
        batch = 0
        while True:
            try:
                try:
                    images = sess.run(next_files)
                except tf.errors.InvalidArgumentError:
                    continue
                
                m, w, h, c = images.shape
                if m != params.batch_size:
                    break
                
                _, total_cost, content_cost, style_cost = sess.run([train_step, J, J_content, J_style], feed_dict={input_placeholder:images})

                if report_fn is None:
                    print('Batch %i, Epoch %i, Cost %s' % (batch, epoch, str(total_cost)))
                else:
                    report_fn(params, batch, epoch, total_cost, content_cost, style_cost)
                batch += 1

            except tf.errors.OutOfRangeError:
                break   

        print('Saving checkpoint')
        saver.save(sess, params.save_path + str(epoch))
    
    print('Saving model to ' + params.save_path)
    saver.save(sess, params.save_path)

    print('Done...')

# params = TrainingParams()
# train(params)