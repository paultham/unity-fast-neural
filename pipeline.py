#%%
import tensorflow as tf
from params import TrainingParams

def process_img(filename, size=None, pad=False):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    
    if pad:
        image_shape = tf.image.extract_jpeg_shape(image_string)
        h = tf.to_int32(image_shape[0]/2)
        w = tf.to_int32(image_shape[1]/2)
        cond = tf.less(w, h)
        w, h = tf.cond(cond, lambda: ((h-w), 0), lambda: (0, (w-h)))
        paddings = tf.stack([
            tf.stack([h, h]),
            tf.stack([w, w]),
            tf.stack([0, 0]),
        ])
        try:
            image_decoded = tf.pad(image_decoded, paddings, mode='REFLECT')
        except:
            pass

    image_resized = tf.image.resize_images(image_decoded, size) if size is not None else image_decoded
    return image_resized

def create_pipeline(sess, params):
    files = tf.data.Dataset.list_files(params.train_path)
    files = files.shuffle(params.total_train_sample)
    files = files.take(params.total_train_sample)
    files = files.map(lambda x: process_img(x, params.input_shape[0:2], pad=True))
    files = files.batch(params.batch_size)
    return files.make_initializable_iterator()

def test_pipeline(sess):
    params = TrainingParams()
    iterator = create_pipeline(sess, params)
    next_files = iterator.get_next()

    for ep in range(2):
        print('it ' + str(ep))
        sess.run(iterator.initializer)
        while True:
            try:
                files = sess.run(next_files)
                print(files)
            except tf.errors.OutOfRangeError:
                break

# tf.reset_default_graph()
# sess = tf.InteractiveSession()
# test_pipeline(sess)

