#%%
import tensorflow as tf
from params import TrainingParams

def process_img(filename, size=None, crop=True):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    
    if crop:
        image_shape = tf.image.extract_jpeg_shape(image_string)
        h = image_shape[0]
        w = image_shape[1]
        s = tf.minimum(h, w)
        image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, s, s)

    image_resized = tf.image.resize_images(image_decoded, size) if size is not None else image_decoded
    return tf.to_float(image_resized)

def create_pipeline(sess, params):
    files = tf.data.Dataset.list_files(params.train_path)
    files = files.shuffle(params.total_train_sample)
    files = files.take(params.total_train_sample)
    files = files.map(lambda x: process_img(x, params.input_shape[0:2], crop=True))
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

