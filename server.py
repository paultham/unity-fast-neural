import os
from flask import Flask, request, url_for
import numpy as np

from params import TransferParams
from pipeline import *
from models import SpriteGenerator

def transfer(path, model_path, out_path=None):
    # init    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # input_image = process_img(path, crop=False).eval()

    input_image = tf.read_file(path)
    input_image = tf.image.decode_image(input_image, channels=3)
    input_image = tf.to_float(input_image)
    input_image = input_image.eval()
    input_shape = [1] + list(input_image.shape)
    
    # make the model
    input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_images')
    gen = SpriteGenerator(input_placeholder, 'SpriteGenerator')
    
    # restore
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    
    # generate and write out
    output = sess.run(gen.output, feed_dict={input_placeholder:np.stack([input_image])})
    if out_path is not None:
        output = tf.image.encode_jpeg(output[0])  
        write = tf.write_file(out_path, output)
        sess.run(write)
        print('Generate Done.')

app = Flask(__name__, static_folder='')

def stylize():
    img_path = request.args.get('path')
    model_name = request.args.get('model')
    root_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(root_dir, 'models', model_name, 'model.ckpt')
    rel_path = os.path.join('data', 'output', 'generated.jpg')
    out_path = os.path.join(root_dir, 'data', 'output', 'generated.jpg')
    transfer(img_path, model_path, out_path)
    return  rel_path

@app.route('/')
def root():
    rel_path = stylize()
    return app.send_static_file(rel_path)