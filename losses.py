#%%
import tensorflow as tf
import numpy as np

def content_loss(vggTrain, vggRef, weight):
    with tf.variable_scope('content_loss'):
        ref = vggRef.content_layer
        gen = vggTrain.content_layer
        _, h, w, c = ref.get_shape().as_list()
        return weight * tf.reduce_sum(tf.squared_difference(ref, gen)) / float(h * w * c)

def style_layer_loss(a_S, a_G):
    m, h, w, c = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [m, -1, c])
    a_G = tf.reshape(a_G, [m, -1, c])

    with tf.variable_scope('gram'):
        size=h*w*c
        GS = tf.matmul(a_S, a_S, transpose_a=True) / float(size)
        GG = tf.matmul(a_G, a_G, transpose_a=True) / float(size)
    
    _, c1, c2 = GS.get_shape().as_list()
    return tf.reduce_sum(tf.squared_difference(GG,GS)) / float(c1*c2)

def style_loss(sess, input_var, vggTrain, vggRef, style_input, style_weight):
    with tf.variable_scope('style_loss'):
        loss = 0

        ref_styles = sess.run(vggRef.style_layers, feed_dict={input_var:style_input})

        for i in range(len(ref_styles)):
            with tf.variable_scope('style_loss_layer_'+str(i)):
                ref = ref_styles[i]
                gen = vggTrain.style_layers[i]
                loss += style_layer_loss(ref, gen) * style_weight
    return loss

def tv_loss(X, weight):
    with tf.variable_scope('tv_loss'):
        ident = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v_array = np.array([[ident], [-1*ident]])
        h_array = np.array([[ident, -1*ident]])
        v_filter = tf.constant(v_array, tf.float32)
        h_filter = tf.constant(h_array, tf.float32)

        vdiff = tf.nn.conv2d(X, v_filter, strides=[1, 1, 1, 1], padding='VALID')
        hdiff = tf.nn.conv2d(X, h_filter, strides=[1, 1, 1, 1], padding='VALID')

        return weight * tf.reduce_sum(tf.square(hdiff)) + tf.reduce_sum(tf.square(vdiff))

def total_loss(sess, input_var, generator, vggTrain, vggRef, input_style, params):
    J_content = content_loss(vggTrain, vggRef, params.content_weight)
    J_style = style_loss(sess, input_var, vggTrain, vggRef, input_style, params.style_weight)
    J_tv = tv_loss(generator.output, params.tv_weight)
    with tf.variable_scope('total_loss'):
        total_loss = J_content + J_style + J_tv
    with tf.variable_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(params.learn_rate).minimize(total_loss)
    return total_loss, train_step    