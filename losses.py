#%%
import tensorflow as tf
import numpy as np

def content_loss(vggTrain, vggRef, weight):
    with tf.variable_scope('content_loss'):
        ref = vggRef.content_layer
        gen = vggTrain.content_layer
        size = tf.size(ref)
        return weight * tf.reduce_sum(tf.squared_difference(ref, gen)) / tf.to_float(size)

def style_layer_loss(a_S, a_G):
    m, h, w, c = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [m, -1, c])
    a_G = tf.reshape(a_G, [m, -1, c])
    size = tf.size(a_S)
    with tf.variable_scope('gram'):
        GS = tf.matmul(a_S, a_S, transpose_a=True) / tf.to_float(size)
        GG = tf.matmul(a_G, a_G, transpose_a=True) / tf.to_float(size)
    
    # squared frobenius norm of the difference
    # means, (gg-gs)*(gg-gs) sum of all (element wise difference and squared)
    return tf.reduce_sum(tf.squared_difference(GS, GG))

def style_loss(sess, input_var, vggTrain, vggRef, style_input, style_weight):
    with tf.variable_scope('style_loss'):
        loss = 0
        ref_styles = sess.run(vggRef.style_layers, feed_dict={input_var:style_input})
        num_layers = len(ref_styles)
        for i in range(num_layers):
            with tf.variable_scope('style_loss_layer_'+str(i)):
                ref = ref_styles[i]
                gen = vggTrain.style_layers[i]
                loss += style_layer_loss(ref, gen)/tf.to_float(num_layers)

    return style_weight*(loss)

def tv_loss(X, weight):
    with tf.variable_scope('tv_loss'):
        return weight * tf.reduce_sum(tf.image.total_variation(X))

def total_loss(sess, input_var, generator, vggTrain, vggRef, input_style, params):
    J_content = content_loss(vggTrain, vggRef, params.content_weight)
    J_style = style_loss(sess, input_var, vggTrain, vggRef, input_style, params.style_weight)
    J_tv = tv_loss(generator.output, params.tv_weight)
    with tf.variable_scope('total_loss'):
        total_loss = J_content + J_style + J_tv
    with tf.variable_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(params.learn_rate).minimize(total_loss)
    return total_loss, train_step, J_content, J_style