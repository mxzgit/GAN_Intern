import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.gridspec as gridspec

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


def model_inputs(image_width, image_height, image_channels, z_dim):
    
    # Tensor for the input real images
    inputs_real = tf.placeholder(tf.float32, (None, image_width, image_height), name='input_real')
    
    # Tensor for the latent space
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    # Tensor for the learning rate 
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    return inputs_real, inputs_z, learning_rate

# Discriminator's model
def discriminator(images, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse):
        
        alpha = 0.2       

        x1 = tf.reshape(images,[-1,28,28,1])
        x1 = tf.layers.conv2d(x1, 64, 5, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        
        
        x2 = tf.layers.conv2d(relu1, 128, 5, strides=1, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True)
        relu2 = tf.maximum(alpha * bn2, bn2)
        
        
        x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)


        flat = tf.reshape(relu3, (-1, 7*7*256))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        
        return out, logits

# Generator's moodel
def generator(z, out_channel_dim, is_train=True):

    with tf.variable_scope('generator', reuse=not is_train):
        
        alpha = 0.2
        
        
        x1 = tf.layers.dense(z, 7*7*256)
        
        
        x1 = tf.reshape(x1, (-1, 7, 7, 256))
        x1 = tf.layers.batch_normalization(x1, training=is_train)
        x1 = tf.maximum(alpha * x1, x1)
        
       
        x2 = tf.layers.conv2d_transpose(x1, 128, 5, strides=1, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        x2 = tf.maximum(alpha * x2, x2)
        
        
        x3 = tf.layers.conv2d_transpose(x2, 64, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=is_train)
        x3 = tf.maximum(alpha * x3, x3)
        
    
        logits = tf.layers.conv2d_transpose(x3, out_channel_dim, 5, strides=2, padding='same')
        
        
        out = tf.sigmoid(logits)
        
        return out

# Define the model calculaating the loss
def model_loss(input_real, input_z, out_channel_dim):

    smooth = 0.1
    g_model = generator(input_z, out_channel_dim, is_train=True)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * (1 - smooth)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss

# Optimization Model
def model_opt(d_loss, g_loss, learning_rate, beta1):

    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
        
        d_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt


# Plot the generated images
def plote(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05,hspace=0.05)
    samples = samples.reshape(16,784)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28),cmap='gray')

    return fig

# The training Function
def train(epoch_count, batch_size, z_dim, learning_rate, beta1):

    i = 0
    # Define inputs
    input_real, input_z, lr = model_inputs(28, 28, 1, z_dim)
    
    # Define loss model
    d_loss, g_loss = model_loss(input_real, input_z, 1)
    
    # Define optimization model
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, lr, beta1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epoch_count):
            if e % 100 !=0:
                print("Epoch {}/{}".format(e+1,epoch_count))
                        
            X_mb = batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]             
            Z_sample = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: X_mb, 
                                              input_z: Z_sample,
                                              lr: learning_rate})
                
            _ = sess.run(g_train_opt, feed_dict={input_real: X_mb, 
                                                     input_z: Z_sample,
                                                     lr: learning_rate})
                
            # For each 10 batches, get the losses and print them out
                
            if e % 100 == 0:
                i += 1
                d_train_loss = d_loss.eval({input_real: X_mb, input_z: Z_sample})
                    
                g_train_loss = g_loss.eval({input_z: Z_sample})
                    
                print("Epoch {}/{}: ".format(e+1, epoch_count),
                         "Discriminator loss = {:.4f} ".format(d_train_loss),
                         "Generator loss = {:.4f}".format(g_train_loss))
                    
            # Show generator output samples so we can see the progress during training               
            if e % 10 == 0:
                z_dim = input_z.get_shape().as_list()[-1]
                Z_noise = np.random.uniform(-1,1,size=[16,z_dim])
                samples = sess.run(generator(input_z,1,False),feed_dict={input_z: Z_noise})
                fig = plote(samples)
                plt.savefig('out_classic/{}.png'.format(str(e).zfill(3)), bbox_inches='tight')
                i +=1
                plt.close(fig)

batch_size = 32 
z_dim = 16
learning_rate = 0.0002
beta1 = 0.5
n_images = 25
epochs = 1000000

with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1)
   