import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Load MNIST DATASET
mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


# Placeholder for the input images and the latent Variable

X = tf.placeholder(tf.float32,shape =[None,784],name='X') # 28*28 
Z = tf.placeholder(tf.float32,shape=[None,16],name='Z') # vector of 16 



# Discriminator architecture
# Hidden layer 1 size 256
D_W1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[784,256],name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[256]),name='D_b1')

# Output layer with size 1 (scalar)
D_W2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[256,1],name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]),name='D_b2')

# List of the discriminator variables
theta_D = [D_W1,D_W2,D_b1,D_b2]


# Generator architecture
# Hidden layer 1 size 256
G_W1 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[16,256],name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[256]),name='G_b1')

# Output layer with size 784 (image of 28*28)
G_W2 = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),shape=[256,784],name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]),name='G_b2')

# List of the generator variables
theta_G = [G_W1,G_W2,G_b1,G_b2]


# Generator's model
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z,G_W1)+G_b1)
    G_log_prob = tf.matmul(G_h1,G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

# Discriminator's model
def discrimonator(x):
    D_h1 = tf.nn.relu(tf.matmul(x,D_W1)+D_b1)
    D_logit = tf.matmul(D_h1,D_W2)+D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob

# Random generator of values for latent space
def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])


# Plot the generated images
def plote(samples):
    fig = plt.figure(figsize=(4,4))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05,hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28),cmap='gray')

    return fig


G_sample = generator(Z) # Initialize a random values from latent space
D_real = discrimonator(X) # Initilaze the first discriminator with Real data
D_fake = discrimonator(G_sample) # Initialize the seconde discriminator with fake data

# Build the model of minmax game between the discriinator and generator
D_loss = -tf.reduce_mean(tf.log(D_real+1e-8)+tf.log(1. - D_fake+1e-8))
G_loss = -tf.reduce_mean(tf.log(D_fake+1e-8))

# Build the model of variables optomization 
D_solver = tf.train.AdamOptimizer().minimize(D_loss,var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss,var_list=theta_G)

mb_size = 32
z_dim = 16

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # initilize the graph

# Create directory to save generates images
if not os.path.exists('out_classic/'):
    os.makedirs('out_classic/')

i=0
for it in range(1000000):
    
    if it%1000==0:
        
        # save picture for every 1000 iteration
        Z_noise = sample_Z(16,z_dim)
        samples = sess.run(G_sample,feed_dict={Z: Z_noise})
        fig = plote(samples)
        plt.savefig('out_classic/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i +=1
        plt.close(fig)


    X_mb, _ = mnist.train.next_batch(mb_size)

    Z_sample = sample_Z(mb_size,z_dim)
    _, D_loss_curr = sess.run([D_solver,D_loss],feed_dict={X:X_mb, Z:Z_sample})
    _, G_loss_curr = sess.run([G_solver,G_loss],feed_dict={Z:Z_sample})

    if it%1000==0:
        print('Iter: {} -- D loss: {:.4} -- G loss: {:.4}'.format(it,D_loss_curr,G_loss_curr))
        print()