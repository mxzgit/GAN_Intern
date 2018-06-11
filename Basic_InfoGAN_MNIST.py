import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1./ tf.sqrt(in_dim /2.)
    return tf.random_normal(shape=size,stddev = xavier_stddev)


# Placeholder for input images, latent Variable, Intermediar space

X = tf.placeholder(tf.float32,shape=[None,784])
Z = tf.placeholder(tf.float32,shape=[None,16])
c = tf.placeholder(tf.float32,shape=[None,10])

# Discriminator architecture
# Hidden layer 1 size 256
D_W1 = tf.Variable(xavier_init([784,256]))
D_b1 = tf.Variable(tf.zeros(shape=[256]))

# Output layer with size 1 (scalar)
D_W2 = tf.Variable(xavier_init([256,1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

# List of the discriminator variables
theta_D = [D_W1,D_W2,D_b1,D_b2]


# Code layer architecture
# Hidden layer 1 size 784
Q_W1 = tf.Variable(xavier_init([784,256]))
Q_b1 = tf.Variable(tf.zeros(shape=[256]))

# Output layer with size 10
Q_W2 = tf.Variable(xavier_init([256,10]))
Q_b2 = tf.Variable(tf.zeros(shape=[10]))

# List of the code layer variables
theta_Q = [Q_W1,Q_W2,Q_b1,Q_b2]



# Generator architecture
# Hidden layer 1 size 256
G_W1 = tf.Variable(xavier_init([26,256]))
G_b1 = tf.Variable(tf.zeros(shape=[256]))

# Output layer with size 784 (image of 28*28)
G_W2 = tf.Variable(xavier_init([256,784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

# List of the generator variables
theta_G = [G_W1,G_W2,G_b1,G_b2]



# Generator's model
def generator(z,c):
    inputs = tf.concat(axis=1,values=[z,c])
    G_h1 = tf.nn.relu(tf.matmul(inputs,G_W1)+G_b1)
    G_log_prob = tf.matmul(G_h1,G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


# Discriminator's model
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x,D_W1)+D_b1)
    D_logit = tf.matmul(D_h1,D_W2)+D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob

# Code's model
def Q(x):
    Q_h1 = tf.nn.relu(tf.matmul(x,Q_W1)+Q_b1)
    Q_prob = tf.nn.softmax(tf.matmul(Q_h1,Q_W2)+Q_b2)

    return Q_prob

# Random generator of values for latent space
def sample_Z(m,n):
    return np.random.uniform(-1.,1.,size=[m,n])

# Random generator of values for code latent space
def sample_c(m):
    return np.random.multinomial(1,10*[0.1],size=m)

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

G_sample = generator(Z,c) # Initialize a random values from latent space
D_real = discriminator(X) # Initilaze the first discriminator with Real data
D_fake = discriminator(G_sample) # Initialize the seconde discriminator with fake data
Q_c    = Q(G_sample) # Initialize values for code space


# Build the model of minmax game between the discriinator and generator
D_loss = -tf.reduce_mean(tf.log(D_real+1e-10)+tf.log(1-D_fake+1e-10))
G_loss = -tf.reduce_mean(tf.log(D_fake+1e-10))
ent = tf.reduce_mean(-tf.reduce_sum(tf.log(c+1e-10)*c,1))
cross_ent = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_given_x+1e-10)*c,1))
Q_loss = ent+cross_ent


# Build the model of variables optomization 
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
Q_solver = tf.train.AdamOptimizer().minimize(Q_loss, var_list=theta_G + theta_Q)

mb_size = 32
Z_dim = 16


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out_infogan/'):
    os.makedirs('out_infogan/')

i = 0
for it in range(1000000):
    
    if it%1000==0:
        
        # save picture for every 1000 iteration
        Z_noise = sample_Z(16,Z_dim)
        idx = np.random.randint(0,10)
        c_noise = np.zeros([16,10])
        c_noise[range(16),idx] = 1

        samples = sess.run(G_sample,feed_dict={Z: Z_noise, c: c_noise})

        fig = plote(samples)
        plt.savefig('out_infogan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i +=1
        plt.close(fig)
    
    X_mb,_ = mnist.train.next_batch(mb_size)
    Z_noise = sample_Z(mb_size,Z_dim)
    c_noise = sample_c(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict={X: X_mb, Z: Z_noise, c: c_noise})
    _, G_loss_curr = sess.run([G_solver, G_loss],feed_dict={Z: Z_noise, c: c_noise})

    sess.run([Q_solver],feed_dict={Z:Z_noise,c:c_noise})

    if it%1000==0:
        print('Iter: {} -- D loss: {:.4} -- G loss: {:.4}'.format(it,D_loss_curr,G_loss_curr))
        print()