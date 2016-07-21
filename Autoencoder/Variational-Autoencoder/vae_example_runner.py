import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from vae_example import *


def generate_data( data_size ):
    z_true = np.random.uniform(0, 1, data_size )
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    x1 = r * np.cos(phi)
    x2 = r * np.sin(phi)

    # Sampling form a Gaussian
    x1 = np.random.normal(x1, 0.10 * np.power(z_true, 2), data_size )
    x2 = np.random.normal(x2, 0.10 * np.power(z_true, 2), data_size )

    # Bringing data in the right form
    data = np.transpose(np.reshape((x1, x2), (2, data_size)))
    data= np.asarray(data, dtype='float32')
    return data

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

#.......................................................................................................................

training_epochs = 500  #Set to 0, for no training
batch_size = 200
display_step = 5
train_samples = 20000
test_samples = 2000

X_train = generate_data( train_samples )
X_test = generate_data( test_samples )

network_architecture = \
    dict(n_hidden_recog_1=5, # 1st layer encoder neurons
         n_hidden_recog_2=6, # 2nd layer encoder neurons
         n_hidden_gener_1=5, # 1st layer decoder neurons
         n_hidden_gener_2=6, # 2nd layer decoder neurons
         n_input=2, # 2 dimension data input
         n_z=1)  # Dimension of the latent space

autoencoder = vae_example(network_architecture, optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

## training
for epoch in range( training_epochs ):
    avg_cost = 0.
    total_batch = int(train_samples / batch_size)
    # Loop over all batches
    for i in range( total_batch ):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / total_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print "Epochs:", '%04d' % (epoch + 1), \
            "cost=", "{:.9f}".format(avg_cost)

autoencoder.save("model_2d/model2d.ckpt")


# testing
autoencoder.restore("model_2d/model2d.ckpt")

x_mu, x_ls2, z = autoencoder.reconstruct( X_test )
plt.plot(x_mu[:,0], x_mu[:,1], '.')
plt.plot(X_test[:,0], X_test[:,1], '+')
plt.show()
# Sampling from z=-2 to z=2
x_mu, x_ls2 = autoencoder.generate( z )
idx = np.linspace(0, test_samples - 1, 20, dtype='int32')
plt.scatter(x_mu[idx,0], x_mu[idx,1], c=z[idx], s=60000* np.mean(np.exp(x_ls2[idx,:]), axis=1))
plt.show()