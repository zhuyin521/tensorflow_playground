import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class vae_example( object ):

    def __init__(self, network_architecture, optimizer = tf.train.AdamOptimizer()):

        self.net_arch = network_architecture

        network_weights = self._initialize_weights()
        self.weights = network_weights

        ## model
        # encoder
        self.x = tf.placeholder(tf.float32, [None, self.net_arch['n_input']])  # batch_size * N_input
        self.h1_encoder = tf.nn.softplus(tf.matmul( self.x, self.weights['W_I_en1']) 
                                         + self.weights['b_I_en1'])  # batch_size * n_hidden_recog_1
        self.h2_encoder = tf.nn.softplus(tf.matmul( self.h1_encoder , self.weights['W_en1_en2']) 
                                         + self.weights['b_en1_en2'] ) # batch_size * n_hidden_recog_2

        # Parameters for the Gaussian
        self.z_mu = tf.add( tf.matmul( self.h2_encoder, self.weights['W_en2_mu']), 
                            self.weights['b_en2_mu'] ) # batch_size * n_z
        self.z_log_sigma_sq = tf.add(tf.matmul(self.h2_encoder, self.weights['W_en2_sig']),
                                     self.weights['b_en2_sig'] ) # batch_size * n_z

        # sample from gaussian distribution
        eps = tf.random_normal(tf.pack([tf.shape(self.x)[0], self.net_arch['n_z']]), 
                               0, 1, dtype=tf.float32)  # Adding a random number  batch_size * n_z
        self.z = tf.add( self.z_mu, 
                         tf.mul(tf.sqrt(tf.exp( self.z_log_sigma_sq )), eps))  # The sampled z   batch_size * n_z

        # decoder
        self.h1_decoder = tf.nn.softplus(tf.matmul( self.z , self.weights['W_z_de1']) 
                                         + self.weights['b_z_de1']) # batch_size * n_hidden_gener_1
        self.h2_decoder = tf.nn.softplus(tf.matmul(self.h1_decoder, self.weights['W_de1_de2']) 
                                         + self.weights['b_de1_de2']) # batch_size * n_hidden_gener_2

        self.x_mu = tf.add(tf.matmul( self.h2_decoder , self.weights['W_de2_mu']), 
                           self.weights['b_de2_mu']) # batch_size * n_input
        self.x_log_sigma_sq = tf.add(tf.matmul( self.h2_decoder, self.weights['W_de2_sig']), 
                                     self.weights['b_de2_sig']) # batch_size * n_input


        # cost and optimizer
        self.reconstr_loss = tf.reduce_sum(0.5 *self. x_log_sigma_sq 
                                           + (tf.square(self.x - self.x_mu) / (2.0 * tf.exp(self.x_log_sigma_sq))), 1)
        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) 
                                                - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean( self.reconstr_loss + self.latent_loss)  # average over batch
        self.optimizer = optimizer.minimize( self.cost )

        # initialize
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)



    def _initialize_weights(self):
        allweights = dict()
        allweights['W_I_en1'] = tf.Variable( tf.truncated_normal( 
            [self.net_arch['n_input'], self.net_arch['n_hidden_recog_1']], stddev=0.1) )
        allweights['b_I_en1'] = tf.Variable( tf.constant(0.1, shape = [self.net_arch['n_hidden_recog_1']]))
        allweights['W_en1_en2'] = tf.Variable( tf.truncated_normal( 
            [self.net_arch['n_hidden_recog_1'], self.net_arch['n_hidden_recog_2']], stddev=0.1) )
        allweights['b_en1_en2'] = tf.Variable( tf.constant(0.1, shape = [self.net_arch['n_hidden_recog_2']]) )

        allweights['W_en2_mu'] = tf.Variable(tf.truncated_normal(
            [self.net_arch['n_hidden_recog_2'], self.net_arch['n_z']], stddev=0.1))
        allweights['b_en2_mu'] = tf.Variable(tf.constant(0.1, shape = [self.net_arch['n_z']]))
        allweights['W_en2_sig'] = tf.Variable(tf.truncated_normal(
            [self.net_arch['n_hidden_recog_2'], self.net_arch['n_z']], stddev=0.1))
        allweights['b_en2_sig'] = tf.Variable(tf.constant(0.1, shape = [self.net_arch['n_z']]))

        #.........................................................................................................................................................
        allweights['W_z_de1'] = tf.Variable( tf.truncated_normal( 
            [self.net_arch['n_z'], self.net_arch['n_hidden_gener_1']], stddev=0.1) )
        allweights['b_z_de1'] = tf.Variable( tf.constant(0.1, shape = [self.net_arch['n_hidden_gener_1']]) )
        allweights['W_de1_de2'] = tf.Variable( tf.truncated_normal( 
            [self.net_arch['n_hidden_gener_1'], self.net_arch['n_hidden_gener_2']], stddev=0.1) )
        allweights['b_de1_de2'] = tf.Variable( tf.constant(0.1, shape = [self.net_arch['n_hidden_gener_2']]) )

        allweights['W_de2_mu'] = tf.Variable( tf.truncated_normal(
            [self.net_arch['n_hidden_gener_2'], self.net_arch['n_input']], stddev=0.1))
        allweights['b_de2_mu'] = tf.Variable( tf.constant(0.1, shape = [self.net_arch['n_input']]))
        allweights['W_de2_sig'] = tf.Variable( tf.truncated_normal(
            [self.net_arch['n_hidden_gener_2'], self.net_arch['n_input']], stddev=0.1))
        allweights['b_de2_sig'] = tf.Variable( tf.constant(0.1, shape = [self.net_arch['n_input']]))

        return allweights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X): # given x, return z_mu
        return self.sess.run(self.z_mu, feed_dict={self.x: X})

    def generate(self, hidden=None): # given z, generate x_hat
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run((self.x_mu, self.x_log_sigma_sq), feed_dict={self.z: hidden})

    def reconstruct(self, X): # given x, reconstruct x_hat
        return self.sess.run( (self.x_mu, self.x_log_sigma_sq, self.z), feed_dict={self.x: X})

    def save(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)  # Saves the weights (not the graph)
        print("Model saved in file: {}".format(save_path))

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
