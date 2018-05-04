import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def corrupt(x):
    r = tf.add(x, tf.cast(tf.random_uniform(shape=tf.shape(x),minval=0,maxval=0.1,dtype=tf.float32), tf.float32))
    # r = tf.multiply(x,tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0.5, maxval=1.5, dtype=tf.float32), tf.float32))
    return r

def kl_divergence(p, p_hat):
    # return tf.reduce_mean(p * tf.log(tf.abs(p)) - p * tf.log(tf.abs(p_hat)) + (1 - p) * tf.log(tf.abs(1 - p)) - (1 - p) * tf.log(tf.abs(1 - p_hat)))
    return tf.reduce_mean(p * tf.log(tf.abs(p)) - p * tf.log(tf.abs(p_hat)) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat))

def autoencoder(dimensions=[784, 512, 256, 64]):

    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)
    noise_input = current_input
    # Build the encoder
    print("========= encoder begin ==========")
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        print("encoder : layer_i - n_output - n_input",layer_i,n_output,n_input)
        W = tf.Variable(tf.random_uniform([n_input, n_output],-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    print("========= encoder finish =========")
    # latent representation
    encoder_out = current_input
    print(encoder_out.shape)
    encoder.reverse()
    # Build the decoder using the same weights
    print("========= decoder begin ==========")
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        print("decoder : layer_i - n_output", layer_i, n_output)
        W = tf.transpose(encoder[layer_i]) #  transpose of the weights
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    print("========= decoder finish =========")
    # now have the reconstruction through the network
    reconstruction = current_input
    # kl = tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=z/0.01))

    p_hat = tf.reduce_mean(encoder_out,0)
    p = np.repeat([-0.05], 200).astype(np.float32)
    dummy = np.repeat([1], 200).astype(np.float32)
    kl = kl_divergence(p_hat,p)

    cost = tf.reduce_mean(tf.square(reconstruction - x)) + 0.01*kl
    # cost = 0.5 * tf.reduce_sum(tf.square(y - x))
    return {
                'x': x,
                'encoder_out': encoder_out,
                'reconstruction': reconstruction,
                'corrupt_prob': corrupt_prob,
                'cost': cost,
                'noise_input' : noise_input,
                'kl' : kl
           }


def train_DOA():
    from get_csv_data import HandleData
    import csv

    ################ TEST DATA ################
    data = HandleData(total_data=880, data_per_angle=110, num_angles=8)
    antenna_data, label_data = data.get_synthatic_data(test_data=False)
    antenna_data_mean = np.mean(antenna_data, axis=0)
    ###########################################

    ################ learning parameters ######
    learning_rate = 0.001
    batch_size = 20
    n_epochs = 1000
    ###########################################

    ################ AutoEncoder ##############
    ae = autoencoder(dimensions=[4, 200])
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    ###########################################

    ################ Training #################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ########### restore ###########
    # saver_restore = tf.train.import_meta_graph('./DAE_save/DenoisingAE_save_noise_add.meta')
    # saver_restore = tf.train.import_meta_graph('DenoisingAE_save_noise_multiply.meta')
    # saver_restore.restore(sess, tf.train.latest_checkpoint('./DAE_save/'))
    ###############################
    train=0
    for epoch_i in range(n_epochs):
        for batch_i in range(data.total_data//batch_size):
            batch_xs, _ = data.next_batch(batch_size)
            train = np.array([img - antenna_data_mean for img in batch_xs])
            # print(train.shape)
            sess.run(optimizer, feed_dict={ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i,sess.run([ae['cost'],ae['kl']], feed_dict={ae['x']: train, ae['corrupt_prob']: [1.0]}))

        ##### debug kl ######
        # tmp=sess.run(ae['encoder_out'], feed_dict={ae['x']: train, ae['corrupt_prob']: [1.0]})
        # p_hat = tf.reduce_mean(tmp, 0)
        # p = np.repeat([-0.05], 200).astype(np.float32)
        # dummy = np.repeat([1], 200).astype(np.float32)
        # p_hat = p_hat+dummy
        # p = p+dummy
        # kl_tmp = p * tf.log(tf.abs(p)) - p * tf.log(tf.abs(p_hat)) + (1 - p) * tf.log(p-1) - (1 - p) * tf.log(p_hat-1)
        # print(sess.run( p_hat ))
        # ######################



    ###########################################
    saver.save(sess, './DAE_save/DenoisingAE_save_noise_add')
    ############### Test Data ################
    data_test = HandleData(total_data=80, data_per_angle=10, num_angles=8)
    antenna_data_test, label_data_test = data_test.get_synthatic_data(test_data=True)
    antenna_data_test_mean = np.mean(antenna_data_test, axis=0)
    ###########################################

    ################ Testing trained data #####
    test_xs_norm = np.array([img - antenna_data_test_mean for img in antenna_data])
    a,b,output_y = sess.run([ae['cost'],ae['noise_input'],ae['reconstruction']], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [1.0]})
    print("Testing trained data avarage cost : ", a)
    ###########################################

    ################ Testing ##################
    test_xs, _ = data_test.next_batch(80)
    test_xs_norm = np.array([img - antenna_data_test_mean for img in test_xs])
    a,b,output_y = sess.run([ae['cost'],ae['noise_input'],ae['reconstruction']], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [1.0]})
    print("avarage cost : ", a)
    for i in range(len(output_y)):
        comp = output_y[i]
        orgi = test_xs[i]
        noise = b[i]
        comp += antenna_data_test_mean
        noise += antenna_data_test_mean
        plt.subplot(8, 10, i + 1)
        plt.plot(comp,color='blue',label='rcon')
        plt.plot(orgi,color='green',label='orgi')
        plt.plot(noise,color='red',label='noise')
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.legend(loc='upper left')
    plt.show()
    print("difference between noise and origial :")
    # print(b-test_xs_norm)
    #############################################

    ################ Test Data ################
    data_test_noise = HandleData(total_data=120, data_per_angle=120, num_angles=8)
    antenna_data_test, label_data_test = data_test_noise.get_synthatic_data(test_data=-1)
    antenna_data_test_mean = np.mean(antenna_data_test, axis=0)
    ###########################################

    ################ Testing ##################
    test_xs, _ = data_test_noise.next_batch(120)
    test_xs_norm = np.array([img - antenna_data_test_mean for img in test_xs])
    a,b,output_y = sess.run([ae['cost'],ae['noise_input'],ae['reconstruction']], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [1.0]})
    print("avarage cost : ", a)
    for i in range(len(output_y)):
        comp = output_y[i]
        orgi = test_xs[i]
        noise = b[i]
        comp += antenna_data_test_mean
        noise += antenna_data_test_mean
        plt.subplot(10, 12, i + 1)
        plt.plot(comp,color='blue',label='rcon')
        plt.plot(orgi,color='green',label='orgi')
        plt.plot(noise,color='red',label='noise')
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    #############################################


train_DOA()