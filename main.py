import tensorflow as tf
import numpy as np
import math
from get_csv_data import HandleData

def corrupt(x):
    r = tf.add(x, tf.cast(tf.random_uniform(shape=tf.shape(x),minval=0,maxval=0.1,dtype=tf.float32), tf.float32))
    # r = tf.multiply(x,tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=0.1, dtype=tf.float32), tf.float32))
    return r

def get_predicted_angle(pred_class):
    return "angle = " + str(pred_class*45)

def autoencoder(dimensions=[784, 512, 256, 64]):

    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)  # artificially corrupting the input signal
    noise_input = current_input
    # Build the encoder
    print("========= encoder begin ==========")
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        print("encoder : ", "n_layer",layer_i, "n_output",n_output, "n_input",n_input)
        W = tf.Variable(tf.random_uniform([n_input, n_output],-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    print("========= encoder end =========")
    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    print("========= decoder begin ==========")
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        print("decoder : ", "n_layer", layer_i,"n_output", n_output)
        W = tf.transpose(encoder[layer_i]) #  transpose of the weights
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    print("========= decoder end =========")
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    return {
                'x': x,
                'z': z,
                'y': y,
                'corrupt_prob': corrupt_prob,
                'cost': cost,
                'noise_input' : noise_input
           }

def getDAE(antenna_data=[]):

    ################ AutoEncoder ##############
    ae = autoencoder(dimensions=[4, 200])
    ###########################################

    ################ Training #################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ########### restore ###########
    saver_restore = tf.train.import_meta_graph('./DAE_save/DenoisingAE_save_noise_add.meta')
    saver_restore.restore(sess, tf.train.latest_checkpoint('./DAE_save/'))
    ###############################

    ################ Testing trained data #####
    return_list = []
    for data in antenna_data:
        antenna_data_mean = np.mean(data, axis=0)
        test_xs_norm = np.array([img - antenna_data_mean for img in data])
        a,b,output_y = sess.run([ae['cost'],ae['noise_input'],ae['y']], feed_dict={ae['x']: test_xs_norm, ae['corrupt_prob']: [1.0]})
        print("DEA avarage cost : ", a)
        return_list.append(output_y)
    tf.reset_default_graph()
    return return_list
    ###########################################

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'],name="DNN1")
    layer_1 = tf.nn.relu(layer_1,name="DNN2")
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'],name="DNN3")
    layer_2 = tf.nn.relu(layer_2,name="DNN4")
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out'],name="DNN5") + biases['out']
    return out_layer

if __name__ == "__main__":
    data = HandleData(total_data=880,data_per_angle=110,num_angles=8)
    antenna_data,label_data = data.get_synthatic_data(test_data=False)

    data_test = HandleData(total_data=80, data_per_angle=10, num_angles=8)
    antenna_data_test, label_data_test = data_test.get_synthatic_data(test_data=True)

    DAE_out = getDAE([antenna_data,antenna_data_test])  # get denoising autoencoder outputs for the train and test data

    data.data_set = DAE_out[0]
    antenna_data = DAE_out[0]

    antenna_data_test = DAE_out[1]
    data_test.data_set = DAE_out[1]

    TRAIN=False

    # Parameters
    learning_rate = 0.0001
    training_epochs = 2000
    batch_size = 5
    display_step = 1
    # Network Parameters
    n_hidden_1 = 12 # 1st layer number of features
    n_hidden_2 = 12 # 2nd layer number of features
    n_input = 4 # antenna_1,antenna_2,antenna_3,antenna_4
    n_classes = 8 # 0,45,90,135,180,225,270,315

    # tf Graph input
    x = tf.placeholder("float", [None, n_input],name='DNN_x')
    y = tf.placeholder("float", [None, n_classes],name='DNN_y')

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='DNN_w1'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='DNN_w2'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name='DNN_w3')
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]),name='DNN_b1'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]),name='DNN_b2'),
        'out': tf.Variable(tf.random_normal([n_classes]),name='DNN_b3')
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name="DNN_cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='DNN_optimizer').minimize(cost)

    # Initializing the Graph
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        if TRAIN:
            ############### Training #################
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(data.total_data/batch_size)
                for i in range(total_batch):
                    batch_x, batch_y = data.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
            print("Optimization Finished!")
            ##########################################

            ########## save ###########
            saver.save(sess, './DAEandDNN_save/DAEandDNN_save')
            ###########################
        else:
            ########### restore ###########
            saver_restore = tf.train.import_meta_graph('./DAEandDNN_save/DAEandDNN_save.meta')
            saver_restore.restore(sess, tf.train.latest_checkpoint('./DAEandDNN_save/'))
            ###############################

        #### Calculate accuracy ###
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Accuracy:", accuracy.eval({x: antenna_data, y: label_data}))

        print("Accuracy:", accuracy.eval({x: antenna_data_test, y: label_data_test}))
