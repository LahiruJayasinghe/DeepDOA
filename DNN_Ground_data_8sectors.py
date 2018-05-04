from __future__ import print_function
from get_csv_data import HandleData
import numpy as np
import tensorflow as tf

data = HandleData(total_data=880,data_per_angle=110,num_angles=8)
antenna_data,label_data = data.get_synthatic_data(test_data=False)

def get_predicted_angle(pred_class):
    return "angle = " + str(pred_class*45)

def corrupt(x):
    r = tf.add(x, tf.cast(tf.random_uniform(shape=tf.shape(x),minval=0,maxval=0.1,dtype=tf.float32), tf.float32))
    # r = tf.multiply(x,tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=0.1, dtype=tf.float32), tf.float32))
    return r

# Parameters
learning_rate = 0.0001
training_epochs = 1000
batch_size = 5
display_step = 1
# Network Parameters
n_hidden_1 = 12 # 1st layer number of features
n_hidden_2 = 12 # 2nd layer number of features
n_input = 4 # antenna_1,antenna_2,antenna_3,antenna_4
n_classes = 8 # 0,45,90,135,180,225,270,315

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(corrupt(x), weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the Graph
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    ########### restore ###########
    saver_restore = tf.train.import_meta_graph('./DNN_save/DNN_GD8_save.meta')
    saver_restore.restore(sess, tf.train.latest_checkpoint('./DNN_save/'))
    ###############################

    ################ Training #################
    # for epoch in range(training_epochs):
    #     avg_cost = 0.
    #     total_batch = int(data.total_data/batch_size)
    #     for i in range(total_batch):
    #         batch_x, batch_y = data.next_batch(batch_size)
    #         _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
    #         avg_cost += c / total_batch
    #     # Display logs per epoch step
    #     if epoch % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
    # print("Optimization Finished!")
    ###########################################

    ########### save ###########
    # saver.save(sess, './DNN_save/DNN_GD8_save')
    ############################

    #### Calculate accuracy ###
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy:", accuracy.eval({x: antenna_data, y: label_data}))

    data_test = HandleData(total_data=80, data_per_angle=10, num_angles=8)
    antenna_data_test, label_data_test = data_test.get_synthatic_data(test_data=True)
    print("Accuracy:", accuracy.eval({x: antenna_data_test, y: label_data_test}))

    data_test_noise = HandleData(total_data=120, data_per_angle=120, num_angles=8)
    antenna_data_test, label_data_test = data_test_noise.get_synthatic_data(test_data=-1)
    print("Accuracy:", accuracy.eval({x: antenna_data_test, y: label_data_test}))

    # pred_result = sess.run(tf.argmax(pred, 1), feed_dict={x: np.array([[24, 38, 20, 9]])})
    # print(get_predicted_angle(pred_result[0]))
    for i in range(0,8):
        x_i, y_i = data.next_batch(110)
        pred_result = sess.run(tf.argmax(pred, 1), feed_dict={x: x_i, y: y_i})
        # print('angle = ',i*45 ,' ', collections.Counter(pred_result))
        unique, counts = np.unique(pred_result, return_counts=True)
        unique_angles = unique * 45
        percentage = (counts/110)*100
        print('angle = ',i*45 ,' ',dict(zip(unique_angles, percentage)))