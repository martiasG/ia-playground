from utils import *
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import argparse
import sys

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 32, print_cost = True, keep_prob=1):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 28, 28, 1)
    Y_train -- test set, of shape (None, n_y = 10)
    X_test -- training set, of shape (None, 28, 28, 1)
    Y_test -- test set, of shape (None, n_y = 10)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []                                        # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    # Initialize parameters
    parameters = initialize_parameters()
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters, keep_prob)
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    #Afther variable initialization
    saver = tf.train.Saver(tf.global_variables())
    # for global variables
    saver_global = tf.train.Saver(var_list=tf.global_variables())
    # Start the session to compute the tensorflow graph
    #Config so it wont allocate all the vram at start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train})
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        #save parameters to latter use when making predictions
        saver.save(sess, './params/model_'+str(getNext()))
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', help='learning rate for the algorithm', default=0.001)
    parser.add_argument('--batch_size', help='size of mini batches', default=32)
    parser.add_argument('--num_epochs', help='iteration number', default=1500)
    parser.add_argument('--predict_image_class', help='predict for image class using the pre trainned weights')
    parser.add_argument('--parameters', help='path of the parameters to use', default='model_0')
    parser.add_argument('--train_size', help='The size of the trainning set', default=0)
    parser.add_argument('--test_size', help='The size of the test set', default=0)
    parser.add_argument('--keep_prob', help='probability of keeping the neuron in the dropout method', default=1)
    parser.add_argument('--predict_all_with_params', help='probability of keeping the neuron in the dropout method')
    args = parser.parse_args()

    predict_all_with_params = args.predict_all_with_params
    image_path = args.predict_image_class
    parameters=args.parameters
    if image_path:
        predict_class(image_path, parameters)
        return

    if predict_all_with_params:
        predictAll(predict_all_with_params)
        return

    train_set_size = int(args.train_size)
    test_set_size = int(args.test_size)
    num_epochs = int(args.num_epochs)
    learning_rate = float(args.learning_rate)
    keep_prob = float(args.keep_prob)

    X_train, Y_train, X_test, Y_test = init_dataset_normalize(train_set_size, test_set_size)
    model(X_train, Y_train, X_test, Y_test, num_epochs=num_epochs, learning_rate=learning_rate, keep_prob=keep_prob)

main()
