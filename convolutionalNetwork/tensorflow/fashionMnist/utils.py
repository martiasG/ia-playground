import math
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd


def init_dataset_normalize(train_set_size, test_set_size):
    """
        this dataset is the shape 5999,786
        in order to use int on a cvnn It needs first to reshape to 59999,28,28,1
        c_n is 1 because is grayscale
    """
    train_df = pd.read_csv('dataset/fashion-mnist_train.csv').sample(n=train_set_size)
    if test_set_size!=0:
        test_df = pd.read_csv('dataset/fashion-mnist_test.csv').sample(n=test_set_size)
    else:
        test_df = pd.read_csv('dataset/fashion-mnist_test.csv')

    Y_train_orig = train_df['label'].values[1:]
    Y_test_orig = test_df['label'].values[1:]

    Y_train = one_hot_matrix(Y_train_orig, 10).T
    Y_test = one_hot_matrix(Y_test_orig, 10).T

    X_train_orig = train_df.T[1:].T.values[1:]
    X_test_orig = test_df.T[1:].T.values[1:]

    X_train_flatten = X_train_orig
    X_test_flatten = X_test_orig

    # Normalize image vectors
    X_train = X_train_flatten/255.
    X_test = X_test_flatten/255.

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return X_train, Y_train, X_test, Y_test

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(dtype=tf.float32, shape=(None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, n_y), name='Y')

    return X, Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8] N_H, N_W, N_Cprev(grayscale so it is 1),N_C
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [3, 3, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

def forward_propagation(X, parameters, keep_prob):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    assert X[0].shape == (28,28,1)
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    assert W1.shape == (3,3,1,8)
    W2 = parameters['W2']
    assert W2.shape == (2,2,8,16)
    # CONV2D: stride of 1, padding 'SAME'
    s=1
    #     [1,s,s,1] one sample at a time and 1 channel over the s in height and w.
    Z1 = tf.nn.conv2d(X, W1, strides=[1,s,s,1], padding='SAME', name='Z1')
    # RELU
    A1 = tf.nn.relu(Z1, name='A1')
    A1dropout = tf.nn.dropout(A1, keep_prob)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    assert A1dropout[0].shape == (28, 28, 8)
    # 3 instead of 8 because divides 27
    f=3
    s=3
    P1 = tf.nn.max_pool(A1dropout, ksize=[1,f,f,1], strides=[1,s,s,1], padding='SAME', name='P1')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    assert P1[0].shape == (10, 10, 8)
    s=1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,s,s,1], padding='SAME', name='Z2')
    # RELU
    A2 = tf.nn.relu(Z2, name='A2')
    A2dropout = tf.nn.dropout(A2, keep_prob)
    assert A2[0].shape == (10, 10, 16)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    f=3
    s=3
    P2 = tf.nn.max_pool(A2dropout, ksize=[1,f,f,1],strides=[1,s,s,1], padding='SAME', name='P2')
    assert P2[0].shape == (4, 4, 16)
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    assert P2[0].shape == 256
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=10, trainable=True, activation_fn=None)
    assert Z3[0].shape==10

    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

def visualizeImage(X_train_orig, index):
    plt.imshow(X_train_orig.T[index].reshape(28,28), cmap='gray')
    plt.show()

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.
    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot

def predict_class(image_path, parameters_path):
    """
    0 T-shirt/top
    1 Trouser
    2 Pullover
    3 Dress
    4 Coat
    5 Sandal
    6 Shirt
    7 Sneaker
    8 Bag
    9 Ankle boot
    """
    import scipy
    from PIL import Image
    from scipy import ndimage

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    original_image = np.array(ndimage.imread(image_path, flatten=False))
    plt.imshow(original_image)
    plt.show()

    graycolor_image = np.array(ndimage.imread(image_path, flatten=True))
    plt.imshow(graycolor_image, cmap='gray')
    plt.show()

    image_flattern = scipy.misc.imresize(graycolor_image, size=(28,28)).reshape((1, 28*28)).T
    print('SHAPE FLATTEN: ', image_flattern.shape)

    recover_image = image_flattern.reshape(1, 28, 28, 1)
    # plt.imshow(recover_image, cmap='gray')
    # plt.show()

    my_image_prediction = predict(recover_image, parameters_path)

    print("Your algorithm predicts: [" + classes[np.squeeze(my_image_prediction)]+']')

def predictAll(parameters_path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir('./testimages') if isfile(join('./testimages', f))]

    for file in onlyfiles:
        print('Image is: ['+ file + ']')
        predict_class('./testimages/'+file, parameters_path)
        print('============================================')

def predict(image_input, parameters_path):
    tf.reset_default_graph()

    parameters = initialize_parameters()
    loader = tf.train.import_meta_graph('./params/'+parameters_path+'.meta')

    X = tf.placeholder("float", [None, image_input.shape[1], image_input.shape[1], 1], name='X')

    Z3 = forward_propagation(X, parameters, 1)
    predict_op = tf.argmax(Z3, 1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        loader = loader.restore(sess, './params/'+parameters_path)
        sess.run(tf.global_variables_initializer())
        prediction = sess.run(predict_op, feed_dict = {X: image_input})

        return prediction
"""
UTILS TO SAVE files
"""
def getNext():
    import json

    with open('sequence.json', 'r') as f:
        sequence_load = json.load(f)

    next = int(sequence_load['next'])

    sequence_next = {'next':next+1}
    with open('./sequence.json', 'w') as f:
        json.dump(sequence_next, f)

    return next
