import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
# IMPORT MODEL
from keras.models import load_model
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from kt_utils import *
from utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow

def ModelFashionMnis(input_shape):
    """
    Implementation of the fashionMnistmodel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    X_train_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_train_input)
    X = Conv2D(32, (7, 7),strides=(1, 1))(X_train_input)
    # normalize the third component colors
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = ZeroPadding2D((2, 2))(X_train_input)
    X = Conv2D(64, (5, 5),strides=(1, 1))(X_train_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = ZeroPadding2D((1, 1))(X_train_input)
    X = Conv2D(128, (3, 3),strides=(1, 1))(X_train_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Flatten()(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(inputs=X_train_input, outputs=X)

    return model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_image_class', help='predict for image class using the pre trainned weights')
    parser.add_argument('--parameters', help='path of the parameters to use', default='params_model_0.h5')
    parser.add_argument('--num_epochs', help='iteration number', default=1500)
    parser.add_argument('--predict_all_with_params', help='probability of keeping the neuron in the dropout method')
    args = parser.parse_args()

    num_epochs = int(args.num_epochs)
    image_path = args.predict_image_class
    parameters = args.parameters
    if image_path:
        model = load_model(parameters)
        image = load_image(image_path)
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        my_image_prediction = model.predict(x=image)
        print("Your algorithm predicts: [" + classes[np.squeeze(np.argmax(my_image_prediction))]+']')
        return

    X_train, Y_train, X_test, Y_test = init_dataset_normalize()
    model = ModelFashionMnis((28,28,1))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train,y=Y_train, epochs=num_epochs)
    model.evaluate(x=X_test, y=Y_test)
    model.save('params_model_'+str(getNext())+'.h5')
    model.summary()

main()
