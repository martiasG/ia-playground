import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
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
    X = Flatten()(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(inputs=X_train_input, outputs=X)

    return model

def main():
    X_train, Y_train, X_test, Y_test = init_dataset_normalize()
    model = ModelFashionMnis((28,28,1))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=X_train,y=Y_train, epochs=1000)
    model.evaluate(x=X_test, y=Y_test)


main()
