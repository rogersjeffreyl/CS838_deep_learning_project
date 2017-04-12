import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
import data_parser as dp
import argparse
import numpy
import glob
import os
from os.path import dirname
import fnmatch

def create_model(input_shape):
    model = Sequential()

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='elu', input_shape=input_shape))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1, activation='elu'))

    model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    return model

def create_training_array(train_data):
    x_train_arr = []
    y_train_arr = []
    count = 0
    for value in dp.parse_data(train_data):
        x_train_arr.append(value[0])
        y_train_arr.append(value[1])
        count+=1

    x_train = numpy.array(x_train_arr)
    y_train = numpy.array(y_train_arr)

    return (x_train, y_train)

def main():
    parser = argparse.ArgumentParser(description='Training Data Folder')
    parser.add_argument(
        '-train_data_folder',
        type=str,
        help='Path to the train data folder'
    )

    args = parser.parse_args()
    model_count=1
    for root, dirnames, filenames in os.walk(args.train_data_folder):
        for filename in fnmatch.filter(filenames, '*.csv'):
            file = os.path.join(root, filename)
            (x_train, y_train) = create_training_array(file)
            print (dirname(file)+"/cnn" + str(model_count)+".h5")
            model = create_model(x_train[0].shape)

            model.fit(x_train, y_train, batch_size=64, epochs = 100, verbose = 1)
            model.save(dirname(file)+"/cnn" + str(model_count)+".h5")
            model_count+=1
main()