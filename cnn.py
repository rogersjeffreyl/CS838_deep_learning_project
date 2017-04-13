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

    x_train = x_train_arr
    y_train = y_train_arr

    return (x_train, y_train)

def main():
    parser = argparse.ArgumentParser(description='Training Data Folder')
    parser.add_argument(
        '-train_data_folder',
        nargs='+',
        type=str,
        help='Path to the train data folder'
    )
    parser.add_argument(
        '-epochs',        
        type=int,
        help='Num epochs'
    )
    parser.add_argument(
        '-batch_size',
        type=str,
        help='Batch Size'
    )

    args = parser.parse_args()
    model_count=1
    final_x_train=[]
    final_y_train=[]
    for train_data_folder in args.train_data_folder:
	    for root, dirnames, filenames in os.walk(train_data_folder):
	        for filename in fnmatch.filter(filenames, '*.csv'):
	            file = os.path.join(root, filename)
	            (x_train, y_train) = create_training_array(file)
	            final_x_train =final_x_train+x_train
	            final_y_train =final_y_train+y_train
	            #print (dirname(file)+"/cnn" + str(model_count)+".h5")
    final_x_train = numpy.array(final_x_train)            
    final_y_train = numpy.array(final_y_train)            
    model = create_model(final_x_train[0].shape)
    model.fit(final_x_train, final_y_train, batch_size=int(args.batch_size), epochs = int(args.epochs), verbose = 1)
    file_name = "_".join(["cnn", str(args.batch_size),str(args.epochs)])+".h5"
    model.save(file_name) 
    print ("Saving model as {0}".format(file_name))           
    
main()