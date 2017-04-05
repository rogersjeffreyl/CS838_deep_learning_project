import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
import data_parser as dp
import argparse
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

	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

def main():
    parser = argparse.ArgumentParser(description='Training Data')
    parser.add_argument(
        '-train_data',
        type=str,
        nargs='+',
        help='Path to the train data'
    )
    args = parser.parse_args()
    for value in dp.parse_data(args.train_data):
    	print (value[0].shape)
	#create_model(input_shape)
main()

