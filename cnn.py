import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers

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
	input_shape = (200, 200, 3)
	create_model(input_shape)
main()

