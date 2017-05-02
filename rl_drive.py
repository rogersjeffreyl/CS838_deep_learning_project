import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import regularizers
from keras.models import load_model

import h5py
from keras import __version__ as keras_version
from pprint import pprint

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
prev_q_vals = None
reward = 10
training_data = []
training_labels = []
alpha = 0.5
batch_size = 10
epochs = 5


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


def create_model(input_shape):
    model = Sequential()

    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape))

    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))

    # model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    # model.add(Dense(10, activation='elu'))
    model.add(Dense(51, activation='linear'))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(lr=1e-4),
                  metrics=['accuracy'])

    return model


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        #image = Image.open(imgString)
        width, height = image.size
        image2 = image.crop((0, 77, width, height))
        image_array = np.asarray(image2)
        # global model
        # if model is None:
        #     model = create_model(image_array.shape)
        #print("Getting the angle now!")
        q_values = model.predict(image_array[None, :, :, :], batch_size=1)[0]
        steering_angle = (np.argmax(q_values) - 25)/25.0
        #print("got the angle" + str(steering_angle))
        throttle = controller.update(float(speed))

        # Now lets do the RL part
        global prev_image_array, prev_q_vals
        if prev_image_array is not None:
            prev_a = np.argmax(prev_q_vals)
            prev_q_vals[prev_a] += alpha * (reward + max(q_values) - prev_q_vals[prev_a])
            training_data.append(prev_image_array)
            training_labels.append(prev_q_vals)
        prev_q_vals = q_values
        prev_image_array = image_array

        if len(training_data) % 250 == 0 and len(training_data) > 0:
            pprint(training_data)
            pprint(training_labels)
            callbacks = []
            callbacks.append(keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto'))
            callbacks.append(keras.callbacks.ModelCheckpoint("./cnn_best_model", monitor='acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10))
            callbacks.append(keras.callbacks.ProgbarLogger(count_mode="samples"))
            if keras.backend.backend() == "tensorflow":
                callbacks.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True))
            print("TRAINING NOW")
            send_control(0, 0)
            model.fit(np.array(training_data), np.array(training_labels), batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks=callbacks)
            print("TRAINING COMPLETE")
            model.save("rl_cnn.h5")
        print(len(training_data))
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        print("MANUAL")
        sio.emit('manual', data={}, skip_sid=True)
        if prev_image_array is not None:
            prev_a = np.argmax(prev_q_vals)
            prev_q_vals[prev_a] += -1 * alpha * reward
            training_data.append(prev_image_array)
            training_labels.append(prev_q_vals)
            prev_image_array = None
            prev_q_vals = None


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default=None,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if args.model:
        # check that model Keras version is same as local Keras version
        f = h5py.File(args.model, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)

        model = load_model(args.model)
    model = create_model([83, 320, 3])

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
