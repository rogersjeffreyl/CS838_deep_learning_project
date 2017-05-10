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
su_model = None

# Remember the previous state for RL training
prev_image_array = None
prev_q_vals = None
prev_steering_angle = None

# Memory used to replay examples for RL training
straight_memory = []
curve_memory = []
fail_memory = []


# RL model learning parameters
alpha = 0.5  # rate as to how much to change the q value
gamma = 0.7  # rate at which to consider future rewards
epsilon = 0.03  # rate at which to consider random decisions
epsilon_decay = 0.995  # decay epsilon so we start taking RL model decisions
min_epsilon = 0.1  # do not decay epsilon below this

batch_size = 1
epochs = 1
episode = 1  # The number of times the model made too large an error
episode_time = 0  # The time that a RL agent stayed on the track without intervention

call_count = 0


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
set_speed = 7
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


def replay(batch_size, memory):
    batches = min(batch_size, len(memory))
    batches = np.random.choice(len(memory), batches)
    for i in batches:
        img, steering_angle, reward, next_img, done = memory[i]
        target = reward
        if not done:
            target = reward + gamma * \
                np.amax(model.predict(next_img[None, :, :, :])[0])
        target_f = model.predict(img[None, :, :, :])
        target_f[0][steering_angle + 25] = target
        try:
            model.fit(img[None, :, :, :], target_f, epochs=1, verbose=1)
        except ValueError as e:
            print(e)

    global epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay


@sio.on('telemetry')
def telemetry(sid, data):
    global model
    global prev_image_array, prev_q_vals, prev_steering_angle
    global episode_time, episode, set_speed, call_count
    try:
        if data:
            # call_count += 1
            # if call_count % 3 != 0:
            #     return
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

            if model is None:
                print('Creating model')
                model = create_model(image_array.shape)

            # print("Getting the angle now!")
            su_angle = su_model.predict(image_array[None, :, :, :], batch_size=1)
            q_values = model.predict(image_array[None, :, :, :], batch_size=1)[0]

            # Use epsilon to choose either random decision or the RL model's decision
            if np.random.rand() <= epsilon:
                print("exploring")
                steering_angle = np.random.randint(-25, 26)
            else:
                steering_angle = np.argmax(q_values) - 25
            #print("got the angle" + str(steering_angle))
            throttle = controller.update(float(speed))

            reward = 10  # Reward for consistently staying on the road this turn
            done = False  # True if we need to correct the model
            # If the deviation from the actual is large penalize the model for incorrect predictions
            print(su_angle[0][0] * 25, steering_angle)
            diff = abs(25 * su_angle[0][0] - steering_angle)
            if diff > 8:
                reward = -100
                done = True
            elif diff > 5:
                reward = -10
            elif diff > 3:
                reward = 0
            else:
                reward = 20

            # Now lets do the RL part
            if prev_image_array is not None:
                #prev_q_vals[prev_steering_angle + 25] += alpha * (reward + gamma*max(q_values) - prev_q_vals[prev_steering_angle + 25])
                if abs(25 * su_angle[0][0]) < 4:
                    straight_memory.append((prev_image_array, prev_steering_angle, reward, image_array, done))
                else:
                    curve_memory.append((prev_image_array, prev_steering_angle, reward, image_array, done))
                if done and epsilon < 0.1:
                    fail_memory.append((prev_image_array, prev_steering_angle, reward, image_array, done))
            prev_image_array = image_array
            prev_steering_angle = steering_angle

            episode_time += 1
            # if len(memory) > 0 and abs(float(data['speed'])) < 0.5 :
            #     replay(32)
            #     controller.set_desired(set_speed)
            #     throttle = controller.update(float(speed))
            if done:
                # print episode number and the time it succeeded
                print("episode: {}, time: {}"
                      .format(episode, episode_time))
                episode += 1
                episode_time = 0
                # send_control(0, 0)
                # if episode % 100 == 0:
                #     model.save("rl_cnn.h5")
                # Train the model on a minibatch
                if len(straight_memory) > 0 and episode % 1000 == 0:
                    for i in range(0, 25):
                        print("training " + str(i))
                        replay(50, straight_memory)
                        replay(50, curve_memory)
                        if epsilon < 0.1:
                            replay(30, fail_memory)
                    model.save("rl_cnn_try2.h5")
                # callbacks = []
                # callbacks.append(keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.0001, patience=10, verbose=1, mode='auto'))
                # callbacks.append(keras.callbacks.ModelCheckpoint("./cnn_best_model", monitor='acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10))
                # callbacks.append(keras.callbacks.ProgbarLogger(count_mode="samples"))
                # if keras.backend.backend() == "tensorflow":
                #     callbacks.append(keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True))
                # print("TRAINING NOW")
                # send_control(0, 0)
                # model.fit(np.array(memory), np.array(training_labels), batch_size=int(batch_size), epochs=int(epochs), verbose=1, callbacks=callbacks)
                # print("TRAINING COMPLETE")
                # model.save("rl_cnn.h5")
                # print(su_angle[0][0], throttle)
                # send_control(su_angle[0][0], throttle)
                send_control(su_angle[0][0], throttle)
            else:
                #print(steering_angle, throttle)
                send_control(steering_angle/25.0, throttle)

            # save frame
            # if args.image_folder != '':
            #     timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            #     image_filename = os.path.join(args.image_folder, timestamp)
            #     image.save('{}.jpg'.format(image_filename))
        else:
            # NOTE: DON'T EDIT THIS.
            print("MANUAL")
            sio.emit('manual', data={}, skip_sid=True)
    except Exception as e:
        print(e)
        print("GOD SAVE MY SORRY SOUL!")
        raise


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
        'su_model',
        type=str,
        help='Path to supervised model h5 file. Model should be on the same path.'
    )
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

    f = h5py.File(args.su_model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    su_model = load_model(args.su_model)

    if args.model:
        # check that model Keras version is same as local Keras version
        f = h5py.File(args.model, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)
        print('loading model')
        model = load_model(args.model)
        f.close()
    #model = create_model([83, 320, 3])

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
