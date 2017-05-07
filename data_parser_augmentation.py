from PIL import Image
from io import BytesIO
import pandas as pd
import os
import base64
import numpy
import ntpath
import random
from scipy.misc import imread, imsave, imresize
import cv2

def randomly_select_image(angle, left_image, center_image, right_image):
  x = random.random()

  if x < 0.33:
    return left_image, (angle + 0.25)

  elif x >=0.33 and x < 0.66:
    return center_image, angle

  return right_image, (angle - 0.25)

def translate_image(img, angle):
  x = 100 * (random.random() - 0.5)
  y = 10 * (random.random() - 0.5)

  new_img_arr = numpy.float32([[1, 0, x], [0, 1, y]])
  height, width = img.shape[:2]
  img = cv2.warpAffine(img, new_img_arr, (width, height))
  angle = angle + (x * 0.0025)
  return img, angle

def flip_image(img, angle):
  x= random.random()
  if x >=0.5:
    img = cv2.flip(img, 1)
    angle = -1 * angle
  return img, angle

def generate_train_data(args):
  arg=args
  print (arg)
  data_file = pd.read_csv(arg, header= None)
  csv_folder, csv_file = os.path.split(arg)
  data_file.columns =["center_cam_image","left_cam_image","right_cam_image",\
                      "angle","forward_throttle","reverse_throttle","speed"]
  center_images =   data_file["center_cam_image"].values
  left_images = data_file["left_cam_image"].values
  right_images = data_file["right_cam_image"].values
  steering_angle =  data_file["angle"].values

  for index,image_file in enumerate(center_images):                 
     
     #image = Image.open(BytesIO(base64.b64decode(image_file)))
     #image_array = np.asarray(image)
     left_image = left_images[index]
     right_image = right_images[index]
     
     image_folder, image_file = ntpath.split(image_file)
     image_file =os.path.join(csv_folder,"IMG",image_file)
     image = Image.open(image_file)
     width, height = image.size
     image = image.crop((0,77 , width, height))
     #image_array = imread(image_file)
     image_array = numpy.array(image)
     yield (image_array,steering_angle[index])

     augemented_image_file, augmented_angle = randomly_select_image(steering_angle[index], left_image, image_file, right_image)
     image_folder, augemented_image_file = ntpath.split(augemented_image_file)
     augemented_image_file =os.path.join(csv_folder,"IMG",augemented_image_file)
     augemented_image_file = Image.open(augemented_image_file)
     width, height = augemented_image_file.size
     augemented_image_file = augemented_image_file.crop((0,77 , width, height))
     image_array = numpy.array(augemented_image_file)
     image_array, augmented_angle = translate_image(image_array, augmented_angle)
     image_array, augmented_angle = flip_image(image_array, augmented_angle)
     yield (image_array,augmented_angle)
        
def parse_data(args):
    
  return generate_train_data(args)


	
