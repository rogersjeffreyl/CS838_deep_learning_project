from PIL import Image
from io import BytesIO
import pandas as pd
import os
import base64
import numpy

from scipy.misc import imread, imsave, imresize
def generate_train_data(args):
  arg=args
  print (arg)
  data_file = pd.read_csv(arg, header= None)
  csv_folder, csv_file = os.path.split(arg)
  data_file.columns =["center_cam_image","left_cam_image","right_cam_image",\
                      "angle","forward_throttle","reverse_throttle","speed"]
  center_images =   data_file["center_cam_image"].values
  steering_angle =  data_file["angle"].values

  for index,image_file in enumerate(center_images):                 
     
     #image = Image.open(BytesIO(base64.b64decode(image_file)))
     #image_array = np.asarray(image)
     image_folder, image_file = os.path.split(image_file)
     image_file =os.path.join(csv_folder,"IMG",image_file)
     image = Image.open(image_file)
     width, height = image.size
     image = image.crop((0,77 , width, height))
     #image_array = imread(image_file)
     image_array = numpy.array(image)
     
     yield (image_array,steering_angle[index])
           
        
def parse_data(args):
    
  return generate_train_data(args)


	
