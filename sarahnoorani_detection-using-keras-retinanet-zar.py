import pandas as pd
# show images inline
%matplotlib inline

import keras
import tensorflow

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf


#Clone Git Repository
!git clone https://github.com/fizyr/keras-retinanet.git
%cd keras-retinanet/
!python setup.py build_ext --inplace

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet import models


df_train = pd.read_csv("/kaggle/input/tensorflow-great-barrier-reef/train.csv")

df_train=df_train.loc[df_train["annotations"].astype(str) != "[]"]
df_train['annotations'] = df_train['annotations'].apply(eval)

df_train['image_path'] = "/kaggle/input/tensorflow-great-barrier-reef/train_images/video_" + df_train['video_id'].astype(str) + "/" + df_train['video_frame'].astype(str) + ".jpg"
df_extrain=df_train.explode('annotations') # Single annotation per row
df_extrain.reset_index(inplace=True)
df_extrain.head()


df_extrain_main=pd.DataFrame(pd.json_normalize(df_extrain['annotations']), columns=['x', 'y', 'width', 'height']).join(df_extrain)
df_extrain_main['class']='Fish'
df_extrain_main=df_extrain_main[['image_path','x','y','width','height','class','video_id','video_frame']]
df_extrain_main.head(10)


!pip install --upgrade git+https://github.com/broadinstitute/keras-resnet
import keras
import keras_resnet
import urllib.request
PRETRAINED_MODEL = './snapshots/_pretrained_model.h5'
#### OPTION 1: DOWNLOAD INITIAL PRETRAINED MODEL FROM FIZYR ####
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, PRETRAINED_MODEL)
print('Downloaded pretrained model to ' + PRETRAINED_MODEL)



def create_tf_example(rowss,data_df):
    """Create a tf.Example entry for a given training image."""
    full_path = os.path.join(rowss.image_path)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    height = image.size[1] # Image height
    width = image.size[0] # Image width
    #print(width,height)
    filename = f'{rowss.video_id}:{rowss.video_frame}'.encode('utf8') # Unique id of the image.
    encoded_image_data = None # Encoded image bytes
    image_format = 'jpeg'.encode('utf8') # b'jpeg' or b'png'

    xmins = [] 
    xmaxs = [] 
    ymins = [] 
    ymaxs = [] 
    
    # Convert ---> [xmin,ymin,width,height] to [xmins,xmaxs,ymins,ymaxs]
    xmin = rowss['x']
    xmax = rowss['x']+rowss['width']
    ymin = rowss['y']
    ymax = rowss['y']+rowss['height']
    

    #main_data.append((rowss['image_path'],xmins,xmaxs,ymins,ymaxs))
    return rowss['image_path'],xmin,ymin,xmax,ymax


import tensorflow as tf
import contextlib2
import io
import IPython
import json
import numpy as np
import os
import pathlib
import pandas as pd
import sys
import tensorflow as tf
import time

tf_example1=[]

from PIL import Image, ImageDraw
for index, row in df_extrain_main.iterrows():
            if index % 1000 == 0:
                print('Processed {0} images.'.format(index))
            image_path,xmins,ymins,xmaxs,ymaxs=create_tf_example(row,df_extrain_main)
            #print(image_path,xmins,xmaxs,ymins,ymaxs)
            df_extrain_main.loc[index,'image_path']=image_path
            df_extrain_main.loc[index,'x']=xmins
            df_extrain_main.loc[index,'y']=ymins
            df_extrain_main.loc[index,'width']=xmaxs
            df_extrain_main.loc[index,'height']=ymaxs



classes=pd.DataFrame([{'class':'Fish','label':0}])
classes.to_csv("classes.csv",index=False,header=False)  # This CSV will be use in training

df_extrain_main['class']='Fish'
df_extrain_main[['image_path','x','y','width','height','class']].to_csv("annotation.csv",index=False,header=False)


!keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights {PRETRAINED_MODEL} --batch-size 1 --steps 100 --epochs 5 csv annotation.csv classes.csv


model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
#print(model_path)

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')  ## Use backbone as resnet50
model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = pd.read_csv('classes.csv',header=None).T.loc[0].to_dict()


THRES_SCORE = 0.4  # Set Score Threshold Value

def df_plot_orinal(drawOG,img_path,df):
    df=df[df['image_path']==img_path]
    for i,r in df.iterrows():
        cv2.rectangle(drawOG, (r['x'], r['y']), (r['width'], r['height']), (255,0,0),2)
    

def img_inference(img_path):
  image = read_image_bgr(img_path)

  # copy to draw on
  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
  drawOG = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image)

  # process image
  start = time.time()
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  df_plot_orinal(drawOG,img_path,df_extrain_main)
  # correct for image scale
  boxes /= scale
  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      #print(score)
      if score < THRES_SCORE:
          continue
      color = label_color(label)
      b = box.astype(int)
      draw_box(draw, b, color=color)
      caption = "{} {:.3f}%".format(labels_to_names[label], score*100)
    
  fig = plt.figure(figsize=(20, 20))
  ax1=fig.add_subplot(1, 2, 1)
  plt.imshow(draw)
  ax2=fig.add_subplot(1, 2, 2)
  plt.imshow(drawOG)

  ax1.title.set_text('Predicted')
  ax2.title.set_text('Actual')
  plt.show()


data=df_extrain_main.sample(n=5)  #Predict on Random 5 Image
for i,r in data.iterrows():
    img_inference(r['image_path'])


# Import the library that is used to submit the prediction result.
import sys
INPUT_DIR = '/kaggle/input/tensorflow-great-barrier-reef/greatbarrierreef/'
sys.path.insert(0, INPUT_DIR)
import greatbarrierreef
env = greatbarrierreef.make_env()   # initialize the environment
iter_test = env.iter_test()


submission_dict = {
    'id': [],
    'prediction_string': [],
}

for (image, sample_prediction_df) in iter_test:
  print(image.shape,sample_prediction_df)

  image = preprocess_image(image)
  image, scale = resize_image(image)

  # process image
  start = time.time()
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  predictions=[]
  # correct for image scale
  boxes /= scale
  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < THRES_SCORE:
          continue
      x_min = int(box[0])  
      y_min = int(box[1])
      x_max = int(box[2])
      y_max = int(box[3])

      bbox_width = x_max - x_min
      bbox_height = y_max - y_min
      predictions.append('{:.2f} {} {} {} {}'.format(score, x_min, y_min, bbox_width, bbox_height))

  prediction_str = ' '.join(predictions)
  sample_prediction_df['annotations'] = prediction_str
  env.predict(sample_prediction_df)
  print('Prediction:', prediction_str)

#my_submission = pd.DataFrame(sample_prediction_df)
# you could use any filename. We choose submission here
#sample_prediction_df.to_csv('submission.csv', index=False)


df_train = pd.read_csv("/kaggle/input/smkfire")

df_train=df_train.loc[df_train["annotations"].astype(str) != "[]"]
df_train['annotations'] = df_train['annotations'].apply(eval)

# df_train['image_path'] = "/kaggle/input/tensorflow-great-barrier-reef/train_images/video_" + df_train['video_id'].astype(str) + "/" + df_train['video_frame'].astype(str) + ".jpg"
# df_extrain=df_train.explode('annotations') # Single annotation per row
# df_extrain.reset_index(inplace=True)
# df_extrain.head()




