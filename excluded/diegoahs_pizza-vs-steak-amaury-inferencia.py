import kagglehub
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import confusion_matrix


df = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak'


modelo = '/kaggle/input/pizza_vs_steak_amaury-2/keras/default/1/pizza_vs_steak_Amaury-2.keras'


im_size = 224


model = models.load_model(modelo)


test_pizza = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak/test/pizza'
test_steak = '/kaggle/input/competencia-02-julio-2025/archive/pizza_steak/test/steak'


pizza = sorted(glob.glob(test_pizza + '/*.jpg'))[:100]
steak = sorted(glob.glob(test_steak + '/*.jpg'))[:100]


final_image = pizza + steak


all_predictions = []
for path in final_image:
    img = tf.keras.utils.load_img(path, target_size = (im_size, im_size))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis = 0)


flipped_img_array = tf.image.flip_left_right(img_array)
flipped_img_array_expanded = np.expand_dims(flipped_img_array, axis=0)


original_pred = model.predict(img_array_expanded, verbose=0)
flipped_pred = model.predict(flipped_img_array_expanded, verbose=0)


avg_pred = (original_pred + flipped_pred) / 2
all_predictions.append(avg_pred)


predictions = np.array(all_predictions).flatten()
predicted_class_indices = (predictions > 0.5).astype("int32").flatten()


class_names = ['pizza', 'steak']
predicted_class_names = [class_names[i] for i in predicted_class_indices]


test_filenames = [path.split('/')[-1] for path in final_image]


#if len(test_filenames) == len(predicted_class_names):
    
    submission = pd.DataFrame({'ID':test_filenames, 'Label':predicted_class_names})


submission.to_csv('/kaggle/working/submission.csv', index = False)




