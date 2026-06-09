import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score , ConfusionMatrixDisplay
from skimage import color, feature, exposure
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize


# load the data
data_train = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')
data_train.head()


# make the datset to an array of images of pixels 
image_array =[]
for i, row in enumerate(data_train.index):
        image = np.fromstring(data_train.loc[row, 'pixels'], dtype=int, sep=' ')
        image_array.append(image.flatten())


image_array


image_array[0].shape


lables = np.array(data_train['emotion']).tolist()
lables


flat_images = np.array(image_array)
target = np.array(lables)

# normalization
flat_images = flat_images / 255


for i in range(5):
    plt.figure(figsize=(1, 2))
    plt.imshow(image_array[i].reshape(48,48), cmap=plt.cm.gray)
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    plt.show()


df = pd.DataFrame(flat_images)
df['target']=target


df


df= df[df['target'] != 5]
df= df[df['target'] != 6]


# names of classes
df['target'].unique()


count0= len(df[df['target'] == 0])
count1= len(df[df['target'] == 1])
count2= len(df[df['target'] == 2])
count3= len(df[df['target'] == 3])
count4= len(df[df['target'] == 4])
print('number of Angry images: ',count0)
print('number of Disgust images: ',count1)
print('number of Fear images: ',count2)
print('number of Happy images: ',count3)
print('number of Sad images: ',count4)


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


def extract_hog_features(image):
    gray_image = image
    # Calculate HOG features
    hog_features, hog_image = feature.hog(gray_image, visualize=True)
    
    # Enhance the contrast of the HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features , hog_image_rescaled


hog_features_list = []
hog_images=[]
for index, row in X.iterrows():
    image_pixels = row.values.reshape(48, 48) 
    hog_features ,hog_image = extract_hog_features(image_pixels)
    hog_features_list.append(hog_features)
    hog_images.append(hog_image)

hog_features_array = np.array(hog_features_list)


for i in range(2,7):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2,1), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image_array[i].reshape(48,48), cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.imshow(hog_images[i], cmap=plt.cm.gray)
    plt.show()

