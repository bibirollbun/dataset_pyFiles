import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sbn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class DataProcessing:
    def __init__(self):
        # self.read_path = read_path
        # self.write_path = write_path
        pass

    def readData(self, read_path:str) -> pd.DataFrame:
        df = pd.read_csv(read_path)
        return df

    def writeData(self, df:pd.DataFrame, write_path:str) -> None:
        df.to_csv(write_path, index=False)


    def dataLoader(self, data_labels:pd.DataFrame, 
                   folder_path:str) -> tuple:
        train_lis = []
        test_lis = []
        for file in os.listdir(folder_path):
            try:
                if file in list(test_labels['id']):
                    test_lis.append(np.load(os.path.join(folder_path, file)))
                else:
                    train_lis.append(np.load(os.path.join(folder_path, file)))
            except:
                print(file, " Missed")
        return train_lis, test_lis

    def bandsPCA(self, image:np.ndarray) -> list:
        h, w, c = image.shape
        pca = PCA(n_components=3)
        best_comps = pca.fit_transform(image.reshape(h * w, c))
        reshaped = best_comps.reshape(h, w, 3)
        max_val, min_val = reshaped.max(), reshaped.min()
        if max_val == min_val:
            max_val = max_val + 0.00001
        result = ((reshaped - max_val) / (max_val - min_val) * 255).astype(np.uint8)
        return result\

    def submission(self, test_df:pd.DataFrame, model:keras.models.Model, test_data:list, write_path:str):
            sub = test_df[['id']]
            predictions = model.predict(test_data)
            predictions = [np.argmax(y_pred) for y_pred in predictions]
            sub['label'] = predictions
            print(sub.shape, len(predictions))
            sub.to_csv(write_path, index=False)


READ_TEST_PATH = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/test.csv"
READ_TRAIN_PATH = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/train.csv"
processor = DataProcessing()
test_labels = processor.readData(READ_TEST_PATH)
train_labels = processor.readData(READ_TRAIN_PATH)
train_labels.head()


train_lis, test_lis = processor.dataLoader(test_labels, folder_path="/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025/ot/ot")
len(train_lis), len(test_lis)


train_lis_pca = []
print("Applying PCA on training data ...")
for img in train_lis:
    res = processor.bandsPCA(img)
    train_lis_pca.append(res)


test_lis_pca = []
print("Applying PCA on testing data ...")
for img in test_lis:
    res = processor.bandsPCA(img)
    test_lis_pca.append(res)
print("PCA application is done for training and testing data ...")


# Collecting images of same shape for training purpose
train_data = []
train_labels_lis = []
for label, img in zip(train_labels['label'], train_lis_pca):
    h, w, c = img.shape
    if h == 128 and w == 128 and c == 3:
        train_data.append(img)
        train_labels_lis.append(label)
len(train_data), len(train_labels_lis)

test_data = []
removed_images = []
test_file_names = []
for img, file_name in zip(test_lis_pca, test_labels['id']):
    h, w, c = img.shape
    if h == 128 and w == 128 and c == 3:
        test_data.append(img)
        test_file_names.append(file_name)
    else:
        removed_images.append(file_name)
        
len(test_data)


len(test_lis_pca), len(test_labels), len(test_data), len(removed_images)


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16') 


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np

X, y = np.array(train_data), np.array(train_labels_lis)
num_classes = 101
epochs = 10
batch_size = 8

model = Sequential([
    Conv3D(32, kernel_size=(3, 3, 1), activation='relu', input_shape=(128, 128, 3, 1)),
    MaxPooling3D(pool_size=(2, 2, 1)),

    Conv3D(64, kernel_size=(3, 3, 1), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 1)),

    Conv3D(128, kernel_size=(3, 3, 1), activation='relu'),
    MaxPooling3D(pool_size=(2, 2, 1)),

    Flatten(),
    Dense(256, activation='relu'),   
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, y_train, X_test, y_test = np.array(X[:-200]), np.array(y[:-200]), np.array(X[-200:]), np.array(y[-200:])

history = model.fit(X_train, y_train, 
                    validation_data = (X_test, y_test),
                    epochs=epochs, 
                    batch_size=batch_size, 
                    )


from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score
X_test, y_test = X[-200:], y[-200:]
y_pred = model.predict(X_test)
y_pred = [np.argmax(val) for val in y_pred]


list1 = history.history['loss']  
list2 = history.history['val_loss']  

epochs = range(1, len(list1) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, list1, label='Training loss', marker='o')
plt.plot(epochs, list2, label='Validation loss"', marker='s')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



history.history


list1 = history.history['accuracy']  
list2 = history.history['val_accuracy']  

epochs = range(1, len(list1) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, list1, label='Training accuracy', marker='o')
plt.plot(epochs, list2, label='Validation accuracy"', marker='s')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# processor.submission(test_df = test_labels, model = model, test_data = np.array(test_data), write_path='Submission.csv')




