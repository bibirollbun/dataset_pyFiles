# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
train.head() #to have a look at our data


train.isnull().sum()


train.duplicated().sum()


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Loop through each column to create histograms
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=train, x=col, hue='Fertilizer Name', kde=True, multiple="stack", palette="Set2")
    plt.title(f'{col} vs Fertilizer Name')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Fertilizer Name', y=col, data=train, palette='Set3')
    plt.title(f'{col} distribution by Fertilizer')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# 1. Encode Fertilizer Name to numeric
train['Fertilizer_numeric'] = train['Fertilizer Name'].astype('category').cat.codes

# 2. Select numeric features + encoded target
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
corr_matrix = train[num_cols + ['Fertilizer_numeric']].corr()

# 3. Extract correlation with target
fertilizer_corr = corr_matrix['Fertilizer_numeric'].drop('Fertilizer_numeric')

# 4. Plot horizontal bar chart
fertilizer_corr.sort_values().plot(kind='barh', figsize=(8, 6), color='skyblue')
plt.title('Correlation of Features with Fertilizer Name')
plt.xlabel('Correlation')
plt.grid(True)
plt.tight_layout()
plt.show()


cat_cols = ['Soil Type', 'Crop Type']

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='Fertilizer Name', data=train, palette='Set2')
    plt.title(f'{col} vs Fertilizer Name')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


heatmap_data = pd.crosstab(train['Crop Type'], train['Soil Type'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Crop Type vs Soil Type Frequency')
plt.tight_layout()
plt.show()


X = train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type']]
y = train['Fertilizer Name']


# check for class imbalance

sns.countplot(x=y)
plt.title("Class distribution of Fertilizer Name")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

# Use your version of X and y
X = train[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type']]
y = train['Fertilizer Name']

# 1. Encode categorical features in X
X = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])

# 2. Normalize numerical features
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 3. Encode target variable properly
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y)
y_cat = to_categorical(y_encoded)

# 4. Split AFTER encoding target!
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=1)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.activations import softmax
import numpy as np

# Build the model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=256,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=1)




import matplotlib.pyplot as plt

# Accuracy
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# MAP@3 function
def mapk(true_labels, predicted_labels, k=3):
    map_total = 0.0
    for true, preds in zip(true_labels, predicted_labels):
        score = 0.0
        for i, pred in enumerate(preds[:k]):
            if pred == true:
                score = 1.0 / (i + 1)
                break
        map_total += score
    map_score = map_total / len(true_labels)
    print(f"MAP@{k} Score: {map_score:.4f}")
    return map_score



# Predict probabilities on validation data
y_val_pred_probs = model.predict(X_val)

# Get top 3 class indices
top_3_indices = np.argsort(y_val_pred_probs, axis=1)[:, -3:][:, ::-1]

# Flatten and inverse transform to class labels
flat_indices = top_3_indices.flatten()
flat_labels = le_y.inverse_transform(flat_indices)
top_3_labels = flat_labels.reshape(top_3_indices.shape)

# Prepare list-of-lists for predicted labels
predicted_labels = [list(row) for row in top_3_labels]

# Get true labels from one-hot to original
true_labels = le_y.inverse_transform(np.argmax(y_val, axis=1))

# Evaluate MAP@3
mapk(true_labels, predicted_labels, k=3)





