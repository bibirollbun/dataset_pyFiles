import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import*

from sklearn.linear_model import *
from sklearn.tree import DecisionTreeClassifier


import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')
train.head()


test = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')
test.head()


# Check for missing values
def check_missing():
    print("Missing values in train:")
    print(train.isnull().sum().sum())
    print("Missing values in test:")
    print(test.isnull().sum().sum())

check_missing()



def label_encode_dataframe(test):
    # Initialiser le LabelEncoder
    le = LabelEncoder()
    
    # Parcourir chaque colonne
    for col in test.columns:
        # Vérifier si la colonne est de type object (généralement les colonnes catégorielles)
        if test[col].dtype == 'object':
            # Appliquer LabelEncoder
            test[col] = le.fit_transform(test[col])
    
    return test

# Appliquer la fonction de Label Encoding
test = label_encode_dataframe(test)


def label_encode_dataframe(train):
    # Initialiser le LabelEncoder
    le = LabelEncoder()
    
    # Parcourir chaque colonne
    for col in train.columns:
        # Vérifier si la colonne est de type object (généralement les colonnes catégorielles)
        if train[col].dtype == 'object':
            # Appliquer LabelEncoder
            train[col] = le.fit_transform(train[col])
    
    return train

# Appliquer la fonction de Label Encoding
train = label_encode_dataframe(train)


# Liste des colonnes à vérifier
train_columns_to_check = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# Parcourir les colonnes et vérifier si la colonne est dans la liste
for col in train.columns:
    if col in train_columns_to_check:  # Si la colonne fait partie de la liste
        print(col, ':', np.unique(train[col]))


# Liste des colonnes à vérifier
test_columns_to_check = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']

# Parcourir les colonnes et vérifier si la colonne est dans la liste
for col in test.columns:
    if col in test_columns_to_check:  # Si la colonne fait partie de la liste
        print(col, ':', np.unique(test[col]))


# Feature selection
X = train.drop(['y', 'ID'], axis=1)
y = train['y']
test_ids = test['ID']
X_test = test.drop(['ID'], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
val_predictions = model.predict(X_val)

# Calculer les métriques adaptées à la régression
mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"MAPE : {mean_absolute_percentage_error(y_val, val_predictions):2f}")
print(f"R-squared (R²): {r2:.2f}")


#Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, bins=30, color="blue")
plt.title("Distribution of Target Variable (y)")
plt.xlabel("y")
plt.ylabel("Frequency")
plt.show()


# Feature importance
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)


# Top 10 important features
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette="viridis")
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# Correlation heatmap
plt.figure(figsize=(15, 10))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# Pairplot for selected features
selected_features = importance_df.head(4)['Feature'].values
pairplot_data = train[list(selected_features) + ['y']]
sns.pairplot(pairplot_data, diag_kind="kde", markers="o")
plt.suptitle("Pairplot of Top Features vs Target", y=1.02)
plt.show()


import xgboost as xgb

# Feature selection
X = train.drop(['y', 'ID'], axis=1)
y = train['y']
test_ids = test['ID']
X_test = test.drop(['ID'], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model with XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,              # Number of trees
    learning_rate=0.05,            # Step size for optimization
    max_depth=6,                   # Maximum depth of the trees
    colsample_bytree=0.8,          # Fraction of features used for each tree
    subsample=0.8,                 # Fraction of samples used for each tree
    random_state=42                # Random state for reproducibility
)

# Fit the model
model.fit(X_train, y_train)

# Predictions
val_predictions = model.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"MAPE : {mean_absolute_percentage_error(y_val, val_predictions):2f}")
print(f"R-squared (R²): {r2:.2f}")



import lightgbm as lgb

# Feature selection
X = train.drop(['y', 'ID'], axis=1)
y = train['y']
test_ids = test['ID']
X_test = test.drop(['ID'], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model with LightGBM
model = lgb.LGBMRegressor(
    n_estimators=100,              # Number of trees
    learning_rate=0.05,            # Step size for optimization
    max_depth=6,                   # Maximum depth of the trees
    colsample_bytree=0.8,          # Fraction of features used for each tree
    subsample=0.8,                 # Fraction of samples used for each tree
    random_state=42                # Random state for reproducibility
)

# Fit the model
model.fit(X_train, y_train)

# Predictions
val_predictions = model.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"MAPE : {mean_absolute_percentage_error(y_val, val_predictions):2f}")
print(f"R-squared (R²): {r2:.2f}")



# Feature selection
X = train.drop(['y', 'ID'], axis=1)
y = train['y']
test_ids = test['ID']
X_test = test.drop(['ID'], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model with Gradient Boosting
model = GradientBoostingRegressor()

# Fit the model
model.fit(X_train, y_train)

# Predictions
val_predictions = model.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"MAPE : {mean_absolute_percentage_error(y_val, val_predictions):2f}")
print(f"R-squared (R²): {r2:.2f}")



# Feature selection
X = train.drop(['y', 'ID'], axis=1)
y = train['y']
test_ids = test['ID']
X_test = test.drop(['ID'], axis=1)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model with Lasso
model = Lasso(alpha=0.01, random_state=42)  # alpha is the regularization strength

# Fit the model
model.fit(X_train, y_train)

# Predictions
val_predictions = model.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_predictions)

# Print the metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"MAPE : {mean_absolute_percentage_error(y_val, val_predictions):2f}")
print(f"R-squared (R²): {r2:.2f}")



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import*

from sklearn.linear_model import *
from sklearn.tree import DecisionTreeClassifier



import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential 
#une autre librairie intéressante : pytorch
from tensorflow.keras.layers import *

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.applications import * #ou VGG16


train_dir = "/kaggle/input/yolo-drone-detection-dataset/drone_dataset/train"
test_dir = "/kaggle/input/yolo-drone-detection-dataset/drone_dataset/valid"
image_size = (224,224)

train = image_dataset_from_directory(train_dir, image_size=image_size)
test = image_dataset_from_directory(test_dir, image_size=image_size)


# Définition du modèle
model = Sequential()
model.add(InputLayer(input_shape=(224, 224, 3)))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10, activation='softmax')) 

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
epochs = 10
history = model.fit(train, validation_data=test, epochs=epochs, verbose=1)

# Affichage des courbes d'apprentissage
plt.plot(range(epochs), history.history['accuracy'], label='Train Accuracy')
plt.plot(range(epochs), history.history['val_accuracy'], color='red')


from tensorflow.keras.applications import VGG16
# Charger le modèle VGG16 pré-entraîné avec input_shape=(224, 224, 3)
vgg = VGG16(include_top=False, input_shape=(224, 224, 3)) 


# Construire le modèle
model = Sequential()
model.add(vgg)
model.add(Flatten())  
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compiler le modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
epochs = 5
history = model.fit(train, validation_data=test, epochs=epochs, verbose=1)

# Affichage des courbes d'apprentissage
plt.plot(range(epochs), history.history['accuracy'])
plt.plot(range(epochs), history.history['val_accuracy'], color='red')


!pip install ultralytics


from  ultralytics import YOLO

import cv2 
from PIL import Image

model = YOLO('yolov8n.pt')


image = "data:image/jpeg;base64,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"

result = model.predict(image)


from matplotlib import pyplot as plt

img = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)

plt.imshow(img)

