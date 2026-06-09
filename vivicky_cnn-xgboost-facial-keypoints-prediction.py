import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix
import seaborn as sns


zip_train = '/kaggle/input/facial-keypoints-detection/training.zip'
zip_test = '/kaggle/input/facial-keypoints-detection/test.zip'
extract_to_path = '/kaggle/working/'

with zipfile.ZipFile(zip_train, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
with zipfile.ZipFile(zip_test, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)


def load_and_process_images(data):
    all_image = []
    for all_pixel in data['Image'].values:
        image = list(map(int, all_pixel.split(' ')))
        all_image.append(image)
    return np.array(all_image).reshape(-1, 96, 96)


def visualize_key_points(all_image, data, y_pred=None, row=5, col=5):
    fig, ax = plt.subplots(row, col, figsize=(16, 10))
    ax = ax.flatten()
    for i in range(len(ax)):
        sample = np.random.randint(0, all_image.shape[0])
        ax[i].imshow(all_image[sample], cmap='gray')

        # Plot the original key points
        x = data.iloc[sample, [i for i in range(0, 30, 2)]]
        y = data.iloc[sample, [i + 1 for i in range(0, 30, 2)]]
        ax[i].scatter(x, y)

        # Plot the predicted key points
        if y_pred is not None:
            x_pred = [y_pred[sample][i] for i in range(0, len(y_pred[0]), 2)]
            y_pred_values = [y_pred[sample][i + 1] for i in range(0, len(y_pred[0]), 2)]
            ax[i].scatter(x_pred, y_pred_values)

        ax[i].axis('off')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, x_test, y_test):
    print(model.__class__.__name__)
    print('--------------------')
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R2: {r2}')

    return [mae, mse, r2]


def custom_accuracy(y_true, y_pred):
    error = np.abs(y_true - y_pred)
    return np.mean(error < 5)


class CNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = self._build_conv_block(16, 16)
        self.conv2 = self._build_conv_block(32, 32)
        self.conv3 = self._build_conv_block(64, 64)
        self.conv4 = self._build_conv_block(128, 128)

        self.fc = keras.Sequential([
            layers.Flatten(),
            layers.Dense(128, activation='leaky_relu'),
            layers.Dense(256, activation='leaky_relu'),
            layers.Dense(30)
        ])

    def _build_conv_block(self, in_channel, out_channel):
        return keras.Sequential([
            layers.Conv2D(in_channel, 3),
            layers.BatchNormalization(),
            layers.Activation('leaky_relu'),
            layers.Conv2D(out_channel, 3),
            layers.BatchNormalization(),
            layers.Activation('leaky_relu'),
            layers.MaxPooling2D((2, 2))
        ])

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


data = pd.read_csv('/kaggle/working/training.csv')
data.head().T


# Process the image data
all_image = load_and_process_images(data)

# Visualize the key points
visualize_key_points(all_image, data, row=2, col=5)

# Remove missing values
data_new = data.dropna()
data_new.head().T


# Split the features and target variables
value = list(range(0, 4)) + list(range(20, 22)) + list(range(28, 30))
target = list(range(4, 20)) + list(range(22, 28))
x_train, x_test, y_train, y_test = train_test_split(data_new.iloc[:, value], data_new.iloc[:, target],
                                                    test_size=0.1, random_state=42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Train the XGBRegressor model
xgb = XGBRegressor()
xgb.fit(x_train, y_train)
score_xgb = evaluate_model(xgb, x_test, y_test)

# Predict the key points
y_pred = xgb.predict(data.iloc[:, value])

# Visualize the predicted key points
visualize_key_points(all_image, data, y_pred, row=5, col=5)


# Fill in the missing values
data_values = data.iloc[:, target].values
y_pred_values = np.array(y_pred)
data.iloc[:, target] = np.where(np.isnan(data_values), y_pred_values, data_values)

# Remove missing values again
data_new = data.dropna()
data_new.shape
print(data_new.head().T)

# Process the image data
all_image = load_and_process_images(data_new)
target = data_new.drop('Image', axis=1).values
x_train, x_test, y_train, y_test = train_test_split(all_image, target,
                                                    test_size=0.1,
                                                    random_state=42)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# Reshape the data to fit the CNN model
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Initialize the CNN model
model = CNN()
model(np.random.rand(32, 96, 96, 1))
model.summary()

# Define the training parameters
learning_rate = 0.001
batch_size = 32
epochs = 50
factor = 0.2

# Compile the model
model.compile(
    loss=keras.losses.LogCosh(),
    metrics=[keras.metrics.LogCoshError()],
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

# Define the callback functions
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=3, min_lr=1e-7)

# Train the model
H = model.fit(
    x_train, y_train, batch_size=batch_size,
    validation_split=0.1,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Plot the training and validation loss curves
score_train = H.history['logcosh']
score_val = H.history['val_logcosh']
num_epoch = len(score_train)
plt.plot(range(1, num_epoch + 1), score_train, label='Train')
plt.plot(range(1, num_epoch + 1), score_val, label='Val')
plt.xlabel('Epochs')
plt.ylabel('LogCosh')
plt.legend()
plt.show()

# Evaluate the CNN model
evaluate_model(model, x_test, y_test)

# Calculate the custom evaluation metric
y_pred_cnn = model.predict(x_test)
custom_acc = custom_accuracy(y_test, y_pred_cnn)
print(f"Custom accuracy of the CNN model: {custom_acc}")

# Save the best model
model.save('best_cnn_model.h5')


importances = xgb.feature_importances_
feature_names = data_new.iloc[:, value].columns

# Plot the feature importance bar chart
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in XGBoost Model')
plt.xticks(rotation=90)
plt.show()


data_test = pd.read_csv('/kaggle/working/test.csv')
data_test.head().T


all_image_test = load_and_process_images(data_test)
all_image_test = all_image_test[..., np.newaxis]
print(all_image_test.shape)


y_pred = model.predict(all_image_test)
print(y_pred.shape)


# Read the IdLookupTable data
pattern = pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')
pattern.head().T


y_pred_df = pd.DataFrame(y_pred)
y_pred_df.columns = data.drop('Image', axis=1).columns
y_pred_df.index = [i for i in range(1, len(y_pred_df) + 1)]


# Fill in the Location column
for i, j in zip(pattern['ImageId'].values, pattern['FeatureName']):
    pattern.loc[(pattern['ImageId'] == i) & (pattern['FeatureName'] == j), 'Location'] = y_pred_df.loc[i, j]


# Generate the submission file
submit = pattern.drop(['ImageId', 'FeatureName'], axis=1)
submit.to_csv('submission.csv', index=False)

