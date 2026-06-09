import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


de_train =   pd.read_parquet("/kaggle/input/open-problems-single-cell-perturbations/de_train.parquet")
id_map = pd.read_csv("/kaggle/input/open-problems-single-cell-perturbations/id_map.csv")
sample_submission = pd.read_csv("/kaggle/input/open-problems-single-cell-perturbations/sample_submission.csv")


de_train


id_map.head()


de_train.isnull().sum()


unique= de_train['sm_name'].unique()
unique, print(f"Count {len(unique)}")


de_train['cell_type'].unique()


de_train[de_train["control"] == True].count  


# shuffle the data
de_train = de_train.sample(frac=1.0, random_state=42)


de_train


# Create features and and labels for reverse model 18211 features and 152 labels for true model
features_columns = ["cell_type", "sm_name"]
labels_columns=["cell_type","sm_name","sm_lincs_id","SMILES","control"]
labels = de_train.drop(columns=labels_columns)
features = pd.DataFrame(de_train, columns=features_columns)


features


labels


# Get test data 
test_data = pd.DataFrame(id_map, columns=features_columns)


from sklearn.preprocessing import OneHotEncoder

# Create an instance of the encoder
encoder = OneHotEncoder()

# Fit the encoder on features
encoder.fit(features)

# Transform the features into one-hot encoded format
one_hot_encode_features = encoder.transform(features)

# Transform the test data(id_map)
one_hot_test = encoder.transform(test_data)


# check shape
one_hot_encode_features.toarray().shape, one_hot_test.toarray().shape


# check one sample
one_hot_encode_features.toarray()[0]


from sklearn.model_selection import train_test_split

# Split the data into 70% training, 15% validation, and 15% testing
X_train, X_temp, y_train, y_temp = train_test_split(one_hot_encode_features, labels.values, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)


# Printing the shapes of the data splits
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_test shape:", y_test.shape)


# We also get full features for final training 
full_features = one_hot_encode_features.toarray()
full_labels = labels.values


print("full_features shape:", full_features.shape)
print("full_labels shape:", full_labels.shape)


from tensorflow.keras.callbacks import ModelCheckpoint

def create_model_checkpoint(filepath, monitor='val_mae', save_best_only=True,
                            save_weights_only=True, mode='auto', verbose=0):
    """
    Create a ModelCheckpoint callback for saving the best model weights during training.

    Args:
        filepath (str): Filepath to save the best weights.
        monitor (str): Metric to monitor (e.g., 'val_loss' or 'val_mae').
        save_best_only (bool): Save only the best weights.
        save_weights_only (bool): Save only the model's weights, not the entire model.
        mode (str): One of {'auto', 'min', 'max'}. In 'min' mode, it saves when the monitored metric decreases.
        verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

    Returns:

        keras.callbacks.ModelCheckpoint: ModelCheckpoint callback.
    """
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode,
        verbose=verbose
    )
    return checkpoint


def plot_training_history(history, metrics):
    """
    Plot training history curves for loss and evaluation metrics on the same line.

    Args:
        history (keras.callbacks.History): Training history object.
        metrics (list): List of metric names to plot.

    Returns:
        None
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss', color="blue")
    plt.plot(epochs, val_loss, label='Validation Loss', color="red")
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot specified evaluation metrics on the same line
    for metric in metrics:
        train_metric_name = f'Training {metric.capitalize()}'
        val_metric_name = f'Validation {metric.capitalize()}'
        train_metric = history.history[metric]
        val_metric = history.history['val_' + metric]

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_metric, label=train_metric_name, color="green")
        plt.plot(epochs, val_metric, label=val_metric_name, color="orange")

    plt.title('Metrics')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



from sklearn.metrics import mean_absolute_error

def calculate_mae_and_mrrmse(model, data, y_true):
    """
    Calculate Mean Absolute Error (MAE) and Mean Rowwise Root Mean Squared Error (MRRMSE).

    Parameters:
    - model: The trained  model.
    - data: The input data for prediction.
    - y_true: The true target values.
    - scaler: The scaler used for data normalization.

    Returns:
    - None
    """
    # Predict using the model
    y_pred_original = model.predict(data, batch_size=1)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true , y_pred_original)
    
    # Calculate Mean Rowwise Root Mean Squared Error (MRRMSE)
    rowwise_rmse = np.sqrt(np.mean(np.square(y_true - y_pred_original), axis=1))
    mrrmse_score = np.mean(rowwise_rmse)
    
    # Print the results
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Rowwise Root Mean Squared Error (MRRMSE): {mrrmse_score}")


def mean_rowwise_rmse_loss(y_true, y_pred):
    """
    Custom loss function to calculate the Mean Rowwise Root Mean Squared Error (RMSE) loss.

    Parameters:
    - y_true: The true target values.
    - y_pred: The predicted values.

    Returns:
    - Mean Rowwise RMSE loss as a scalar tensor.
    """
    # Calculate RMSE for each row
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    # Calculate the mean of RMSE values across all rows
    mean_rmse = tf.reduce_mean(rmse_per_row)
    
    return mean_rmse


def custom_mean_rowwise_rmse(y_true, y_pred):
    """
    Custom metric to calculate the Mean Rowwise Root Mean Squared Error (RMSE).

    Parameters:
    - y_true: The true target values.
    - y_pred: The predicted values.

    Returns:
    - Mean Rowwise RMSE as a scalar tensor.
    """
    # Calculate RMSE for each row
    rmse_per_row = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=1))
    # Calculate the mean of RMSE values across all rows
    mean_rmse = tf.reduce_mean(rmse_per_row)
    
    return mean_rmse


from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential


tf.random.set_seed(42)

model_0 = Sequential([
    Dense(512, activation="tanh"),
    Dense(18211, activation="linear")
])

model_0.compile(loss=mean_rowwise_rmse_loss, 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[custom_mean_rowwise_rmse])

history_0 = model_0.fit(X_train, y_train,
                       epochs=10,
                       validation_data=(X_val,y_val),
                       batch_size=32,
                       callbacks=[create_model_checkpoint("model_0", monitor="val_custom_mean_rowwise_rmse")])


# Loading weights 
model_0.load_weights("model_0")
calculate_mae_and_mrrmse(model=model_0, data=X_test, y_true=y_test)


# Model performance on full data 
calculate_mae_and_mrrmse(model=model_0, data=full_features, y_true=full_labels)


# Visualize the learning from our helper functions
plot_training_history(history_0, metrics=["custom_mean_rowwise_rmse"])


tf.random.set_seed(42)

model_1 = Sequential([
    Dense(1024, activation="tanh"),
    Dense(512, activation="tanh"),
    Dense(18211, activation="linear")
])

model_1.compile(loss=mean_rowwise_rmse_loss, 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[custom_mean_rowwise_rmse])

history_1 = model_1.fit(X_train, y_train,
                       epochs=30,
                       validation_data=(X_val,y_val),
                       callbacks=[create_model_checkpoint("model_1", monitor="val_custom_mean_rowwise_rmse")])


# Loading weights 
model_1.load_weights("model_1")
calculate_mae_and_mrrmse(model=model_1, data=X_test, y_true=y_test)


# Model performance on full data 
calculate_mae_and_mrrmse(model=model_1, data=full_features, y_true=full_labels)


# Visualize the learning from our helper functions
plot_training_history(history_1, metrics=["custom_mean_rowwise_rmse"])


tf.random.set_seed(42)

model_2 = Sequential([ 
    Dense(256),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(18211,activation= "linear")
])


model_2.compile(loss="mae", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])

history_2 = model_2.fit(X_train, y_train,
                       epochs=30,
                       verbose=0, #train in silent mode
                       validation_data=(X_val,y_val),
                       callbacks=[create_model_checkpoint("model_2", monitor="val_mae")])


model_2.load_weights("model_2")
calculate_mae_and_mrrmse(model=model_2, data=X_test, y_true=y_test)


calculate_mae_and_mrrmse(model=model_2, data=full_features, y_true=full_labels)


#  Visualize the learning from our helper functions
plot_training_history(history_2, metrics=["mae"])


tf.random.set_seed(42)

# clone model 2
model_3 = tf.keras.models.clone_model(model_2)

model_3.compile(loss="mae", 
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0027),
                metrics=[custom_mean_rowwise_rmse])

history_3 = model_3.fit(X_train, y_train,
                       epochs=25,
                       verbose=0, #train in silent mode
                       validation_data=(X_val,y_val),
                       callbacks=[create_model_checkpoint("model_3", monitor="val_custom_mean_rowwise_rmse")])


model_3.load_weights("model_3")
calculate_mae_and_mrrmse(model=model_3, data=X_test, y_true=y_test)


calculate_mae_and_mrrmse(model=model_3, data=full_features, y_true=full_labels)


plot_training_history(history_3, metrics=["custom_mean_rowwise_rmse"])


tf.random.set_seed(42)

# clone model 2
model_4 = Sequential([ 
    Dense(256),
    BatchNormalization(),
    Activation("relu"),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dropout(0.1),
    Dense(18211,activation= "linear")
])

model_4.compile(loss="mae", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[custom_mean_rowwise_rmse])


from sklearn.model_selection import KFold

# Define the number of folds (K)
num_folds = 5 # You can change this value as needed

# Initialize lists to store the model's performance scores
mae_scores = []
mrrmse_scores = []

# Initialize the KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=51)

# Loop through the K folds
for train_index, val_index in kf.split(full_features):
    # Convert indices to integers and split the data
    train_index = train_index.astype(int)
    val_index = val_index.astype(int)
    X_train_, X_val_ = full_features[train_index], full_features[val_index]
    y_train_, y_val_ = full_labels[train_index], full_labels[val_index]

    # Train your model on X_train and y_train
    model_4.fit(X_train_, y_train_, epochs=50, verbose=0)

    # Make predictions on the validation set
    y_preds = model_4.predict(X_val_)

    # Calculate the Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_val_, y_preds)
    mae_scores.append(mae)

    # Calculate the Mean Rowwise Root Mean Square Error (MRRMSE)
    rowwise_rmse = np.sqrt(np.mean(np.square(y_val_ - y_preds), axis=1))
    mrrmse_score = np.mean(rowwise_rmse)
    mrrmse_scores.append(mrrmse_score)

# Calculate the mean and standard deviation of MAE and MRRMSE scores
mean_mae = np.mean(mae_scores)
mean_mrrmse = np.mean(mrrmse_scores)

# Print the results
print(f'Average MAE across {num_folds} folds: {mean_mae:.4f} ')
print(f'Average MRRMSE across {num_folds} folds: {mean_mrrmse:.4f}')


model_4.compile(loss="mae", 
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[custom_mean_rowwise_rmse])

history_4 = model_4.fit(full_features, full_labels,
                       epochs=50,
                       verbose=0)


calculate_mae_and_mrrmse(model=model_4, data=full_features, y_true=full_labels)


preds = model_4.predict(one_hot_test.toarray(), batch_size=1)


preds.shape 


sample_columns = sample_submission.columns
sample_columns= sample_columns[1:]
submission_df = pd.DataFrame(preds, columns=sample_columns)


submission_df.insert(0, 'id', range(255))


sample_submission


submission_df


submission_df.to_csv("submission_df.csv", index=False)


!zip submission_preds.zip /kaggle/working/submission_df.csv

