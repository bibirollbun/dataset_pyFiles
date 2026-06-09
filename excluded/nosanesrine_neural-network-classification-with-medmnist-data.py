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


data_pa = np.load("/kaggle/input/tensor-reloaded-multi-task-med-mnist/data/pathmnist.npz")
print(len(data_pa))
data_pa.files


train_x_pa = data_pa.get("train_images")
train_y_pa = data_pa.get("train_labels")
valid_x_pa = data_pa.get("val_images")
valid_y_pa = data_pa.get("val_labels")
test_x_pa = data_pa.get("test_images")
test_y_pa = data_pa.get("test_labels")
train_x_pa[0], train_y_pa

# Check the shape of our data
train_x_pa.shape, train_y_pa.shape, test_x_pa.shape, test_y_pa.shape


# Check shape of a single example
train_x_pa[0].shape, train_y_pa[0].shape


# Plot a single example
import matplotlib.pyplot as plt
sample_index = 1
plt.imshow(train_x_pa[sample_index]);

# Check our samples label
print(f"Class Label for this Image {train_y_pa[sample_index]}")


# Plot multiple random images of fashion MedMNIST
import random
plt.figure(figsize=(7, 7))
for i in range(4):
  ax = plt.subplot(2, 2, i + 1)
  rand_index = random.choice(range(len(train_x_pa)))
  plt.imshow(train_x_pa[rand_index], cmap=plt.cm.binary)
  plt.title(train_y_pa[rand_index])
  plt.axis(False)


# Flatten image labels (Y data) into 1-D Tensor
train_y_pa = np.array(train_y_pa).ravel()
test_y_pa = np.array(test_y_pa).ravel()
valid_y_pa = np.array(valid_y_pa).ravel()
train_y_pa.shape, test_y_pa.shape, valid_y_pa.shape


import tensorflow as tf
tf.random.set_seed(12)

# Model creation
model_pa_1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(8, activation="tanh"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_1.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Fit the model
history_pa_1 = model_pa_1.fit(train_x_pa, train_y_pa, epochs=10, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))


# Check model summary
model_pa_1.summary()


# Check the min and max values of the training data
train_x_pa.min(), train_x_pa.max()


train_x_pa = train_x_pa / 255
test_x_pa = test_x_pa / 255
valid_x_pa = valid_x_pa / 255

train_x_pa.min(), train_x_pa.max()


# MODEL-2: With the Normalizer version of Input data
model_pa_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_2.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.01
    metrics=["accuracy"]
)

# Fit the model
history_pa_2 = model_pa_2.fit(train_x_pa, train_y_pa, epochs=10, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))


# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# and Made with ML's introductory notebook - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
import itertools
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)
  fig.delaxes(fig.axes[1])

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)


# Make predictions with the most recent model
y_probs_2 = model_pa_2.predict(test_x_pa) # "probs" is short for probabilities

# View the first 5 predictions
y_probs_2[:3]


# Convert all of the predictions from probabilities to labels
y_preds_2 = y_probs_2.argmax(axis=1)

# View the first 10 prediction labels
y_preds_2[:10]


# Check out the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true=test_y_pa,
                 y_pred=y_preds_2)
# Make a prettier confusion matrix
make_confusion_matrix(y_true=test_y_pa,
                      y_pred=y_preds_2,
                      classes=[0,1,2,3,4,5,6,7,8],
                      figsize=(11, 10),
                      text_size=8)


# Optimizer Learning rate parameter tuning
model_pa_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_3.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(), #learning_rate=0.01
    metrics=["accuracy"]
)

# Create the learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10**(epoch/10))

# Fit the model
history_pa_3 = model_pa_3.fit(train_x_pa, train_y_pa, epochs=40, verbose=False,
                             validation_data=(valid_x_pa, valid_y_pa),
                             callbacks=[lr_scheduler])

# Plot the learning rate decay curve
lrs = 1e-6 * (10**(np.arange(40)/10))
plt.semilogx(lrs, history_pa_3.history["loss"]) # want the x-axis to be log-scale
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate");


# With optimum lr
model_pa_4 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_4.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_4 = model_pa_4.fit(train_x_pa, train_y_pa, epochs=50, verbose=False,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_4.history).plot(title="Loss curve Model-4")


# Make predictions with the most recent model
y_probs_4 = model_pa_4.predict(test_x_pa) # "probs" is short for probabilities

# View the first 5 predictions
y_probs_4[:5]

# Convert all of the predictions from probabilities to labels
y_preds_4 = y_probs_4.argmax(axis=1)

# View the first 10 prediction labels
y_preds_4[:10]


# Check out the confusion matrix
confusion_matrix(y_true=test_y_pa,
                 y_pred=y_preds_4)
# Make a prettier confusion matrix
make_confusion_matrix(y_true=test_y_pa,
                      y_pred=y_preds_4,
                      classes=[0,1,2,3,4,5,6,7,8],
                      figsize=(11, 10),
                      text_size=8)


model_pa_5 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(8, activation="tanh"),
    tf.keras.layers.Dense(4, activation="tanh"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_5.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_5 = model_pa_5.fit(train_x_pa, train_y_pa, epochs=50, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_5.history).plot(title="Loss curve Model-5")


# Make predictions with the most recent model
y_probs_5 = model_pa_5.predict(test_x_pa) # "probs" is short for probabilities

# View the first 5 predictions
y_probs_5[:5]

# Convert all of the predictions from probabilities to labels
y_preds_5 = y_probs_5.argmax(axis=1)

# View the first 10 prediction labels
y_preds_5[:10]

# Check out the confusion matrix
confusion_matrix(y_true=test_y_pa,
                 y_pred=y_preds_5)
# Make a prettier confusion matrix
make_confusion_matrix(y_true=test_y_pa,
                      y_pred=y_preds_5,
                      classes=[0,1,2,3,4,5,6,7,8],
                      figsize=(11, 10),
                      text_size=8)


model_pa_6 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(24, activation="tanh"),
    tf.keras.layers.Dense(12, activation="tanh"),
    tf.keras.layers.Dense(9, activation="tanh"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_6.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_6 = model_pa_6.fit(train_x_pa, train_y_pa, epochs=20, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_6.history).plot(title="Loss curve Model-5")


# Make predictions with the most recent model
y_probs_6 = model_pa_6.predict(test_x_pa) # "probs" is short for probabilities

# Convert all of the predictions from probabilities to labels
y_preds_6 = y_probs_6.argmax(axis=1)

# Check out the confusion matrix
confusion_matrix(y_true=test_y_pa,
                 y_pred=y_preds_6)
# Make a prettier confusion matrix
make_confusion_matrix(y_true=test_y_pa,
                      y_pred=y_preds_6,
                      classes=[0,1,2,3,4,5,6,7,8],
                      figsize=(11, 10),
                      text_size=8)


model_pa_7 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(40, activation="tanh"),
    tf.keras.layers.Dense(20, activation="tanh"),
    tf.keras.layers.Dense(9, activation="tanh"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_7.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_7 = model_pa_7.fit(train_x_pa, train_y_pa, epochs=20, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_7.history).plot(title="Loss curve Model-7")


model_pa_8 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(84, activation="tanh"),
    tf.keras.layers.Dense(40, activation="tanh"),
    tf.keras.layers.Dense(20, activation="tanh"),
    tf.keras.layers.Dense(9, activation="tanh"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_8.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_8 = model_pa_8.fit(train_x_pa, train_y_pa, epochs=20, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_8.history).plot(title="Loss curve Model-8")


model_pa_9 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(168, activation="tanh"),
    tf.keras.layers.Dense(84, activation="tanh"),
    tf.keras.layers.Dense(40, activation="tanh"),
    tf.keras.layers.Dense(30, activation="tanh"),
    tf.keras.layers.Dense(18, activation="tanh"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_9.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_9 = model_pa_9.fit(train_x_pa, train_y_pa, epochs=20, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_9.history).plot(title="Loss curve Model-9")


model_pa_10 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)), # reshape input array to vector
    tf.keras.layers.Dense(200, activation="tanh"),
    tf.keras.layers.Dense(120, activation="tanh"),
    tf.keras.layers.Dense(80, activation="tanh"),
    tf.keras.layers.Dense(40, activation="tanh"),
    tf.keras.layers.Dense(18, activation="tanh"),
    tf.keras.layers.Dense(9, activation="softmax") # Output layer
])

# Compile model
model_pa_10.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics=["accuracy"]
)

# Fit the model
history_pa_10 = model_pa_10.fit(train_x_pa, train_y_pa, epochs=150, verbose=True,
                             validation_data=(valid_x_pa, valid_y_pa))

import pandas as pd
# Plot loss curves
pd.DataFrame(history_pa_10.history).plot(title="Loss curve Model-10")


# Make predictions with the most recent model
y_probs_10 = model_pa_10.predict(test_x_pa) # "probs" is short for probabilities

# Convert all of the predictions from probabilities to labels
y_preds_10 = y_probs_10.argmax(axis=1)

# Check out the confusion matrix
confusion_matrix(y_true=test_y_pa,
                 y_pred=y_preds_10)
# Make a prettier confusion matrix
make_confusion_matrix(y_true=test_y_pa,
                      y_pred=y_preds_10,
                      classes=[0,1,2,3,4,5,6,7,8],
                      figsize=(11, 10),
                      text_size=8)


# Save a model using the SavedModel format
model_pa_10.save('pathmnist_model_70percent_accuracy_SavedModel.keras')


model_pa_10.summary()


# Model evaluation
loss, accuracy = model_pa_10.evaluate(test_x_pa, test_y_pa)
print(f"Model loss on test set: {loss}")
print(f"Model accuracy on test set: {(accuracy*100):.2f}%")
print("---")

from sklearn.metrics import classification_report
print(classification_report(test_y_pa, y_preds_10, labels=[0,1,2,3,4,5,6,7,8]))




