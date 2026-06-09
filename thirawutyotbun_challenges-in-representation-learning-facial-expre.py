import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Input, Add, GlobalAveragePooling2D, SpatialDropout2D, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
data = data.rename(columns = {" pixels": "pixels", " Usage": "usage"})

train_data = data.loc[data['usage'] == 'Training'].drop('usage', axis = 1).reset_index(drop = True)
validation_data = data.loc[data['usage'] == 'PublicTest'].drop('usage', axis = 1).reset_index(drop = True)
test_data = data.loc[data['usage'] == 'PrivateTest'].drop('usage', axis = 1).reset_index(drop = True)

train_size = len(train_data)
validation_size = len(validation_data)

n = 48

emotion_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}


train_data["image"] = train_data["pixels"].map(lambda x: np.reshape(np.array([int(i) for i in x.split()]), (-1, 48)))/255
validation_data["image"] = validation_data["pixels"].map(lambda x: np.reshape(np.array([int(i) for i in x.split()]), (-1, 48)))/255
test_data["image"] = test_data["pixels"].map(lambda x: np.reshape(np.array([int(i) for i in x.split()]), (-1, 48)))/255


px = 1/plt.rcParams['figure.dpi']
fig = plt.figure(figsize = (720*px, 720*px)) 
rows = 3
columns = 3
for i in range(rows*columns):
    fig.add_subplot(rows, columns, i + 1) 
    plt.imshow(train_data['image'][i], cmap = 'gray') 
    plt.axis('off') 
    plt.title(emotion_dict.get(train_data['emotion'][i]))


X_train = np.stack(train_data['image']).reshape(-1, n, n, 1)
print(f"X_train shape: {np.stack(X_train).shape}")

y_train = to_categorical(train_data['emotion'], num_classes = len(emotion_dict))
print(f"y_train shape: {y_train.shape}")

X_validation = np.stack(validation_data['image']).reshape(-1, n, n, 1)
print(f"X_validation shape:  {np.stack(X_validation).shape}")

y_validation = to_categorical(validation_data['emotion'], num_classes = len(emotion_dict))
print(f"y_validation shape: {y_validation.shape}")

X_test = np.stack(test_data['image']).reshape(-1, n, n, 1)
print(f"X_test shape:  {np.stack(X_test).shape}")

y_test = to_categorical(test_data['emotion'], num_classes = len(emotion_dict))
print(f"y_test shape: {y_test.shape}")


n = 48
batch_size = 32

datagen = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_generator = datagen.flow(X_train, y_train, batch_size = batch_size)


callback = EarlyStopping(monitor = 'val_accuracy', patience = 50, restore_best_weights = True)

checkpoint = ModelCheckpoint(
    filepath = 'best_cnn_model.keras',
    monitor = 'val_accuracy',
    save_best_only = True,
    mode = 'max',
    verbose = 1
)

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = 10,
    verbose = 1,
    min_lr = 0.000001
)


def Residual_Block(input_tensor, filters, kernel_size = 3, stride = 1):
    shortcut = input_tensor
    if input_tensor.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size = 1, strides = stride, padding = 'same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Conv2D(filters, kernel_size = kernel_size, strides = stride, padding = 'same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size = kernel_size, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x


input_layer = Input(shape = (n, n, 1))
x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.3)(x)

x = Residual_Block(x, 128)
x = MaxPooling2D((2, 2))(x)
x = SpatialDropout2D(0.3)(x)

x = Residual_Block(x, 256)
x = MaxPooling2D((2, 2))(x)
x = SpatialDropout2D(0.3)(x)

x = Flatten()(x)
x = Dense(512, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

output_layer = Dense(len(emotion_dict), activation = 'softmax')(x)

model = Model(inputs = input_layer, outputs = output_layer)

model.compile(
    optimizer = Adam(learning_rate = 0.001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


%%time

history = model.fit(
    train_generator,
    validation_data = (X_validation, y_validation),
    epochs = 1000,
    steps_per_epoch = len(X_train) // 32,
    callbacks = [callback, checkpoint, reduce_lr]
)


best_model = load_model('best_cnn_model.keras')

test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize = (12, 6))
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = emotion_dict.values())
disp.plot(cmap = 'viridis', xticks_rotation = 'vertical')
plt.title('Confusion Matrix')
plt.show()


print(classification_report(y_true, y_pred_classes, target_names = emotion_dict.values()))


import numpy as np
from sklearn.metrics import confusion_matrix

# Load the best model
best_model = load_model('best_cnn_model.keras')

# Make predictions on the test set
y_pred_prob = best_model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred_prob, axis=1)  # Predicted class labels

# Convert y_test to class labels
y_test_classes = np.argmax(y_test, axis=1)  # True class labels

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Define emotion_dict if not already defined
emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Sad'}  # Replace with your actual emotion dictionary

# Calculate Sensitivity for each class
sensitivity = {}
for i in range(len(emotion_dict)):  # Assuming emotion_dict has the same number of classes as in your dataset
    TP = conf_matrix[i, i]  # True Positives for class i
    FN = np.sum(conf_matrix[i, :]) - TP  # False Negatives for class i
    sensitivity[i] = TP / (TP + FN) if (TP + FN) > 0 else 0  # Avoid division by zero

# Print Sensitivity for each class
for i, emotion in enumerate(emotion_dict.values()):
    print(f"Sensitivity (Recall/TPR) for {emotion}: {sensitivity[i]:.2f}")


import numpy as np
from sklearn.metrics import confusion_matrix

# Assuming y_test_classes contains the true labels and y_pred_classes contains the predicted labels
# Example: y_test_classes = np.array([0, 1, 2, 2, 1])  # True labels
# Example: y_pred_classes = np.array([0, 0, 2, 2, 1])  # Predicted labels

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Define emotion_dict if not already defined
emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Sad'}  # Replace with your actual emotion dictionary

# Calculate Specificity for each class
specificity = {}
for i in range(len(emotion_dict)):  # Assuming emotion_dict has the same number of classes as in your dataset
    # Calculate True Negatives (TN) for class i
    TN = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
    
    # Calculate False Positives (FP) for class i
    FP = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
    
    # Calculate Specificity
    specificity[i] = TN / (TN + FP) if (TN + FP) > 0 else 0  # Avoid division by zero

# Print Specificity for each class
for i, emotion in enumerate(emotion_dict.values()):
    print(f"Specificity (True Negative Rate / TNR) for {emotion}: {specificity[i]:.2f}")


import numpy as np
from sklearn.metrics import confusion_matrix

# Assuming y_test_classes contains the true labels and y_pred_classes contains the predicted labels
# Example: y_test_classes = np.array([0, 1, 2, 2, 1])  # True labels
# Example: y_pred_classes = np.array([0, 0, 2, 2, 1])  # Predicted labels

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Define emotion_dict if not already defined
emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Sad'}  # Replace with your actual emotion dictionary

# Calculate PPV and NPV for each class
ppv = {}
npv = {}
for i in range(len(emotion_dict)):  # Assuming emotion_dict has the same number of classes as in your dataset
    # Calculate True Positives (TP) for class i
    TP = conf_matrix[i, i]
    
    # Calculate False Positives (FP) for class i
    FP = np.sum(conf_matrix[:, i]) - TP
    
    # Calculate PPV
    ppv[i] = TP / (TP + FP) if (TP + FP) > 0 else 0  # Avoid division by zero
    
    # Calculate True Negatives (TN) for class i
    TN = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
    
    # Calculate False Negatives (FN) for class i
    FN = np.sum(conf_matrix[i, :]) - conf_matrix[i, i]
    
    # Calculate NPV
    npv[i] = TN / (TN + FN) if (TN + FN) > 0 else 0  # Avoid division by zero

# Print PPV and NPV for each class
for i, emotion in enumerate(emotion_dict.values()):
    print(f"Positive Predictive Value (PPV) for {emotion}: {ppv[i]:.2f}")
    print(f"Negative Predictive Value (NPV) for {emotion}: {npv[i]:.2f}")


import matplotlib.pyplot as plt

# หลังจากการฝึกโมเดลแล้ว
history = model.fit(
    train_generator,
    validation_data=(X_validation, y_validation),
    epochs=100,
    steps_per_epoch=len(X_train) // 32,
    callbacks=[callback, checkpoint, reduce_lr]
)

# แสดงกราฟฝึกและทดสอบ
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

