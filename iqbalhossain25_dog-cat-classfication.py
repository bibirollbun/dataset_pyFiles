import numpy as np
import pandas as pd 
import random
import os
import zipfile
from PIL import Image
import seaborn as sns
import shutil
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization,Input,GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,
                             EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split


print(os.listdir("../input"))


# Set the path for the input directory where the zip files are located
input_path = '/kaggle/input/dogs-vs-cats'

# Define the paths for the train and test zip files
train_zip = os.path.join(input_path, 'train.zip')
test_zip = os.path.join(input_path, 'test1.zip')

# Set the output paths where the unzipped data will be extracted
output_train_path = '/kaggle/working/train'
output_test_path = '/kaggle/working/test'

# Unzip the training data to the specified output path
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall(output_train_path)
    print("Unzipping training data complete.")

# Unzip the test data to the specified output path
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall(output_test_path)
    print("Unzipping testing data complete.")



train_data_path='/kaggle/working/train/train'
test_data_path='/kaggle/working/test/test1'

train_images=os.listdir(train_data_path)
test_images=os.listdir(test_data_path)


image_name=[]
category=[]
code=[]
size=[]
aspect_ratio=[]

for image in train_images:
    image_name.append(image)
    cate=image.split('.')[0]
    category.append(cate)
    code.append(1) if cate=='dog'  else code.append(0)
    
    img_path=os.path.join(train_data_path,image)    
    # Read the image to get its size (height, width)
    img=cv2.imread(img_path)
    size.append((img.shape[0],img.shape[1]))
    # Calculate and append the aspect ratio (height/width) of the image
    aspect_ratio.append(img.shape[0]/img.shape[1])

train_df=pd.DataFrame({'Image_Name':image_name,'Category':category,'Code':code,"Size":size,'Aspect_ratio':aspect_ratio})
train_df


# Define a function to display the first 9 images from a DataFrame 
def display_first_9(path,df):
    plt.figure(figsize=[30,30])
    for i in range(9):
        plt.subplot(3,3,i+1)
        img_path=os.path.join(path,df['Image_Name'].iloc[i])
        plt.title(df['Image_Name'][i])
        img=Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


df_sorted=train_df.sort_values(by='Aspect_ratio',ascending=False)
display_first_9(train_data_path,df_sorted)


df_sorted=train_df.sort_values(by='Aspect_ratio',ascending=True)
display_first_9(train_data_path,df_sorted)


ax=sns.countplot(data=train_df,x='Category',palette='viridis')
ax.bar_label(ax.containers[0])
plt.show()


# First split: 80% training and 20% temporary (validation + test)
train_df, temp_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Category'])

# Second split: 50% of temp_df for validation and 50% for test (10% of original data each)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Category'])


train_df.sort_values(by='Size',ascending=False)


image_size=(224,224)
image_channels=3
image_shape=(image_size[0],image_size[1],3)
batch_size = 32
epochs = 5


train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    horizontal_flip=True,
                                    zoom_range=0.2,
                                    width_shift_range=0.3,
                                    height_shift_range=0.3,
                                    rotation_range=0.3,
                                    shear_range=0.2,
                                    fill_mode='nearest')

train_generator = train_data_gen.flow_from_dataframe(
                                    dataframe=train_df,          
                                    directory=train_data_path,   
                                    x_col='Image_Name',
                                    y_col='Category',
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    class_mode="binary",
                                    target_size=image_size)



val_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_data_gen.flow_from_dataframe(
                                    dataframe=val_df,            
                                    directory=train_data_path,   
                                    x_col='Image_Name',
                                    y_col='Category',
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    class_mode="binary",
                                    target_size=image_size)


test_generator = val_data_gen.flow_from_dataframe(
                                    dataframe=test_df,            
                                   directory=train_data_path,   
                                    x_col='Image_Name',
                                    y_col='Category',
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    class_mode="binary",
                                    target_size=image_size)


sample_batch=next(train_generator)
images,labels=sample_batch



plt.figure(figsize=(15,15))
for i in range(batch_size):
    plt.subplot(8,4,i+1)
    plt.imshow(images[i])
    label_index = np.argmax(labels[i])
    label='Cat' if label_index ==0 else 'Dog'
    plt.title(label)
    plt.axis('off')
plt.tight_layout()
plt.show()


ResNet=ResNet50(weights='imagenet',include_top=False,input_shape=(image_size[0],image_size[1],3))


ResNet.summary()


# Get the output from the pre-trained ResNet model
resnet_output = ResNet.output

# Apply Global Average Pooling to reduce the dimensionality of the feature maps
gb = GlobalAveragePooling2D()(resnet_output)

# Add a Dropout layer to help prevent overfitting during training
drop = Dropout(0.5)(gb)

# Add a fully connected Dense layer with 1024 units and ReLU activation
# Apply L2 regularization to prevent overfitting
dense = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(drop)

# Add another Dropout layer after the Dense layer
drop = Dropout(0.5)(dense)

# Create the output layer with 1 unit (binary classification: dog or cat) using sigmoid activation
predictions = Dense(1, activation='sigmoid')(drop)

# Define the model with the specified inputs (ResNet input) and outputs (predictions)
model = Model(inputs=ResNet.input, outputs=predictions)

# Display the model summary to show the architecture
model.summary()


checkpoint = ModelCheckpoint('../working/Resnet50_best.weights.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

callbacks_list = [checkpoint, reduceLROnPlat, early]


#the total number of images we have:
train_size = len(train_generator.filenames)
#train_steps is how many steps per epoch Keras runs the genrator. One step is batch_size*images
train_steps = train_size/batch_size
#use 2* number of images to get more augmentations in. some do, some dont. up to you
train_steps = int(2*train_steps)

#same for the validation set
valid_size = len(val_generator.filenames)
valid_steps = valid_size/batch_size
valid_steps = int(2*valid_steps) 

#same for the test set
test_size = len(test_generator.filenames)
test_steps = test_size/batch_size
test_steps = int(2*test_steps) 


# Step 1: Freeze the ResNet layers
for layer in ResNet.layers:
    layer.trainable = False

# Step 2: Compile the model with frozen layers
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the model with frozen ResNet layers
history=model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=valid_steps,
    callbacks=callbacks_list,
    verbose=1
)



# Step 4: Unfreeze the ResNet layers for fine-tuning
for layer in ResNet.layers:
    layer.trainable = True

# Step 5: Re-compile the model after unfreezing the layers (with potentially a lower learning rate)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the model again with the unfrozen ResNet layers (fine-tuning)
new_history=model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=valid_steps,
    callbacks=callbacks_list,
    verbose=1
)


history.history['loss'] += new_history.history['loss']
history.history['val_loss'] += new_history.history['val_loss']
history.history['accuracy'] += new_history.history['accuracy']
history.history['val_accuracy'] += new_history.history['val_accuracy']


history.history.keys()


loss=history.history['loss']
val_loss=history.history['val_loss']
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

# Find the epoch with the minimum validation loss (best performance in terms of loss)
index_loss = np.argmin(val_loss)
# Find the epoch with the maximum validation accuracy (best performance in terms of accuracy)
index_acc = np.argmax(val_acc)

# Get the lowest validation loss value at the best epoch
val_lowest = val_loss[index_loss]
# Get the highest validation accuracy value at the best epoch
val_highest = val_acc[index_acc]

# Create a list of epoch numbers, starting from 1, for plotting purposes
epochs=[i+1 for i in range(len(acc))]

# Create labels for plotting the best epoch in terms of loss and accuracy
loss_label = f'Best Epoch = {str(index_loss + 1)}'
acc_label = f'Best Epoch = {str(index_acc + 1)}'


# Set the style of the plot using 'fivethirtyeight' for a cleaner look
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15,8))

# Create the first subplot (1 row, 2 columns, this is the first plot)
plt.subplot(1,2,1)
# Plot training loss across epochs
plt.plot(epochs, loss, 'r', label='Training Loss')
# Plot validation loss across epochs
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
# Highlight the point with the lowest validation loss with a blue dot
plt.scatter((index_loss+1), val_lowest, label=loss_label, s=150, c='b')
# Set the title, x-axis label, and y-axis label for the loss plot
plt.title('Training vs Validation (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# Show the legend to differentiate between training and validation curves
plt.legend()

# Create the second subplot (2nd plot of the 1 row, 2 column layout)
plt.subplot(1,2,2)
# Plot training accuracy across epochs
plt.plot(epochs, acc, 'r', label='Training Accuracy')
# Plot validation accuracy across epochs
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
# Highlight the point with the highest validation accuracy with a blue dot
plt.scatter((index_acc+1), val_highest, label=acc_label, s=150, c='b')
# Set the title, x-axis label, and y-axis label for the accuracy plot
plt.title('Training vs Validation (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# Show the legend to differentiate between training and validation curves
plt.legend()

plt.tight_layout()
plt.show()


history.history


# Exclude learning_rate if it's of a different length
history_dict = {k: v for k, v in history.history.items() if len(v) == len(history.history['accuracy'])}
    
# Create DataFrame
df_history = pd.DataFrame(history_dict)
df_history


val_generator.reset()
df_valid = pd.DataFrame()


# Initialize lists to store differences, predictions, categories, and labels
diffs = []
predictions = []
cat_or_dog = []
labels = []

# Iterate through each filename in the validation generator
for file in val_generator.filenames:
    # Create the full image path
    img_path = os.path.join(train_data_path, file)
    img = Image.open(img_path)
    # Resize the image to the specified dimensions
    img = img.resize((image_size[0], image_size[1]))
    # Convert the image to a NumPy array
    img = np.array(img)
    
    # Determine the reference label based on the filename
    if 'cat' == file.split('.')[0]:
        ref = 0  
        cat_or_dog.append('cat')   
    else:
        ref = 1   
        cat_or_dog.append('dog')   
    
    labels.append(ref)
    pred = model.predict(preprocess_input(img[np.newaxis]))
    predictions.append(pred)
    diffs.append(np.abs(pred[0][0] - ref))


df_valid["filename"] = val_generator.filenames
df_valid["cat_or_dog"] = cat_or_dog
df_valid["label"] = labels
df_valid["diff"] = diffs
df_valid["prediction"] = predictions


sorted_diffs=df_valid.sort_values(by='diff',ascending=False)



# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



from sklearn.metrics import classification_report, confusion_matrix

# Predict on the validation set
predictions = model.predict(val_generator, verbose=1)
predictions = (predictions > 0.5)  # Convert probabilities to binary class

# Evaluate the performance with confusion matrix and classification report
print("Confusion Matrix")
print(confusion_matrix(val_generator.classes, predictions))
print("Classification Report")
print(classification_report(val_generator.classes, predictions))



# Display the model architecture
model.summary()



def plot_top_N(N, sorted_df):
    from math import ceil
    # Initialize index for subplot tracking
    i = 0
    rows = int(ceil(N / 3))  # Calculate number of rows needed
    height = rows * 10  # Determine plot height based on rows
    plt.figure(figsize=[30, height])  # Set figure size (width fixed, height scales)

    # Iterate over each row in the sorted DataFrame
    for index, row in sorted_df.iterrows():
        # Create subplot for the current image
        plt.subplot(rows, 3, i + 1)
        
        # Extract data from the DataFrame row
        file_name = row["filename"]
        category = row["cat_or_dog"]  # True label as string ('cat' or 'dog')
        true_label = row["label"]  # True label as 0 or 1
        diffs = row["diff"]  # Difference between prediction and actual
        predictions = row["prediction"]  # Predicted probability

        # Determine predicted label based on model prediction
        pred_label = 1 if predictions[0][0] > 0.5 else 0  # 1 = dog, 0 = cat
        predicted_category = 'dog' if pred_label == 1 else 'cat'  # Predicted category as string
        
        # Open and display image
        img = Image.open(f"{train_data_path}/{file_name}")
        plt.imshow(img)

        # Set title with model prediction and difference
        plt.title(f'It is a {category}, model predicted as {predicted_category} with {predictions[0][0]:.2f}, error {diffs:.2f}')

        i += 1  
        img.close()  # Close image to free memory

        if i >= N:  # Stop when the top N images are plotted
            break


plot_top_N(10,sorted_diffs)

