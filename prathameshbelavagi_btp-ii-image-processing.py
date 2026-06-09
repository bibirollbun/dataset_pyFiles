# Importing Libararies
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# Remove Warnings
import warnings
warnings.filterwarnings('ignore')


# Lets Load the Training Data
data=pd.read_csv('/kaggle/input/paddy-disease-classification/train.csv')
data.head()


# Lets print the shape of the Training Data
print('The shape of the Training Data is:', {data.shape})
print('The Number of Rows  in the Training Data is:', data.shape[0])
print('The Number of Column  in the Training Data is:', data.shape[1])


# Lets check the unique value list of the label column
unique_values = data['label'].nunique()
print("Total number of unique values in label column are :", unique_values)
print('-----------------------------------------')
print('Unique values of label column are:')
data['label'].unique().tolist()


# Lets check the unique value list of the label column
unique_values_variety = data['variety'].nunique()
print("Total number of unique values in variety column are :", unique_values_variety)
print('-----------------------------------------')
print('Unique values of variety column are:')
data['variety'].unique().tolist()


fig, axes = plt.subplots(1, 1, figsize=(12, 6))
sns.countplot(data=data, x='variety', ax=axes, palette='magma')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Variety', fontsize=12)
plt.title('Count plot of the Variety column', fontsize=14)

# Adding count labels to the bars
for i in axes.containers:
    axes.bar_label(i, label_type='edge', fontsize=10, padding=5)

plt.show()



fig, ax = plt.subplots(figsize=(20, 5))

sns.countplot(x='label', data=data, ax=ax, palette='Dark2')
ax.set_title('Count plot of the Label column', fontsize=15)
ax.set_xlabel('Label', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=10)

# Add text labels to each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='white', xytext=(0, 5),
                textcoords='offset points', weight='bold')
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, -10),
                textcoords='offset points', weight='bold')

plt.show()



data['age'].describe()


max_age_variety = data[data['age'] == 82]['variety'].iloc[0]
print(f"The variety with the maximum age of 82 is **{max_age_variety}**.")





fig = px.histogram(data_frame=data, x='age', color='variety', nbins=20, color_discrete_sequence=px.colors.qualitative.Dark24)
fig.show()



# Lets Filter the label column based on Normal
normal=data[data['label']=='normal']
# Lets Filter the variety column based on ADT45
normal=normal[normal['variety']=='ADT45']
five_normal=normal.image_id[:5].values
five_normal.tolist()


# Lets Filter the label column based on dead_heart
dead=data[data['label']=='dead_heart']
# Lets Filter the variety column based on ADT45
dead=dead[dead['variety']=='ADT45']
five_dead=dead.image_id[:5].values
five_dead.tolist()


plt.figure(figsize=(20,10))
columns=5
path='/kaggle/input/paddy-disease-classification/train_images/'
for i, image_loc in enumerate(np.concatenate((five_normal, five_dead))):
    plt.subplot(10//columns + 1, columns, i+1)
    
    if i<5:
        
        image=plt.imread(path+'normal/'+image_loc)
        plt.title('normal')
    else:
        image=plt.imread(path+'dead_heart/'+image_loc)
        plt.title('dead_heart')

    plt.imshow(image)   




images=['/kaggle/input/paddy-disease-classification/train_images/hispa/100003.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/tungro/100013.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_blight/100126.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/downy_mildew/100037.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/blast/100040.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_streak/100178.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/normal/100066.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/brown_spot/100064.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/dead_heart/100047.jpg',\
       '/kaggle/input/paddy-disease-classification/train_images/bacterial_panicle_blight/100352.jpg']
diseases = ['hispa','tungro','bacterial_leaf_blight','downy_mildew','blast','bacterial_leaf_streak',\
           'normal','brown_spot','dead_heart','bacterial_panicle_blight']
diseases = [disease + ' image' for disease in diseases]
plt.figure(figsize=(20,10))
columns = 5
for i, image_loc in enumerate(images):
    plt.subplot(len(images)//columns + 1, columns, i + 1)
    image=plt.imread(image_loc)
    plt.title(diseases[i])
    plt.imshow(image)



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder()
data['label']=onehotencoder.fit_transform(data[['label']]).todense()
data['variety']=onehotencoder.fit_transform(data[['variety']]).todense()
data.head()


batch_size=32
image_width=224
image_height=224


import tensorflow as tf 
train_ds=tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(image_width, image_height),
    batch_size=batch_size)


import tensorflow as tf 
val_ds=tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_width, image_height),
    batch_size=batch_size)


class_name=train_ds.class_names
print('class Names:\n',class_name)


import matplotlib.pyplot as plt

# Get the first batch of images and labels from the training dataset
images, labels = next(iter(train_ds))

# Define the class names
class_name = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 
              'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']

# Create subplots with increased figsize
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10))

# Iterate over the subplots and plot the images with corresponding class names
for i, ax in enumerate(axes.flatten()):
    # Plot the image
    ax.imshow(images[i].numpy().astype("uint8"))
    
    # Check if the label index is within the range of class_name list
    if labels[i] < len(class_name):
        # Set the title as the corresponding class name
        ax.set_title(class_name[labels[i]])
    else:
        # Set a generic title if the label index is out of range
        ax.set_title("Unknown Class")
    
    # Remove axis labels
    ax.axis("off")

# Adjust subplot spacing
plt.subplots_adjust(hspace=0.25, wspace=0.35)

# Show the plot
plt.show()


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


normalized_layer=tf.keras.layers.Rescaling(1./255)


normalized_ds=train_ds.map(lambda x,y:(normalized_layer(x), y))
image_batch, labels_ds=next(iter(normalized_ds))
# print("Image Data Shape: ", image_ds.shape)
first_image=image_batch[0]
print(np.min(first_image), np.max(first_image))


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes=len(class_name)
num_classes


model=tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# Compile the Model
model.compile(optimizer='adam',  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# # Fit the model
# model.fit(train_ds, validation_data=val_ds, epochs=1)


%%time
early_stopping=EarlyStopping(patience=15)
# Fit the model
history=model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[early_stopping])
# Evaluate the model
loss, accuracy=model.evaluate(val_ds)
# plot the Training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Model loss')
plt.show()
# plot the Training  and validation Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.title('Model accuracy')
plt.show()


# Load and preprocess the test images
test_image_paths = ['/kaggle/input/paddy-disease-classification/test_images/200001.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200002.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200003.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200004.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200005.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200006.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200007.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200008.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200009.jpg',\
                    '/kaggle/input/paddy-disease-classification/test_images/200010.jpg']# List of test image paths

test_images = []
for image_path in test_image_paths:
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(image_height, image_width))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # Rescale pixel values to [0, 1]
    test_images.append(image)

test_images = np.array(test_images)

# Predict on the test images
predictions = model.predict(test_images)


class_labels = np.argmax(predictions, axis=1)

submission_df = pd.DataFrame({'ImageID': test_image_paths, 'Label': class_labels})
submission_df.to_csv('output.csv', index=False)


# Load the test data
test_dir='/kaggle/input/paddy-disease-classification/test_images'
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size)

# Predict the labels of the test set
predictions = model.predict(test_ds)

# Convert the predictions to class labels
predicted_labels = tf.argmax(predictions, axis=1)

# Create a submission file
output_df = pd.DataFrame({'image_id': test_ds.file_paths, 'label': predicted_labels})
output_df['image_id'] = output_df['image_id'].apply(lambda x: x.split('/')[-1])
output_df.to_csv('output.csv', index=False)

