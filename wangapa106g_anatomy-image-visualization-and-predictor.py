import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings


train = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv')


print("Total Cases: ", len(train))


train.columns


figure, axis = plt.subplots(1,3, figsize=(20,5)) 
for idx, d in enumerate(['foraminal', 'subarticular', 'canal']):
    diagnosis = list(filter(lambda x: x.find(d) > -1, train.columns))
    dff = train[diagnosis]
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        value_counts = dff.apply(pd.value_counts).fillna(0).T
    value_counts.plot(kind='bar', stacked=True, ax=axis[idx])
    axis[idx].set_title(f'{d} distribution')


# List out all of the Studies we have on patients.
part_1 = os.listdir('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images')
part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))


df_meta_f = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')


p1 = [(x, f"/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/{x}") for x in part_1]
meta_obj = { p[0]: { 'folder_path': p[1], 
                    'SeriesInstanceUIDs': [] 
                   } 
            for p in p1 }


for m in meta_obj:
    meta_obj[m]['SeriesInstanceUIDs'] = list(
        filter(lambda x: x.find('.DS') == -1, 
               os.listdir(meta_obj[m]['folder_path'])
              )
    )


# grabs the correspoding series descriptions
for k in tqdm(meta_obj):
    for s in meta_obj[k]['SeriesInstanceUIDs']:
        if 'SeriesDescriptions' not in meta_obj[k]:
            meta_obj[k]['SeriesDescriptions'] = []
        try:
            meta_obj[k]['SeriesDescriptions'].append(
                df_meta_f[(df_meta_f['study_id'] == int(k)) & 
                (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
        except:
            print("Failed on", s, k)


meta_obj[list(meta_obj.keys())[1]]


patient = train.iloc[1]


ptobj = meta_obj[str(patient['study_id'])]


print(ptobj)


# Get data into the format
"""
im_list_dcm = {
    '{SeriesInstanceUID}': {
        'images': [
            {'SOPInstanceUID': ...,
             'dicom': PyDicom object
            },
            ...,
        ],
        'description': # SeriesDescription
    },
    ...
}
"""
im_list_dcm = {}
for idx, i in enumerate(ptobj['SeriesInstanceUIDs']):
    im_list_dcm[i] = {'images': [], 'description': ptobj['SeriesDescriptions'][idx]}
    images = glob.glob(f"{ptobj['folder_path']}/{ptobj['SeriesInstanceUIDs'][idx]}/*.dcm")
    for j in sorted(images, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
        im_list_dcm[i]['images'].append({
            'SOPInstanceUID': j.split('/')[-1].replace('.dcm', ''), 
            'dicom': pydicom.dcmread(j) })


# Function to display images
def display_images(images, title, max_images_per_row=4):
    # Calculate the number of rows needed
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # Ceiling division

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5, 1.5 * num_rows))
    
    # Flatten axes array for easier looping if there are multiple rows
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for consistency

    # Plot each image
    for idx, image in enumerate(images):
        ax = axes[idx]
        ax.imshow(image, cmap='gray')  # Assuming grayscale for simplicity, change cmap as needed
        ax.axis('off')  # Hide axes

    # Turn off unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()


for i in im_list_dcm:
    display_images([x['dicom'].pixel_array for x in im_list_dcm[i]['images']], 
                   im_list_dcm[i]['description'])


df_coor = pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')


df_coor.head()


def display_coor_on_img(c, i, title):
    center_coordinates = (int(c['x']), int(c['y']))
    radius = 10
    color = (255, 0, 0)  # Red color in BGR
    thickness = 2
    IMG = i['dicom'].pixel_array
    IMG_normalized = cv2.normalize(IMG, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    IMG_with_circle = cv2.circle(IMG_normalized.copy(), center_coordinates, radius, color, thickness)
    
    # Convert the image from BGR to RGB for correct color display in matplotlib
    IMG_with_circle = cv2.cvtColor(IMG_with_circle, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(IMG_with_circle)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.title(title)
    plt.show()



coor_entries = df_coor[df_coor['study_id'] == int(patient['study_id'])]


print("Only showing severe cases for this patient")
for idc, c in coor_entries.iterrows():
    for i in im_list_dcm[str(c['series_id'])]['images']:
        if int(i['SOPInstanceUID']) == int(c['instance_number']):
            try:
                patient_severity = patient[
                    f"{c['condition'].lower().replace(' ', '_')}_{c['level'].lower().replace('/', '_')}"
                ]
            except Exception as e:
                patient_severity = "unknown severity"
            title = f"{i['SOPInstanceUID']} \n{c['level']}, {c['condition']}: {patient_severity} \n{c['x']}, {c['y']}"
            if patient_severity == 'Severe':
                display_coor_on_img(c, i, title)


train_label_coordinates=pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv')
train_label_coordinates.head()


train_series_descriptions=pd.read_csv('/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')
train_series_descriptions.head(2)


train_label_coordinates['label']=train_label_coordinates['condition'].str.lower().str.replace(' ','_')+'_'+train_label_coordinates['level'].str.lower().str.replace("/", "_").str.replace(" ", "_")
train_label_coordinates.head(2)


final_df = pd.merge(train_label_coordinates, train, on='study_id', how='left')

def get_case(row):
    # Check if the label in the final_df matches any of the columns in df
    if row['label'] in train.columns:
        return row[row['label']]
    else:
        return None
final_df['case'] = final_df.apply(get_case, axis=1)

final_df = final_df[['study_id', 'series_id', 'instance_number', 'label', 'x', 'y', 'case']]
final_df = pd.merge(final_df, train_series_descriptions[['series_id', 'series_description']],
                    on='series_id', how='left')


final_df.head(2)


import os

# Function to return DICOM image paths and their cases
def get_dicom_image_paths(final_df, im_list_dcm, main_directory):
    
    study_dicoms = {}

    for label in final_df['label'].unique():
        study_dicoms[label] = {}

        label_df = final_df[final_df['label'] == label]

        # Loop through each study_id for the current label
        for study_id in label_df['study_id'].unique():
            # Get the corresponding series_ids and case for the study_id
            study_data = label_df[label_df['study_id'] == study_id]

            # Initialize the dictionary to hold series_id -> paths mapping
            study_dicoms[label][study_id] = {}

            # Loop through each row in the filtered study_data
            for _, row in study_data.iterrows():
                series_id = row['series_id']
                series_description = row['series_description']
                case = row['case']
                instance_number = row['instance_number']

                # If this study_id and series_id combination doesn't exist yet in the dictionary, create it
                if series_id not in study_dicoms[label][study_id]:
                    study_dicoms[label][study_id][series_id] = {
                        'series_description': series_description,
                        'dicom_paths': [],
                        'case': case
                    }

                # Generate the DICOM path based on the new logic
                dicom_path = os.path.join(main_directory, str(study_id), str(series_id), f"{instance_number}.dcm")
                
                # Append the generated DICOM path
                study_dicoms[label][study_id][series_id]['dicom_paths'].append(dicom_path)
            
    return study_dicoms

main_directory = '/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images'
study_dicoms = get_dicom_image_paths(final_df, im_list_dcm, main_directory)



# Visualize the first element in the dictionary
first_label = list(study_dicoms.keys())[0]  # Get the first label
first_study_id = list(study_dicoms[first_label].keys())[0]  # Get the first study_id for that label
first_series_id = list(study_dicoms[first_label][first_study_id].keys())[0]  # Get the first series_id for that study_id

# Get the details for the first series_id
first_series_data = study_dicoms[first_label][first_study_id][first_series_id]

# Print the results
print(f"First Label: {first_label}")
print(f"First Study ID: {first_study_id}")
print(f"First Series ID: {first_series_id}")
print(f"Series Description: {first_series_data['series_description']}")
print(f"Case: {first_series_data['case']}")
print(f"DICOM Paths: {first_series_data['dicom_paths']}")



# Function to extract all DICOM paths and their corresponding cases for a specific label
def get_dicom_paths_for_label(study_dicoms, label):
    dicom_paths = []
    cases = []

    # Loop through each study_id and series_id for the given label
    if label in study_dicoms:
        for study_id in study_dicoms[label]:
            for series_id in study_dicoms[label][study_id]:
                # Get the dicom paths and case for the current series_id
                dicom_paths.extend(study_dicoms[label][study_id][series_id]['dicom_paths'])
                cases.extend([study_dicoms[label][study_id][series_id]['case']] * len(study_dicoms[label][study_id][series_id]['dicom_paths']))

    return dicom_paths, cases




import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def process_labels(study_dicoms, label):
    """
    Processes the given label by performing the following steps:
    1. Extracts DICOM paths and cases for the label.
    2. Cleans the case labels by removing special characters.
    3. Applies OneHotEncoding to the cleaned labels.
    4. Prepares a TensorFlow dataset with image paths and encoded labels.
    
    Args:
        study_dicoms (dict): Dictionary containing study DICOM paths.
        label (str): The label to process.
    
    Returns:
        tf.data.Dataset: A TensorFlow dataset containing image paths and one-hot encoded labels.
    """
    dicom_paths, cases = get_dicom_paths_for_label(study_dicoms, label)
    
    df_ylabel = pd.DataFrame(cases, columns=['case'])
    
    pattern = r'[^\w\s]'
    df_ylabel = df_ylabel.replace(pattern, '', regex=True)
    
    # One-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = onehot_encoder.fit_transform(df_ylabel[['case']])
    
    # Convert to DataFrame
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=onehot_encoder.categories_[0])
    
    # Concatenate with original DataFrame
    df_train = pd.concat([df_ylabel, one_hot_df], axis=1)
    
    # Ensure file names are properly cast to string
    file_names = tf.convert_to_tensor([str(path) for path in dicom_paths], dtype=tf.string)
    labels = tf.cast(df_train.iloc[:, 1:].values, dtype=tf.float32)  # Exclude the 'case' column
    
    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
    
    return dataset
    
dataset = process_labels(study_dicoms, 'spinal_canal_stenosis_l1_l2')



for elements in dataset.take(6):
  print(elements[0].numpy(),elements[1].numpy())



import tensorflow as tf
import tensorflow_io as tfio

def load_image(file_name, label):
    #file_name = tf.py_function(func=process_dicom, inp=[file_name], Tout=tf.string)

    # Skip corrupt files
    if tf.strings.length(file_name) == 0:
        return tf.zeros([128, 128, 1], dtype=tf.float32), tf.zeros([3], dtype=tf.float32)

    raw = tf.io.read_file(file_name)
    dicom_array = tfio.image.decode_dicom_image(raw, dtype=tf.uint16)
    tensor = tf.cast(dicom_array, tf.float32)
    
    # Check if the tensor has the correct number of dimensions (3D: height, width, channels)
    tensor_shape = tf.shape(tensor)
    if len(tensor_shape) != 3:  # It should be 3D (height, width, channels)
        print(f"❌ Skipping file due to incorrect shape: {tensor_shape}")
        return tf.zeros([128, 128, 1], dtype=tf.float32), tf.zeros([3], dtype=tf.float32)

    tensor = tf.image.resize(tensor, [128, 128])  # Resize only if valid shape
    tensor = tf.reshape(tensor, [128, 128, 1])  # Ensure it's 3D (height, width, channels)
    tensor = tf.cast(tensor, tf.float32) / 255.0  # Normalize

    target = tf.keras.utils.to_categorical(label, num_classes=3)
    label = tf.convert_to_tensor(target, dtype=tf.float32)

    return tensor, label



dataset=dataset.map(lambda x,y:load_image(x,y))


for elements in dataset.take(6):
  print("image shape:",elements[0].shape,"labels shape:",elements[1].shape)


BUFFER_SIZE = 1000
BATCH_SIZE = 16
NUM_EPOCHS = 20


train_size = int(0.8 * 1903)  
val_size = 1903 - train_size

x_train = dataset.take(train_size).batch(BATCH_SIZE)
x_val = dataset.skip(train_size).batch(BATCH_SIZE)


for elements in x_train.take(6):
  print("image shape:",elements[0].shape,"labels shape:",elements[1].shape)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

model = Sequential([
    Conv2D(32, (3, 3), padding='same', data_format='channels_last',
           activation='relu', name='conv_1', input_shape=(128, 128, 1)),  # Adjusted input shape
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),

    Conv2D(64, (3, 3), padding='same', data_format='channels_last',
           activation='relu', name='conv_2'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),

    Conv2D(128, (3, 3), padding='same', data_format='channels_last',
           activation='relu', name='conv_3'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),

    Flatten(),
    Dense(1024, activation='relu', name='dense_1'),
    Dense(3, activation='softmax', name='output')  # Adjusted output activation to sigmoid
])



model.build(input_shape=(None , 128, 128, 1))
model.summary()


from tensorflow.keras.optimizers.schedules import ExponentialDecay

lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


history = model.fit(
    x_train,  # Training data
    epochs=10,  # Number of epochs to train the model
    validation_data=x_val,
    verbose=1 
)


# Function to plot training history
def plot_training_history(history):
    # Get the loss and accuracy values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Call the function to visualize training
plot_training_history(history)





model2 = Sequential([
    Conv2D(32, (3, 3), padding='same', data_format='channels_last',
           activation='relu', name='conv_1', input_shape=(128, 128, 1)),  # Adjusted input shape
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),

    Conv2D(64, (3, 3), padding='same', data_format='channels_last',
           activation='relu', name='conv_2'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),

    Conv2D(128, (3, 3), padding='same', data_format='channels_last',
           activation='relu', name='conv_3'),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),

    Flatten(),
    Dense(1024, activation='relu', name='dense_1'),
    Dense(3, activation='softmax', name='output')  # Adjusted output activation to sigmoid
])



model2.build(input_shape=(None , 128, 128, 1))
model2.summary()


lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9
)
optimizer = Adam(learning_rate=lr_schedule)

model2.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


label1 = 'left_neural_foraminal_narrowing_l5_s1'
dataset2 = process_labels(study_dicoms, label1)
for elements in dataset2.take(6):
  print(elements[0].numpy(),elements[1].numpy())


dataset2=dataset2.map(lambda x,y:load_image(x,y))


train_size = int(0.6 * 1903)  # Adjust the split ratio
val_size = 1903 - train_size

x_train1 = dataset2.take(train_size).batch(BATCH_SIZE)
x_val1 = dataset2.skip(train_size).batch(BATCH_SIZE)


history = model2.fit(
    x_train1,  # Training data
    epochs=10,  # Number of epochs to train the model
    validation_data=x_val1,
    verbose=1 
)


label2 = 'spinal_canal_stenosis_l2_l3'
dataset3 = process_labels(study_dicoms, label2)


dataset3=dataset3.map(lambda x,y:load_image(x,y))



inputs_test = []
labels_test = []

for batch_inputs, batch_labels in dataset3.batch(BATCH_SIZE):
    inputs_test.append(batch_inputs)
    labels_test.append(batch_labels)

# Convert inputs and labels to numpy arrays (if needed)
inputs_test = np.concatenate(inputs_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)

# Ensure your model has been trained and is ready for prediction

# Predict on the test set
predictions = model2.predict(inputs_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(labels_test, axis=1)  # Convert one-hot labels to class indices

# Evaluate the model
test_loss, test_accuracy = model2.evaluate(inputs_test, labels_test, verbose=0)

# Print results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Optionally, print some comparison of predicted vs actual values
print(f"Predicted Classes: {predicted_classes[:100]}")
print(f"True Classes: {true_classes[:100]}")





