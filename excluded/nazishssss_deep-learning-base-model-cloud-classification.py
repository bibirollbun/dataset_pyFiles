import os
import io
import cv2
import time
import math
import random
import logging
import warnings

warnings.filterwarnings("ignore")
#logging.getLogger("tensorflow").setLevel(logging.WARNING)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import pandas as pd
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


print("GPU available!" if tf.test.is_gpu_available() else "GPU is not available")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


SEED = 42

IMG_WIDTH = 2100
IMG_HEIGHT = 1400

# training params
R_WIDTH = 576
R_HEIGHT = 384
NUM_CHANNELS = 3
NUM_CLASSES = 4
NUM_EPOCHS = 20 
BATCH_SIZE = 4
TEST_BATCH_SIZE = 32



print(tf.__version__)


!pip install -q segmentation-models



import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm


import tensorflow as tf
import segmentation_models as sm

print("TensorFlow version:", tf.__version__)
print("Segmentation Models version:", sm.__version__)



data_dir = '../input/understanding_cloud_organization'
train_csv_path = os.path.join(data_dir,'train.csv')
test_csv_path = os.path.join(data_dir,"sample_submission.csv")
train_image_path = os.path.join(data_dir,'train_images')
test_image_path = os.path.join(data_dir,'test_images')


train_df = pd.read_csv(train_csv_path).fillna(-1)
train_df.head()


train_df['Image_Id'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])
train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])
train_df.head()


train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)
train_df.head()


grouped_EncodedPixels = train_df.groupby('Image_Id')['Label_EncodedPixels'].apply(list)
grouped_EncodedPixels.head()
grouped_EncodedPixels.info()


train_df = grouped_EncodedPixels.to_frame().reset_index()
train_df.head()


labels = ['Fish', 'Flower', 'Gravel', 'Sugar']

for label in labels:
    train_df = train_df.assign(**{label: 0})
for index, row in train_df.iterrows():
    for item in row['Label_EncodedPixels']:
        label, value = item
        if value == -1:
            bool_value = 0
        else:
            bool_value = 1
        train_df.loc[index, label] = bool_value

train_df['classes'] = train_df.apply(lambda row: [col for col in labels if row[col] == 1], axis=1)

train_df.head()


train_df[labels].sum().plot(kind='bar')
plt.xlabel('Columns')
plt.ylabel('Frequency')


train_df.info()


# Finding the index of images having all the masks
for ix,item in enumerate(train_df['Label_EncodedPixels'][:100]):
    c1=item[0][-1]!=-1
    c2=item[1][-1]!=-1
    c3=item[2][-1]!=-1
    c4=item[3][-1]!=-1
    if c1 and c2 and c3 and c4:
        print(ix)
        


train_df.loc[28]


for ix,item in enumerate(os.listdir(train_image_path)):
    if item == "015aa06.jpg":
        print(ix)


test_df = pd.read_csv(test_csv_path).fillna(-1)
test_df.head()


test_df['Image_Id'] = test_df['Image_Label'].apply(lambda x: x.split('_')[0])
test_df['Label'] = test_df['Image_Label'].apply(lambda x: x.split('_')[1])
test_df.head()


test_df['Label_EncodedPixels'] = test_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)
test_df.head()


grouped_EncodedPixels = test_df.groupby('Image_Id')['Label_EncodedPixels'].apply(list)
grouped_EncodedPixels.head()
grouped_EncodedPixels.info()


test_df = grouped_EncodedPixels.to_frame().reset_index()
test_df.head()


INDEX = 5019


indexes =  [0, 1, 2, 3]
labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
colors = ['maroon', 'darkblue', 'purple','teal']
colormaps = ['PuRd_r', 'Blues_r', 'Purples_r','winter_r']
rgb_colors = [(56, 255, 255),(255, 70, 90),(48, 255, 99),(255, 255, 102)]

label_to_idx = dict(zip(labels,indexes))
idx_to_label = dict(zip(indexes,labels))

label_to_color = dict(zip(labels,colors))
idx_to_color = dict(zip(indexes,colors))

label_to_rgb_color =  dict(zip(labels,rgb_colors))
idx_to_rgb_color = dict(zip(indexes,rgb_colors))

label_to_colormap = dict(zip(labels,colormaps))
idx_to_colormap = dict(zip(indexes,colormaps))


class COLOR:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'




# Function to count nonzero pixels from an RLE string
def count_nonzero_pixels_from_rle(rle_string, height, width):
    if not isinstance(rle_string, str) or rle_string.strip() == "-1":
        return 0  # Return 0 for missing RLE

    try:
        rle_numbers = list(map(int, rle_string.strip().split()))

        # Ensure RLE format is correct (pairs of numbers)
        if len(rle_numbers) % 2 != 0:
            raise ValueError("Invalid RLE format.")

        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        return rle_pairs[:, 1].sum()  # Efficiently sum pixel lengths

    except Exception as e:
        print(f"Error processing RLE: {rle_string}. Error: {e}")
        return 0

# Class-wise pixel count dictionary
class_wise_pixel_count = {class_name: 0 for class_name in ["Fish", "Flower", "Gravel", "Sugar"]}

# Image dimensions
img_width = 2100
img_height = 1400

# Ensure train_df exists and has the correct column
if isinstance(train_df, pd.DataFrame) and "Label_EncodedPixels" in train_df.columns:
    for _, row in train_df.iterrows():
        rle_list = row.get("Label_EncodedPixels")

        # Ensure it's a valid list with 4 elements
        if isinstance(rle_list, list) and len(rle_list) == 4:
            for class_name, rle_data in zip(class_wise_pixel_count.keys(), rle_list):
                try:
                    # Extract RLE string (handling tuple cases)
                    rle_string = str(rle_data[1]).strip() if isinstance(rle_data, tuple) else str(rle_data).strip()

                    # Use the function to count nonzero pixels
                    class_wise_pixel_count[class_name] += count_nonzero_pixels_from_rle(rle_string, img_height, img_width)

                except Exception as e:
                    print(f"Error processing row: {row}\nError: {e}")

# Print final pixel counts
print(class_wise_pixel_count)





# Create figure
fig = plt.figure(figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
fig.suptitle('Classwise Pixel Count', fontsize=20)

# Create subplot
ax = plt.subplot(1, 2, 1)
bar = plt.bar(class_wise_pixel_count.keys(), class_wise_pixel_count.values(), color=colors)  # Using previous `colors`

# Add value labels on top of bars
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, f'{height:.3E}',
             ha='center', va='bottom', fontsize=10)

# Labels
plt.xlabel("Cloud Types")
plt.ylabel("Total Pixels")

# Adjust layout & show plot
plt.tight_layout()
plt.show()



ax = plt.subplot(1,2,2)
plt.pie(class_wise_pixel_count.values(),
        labels=class_wise_pixel_count.keys(),
        autopct='%1.1f%%',
        explode=[0.1,0,0,0],
        colors=colors,
        shadow=True,
        startangle=0);

print(COLOR.BOLD +COLOR.GREEN+ "Observation: The pixel distribution of the classes is somewhat balanced." + COLOR.END)


# Ensure train_df is not empty
if "train_df" in locals() and not train_df.empty:
    total_pixels = img_height * img_width * len(train_df)
    mask_pixels = sum(class_wise_pixel_count.values())

    # Avoid division by zero
    if total_pixels > 0:
        pixel_distribution = {
            "Background": (total_pixels - mask_pixels) / total_pixels * 100,
            "Fish": class_wise_pixel_count["Fish"] / total_pixels * 100,
            "Flower": class_wise_pixel_count["Flower"] / total_pixels * 100,
            "Gravel": class_wise_pixel_count["Gravel"] / total_pixels * 100,
            "Sugar": class_wise_pixel_count["Sugar"] / total_pixels * 100,
        }

        print(pixel_distribution)
    else:
        print("Error: Total pixels calculated as zero.")
else:
    print("Error: train_df is not defined or empty.")



fig = plt.figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
fig.suptitle('Pixel Distribution', fontsize=24)
fig.tight_layout();

ax = plt.subplot(1,2,1)
bar = plt.bar(pixel_distribution.keys(), pixel_distribution.values(), color=["darkslategrey"]+colors);
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, '%.3f %%' % height,
             ha='center', va='bottom',fontsize=10)

plt.xlabel("Pixel Types");
plt.ylabel("Parcentage of Pixels");



ax = plt.subplot(1,2,2)
plt.pie(pixel_distribution.values(),
        labels=pixel_distribution.keys(),
        autopct='%1.1f%%',
        explode=[0.1,0,0,0,0],
        shadow=True,
        colors=["darkslategrey"]+colors,
        startangle=180);


comment = '''Observation: The pixel distribution of the overall dataset is not balanced,
             most of the pixels does not have any associated label.'''

print(COLOR.BOLD +COLOR.GREEN+ comment + COLOR.END)


fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout();

class_frequency = dict(train_df[labels].sum())
bar = plt.bar(class_frequency.keys(),class_frequency.values(), color=colors)
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, height,
             ha='center', va='bottom',fontsize=10)

plt.xlabel("Labels",fontsize=16);
plt.ylabel('Frequency',fontsize=16)
plt.title("Mask Class Frequency",fontsize=20)
print(COLOR.BOLD +COLOR.GREEN+ "Observation: The Dataset is somewhat balanced for classifcation tasks." + COLOR.END)


fig = plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout();

class_frequency_per_image = dict(train_df["classes"].apply(len).value_counts())

bar = plt.bar(class_frequency_per_image.keys(), class_frequency_per_image.values(), color=colors);
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, height,
             ha='center', va='bottom',fontsize=10)
plt.xlabel("No. of Labels in a Single Image",fontsize=16);
plt.xticks(ticks=[1,2,3,4]);
plt.ylabel("Frequency",fontsize=16);
plt.title("Class Frequency per Image",fontsize=20);

print(COLOR.BOLD +COLOR.GREEN+ "Observation: Most of the images have 2 labels." + COLOR.END)




# Calculate average mask area
average_mask_area = dict()

# Ensure class_frequency is defined before this loop
for key in class_frequency.keys():
    if class_frequency[key] == 0:  # Avoid division by zero
        average_mask_area[key] = 0
    else:
        average_mask_area[key] = class_wise_pixel_count[key] // class_frequency[key]

# Plot the average mask area
fig = plt.figure(figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout()

# Ensure colors has the same length as number of classes
if 'colors' not in locals():
    colors = plt.cm.Paired.colors[:len(average_mask_area)]  # Auto-assign colors if undefined

# Bar plot for average mask area
bar = plt.bar(average_mask_area.keys(), average_mask_area.values(), color=colors)
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2, height, '%.3E' % height,  # Exponential notation for readability
             ha='center', va='bottom', fontsize=10)

# Labels and Title
plt.xlabel("Labels", fontsize=16)
plt.ylabel("Average No. of Pixels per Mask", fontsize=16)
plt.title("Average Mask Area", fontsize=20)

plt.show()

# Print observation
print(f"\033[1m\033[32mObservation: Average area (pixel count) for each mask is somewhat close.\033[0m")



from itertools import combinations

# Ensure `labels` is defined and contains class names
classes = labels if 'labels' in locals() else ["Fish", "Flower", "Gravel", "Sugar"]

# Generate all possible label combinations (from 1 to 4 classes)
combinations_list = [combo for i in range(1, len(classes) + 1) for combo in combinations(classes, i)]

# Check if train_df exists and contains the "classes" column
if 'train_df' in locals() and 'classes' in train_df.columns:
    # Count occurrences of each combination in the dataset efficiently
    label_counts = {
        tuple(combination): train_df["classes"].apply(lambda x: set(combination).issubset(set(x)) if isinstance(x, list) else False).sum()
        for combination in combinations_list
    }
    
    # Function to format dictionary keys by removing unwanted characters
    def format_tuple_key(t):
        return ", ".join(t)  # More readable than removing all characters

    # Apply formatting to dictionary keys
    cleaned_label_counts = {format_tuple_key(k): v for k, v in label_counts.items()}

    # Display results
    print(cleaned_label_counts)
else:
    print("Error: 'train_df' is not defined or 'classes' column is missing.")





# Ensure `labels` and `train_df["classes"]` are defined
labels = ["Fish", "Flower", "Gravel", "Sugar"]  # Replace with actual class labels
train_df["class_sets"] = train_df["classes"].map(lambda x: set(x) if isinstance(x, list) else set())

# Generate all possible label combinations
combinations_list = [combo for i in range(1, len(labels) + 1) for combo in combinations(labels, i)]

# Count occurrences of each combination in the dataset
label_counts = {
    tuple(combination): train_df["class_sets"].map(lambda x: set(combination).issubset(x)).sum()
    for combination in combinations_list
}

# Sort combinations by frequency
sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
label_names = [", ".join(k) for k, v in sorted_labels]  # Convert tuples to readable strings
frequencies = [v for k, v in sorted_labels]

# Set Matplotlib style **before** creating the figure
plt.style.use("ggplot")  # Alternative to seaborn-darkgrid

# Create figure
plt.figure(figsize=(12, 6), dpi=80)
plt.barh(label_names, frequencies, color='royalblue')

# Add frequency labels on bars
for index, value in enumerate(frequencies):
    plt.text(value + max(frequencies) * 0.01, index, str(value), ha='left', va='center',
             fontsize=10, fontweight="bold")

# Labels and title
plt.xlabel("Frequency (No. of Images)", fontsize=14)
plt.ylabel("Class Combinations", fontsize=14)
plt.title("Class Combination Frequency", fontsize=16, fontweight="bold")

# Improve readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="x", linestyle="--", alpha=0.6)

# Show plot
plt.show()



def batchDataLoader(image_dir,img_w= 512, img_h=512, num_channel =4, Batch_Size=32):
    
    while True:
        k=0
        image_ids = os.listdir(image_dir)
        num_batches = math.ceil(len(image_ids)/Batch_Size)
        
        for batch_no in range(1,num_batches+1): 
            if batch_no < num_batches:
                batch_size = Batch_Size
                batch_image_ids = image_ids[k:k+batch_size]
                image_batch = np.zeros((batch_size, img_h, img_w, num_channel),dtype=np.uint8)
                for i in range(batch_size):
                    path = os.path.join(image_dir, image_ids[i])
                    img = cv2.imread(path)
                    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_batch[i] = img
            # for the last batch which could be fractional
            if batch_no == num_batches:
                batch_image_ids = image_ids[k:]
                batch_size = len(batch_image_ids)
                image_batch = np.zeros((batch_size, img_h, img_w, num_channel),dtype=np.uint8)
                for i in range(batch_size):
                    path = os.path.join(image_dir, image_ids[i])
                    img = cv2.imread(path)
                    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_batch[i] = img
            
            k = k+batch_size
            print(f"batch_no = {batch_no}")
            yield image_batch


img_width = 2100
img_height = 1400
num_channel = 3
BATCH_SIZE = 32
current_batch = batchDataLoader(train_image_path,img_width,img_height, num_channel, BATCH_SIZE)



images = next(current_batch)
print(images.shape)
plt.figure(figsize=(24,8))
for i in range(8):
    ax = plt.subplot(2,4, i+1)
    plt.imshow(images[i])
    #plt.title(labels[i])
    plt.axis("off")


def rle_to_mask(rle_string, height, width):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
    
    rows, cols = height, width
    
    if rle_string == -1:
        return np.zeros((height,width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1,2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        img = img/255.0
        return img



def get_masks_by_img_id(dataframe, image_id):
    masks = np.zeros((img_height,img_width,4))
    rle_masks = list(dataframe[dataframe['Image_Id'] == image_id]['Label_EncodedPixels'])[0]
    fish_mask = rle_to_mask(rle_masks[0][1], img_height, img_width)
    flower_mask = rle_to_mask(rle_masks[1][1], img_height, img_width)
    gravel_mask = rle_to_mask(rle_masks[2][1], img_height, img_width)
    sugar_mask = rle_to_mask(rle_masks[3][1], img_height, img_width)
    mask_list = [fish_mask,flower_mask,gravel_mask,sugar_mask]
    for ix, mask in enumerate(mask_list):
        masks[:,:,ix] = mask
    return masks


image_id = '0011165.jpg'
path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rle = list(train_df[train_df['Image_Id'] == image_id]['Label_EncodedPixels'])[0][0][1]

m = rle_to_mask(rle,img_height,img_width)
m = cv2.resize(m, (384,256),interpolation=cv2.INTER_LINEAR)
m = (m>0).astype(int)
plt.imshow(m)
print(m.shape)
print(np.unique(m))
print(np.argwhere(m==1)[0])


image_id = os.listdir(train_image_path)[1]
#image_id = 'f516a20.jpg'
masks = get_masks_by_img_id(train_df, image_id)
masks.shape


image_id = os.listdir(train_image_path)[5390]
#image_id = 'f516a20.jpg'
masks = get_masks_by_img_id(train_df, image_id)
print(image_id)
plt.figure(figsize=(24,4))
for ix in range(masks.shape[-1]):
    ax = plt.subplot(1,4, ix+1)
    plt.imshow(masks[:,:,ix],cmap=None)
    plt.axis("off");


def draw_label_on_mask(mask, label, obj=plt):
    '''
    Function to add labels to the image.
    '''
    if np.sum(mask) > 0:
        y,x = 0,0
        y,x = np.argwhere(mask==1)[0]
        y,x = y+50,x+20      
        obj.text(x,y,label,color='white',)
    return None


image_id = os.listdir(train_image_path)[5390]
path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(384,256))
img = img.astype(np.float32)
img = img/255.0
#img -= img.mean()
#img /= img.std()
#standarization changes the color
print(img.shape)
plt.imshow(img);


image_id = os.listdir(train_image_path)[5390]
path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = get_masks_by_img_id(train_df, image_id)
mask = masks[:,:,1]
mask = np.clip(mask,0,1)
mask = np.ma.masked_where(mask == 0, mask)
plt.imshow(img)
plt.imshow(mask,alpha=0.7,cmap='PuRd_r')
draw_label_on_mask(mask,"Flower")
plt.axis('off');


image_id = os.listdir(train_image_path)[5390]
path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = get_masks_by_img_id(train_df, image_id)
mask = masks[:,:,1]
mask = np.clip(mask,0,1)
mask = np.ma.masked_where(mask == 0, mask)
bbox = cv2.boundingRect(mask.astype(np.uint8))
cv2.rectangle(img, bbox, (0, 255, 0), 5)
plt.imshow(img)
#plt.imshow(mask,alpha=0.7,cmap='PuRd_r')
draw_label_on_mask(mask,"Flower")
plt.axis('off');


image_id = os.listdir(train_image_path)[5390]
masks = get_masks_by_img_id(train_df, image_id)
masks = (masks[:,:,0], masks[:,:,1],masks[:,:,2],masks[:,:,3])

path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
colormaps = ['PuRd_r', 'Blues_r', 'Purples_r','winter_r'] # colormap_r = inverse colormap
mask_labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
plt.figure(figsize=(15,10))
for i,(mask,cmap,label)in enumerate(zip(masks,colormaps,mask_labels)):
    mask = np.clip(mask,0,1)
    mask = np.ma.masked_where(mask == 0, mask)
    ax = plt.subplot(2,2, i+1)
    plt.imshow(img)
    plt.imshow(mask,alpha=0.7,cmap=cmap) 
    draw_label_on_mask(mask,label)
    plt.axis("off")


image_id = os.listdir(train_image_path)[5390]
masks = get_masks_by_img_id(train_df, image_id)
masks = (masks[:,:,0], masks[:,:,1],masks[:,:,2],masks[:,:,3])

path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

colormaps = ['PuRd_r', 'Blues_r', 'Purples_r','winter_r']
mask_labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
plt.figure(figsize=(32,8))
plt.imshow(img)
for i,(mask,cmap,label) in enumerate(zip(masks,colormaps,mask_labels)):
    mask = np.clip(mask,0,1)
    mask = np.ma.masked_where(mask == 0, mask)
    plt.imshow(mask,alpha=0.7,cmap=cmap) # colormap_r = inverse colormap
    draw_label_on_mask(mask,label)
    plt.axis("off")


import cv2
import numpy as np
import matplotlib

def show_bounding_boxes(image, mask, labels, colors):
    """Shows the bounding boxes surrounding the polygon in the image, and
    adds labels to the bounding boxes.

    Args:
    image: The image.
    mask: The binary polygon mask.
    labels: The labels of the objects in the mask.
    colors: A list of colors to use for the bounding boxes.

    Returns:
    The image with the bounding boxes and labels drawn on it.
    """

    # Find the bounding boxes of the polygon.
    bounding_boxes = []
    for i in range(mask.shape[-1]):
        bbox = cv2.boundingRect(mask[:, :, i])
        bounding_boxes.append(bbox)

    # Draw the bounding boxes on the image.
    for bbox, label, color_name in zip(bounding_boxes, labels, colors):
        rgb_color = matplotlib.colors.to_rgb(color_name)
        rgb_color = tuple(value * 255 for value in rgb_color)
        cv2.rectangle(image, bbox, rgb_color, 10)
        cv2.putText(image, label, (bbox[0], bbox[1] + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 6)

    return image



import matplotlib
rgb_color = matplotlib.colors.to_rgb('darkblue')
rgb_color = tuple(value * 255 for value in rgb_color)
rgb_color


image_id = os.listdir(train_image_path)[5390]
masks = get_masks_by_img_id(train_df, image_id)
masks = masks.astype(np.uint8)


path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

colors = ['maroon', 'darkblue', 'purple','teal']
labels = ['Fish', 'Flower', 'Gravel', 'Sugar']

img = show_bounding_boxes(img,masks,labels,colors)

plt.figure(figsize=(32,8))
plt.imshow(img)
plt.axis("off");


def show_img_with_masks(img,masks,comment=""):
    
    colormaps = ['PuRd_r', 'Blues_r', 'Purples_r','winter_r']
    mask_labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    
    fig, axes = plt.subplots(1,6,figsize=(36,4))
    axes = axes.ravel()
    
    if img.shape[-1]!=3:
        img_cmap = 'gray'
    else:
        img_cmap=None
        
    for ix,axis in enumerate(axes):
        ix = ix%6
        axis.imshow(img,cmap=img_cmap)
        axis.axis('off')
        if ix==0:
            axis.set_title("Main Image")
        elif ix==1:
            for i,(mask,cmap,label) in enumerate(zip(masks,colormaps,mask_labels)):
                mask = np.clip(mask,0,1)
                mask = np.ma.masked_where(mask == 0, mask)
                axis.imshow(mask,alpha=0.7,cmap=cmap)
                axis.set_title(f"All the mask {comment}")
                draw_label_on_mask(mask,label,axis)
        elif ix>=2:
            for i,(mask,cmap,label) in enumerate(zip(masks,colormaps,mask_labels)):
                mask = np.clip(mask,0,1)
                mask = np.ma.masked_where(mask == 0, mask)
                axis = axes[2+i]
                axis.imshow(mask,alpha=0.4,cmap=cmap)
                axis.set_title(f"{label} {comment}")
                draw_label_on_mask(mask,label,axis)
    plt.show()
    
    return None


image_id = os.listdir(train_image_path)[5390]
path = os.path.join(train_image_path,image_id)
img = cv2.imread(path)
img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = get_masks_by_img_id(train_df, image_id)
masks = (masks[:,:,0], masks[:,:,1],masks[:,:,2],masks[:,:,3])
show_img_with_masks(img,masks)


image_ids = os.listdir(train_image_path)[13:16]
for image_id in image_ids:
    path = os.path.join(train_image_path,image_id)
    img = cv2.imread(path)
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = get_masks_by_img_id(train_df, image_id)
    masks = (masks[:,:,0], masks[:,:,1],masks[:,:,2],masks[:,:,3])
    show_img_with_masks(img,masks)


import numpy as np
import matplotlib.pyplot as plt

def rle_to_mask(rle_string, height, width):
    '''
    Convert RLE (run-length encoding) string to numpy array.

    Parameters:
    rle_string (str): string of RLE encoded mask.
    height (int): height of the mask.
    width (int): width of the mask.

    Returns:
    numpy.array: numpy array of the mask.
    '''
    rows, cols = height, width

    if rle_string == -1:  # Handle cases where no mask exists
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)

        for start, length in rle_pairs:
            start -= 1  # RLE is 1-indexed, so adjust to 0-indexed
            img[start:start+length] = 255  # Set the specified region as part of the object

        img = img.reshape(cols, rows)  # Reshape it to (cols, rows)
        img = img.T  # Transpose to (height, width)
        img = img / 255.0  # Normalize to 0-1

        return img

def get_masks_by_img_id(dataframe, image_id, img_height, img_width):
    '''
    Get the masks for a given image ID.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing RLE encodings.
    image_id (str): The image ID to fetch the masks for.
    img_height (int): Height of the mask.
    img_width (int): Width of the mask.

    Returns:
    numpy.array: A mask array of shape (height, width, num_classes).
    '''
    masks = np.zeros((img_height, img_width, 4))  # Assuming 4 classes: Fish, Flower, Gravel, Sugar
    rle_masks = list(dataframe[dataframe['Image_Id'] == image_id]['Label_EncodedPixels'])[0]

    # Get the RLE masks for each class
    fish_mask = rle_to_mask(rle_masks[0][1], img_height, img_width)
    flower_mask = rle_to_mask(rle_masks[1][1], img_height, img_width)
    gravel_mask = rle_to_mask(rle_masks[2][1], img_height, img_width)
    sugar_mask = rle_to_mask(rle_masks[3][1], img_height, img_width)

    mask_list = [fish_mask, flower_mask, gravel_mask, sugar_mask]

    # Fill the mask array with the individual class masks
    for ix, mask in enumerate(mask_list):
        masks[:, :, ix] = mask

    return masks

def draw_label_on_mask(mask, label, obj=plt):
    '''
    Function to add labels to the image.

    Parameters:
    mask (numpy.array): Binary mask of the object.
    label (str): The label to be displayed.
    obj: Object for plotting (default: plt).
    '''
    if np.sum(mask) > 0:  # Only draw the label if the mask has content
        y, x = np.argwhere(mask == 1)[0]  # Get the first coordinate where the mask is present
        y, x = y + 10, x + 5  # Offset to place the label slightly inside the mask
        obj.text(x, y, label, color='white', fontsize=12, weight='bold')  # Draw the label

    return None



import tensorflow as tf
import numpy as np
import math
import os
import cv2

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
                 dataframe=None,
                 root_dir=".",
                 img_width=2100,
                 img_height=1400,
                 resize=False,
                 resize_width=384,
                 resize_height=256,
                 mode='fit',
                 augmentations=None,
                 num_channels=3,
                 num_classes=4,
                 batch_size=32,
                 shuffle=True, 
                 random_state=42): 
        
        self.dataframe = dataframe
        self.filenames = list(dataframe['Image_Id']) if dataframe is not None else os.listdir(root_dir)
        self.root_dir = root_dir
        self.img_width = img_width
        self.img_height = img_height
        self.resize = resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.mode = mode
        self.augmentations = augmentations
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.total_samples = len(self.filenames)
        self.indexes = np.arange(len(self.filenames))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        
        self.on_epoch_end()
    
    @property
    def image_shape(self):
        if not self.resize:
            img_shape = (self.img_height, self.img_width, self.num_channels)
        else:
            img_shape = (self.resize_height, self.resize_width, self.num_channels)
        return img_shape

    def __len__(self):
        num_batches_total = math.ceil(self.total_samples / self.batch_size)
        return num_batches_total

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, self.total_samples)
        batch_files = self.filenames[low:high]
        
        if self.mode == 'fit':
            batch_X = self.__generate_X(batch_files)
            batch_y = self.__generate_y(batch_files)
            if self.augmentations is not None:
                batch_X, batch_y = self.augment_batch(batch_X, batch_y)
            batch_X, batch_y = batch_X.astype(np.float32), batch_y.astype(np.float32)
            return batch_X, batch_y
        elif self.mode == 'predict':
            batch_X = self.__generate_X(batch_files)
            batch_X = batch_X.astype(np.float32)
            return batch_X
        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def __generate_X(self, batch_files):
        if self.resize:
            img_size = (self.resize_height, self.resize_width)
        else:
            img_size = (self.img_height, self.img_width)
            
        batch_images = np.zeros((len(batch_files), *img_size, self.num_channels), dtype=np.uint8)

        for ix, filename in enumerate(batch_files):
            img_path = os.path.join(self.root_dir, filename)
            if self.num_channels == 3:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(img_path, 0)
                img = np.expand_dims(img, axis=-1)
            if self.resize:
                img = cv2.resize(img, tuple(reversed(img_size)))
            img = img.astype(np.uint8)
            batch_images[ix] = img
                               
        return batch_images

    def __generate_y(self, batch_files):
        if self.resize:
            mask_size = (self.resize_height, self.resize_width)
        else:
            mask_size = (self.img_height, self.img_width)
        
        batch_masks = np.zeros((len(batch_files), *mask_size, self.num_classes), dtype=np.uint8)
        
        for ix, filename in enumerate(batch_files):
            masks = self.get_masks_by_img_id(self.dataframe, filename)
            batch_masks[ix] = masks
        batch_masks = (batch_masks > 0).astype(np.uint8)
        return batch_masks
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.total_samples)
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def get_masks_by_img_id(self, dataframe, image_id):
        rle_masks = list(dataframe[dataframe['Image_Id'] == image_id]['Label_EncodedPixels'])[0]
        fish_mask = rle_to_mask(rle_masks[0][1], self.img_height, self.img_width)
        flower_mask = rle_to_mask(rle_masks[1][1], self.img_height, self.img_width)
        gravel_mask = rle_to_mask(rle_masks[2][1], self.img_height, self.img_width)
        sugar_mask = rle_to_mask(rle_masks[3][1], self.img_height, self.img_width)
        
        mask_list = [fish_mask, flower_mask, gravel_mask, sugar_mask]
        
        if self.resize:
            resized_mask_list = []
            for mask in mask_list:
                mask = cv2.resize(mask, (self.resize_width, self.resize_height))
                resized_mask_list.append(mask)
            masks = np.zeros((self.resize_height, self.resize_width, self.num_classes))
            for ix, mask in enumerate(resized_mask_list):
                masks[:, :, ix] = mask
        else:
            masks = np.zeros((self.img_height, self.img_width, self.num_classes))   
            for ix, mask in enumerate(mask_list):
                masks[:, :, ix] = mask
                
        return masks

    def augment_batch(self, batch_images, batch_masks):
        batch_X = np.zeros(batch_images.shape, dtype=np.float32)
        batch_y = np.zeros(batch_masks.shape, dtype=np.float32)
        for ix, (img, masks) in enumerate(zip(batch_images, batch_masks)):
            augmented = self.augmentations(image=img, mask=masks)
            img = augmented['image']
            masks = augmented['mask']
            batch_X[ix] = img
            batch_y[ix] = masks
        return batch_X, batch_y



import albumentations as A

augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.GridDistortion(p=0.5),
    A.OpticalDistortion(p=0.5),
    A.RandomGamma(),
    A.RandomBrightnessContrast(),
    A.CLAHE(),
    A.Equalize(),
    A.ToFloat()
])


data_generator = DataGenerator(train_df,
                               train_image_path,
                               resize=True,
                               resize_width=576,
                               resize_height=384,
                               num_channels=3,
                               augmentations=augmentations
                                )

print(data_generator.total_samples)
print(len(data_generator.indexes))
print(data_generator.__len__())
print(data_generator.image_shape)


batch_X, batch_y = data_generator.__getitem__(1)
print(batch_X.shape)
print(batch_y.shape)
print(batch_X.dtype)
print(batch_y.dtype)


plt.imshow(batch_X[13])


plt.imshow(batch_y[13][:,:,1])


for img,masks in zip(batch_X[3:6],batch_y[3:6]):
    masks = (masks[:,:,0], masks[:,:,1],masks[:,:,2],masks[:,:,3])
    show_img_with_masks(img,masks)


from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['classes'])
print(df_train.shape)
print(df_val.shape)


R_WIDTH = 576
R_HEIGHT = 384
NUM_CHANNELS = 3
BATCH_SIZE = 4
TEST_BATCH_SIZE = 32


LABELS = ["Fish", "Sugar", "Flower", "Gravel"]



train_generator = DataGenerator(dataframe=df_train,
                                root_dir=train_image_path,
                                mode="fit",
                                resize=True,
                                shuffle=True,
                                resize_width=R_WIDTH,
                                resize_height=R_HEIGHT,
                                num_channels=NUM_CHANNELS,
                                batch_size=BATCH_SIZE,
                                augmentations=augmentations)

print(train_generator.total_samples)
print(len(train_generator.indexes))
print(train_generator.__len__())



batch_X, batch_y = train_generator.__getitem__(4)
print(batch_X.shape)
print(batch_y.shape)
print(batch_X.dtype)
print(batch_y.dtype)


val_generator =  DataGenerator(dataframe=df_val,
                               root_dir=train_image_path,
                               mode="fit",
                               resize=True,
                               shuffle=True,
                               resize_width=R_WIDTH,
                               resize_height=R_HEIGHT,
                               num_channels=NUM_CHANNELS,
                               batch_size=BATCH_SIZE,
                               augmentations=augmentations)

print(val_generator.total_samples)
print(len(val_generator.indexes))
print(val_generator.__len__())
batch_X, batch_y = val_generator.__getitem__(1)
print(batch_X.shape)
print(batch_y.shape)
print(batch_X.dtype)
print(batch_y.dtype)


test_generator = DataGenerator(
    dataframe=None, # Set to None to use os.listdir for filenames
    root_dir=test_image_path,  # Path to the test images
    mode="predict",  # For inference mode
    resize=True,  # Enable resizing
    shuffle=False,  # No need to shuffle for prediction
    resize_width=R_WIDTH,
    resize_height=R_HEIGHT,
    num_channels=NUM_CHANNELS,
    batch_size=TEST_BATCH_SIZE
)
print(test_generator.total_samples)
print(len(test_generator.indexes))
print(test_generator.__len__())


batch_X = test_generator.__getitem__(1)
print(batch_X.shape)
print(batch_X.dtype)


eval_generator =  DataGenerator(dataframe=df_val,
                               root_dir=train_image_path,
                               mode="fit",
                               resize=True,
                               shuffle=False,
                               resize_width=R_WIDTH,
                               resize_height=R_HEIGHT,
                               num_channels=NUM_CHANNELS,
                               batch_size=TEST_BATCH_SIZE,
                               augmentations=augmentations)

print(eval_generator.total_samples)
print(len(eval_generator.indexes))
print(eval_generator.__len__())


batch_X, batch_y = eval_generator.__getitem__(1)
print(batch_X.shape)
print(batch_y.shape)
print(batch_X.dtype)
print(batch_y.dtype)


from PIL import Image
import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def viz_model(model):
    # Clear any previous session to avoid memory issues
    tf.keras.backend.clear_session()

    # Generate the plot of the model architecture
    model_plot = tf.keras.utils.plot_model(model, show_shapes=True, rankdir='TB')

    # Convert the model plot to an image
    model_plot = io.BytesIO(model_plot.data)
    model_plot = Image.open(model_plot)
    model_plot = np.array(model_plot)

    # Transpose the image to fix orientation
    model_plot = np.transpose(model_plot, (1, 0, 2))
    model_plot = Image.fromarray(model_plot)

    # Flip the image vertically
    model_plot = model_plot.transpose(Image.FLIP_TOP_BOTTOM)
    model_plot = np.array(model_plot)

    # Create a larger figure for better visualization
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    # Display the image
    ax.imshow(model_plot)
    ax.axis('off')  # Hide axis for better look
    plt.show()



import tensorflow as tf
base = tf.keras.applications.ResNet50(include_top=False,
                                            weights="imagenet",
                                            input_shape=(256,384, 3)
                                            )


tf.keras.backend.clear_session()


for layer in base.layers:
    print(layer.name)



# Low-level feature (early)
base.get_layer('conv2_block3_out').output

# Mid-high feature
base.get_layer('conv4_block6_out').output

# Deep feature (before ASPP or decoder)
base.get_layer('conv5_block3_out').output



import tensorflow as tf

def hw_flatten(x):
    return tf.keras.layers.Reshape(target_shape=(-1, x.shape[-1]))(x)

def self_attention_cnn(x):
    q = layers.Conv2D(filters=x.shape[-1] // 8, kernel_size=(1, 1))(x)
    k = layers.Conv2D(filters=x.shape[-1] // 8, kernel_size=(1, 1))(x)
    v = layers.Conv2D(filters=x.shape[-1], kernel_size=(1, 1))(x)

    qk = layers.Dot(axes=2)([hw_flatten(q), hw_flatten(k)])
    softmax = layers.Activation("softmax")(qk)

    attention = layers.Dot(axes=1)([softmax, hw_flatten(v)])
    attention = layers.Reshape(target_shape=x.shape[1:])(attention)

    y = layers.Add()([x, attention])
    return y


def ASPP(inputs):
    shape = inputs.shape

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y_pool = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization()(y_pool)
    y_pool = tf.keras.layers.Activation('relu')(y_pool)
    y_pool = tf.keras.layers.UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = tf.keras.layers.Conv2D(256, 1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = tf.keras.layers.BatchNormalization()(y_1)
    y_1 = tf.keras.layers.Activation('relu')(y_1)

    y_6 = tf.keras.layers.Conv2D(256, 3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = tf.keras.layers.BatchNormalization()(y_6)
    y_6 = tf.keras.layers.Activation('relu')(y_6)

    y_12 = tf.keras.layers.Conv2D(256, 3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = tf.keras.layers.BatchNormalization()(y_12)
    y_12 = tf.keras.layers.Activation('relu')(y_12)

    y_18 = tf.keras.layers.Conv2D(256, 3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = tf.keras.layers.BatchNormalization()(y_18)
    y_18 = tf.keras.layers.Activation('relu')(y_18)

    y = tf.keras.layers.Concatenate()([y_pool, y_1, y_6, y_12, y_18])
    y = tf.keras.layers.Conv2D(256, 1, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)

    return y

def ResNet50AttentionDeepLabV3Plus(num_classes, input_shape=(256, 384, 3)):
    inputs = tf.keras.layers.Input(input_shape)

    backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    image_features = backbone.get_layer('conv4_block6_out').output  # Replace EfficientNet block7a
    x_a = ASPP(image_features)
    x_a = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x_a)

    x_b = backbone.get_layer('conv2_block3_out').output  # Replace EfficientNet block2a
    x_b = self_attention_cnn(x_b)
    x_b = tf.keras.layers.Conv2D(48, 1, padding='same', use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization()(x_b)
    x_b = tf.keras.layers.Activation('relu')(x_b)

    x_a_shape = tf.keras.backend.int_shape(x_a)
    x_b = tf.keras.layers.UpSampling2D(
        size=(x_a_shape[1] // tf.keras.backend.int_shape(x_b)[1],
              x_a_shape[2] // tf.keras.backend.int_shape(x_b)[2]),
        interpolation="bilinear")(x_b)

    x = tf.keras.layers.Concatenate()([x_a, x_b])
    x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), name='output_layer')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model



from tensorflow.keras import layers



input_shape=(256,384, 3)
inputs = tf.keras.layers.Input(input_shape)
backbone = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",input_tensor=inputs)


image_features = backbone.get_layer('conv5_block3_out').output
print(image_features.shape)
x_a = ASPP(image_features)
x_a = tf.keras.layers.UpSampling2D((8, 8), interpolation="bilinear")(x_a)
print(x_a.shape)
""" Get low-level features """
x_b = backbone.get_layer('conv2_block3_out').output
x_b = self_attention_cnn(x_b)
print(x_b.shape)
x_b = tf.keras.layers.Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
x_b = tf.keras.layers.BatchNormalization()(x_b)
x_b = tf.keras.layers.Activation('relu')(x_b)
print(x_b.shape)
x = tf.keras.layers.Concatenate()([x_a, x_b])
print(x.shape)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
print(x.shape)
x = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x)

""" Outputs """
x = tf.keras.layers.Conv2D(1, (1, 1), name='output_layer')(x)
x = tf.keras.layers.Activation('sigmoid')(x)

""" Model """
model = tf.keras.models.Model(inputs=inputs, outputs=x)


model = ResNet50AttentionDeepLabV3Plus(num_classes=4,
                                        input_shape = (R_HEIGHT, R_WIDTH, NUM_CHANNELS))
total_param = format(model.count_params(),",")
print(f"Total no of parameters = {total_param}")
viz_model(model)


total_param = format(model.count_params(),",")
print(f"Total no of parameters = {total_param}")
viz_model(model)


class DeepLabV3Plus:
    def __init__(self,
                 input_shape=(256,384, 3),
                 num_classes=4,
                 activation='relu'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.model = self.build_model()
        self.backbone = tf.keras.applications.ResNet50(include_top=False,
                                                            weights="imagenet",
                                                            input_shape=input_shape,
                                                            )
        self.encoder = self.backbone.get_layer("block7a_dwconv").output
        
        pass


import tensorflow as tf

class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes=1, threshold=0.5, name='dice_score', **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)

        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, dtype=tf.float32)
        else:
            y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.num_classes)
            y_true = tf.one_hot(tf.argmax(y_true, axis=-1), depth=self.num_classes)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        epsilon = tf.keras.backend.epsilon()
        dice = (2.0 * self.intersection) / (self.union + epsilon)
        return dice

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class IoUScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes=1, threshold=0.5, name='iou_score', **kwargs):
        super(IoUScore, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)

        if self.num_classes == 1:
            y_pred = tf.cast(y_pred > self.threshold, dtype=tf.float32)
        else:
            y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.num_classes)
            y_true = tf.one_hot(tf.argmax(y_true, axis=-1), depth=self.num_classes)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        epsilon = tf.keras.backend.epsilon()
        iou = (self.intersection) / (self.union + epsilon)
        return iou

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)



import numpy as np
import tensorflow as tf

y_pred = np.array([[0., 0., 1., 0.]])
y_true = np.array([[1., 1., 1., 0.]])

iou = IoUScore()
iou.update_state(y_true, y_pred)
print("IoU Score:", iou.result().numpy())  # Should be 0.3333

dice = DiceScore()
dice.update_state(y_true, y_pred)
print("Dice Score:", dice.result().numpy())  # Should be 0.5



import tensorflow as tf

class BCELoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name='bce_loss'):
        super().__init__(name=name)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true, y_pred):
        return self.bce(y_true, y_pred)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2.0, from_logits=False, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        bce = self.bce(y_true, y_pred)
        pt = tf.exp(-bce)  # Calculate pt (probability of true class)
        focal_loss = self.alpha * tf.pow(1. - pt, self.gamma) * bce
        return tf.reduce_mean(focal_loss)


class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name='dice_loss'):
        super().__init__(name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        intersection = tf.reduce_sum(y_true * y_pred)
        dice = (2.0 * intersection + epsilon) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon)
        return 1.0 - dice


class IoULoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name='iou_loss'):
        super().__init__(name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))
        iou = (intersection + epsilon) / (union + epsilon)
        return 1.0 - iou


class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name='dice_bce_loss'):
        super().__init__(name=name)
        self.from_logits = from_logits
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true, y_pred):
        bce_loss = self.bce(y_true, y_pred)
        dice_loss = DiceLoss(from_logits=self.from_logits).call(y_true, y_pred)
        return dice_loss + bce_loss


class DiceFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2.0, from_logits=False, name='dice_focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        dice_loss = DiceLoss(from_logits=self.from_logits).call(y_true, y_pred)
        focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma, from_logits=self.from_logits).call(y_true, y_pred)
        return dice_loss + focal_loss


class DiceFocalBCELoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2.0, from_logits=False, name='dice_focal_bce_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        dice_loss = DiceLoss(from_logits=self.from_logits).call(y_true, y_pred)
        focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma, from_logits=self.from_logits).call(y_true, y_pred)
        bce_loss = BCELoss(from_logits=self.from_logits).call(y_true, y_pred)
        return dice_loss + focal_loss + bce_loss



class BCELoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name='bce_loss'):
        super().__init__(name=name)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def call(self, y_true, y_pred):
        return self.bce(y_true, y_pred)



y_pred = tf.convert_to_tensor([0., 0., 1., 0.], dtype=tf.float32)
y_true = tf.convert_to_tensor([[1., 1., 1., 0.]], dtype=tf.float32)

# Instantiate and test each loss
criterion = BCELoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")

criterion = FocalLoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")

criterion = DiceLoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")

criterion = IoULoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")

criterion = DiceBCELoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")

criterion = DiceFocalLoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")

criterion = DiceFocalBCELoss()
loss = criterion.call(y_true, y_pred)
print(f"{criterion.name} = {loss.numpy()}")


'''import os
import glob
import keras
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from tensorflow.keras.callbacks import Callback

class PrAucCallback(Callback):
    def __init__(self, data_generator, num_workers=4, 
                 early_stopping_patience=5, 
                 plateau_patience=3, reduction_rate=0.5,
                 stage='train', checkpoints_path='checkpoints/'):
        super(PrAucCallback, self).__init__()
        self.data_generator = data_generator
        self.num_workers = num_workers
        self.class_names = ['Fish', 'Flower', 'Sugar', 'Gravel']
        self.history = [[] for _ in range(len(self.class_names) + 1)]  # to store per class and mean PR AUC
        self.early_stopping_patience = early_stopping_patience
        self.plateau_patience = plateau_patience
        self.reduction_rate = reduction_rate
        self.stage = stage
        self.best_pr_auc = -float('inf')

        # Ensure checkpoints path exists
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        self.checkpoints_path = checkpoints_path

    def compute_pr_auc(self, y_true, y_pred):
        pr_auc_mean = 0
        print(f"\n{'#'*30}\n")
        for class_i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true[:, class_i], y_pred[:, class_i])
            pr_auc = auc(recall, precision)
            pr_auc_mean += pr_auc / len(self.class_names)
            print(f"PR AUC {self.class_names[class_i]}, {self.stage}: {pr_auc:.3f}\n")
            self.history[class_i].append(pr_auc)        
        print(f"\n{'#'*20}\n PR AUC mean, {self.stage}: {pr_auc_mean:.3f}\n{'#'*20}\n")
        self.history[-1].append(pr_auc_mean)
        return pr_auc_mean

    def is_patience_lost(self, patience):
        if len(self.history[-1]) > patience:
            best_performance = max(self.history[-1][-(patience + 1):-1])
            return best_performance == self.history[-1][-(patience + 1)] and best_performance >= self.history[-1][-1]

    def early_stopping_check(self, pr_auc_mean):
        if self.is_patience_lost(self.early_stopping_patience):
            self.model.stop_training = True

    def model_checkpoint(self, pr_auc_mean, epoch):
        if pr_auc_mean > self.best_pr_auc:
            # Clean old checkpoints
            for checkpoint in glob.glob(os.path.join(self.checkpoints_path, 'classifier_densenet169_epoch_*')):
                os.remove(checkpoint)
            self.best_pr_auc = pr_auc_mean
            # Save the model with the best PR AUC in validation
            self.model.save(os.path.join(self.checkpoints_path, f'classifier_densenet169_epoch_{epoch}_val_pr_auc_{pr_auc_mean}.h5'))
            print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")

    def reduce_lr_on_plateau(self):
        if self.is_patience_lost(self.plateau_patience):
            new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * self.reduction_rate
            keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\n{'#'*20}\nReduced learning rate to {new_lr}.\n{'#'*20}\n")

    def on_epoch_end(self, epoch, logs={}):
        # Get predictions from the model using the data generator
        y_pred = self.model.predict(self.data_generator, workers=self.num_workers)
        y_true = self.data_generator.get_labels()  # This assumes you have a method to fetch true labels
        
        # Compute PR AUC
        pr_auc_mean = self.compute_pr_auc(y_true, y_pred)
        
        if self.stage == 'val':
            # Early stop after early_stopping_patience epochs of no improvement
            self.early_stopping_check(pr_auc_mean)

            # Save model checkpoint with best PR AUC
            self.model_checkpoint(pr_auc_mean, epoch)

            # Reduce learning rate on plateau
            self.reduce_lr_on_plateau()

    def get_pr_auc_history(self):
        return self.history
'''


import tensorflow as tf

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=3, monitor=["val_dice_score", "val_loss"], mode="max", verbose=1):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_weights = None
        self.best_metric_values = [float('-inf')] * len(monitor)  # For "max" mode
        self.wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        stop_training = False
        for i, metric in enumerate(self.monitor):
            current_metric_value = logs.get(metric)
            if current_metric_value is None:
                continue

            # Compare current value with the best value
            if self.mode == "max":
                if current_metric_value > self.best_metric_values[i]:
                    self.best_metric_values[i] = current_metric_value
                    self.wait = 0
                else:
                    self.wait += 1
            else:  # min mode (e.g., for loss)
                if current_metric_value < self.best_metric_values[i]:
                    self.best_metric_values[i] = current_metric_value
                    self.wait = 0
                else:
                    self.wait += 1

            if self.wait >= self.patience:
                stop_training = True

        if stop_training:
            if self.verbose > 0:
                print("Early stopping triggered")
            self.model.stop_training = True
            self.model.set_weights(self.best_weights)  # Restore best weights

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()  # Initialize best weights



import os
import tensorflow as tf

work_dir = "/kaggle/working/"
model_name = "cloud_1e_ResNet50AttentionDeepLabV3Plus"

# Learning rate scheduler
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_dice",  # Monitor val_dice
    mode="max",  # 'max' because higher Dice score is better
    factor=0.5,
    min_lr=1e-6,
    patience=1,
    verbose=1
)

# Model Checkpoint (saving every epoch)
ckpt_filepath = os.path.join(work_dir, model_name, "ckpts", "weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5")
os.makedirs(os.path.dirname(ckpt_filepath), exist_ok=True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_filepath,
    monitor="val_loss",  # Monitor validation loss
    mode="min",  # Mode is 'min' because lower loss is better
    save_best_only=False,
    save_freq="epoch",
    verbose=1
)

# CSV Logger (log training process)
csv_filepath = os.path.join(work_dir, model_name, "logs", "training.csv")
os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

csv_log = tf.keras.callbacks.CSVLogger(csv_filepath)

# Callbacks list (without early stopping)
callbacks = [reduce_lr, model_checkpoint, csv_log]




config = dict(
    input_shape=(R_HEIGHT, R_WIDTH, NUM_CHANNELS),
    batch_size=BATCH_SIZE,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=DiceBCELoss(),
    metrics=[
        DiceScore(name="dice"),
        IoUScore(name="iou"),
        DiceScore(name="val_dice"),
        IoUScore(name="val_iou")
    ],
    epochs=5,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice",  # Monitor validation dice score
            mode="max",  # 'max' for Dice score since higher is better
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="path_to_save_model",  # Update this path
            monitor="val_loss",  # Monitor validation loss
            mode="min",  # Mode is 'min' because lower loss is better
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv")
    ]
)



import segmentation_models as sm

# Define model with ResNet50 backbone (lowercase 'r' in 'resnet50')
model = sm.Unet('resnet50',  # Change 'Resnet50' to 'resnet50'
                input_shape=config["input_shape"],
                classes=4,
                activation= 'softmax',
                encoder_weights='imagenet')

# Compile the model
model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=config['metrics'])

# Check model parameters
total_param = format(model.count_params(), ",")
print(f"Total number of parameters = {total_param}")



tf.keras.backend.clear_session()

model = ResNet50AttentionDeepLabV3Plus(num_classes=4,
                                        input_shape = (R_HEIGHT, R_WIDTH, NUM_CHANNELS))


model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=config['metrics'])


total_param = format(model.count_params(),",")
print(f"Total no of parameters = {total_param}")


print(model.input_shape,'\t',train_generator.image_shape,'\t',val_generator.image_shape)
print("GPU available!" if tf.test.is_gpu_available() else "GPU is not available")
print(model.optimizer)


import multiprocessing
num_cores = multiprocessing.cpu_count()
print(num_cores)


import tensorflow as tf

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})





history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=config['epochs'],
                    callbacks=callbacks,
                    workers=num_cores,
                    use_multiprocessing=False
                    )



import matplotlib.pyplot as plt

def plot_accuracy_loss(history):
    plt.figure(figsize=(16, 5))

    # Plot Dice
    plt.subplot(1, 3, 1)
    plt.plot(history.history['dice'], label='Train Dice')
    plt.plot(history.history['val_dice'], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.legend()

    # Plot IoU
    plt.subplot(1, 3, 2)
    plt.plot(history.history['iou'], label='Train IoU')
    plt.plot(history.history['val_iou'], label='Val IoU')
    plt.title('IoU Score')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function
plot_accuracy_loss(history)



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)



# Freeze all layers except the last one (the final output conv layer)
for base_layer in model.layers[:-1]:
    base_layer.trainable = False

# Compile with categorical crossentropy (for multi-class segmentation)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=[dice_coef, iou_score]
)

# Train head-only
history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,  # start with 5, you can adjust
    callbacks=callbacks,
    workers=num_cores,
    use_multiprocessing=False,
    verbose=1
)



# Unfreeze last 40 layers of the backbone
for layer in model.backbone.layers[-40:]:
    layer.trainable = True

# Re-compile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=[dice_coef, iou_score]
)

# Fine-tune model
history_phase2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks,
    workers=num_cores,
    use_multiprocessing=False,
    verbose=1
)



import tensorflow as tf
import multiprocessing
import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Define metrics
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Model Definition
def ResNet50AttentionDeepLabV3Plus(num_classes, input_shape=(256, 384, 3)):
    inputs = tf.keras.layers.Input(input_shape)

    backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    image_features = backbone.get_layer('conv4_block6_out').output  # High-level features
    x_a = ASPP(image_features)
    x_a = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x_a)

    x_b = backbone.get_layer('conv2_block3_out').output  # Low-level features
    x_b = self_attention_cnn(x_b)
    x_b = tf.keras.layers.Conv2D(48, 1, padding='same', use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization()(x_b)
    x_b = tf.keras.layers.Activation('relu')(x_b)

    x_a_shape = tf.keras.backend.int_shape(x_a)
    x_b = tf.keras.layers.UpSampling2D(
        size=(x_a_shape[1] // tf.keras.backend.int_shape(x_b)[1],
              x_a_shape[2] // tf.keras.backend.int_shape(x_b)[2]),
        interpolation="bilinear")(x_b)

    x = tf.keras.layers.Concatenate()([x_a, x_b])

    x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), name='output_layer')(x)
    x = tf.keras.layers.Activation('softmax')(x)  # Using softmax for multi-class segmentation

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

    # Attach backbone for fine-tuning
    model.backbone = backbone

    return model

# Initialize Model
model = ResNet50AttentionDeepLabV3Plus(num_classes=4, input_shape=(R_HEIGHT, R_WIDTH, NUM_CHANNELS))

# Unfreeze the last 40 layers of the backbone (for fine-tuning)
for layer in model.backbone.layers[:-40]:  # Freeze all except the last 40 layers
    layer.trainable = False

for layer in model.backbone.layers[-40:]:
    layer.trainable = True

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate
    loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class segmentation
    metrics=[dice_coef, iou_score]
)

# Callbacks
early_stopping = EarlyStopping(
    patience=5,  # Early stopping patience to prevent overfitting
    restore_best_weights=True,
    monitor='val_loss',
    mode='min'
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,  # Reduce LR if no improvement after 3 epochs
    verbose=1,
    mode='min'
)

model_checkpoint = ModelCheckpoint(
    'best_weights.h5',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

callbacks = [early_stopping, lr_scheduler, model_checkpoint]

# Get number of cores for parallel processing
num_cores = multiprocessing.cpu_count()

# Train model with callbacks
start = time.perf_counter()

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks,
    workers=num_cores,
    use_multiprocessing=False
)

end = time.perf_counter()
print(f"Phase 1 training time: {end-start:.2f} seconds")





import numpy as np
import tensorflow as tf

def dice_score_per_class(y_true, y_pred, smooth=1e-6):
    num_classes = y_true.shape[-1]
    dice_scores = []

    for i in range(num_classes):
        y_true_f = tf.reshape(y_true[..., i], [-1])
        y_pred_f = tf.reshape(y_pred[..., i], [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice_scores.append(dice.numpy())
    
    return dice_scores

def iou_score_per_class(y_true, y_pred, smooth=1e-6):
    num_classes = y_true.shape[-1]
    iou_scores = []

    for i in range(num_classes):
        y_true_f = tf.reshape(y_true[..., i], [-1])
        y_pred_f = tf.reshape(y_pred[..., i], [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.numpy())
    
    return iou_scores






import numpy as np
from sklearn.metrics import accuracy_score

# Function to calculate overall pixel accuracy
def calculate_pixel_accuracy(y_true, y_pred):
    # Convert lists to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Flatten the arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate accuracy
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    return accuracy

# Function to calculate per-class accuracy
def calculate_class_accuracy(y_true, y_pred, num_classes):
    # Convert lists to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Initialize list to hold class-wise accuracies
    class_accuracies = []

    for i in range(num_classes):
        # Mask for the current class
        y_true_class = (y_true == i)
        y_pred_class = (y_pred == i)
        
        # Calculate the accuracy for the current class
        accuracy = accuracy_score(y_true_class.flatten(), y_pred_class.flatten())
        class_accuracies.append(accuracy)

    return class_accuracies

# Example: Ground truth labels and predicted labels for 4 classes
y_true = [[0, 1], [2, 3]]  # Ground truth labels (Fish=0, Flower=1, Sugar=2, Gravel=3)
y_pred = [[0, 1], [2, 2]]  # Predicted labels (Fish=0, Flower=1, Sugar=2, Gravel=3)

# Number of classes (Fish=0, Flower=1, Sugar=2, Gravel=3)
num_classes = 4 

# Calculate overall accuracy
accuracy = calculate_pixel_accuracy(y_true, y_pred)
print(f"Overall Pixel Accuracy: {accuracy:.4f}")

# Class names for interpretation
class_names = ["Fish", "Flower", "Sugar", "Gravel"]

# Calculate class-specific accuracy
class_accuracies = calculate_class_accuracy(y_true, y_pred, num_classes)
for i, acc in enumerate(class_accuracies):
    print(f"Accuracy for class {class_names[i]}: {acc:.4f}")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Function to calculate overall pixel accuracy
def calculate_pixel_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    return accuracy

# Function to calculate per-class accuracy
def calculate_class_accuracy(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    class_accuracies = []

    for i in range(num_classes):
        y_true_class = (y_true == i)
        y_pred_class = (y_pred == i)
        
        accuracy = accuracy_score(y_true_class.flatten(), y_pred_class.flatten())
        class_accuracies.append(accuracy)

    return class_accuracies

# Example data
y_true = [[0, 1], [2, 3]]  # Ground truth labels
y_pred = [[0, 1], [2, 2]]  # Predicted labels
num_classes = 4 

# Calculate class-specific accuracy
class_accuracies = calculate_class_accuracy(y_true, y_pred, num_classes)
class_names = ["Fish", "Flower", "Sugar", "Gravel"]

# Function to plot accuracy graph
def plot_accuracy_graph(class_accuracies, class_names):
    plt.figure(figsize=(8, 6))
    plt.bar(class_names, class_accuracies, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1)
    plt.show()

# Generate the graph
plot_accuracy_graph(class_accuracies, class_names)



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Example Data
y_true = [[0, 1], [2, 3]]
y_pred = [[0, 1], [2, 2]]
num_classes = 4
class_names = ["Fish", "Flower", "Sugar", "Gravel"]

# Functions to calculate accuracies
def calculate_pixel_accuracy(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return accuracy_score(y_true, y_pred)

def calculate_class_accuracy(y_true, y_pred, num_classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    class_accuracies = []
    for i in range(num_classes):
        y_true_class = (y_true == i)
        y_pred_class = (y_pred == i)
        acc = accuracy_score(y_true_class.flatten(), y_pred_class.flatten())
        class_accuracies.append(acc)
    return class_accuracies

# Calculate accuracies
overall_accuracy = calculate_pixel_accuracy(y_true, y_pred)
class_accuracies = calculate_class_accuracy(y_true, y_pred, num_classes)

# Plot similar style graph
def plot_accuracy_graph(overall_accuracy, class_accuracies, class_names):
    f, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax = ax.ravel()

    # Overall Pixel Accuracy
    ax[0].bar(['Overall Pixel Accuracy'], [overall_accuracy], color='lightgreen')
    ax[0].set_ylim(0, 1)
    ax[0].set_title('Overall Pixel Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].grid(True, axis='y')

    # Per-Class Accuracy
    ax[1].bar(class_names, class_accuracies, color='skyblue')
    ax[1].set_ylim(0, 1)
    ax[1].set_title('Per-Class Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid(True, axis='y')

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_accuracy_graph(overall_accuracy, class_accuracies, class_names)



# Plot the accuracy - Line Plot
plt.figure(figsize=(8, 6))
plt.plot(class_names, class_accuracies, marker='o', color='skyblue', linewidth=2)
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.ylim(0, 1)
plt.grid(True, linestyle='--')

# Show overall accuracy
plt.text(0, 0.95, f'Overall Pixel Accuracy: {accuracy:.4f}', fontsize=12, color='red')

plt.tight_layout()
plt.show()


