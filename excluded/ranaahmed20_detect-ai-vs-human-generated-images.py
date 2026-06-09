import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import os

image_dataset_path = "/kaggle/input/ai-vs-human-generated-dataset/"

train_df = pd.read_csv(image_dataset_path + "train.csv")

train_df["file_name"] = image_dataset_path + train_df["file_name"]


train_df.head()



train_df.info()


train_df=train_df.drop('Unnamed: 0',axis=1)


import os

missing_files = train_df[~train_df["file_name"].apply(os.path.exists)]
print(f"Missing train images: {len(missing_files)}")


train_df.info()


import random
from PIL import Image

ai_images = train_df[train_df["label"] == 1]["file_name"].tolist()
human_images = train_df[train_df["label"] == 0]["file_name"].tolist()

# Function to display images
def show_images(image_paths, title, num_images=5):
    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(random.sample(image_paths, num_images)):
        img = Image.open(img_path)  # Open image
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.show()


show_images(ai_images, "AI-Generated Images")



show_images(human_images, "Human-Generated Images")


import cv2

def show_ai_vs_human(df, num_images=5):
    ai_images = df[df["label"] == 1]["file_name"].dropna().sample(num_images, random_state=42).values
    human_images = df[df["label"] == 0]["file_name"].dropna().sample(num_images, random_state=42).values

    plt.figure(figsize=(15, num_images * 2))
    
    for i, (ai_img_path, human_img_path) in enumerate(zip(ai_images, human_images)):
        # Read AI Image
        ai_img = cv2.imread(ai_img_path)
        human_img = cv2.imread(human_img_path)

        if ai_img is None or human_img is None:
            print(f"Skipping missing images: {ai_img_path} or {human_img_path}")
            continue  # Skip missing images
        
        ai_img = cv2.cvtColor(ai_img, cv2.COLOR_BGR2RGB)
        human_img = cv2.cvtColor(human_img, cv2.COLOR_BGR2RGB)

        # Show Human Image
        plt.subplot(num_images, 2, i * 2 + 1)
        plt.imshow(human_img)
        plt.axis("off")
        plt.title("Human-Created")

        # Show AI Image
        plt.subplot(num_images, 2, i * 2 + 2)
        plt.imshow(ai_img)
        plt.axis("off")
        plt.title("AI-Generated")

    plt.suptitle("AI vs Human-Generated Images", fontsize=16)
    plt.tight_layout()
    plt.show()

# Display images
show_ai_vs_human(train_df)


import pandas as pd

# Define dataset path
image_dataset_path = "/kaggle/input/ai-vs-human-generated-dataset/"

# Load test.csv
test_df = pd.read_csv(image_dataset_path + "test.csv")

# Construct full image file paths
test_df["file_path"] = image_dataset_path + test_df["id"]  # id already contains 'test_data_v2/filename.jpg'

# Verify updated paths
test_df.head()


import os

missing_files = test_df[~test_df["file_path"].apply(os.path.exists)]

print(f"Missing test images: {len(missing_files)}")



import matplotlib.pyplot as plt
import cv2

# Function to display images
def show_test_images(df, num_images=5):
    sample_images = df.sample(num_images, random_state=42)["file_path"].values

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Test Image {i+1}")
        
    plt.show()

# Show test images
show_test_images(test_df)


print(train_df.head())
print(test_df.head())


train_df["label"] = train_df["label"].astype(str)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(rescale=1/255, validation_split=0.3)


train_generator = data_generator.flow_from_dataframe(
    train_df,
    x_col="file_name",   # Path to images
    y_col="label",       # Labels (as string)
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode="binary",  # Binary classification
    subset="training"  # Training split
)



valid_generator = data_generator.flow_from_dataframe(
    train_df,
    x_col="file_name",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)


from keras.applications.vgg19 import VGG19
vgg19 = VGG19()
vgg19.summary()


Model: "vgg19"



vgg19_layers = vgg19.layers
for i in vgg19_layers:
    print(i)


from keras.models import Sequential

vgg19_model = Sequential()
for i in range(len(vgg19_layers)-1):
    vgg19_model.add(vgg19_layers[i])


for layers in vgg19_model.layers:
    layers.trainable = False


from keras.layers import *
vgg19_model.add(Dense(1, activation = 'sigmoid'))


vgg19_model.summary()



vgg19_model.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


model_history=vgg19_model.fit(train_generator,epochs=10,validation_data=valid_generator)



# Evaluate the model on the training data
train_loss, train_accuracy = vgg19_model.evaluate(train_generator)

print(f"Accuracy on train data: {train_accuracy:.2%} | Loss: {train_loss:.4f}")



test_generator = data_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="file_path",   # Path to images in the test data
    y_col=None,          # No labels in the test data
    target_size=(224, 224),  # Resize images to the model's input size
    batch_size=32,
    class_mode=None,     # No labels in test data (we'll predict)
    shuffle=False        # Do not shuffle for evaluation
)


predictions = vgg19_model.predict(test_generator)

predicted_labels = (predictions > 0.5).astype(int)


submission_df = pd.DataFrame({
    'id': test_df['id'],  
    'label': predicted_labels.flatten()  
})

submission_df.to_csv('submission.csv', index=False)

print(submission_df.head())


submission_df['label'].value_counts()

