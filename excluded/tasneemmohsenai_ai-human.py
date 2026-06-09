#hide
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_path =  r'/kaggle/input/newimg/WhatsApp Image 2025-03-07 at 10.15.47 PM.jpeg'
img = mpimg.imread(img_path)

# Ø¥Ø¹Ø¯Ø§Ø¯ Ø§Ù„Ù…Ø®Ø·Ø·
plt.figure(figsize=(10, 8))

# Ø¥Ø¶Ø§Ù�Ø© Ø§Ù„Ù†Øµ
text = """AI vs. Human-Generated Images: A Cutting-Edge Classification Dataset

In the rapidly evolving era of artificial intelligence, distinguishing between AI-generated images and authentic human-created visuals has become a crucial challenge. 
This dataset aims to push the boundaries of machine learning by providing a diverse collection of images â€” both AI-generated and human-crafted â€” to train and test advanced classification models. 
The goal is to develop algorithms capable of identifying subtle patterns, facial structures, and artistic nuances that differentiate real content from synthetic imagery.

This dataset serves as a valuable resource for researchers, data scientists, and AI enthusiasts striving to enhance AI transparency, combat deepfakes, and advance the field of computer vision."""

plt.text(
    0.5, 0.95, text, size=10, color='white', ha='center', va='top',
    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5')
)

# Ø¹Ø±Ø¶ Ø§Ù„ØµÙˆØ±Ø©
plt.imshow(img)
plt.axis('off')
plt.show()



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



data_path = "/kaggle/input/ai-vs-human-generated-dataset"
df = pd.read_csv(os.path.join(data_path, "train.csv"))
df.head()


df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()


# Plot class distribution
sns.countplot(x=df['label'])
plt.title("Distribution of AI vs. Human-Generated Images")
plt.xlabel("Label (0 = Human, 1 = AI)")
plt.ylabel("Count")
plt.show()


df["label"] = df["label"].astype(str)



import pandas as pd
import os

image_dataset_path = "/kaggle/input/ai-vs-human-generated-dataset/"

train_df = pd.read_csv(image_dataset_path + "train.csv")

train_df["file_name"] = image_dataset_path + train_df["file_name"]


train_df.head()


import pandas as pd


image_dataset_path = "/kaggle/input/ai-vs-human-generated-dataset/"


test_df = pd.read_csv(image_dataset_path + "test.csv")


test_df["file_path"] = image_dataset_path + test_df["id"] 


test_df.head()


train_df=train_df.drop('Unnamed: 0',axis=1)


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


import os

missing_files = train_df[~train_df["file_name"].apply(os.path.exists)]
print(f"Missing train images: {len(missing_files)}")


# Define image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create ImageDataGenerator
datagen = ImageDataGenerator(
      
      horizontal_flip=True,           # Ù‚Ù„Ø¨ Ø§Ù„ØµÙˆØ± Ø£Ù�Ù‚ÙŠØ§Ù‹
    validation_split=0.2,
    rescale=1./255)  # Normalize images

# Load images in batches
# make datagen into 2 
train_generator = datagen.flow_from_directory(
   data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
     shuffle=True
)

val_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)



from collections import Counter

# Ø¹Ø¯ Ø¹Ø¯Ø¯ Ø§Ù„ØµÙˆØ± Ù�ÙŠ ÙƒÙ„ Ù�Ø¦Ø©
label_counts = Counter(train_df['label'])
print("ØªÙˆØ²ÙŠØ¹ Ø§Ù„Ù�Ø¦Ø§Øª Ù�ÙŠ Ø¨ÙŠØ§Ù†Ø§Øª Ø§Ù„ØªØ¯Ø±ÙŠØ¨:", label_counts)

# ØªØµÙˆØ± Ø§Ù„ØªÙˆØ²ÙŠØ¹
import matplotlib.pyplot as plt

plt.bar(label_counts.keys(), label_counts.values(), color=['skyblue', 'pink'])
plt.title('Distribution of Classes in Training Data')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()












from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization


# ØªØ­Ù…ÙŠÙ„ Ù†Ù…ÙˆØ°Ø¬ ConvNeXt Ù…Ø¯Ø±Ø¨ Ù…Ø³Ø¨Ù‚Ù‹Ø§ (pre-trained)
base_model = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(224,224, 3))

# ØªØ¬Ù…ÙŠØ¯ Ø·Ø¨Ù‚Ø§Øª ConvNeXt Ø­ØªÙ‰ Ù„Ø§ ÙŠØªÙ… ØªØ­Ø¯ÙŠØ« Ø£ÙˆØ²Ø§Ù†Ù‡Ø§ Ø£Ø«Ù†Ø§Ø¡ Ø§Ù„ØªØ¯Ø±ÙŠØ¨
base_model.trainable = False

# Ø¥Ø¶Ø§Ù�Ø© Ø§Ù„Ø·Ø¨Ù‚Ø§Øª Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠØ© (Ø±Ø£Ø³ Ø§Ù„ØªØµÙ†ÙŠÙ�)
x = base_model.output
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)

output = Dense(1, activation='sigmoid')(x)

# Ø¨Ù†Ø§Ø¡ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬ Ø§Ù„Ù†Ù‡Ø§Ø¦ÙŠ
model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Ø·Ø¨Ø§Ø¹Ø© Ù…Ù„Ø®Øµ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
model.summary()



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


param_grid = {
    'learning_rate': [0.001, 0.0001],
    'batch_size': [16,32]
}


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)



train_df["label"] = train_df["label"].astype(str)











from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ØªÙ‚Ù„ÙŠÙ„ Ù…Ø¹Ø¯Ù„ Ø§Ù„ØªØ¹Ù„Ù…
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

# Ø¥ÙŠÙ‚Ø§Ù� Ù…Ø¨ÙƒØ±
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ØªØ¯Ø±ÙŠØ¨ Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
history = model.fit(
    train_generator,
    epochs=8,
    validation_data=val_generator,
    callbacks=[early_stopping]
)



# ØªÙ‚ÙŠÙŠÙ… Ø§Ù„Ù†Ù…ÙˆØ°Ø¬
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy:.2f}")



print(train_df['label'].value_counts())
print(train_df['label'].unique())







import numpy as np

# ØªØ­ÙˆÙŠÙ„ labels Ø¥Ù„Ù‰ Ù…ØµÙ�ÙˆÙ�Ø© Ù„Ù„Ø­ØµÙˆÙ„ Ø¹Ù„Ù‰ Ø´ÙƒÙ„Ù‡Ø§
labels = np.array(train_generator.labels)
print("Labels shape:", labels.shape)
print("First 10 labels:", labels[:10])



train_df['label'] = train_df['label'].astype(str)


valid_generator =datagen.flow_from_dataframe(
    train_df,
    x_col="file_name",
    y_col="label",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)


for i in range(3):
    x_batch, y_batch = next(train_generator)
    print(f"Batch {i+1} labels:", y_batch)



print("Input shape:", next(iter(train_generator))[0].shape)


x_batch, y_batch = next(train_generator)
print("Batch input shape:", x_batch.shape)
print("Batch labels shape:", y_batch.shape)
print("Sample labels:", y_batch[:10])

# Ø¬Ø±Ø¨ÙŠ ØªÙ…Ø±ÙŠØ± Ø¯Ù�Ø¹Ø© ÙˆØ§Ø­Ø¯Ø© Ù„Ù„Ù†Ù…ÙˆØ°Ø¬
preds = model.predict(x_batch)
print("Predictions shape:", preds.shape)
print("Sample predictions:", preds[:10])



x_batch, y_batch = next(train_generator)
print("Batch shapes:", x_batch.shape, y_batch.shape)
print("Sample labels:", y_batch[:10])



from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label'].values
)
class_weights = dict(enumerate(class_weights))

model.fit(
    train_generator,
    epochs=5,
    class_weight=class_weights,
    callbacks=[early_stopping],
    validation_data=valid_generator
)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# ØªØ­Ù…ÙŠÙ„ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ù…Ù† CSV
test_df = pd.read_csv('/kaggle/input/ai-vs-human-generated-dataset/test.csv')

# ØªÙˆÙ„ÙŠØ¯ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª
test_datagen = ImageDataGenerator(rescale=1.0/255)

# ØªØ­Ø¯ÙŠØ¯ Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ù„ÙŠ Ù�ÙŠÙ‡Ø§ Ù…Ø³Ø§Ø± Ø§Ù„ØµÙˆØ± ÙˆØ§Ù„ØªØµÙ†ÙŠÙ�Ø§Øª
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory='/kaggle/input/ai-vs-human-generated-dataset/',
    x_col='id',  # Ø¹Ù…ÙˆØ¯ Ø§Ù„Ù…Ø³Ø§Ø±Ø§Øª
    y_col=None,  # Ù„Ø§ ÙŠÙˆØ¬Ø¯ labels
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # Ø¨Ø¯ÙˆÙ† labels
    shuffle=False
)





predictions = model.predict(test_generator)

predicted_labels = (predictions > 0.5).astype(int) 


import os
print(os.listdir('/kaggle/input/newimg'))



from tensorflow.keras.preprocessing import image
import numpy as np


img_path = r'/kaggle/input/newimg/WhatsApp Image 2025-03-07 at 10.15.47 PM.jpeg'


img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
prediction = model.predict(img_array)


if prediction[0][0] > 0.5:
    print("The image is AI-Generated.")
else:
    print("The image is Human-Generated.")


import pandas as pd


submission = pd.DataFrame({
    'id': test_df['id'],  
    'label': predicted_labels.flatten()    
})

# Ø­Ù�Ø¸ Ø§Ù„Ù…Ù„Ù�
submission.to_csv('submission.csv', index=False)



print(submission.head())


