#Importing the required modules for building the mnodel.

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


#Getting the training labels from the csv file.
train_csv = pd.read_csv('/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train_labels.csv')
train_csv.head()


#getting the training image directory

train_path = '/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/train'

# the first one
idx = 0
row = train_csv.iloc[idx]
img_path = os.path.join(train_path, row.filename)
img = Image.open(img_path).convert('RGB')

plt.imshow(img)
plt.title("Label: %s" % row.label, fontsize = 14)
plt.axis('off')
plt.show()


#getting the test image directory
test_path = "/kaggle/input/sheep-classification-challenge-2025/Sheep Classification Images/test"

# the first one
img_path = os.path.join(test_path, "0306fa89.jpg")
img = Image.open(img_path).convert('RGB')

plt.imshow(img)
plt.title("No label", fontsize = 14)
plt.axis('off')
plt.show()


IMG_SIZE   = (224, 224)       
BATCH_SIZE = 32          
SEED = 42


train_csv['filepath'] = train_csv['filename'].apply(lambda x: os.path.join(train_path, x))
train_csv.head()


test_csv = pd.DataFrame({'filepath': glob.glob(os.path.join(test_path, '*.jpg'))})
test_csv.head()


id2label = {i:lab for i,lab in enumerate(sorted(train_csv.label.unique()))}
label2id = {v:k for k,v in id2label.items()}


id2label


label2id




train_data, val_data = train_test_split(
    train_csv,
    test_size = 0.2,
    stratify = train_csv['label'],
    random_state=SEED
)

train_datagen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,
    directory = train_path,
    x_col = 'filename',
    y_col = 'label',
    target_size = IMG_SIZE,       
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    seed = SEED
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe = val_data,
    directory = train_path,
    x_col = 'filename',
    y_col = 'label',
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    seed = SEED
)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_csv,
    directory=test_path,
    x_col='filepath',
    y_col=None,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)


def create_model_large():
    base_model = applications.ConvNeXtXLarge(
        include_top = False,
        weights = 'imagenet',
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(7, activation = 'softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
early_stopping = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 5,
    restore_best_weights = True
)
model_checkpoint = ModelCheckpoint(
    "best_model_convnextlarge.h5",
    monitor = 'val_accuracy',
    save_best_only = True,
    verbose = 1
)
model = create_model_large()
model.summary()


history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 100,
    callbacks = [early_stopping, model_checkpoint], 
    verbose = 2
)


def plot_history_matplotlib(history):
    h = history.history
    epochs = list(range(1, len(h['accuracy']) + 1))

    plt.figure(figsize = (8, 5))
    plt.plot(epochs, h['accuracy'],  marker = 'o', linestyle = '-',  label = 'Train Acc')
    plt.plot(epochs, h['val_accuracy'], marker = 's', linestyle = '--', label = 'Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle = ':', alpha = 0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (8, 5))
    plt.plot(epochs, h['loss'],  marker = 'D', linestyle = '-',  label = 'Train Loss')
    plt.plot(epochs, h['val_loss'], marker = '^', linestyle = '--', label = 'Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle = ':', alpha = 0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot_history_matplotlib(history)


model.load_weights("best_model_convnextlarge.h5")
preds = model.predict(test_generator, verbose=0)
pred_ids = tf.argmax(preds, axis=1).numpy()

fullpaths = test_generator.filenames 

basenames = [os.path.basename(fp) for fp in fullpaths] 

class_names = list(train_generator.class_indices.keys())
pred_labels = [class_names[i] for i in pred_ids]


submission = pd.DataFrame({
    'filename': basenames,
    'label':    pred_labels
})


from IPython.display import HTML
import base64  
import pandas as pd  

def create_download_link( df, title = "Download CSV file", filename = "submission_goat_4.csv"):  
    csv = df.to_csv(index =False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(submission)

