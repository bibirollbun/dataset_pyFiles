# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import os
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from fastai.callback.all import SaveModelCallback, CSVLogger

import os
from fastai.vision.all import *
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50, densenet201, efficientnet_b3
from fastai.tabular.all import *
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
train_df.head()


train_df.info()


train_df.duplicated().sum()


train_df['img_path'] = train_df['id'].apply(lambda x: os.path.join(
    '/kaggle/input/histopathologic-cancer-detection/train',f"{x}.tif"
))
def view_sample_images(df, no_sample =None,label_col =None, img_col='id'):
    sample_df = df.sample(no_sample)  
    plt.figure(figsize=(10, 6))
    
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img = mpimg.imread(row[img_col])
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        if label_col:
            plt.title(str(row[label_col]))
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
view_sample_images(train_df, no_sample =6,label_col ='label',img_col='img_path')


#Get image shape 
img = mpimg.imread(train_df['img_path'][0])
img.shape


# Check class distribution 
labels = train_df['label'].value_counts(normalize=True)
plt.pie(labels,autopct='%1.2f%%',labels=labels.index)
plt.title('Label Distribution');


def build_model(model_arch, batch_size=64, image_size=96, 
                validation_pct=0.2, 
                test_path='/kaggle/input/histopathologic-cancer-detection/test', 
                model_name='model'):
    """
    Train a CNN on train_df and save validation + test predictions for stacking.
    Will save best weights and reload automatically if already trained.
    """
    # DataBlock & DataLoaders
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader('img_path'),
        get_y=ColReader('label'),
        splitter=RandomSplitter(valid_pct=validation_pct, seed=42),
        item_tfms=Resize(image_size),
        batch_tfms=[*aug_transforms(size=image_size), Normalize.from_stats(*imagenet_stats)]
    )
    dls = dblock.dataloaders(train_df, bs=batch_size)

    # Learner
    learn = vision_learner(dls, model_arch, metrics=[accuracy, RocAucBinary()])
    learn.to_fp16()

    # Path to save model
    model_path = f'{model_name}_checkpoint'

    # Try to load best model if exists
    if (learn.path/f'models/{model_path}.pth').exists():
        learn.load(model_path)
        print(f"Loaded checkpoint: {model_path}")
    else:
        print("No checkpoint found, starting fresh training.")
        cbs = [
            SaveModelCallback(monitor='roc_auc_score', fname=model_path),  # save best model
            CSVLogger(fname=f'{model_name}_history.csv')  # training log
        ]
        learn.fine_tune(5, cbs=cbs)

    # Save validation predictions
    val_preds, val_targs = learn.get_preds(ds_idx=1)
    pd.DataFrame({
        'val_0': val_preds[:,0].numpy(),
        'val_1': val_preds[:,1].numpy(),
        'ground_truth_label': val_targs.numpy()
    }).to_csv(f'{model_name}_val_preds.csv', index=False)

    # Save test predictions
    test_files = get_image_files(test_path)
    test_dl = dls.test_dl(test_files)
    test_preds, _ = learn.get_preds(dl=test_dl)
    pd.DataFrame({
        'id': [f.stem for f in test_files],
        'pred_0': test_preds[:,0].numpy(),
        'pred_1': test_preds[:,1].numpy()
    }).to_csv(f'{model_name}_test_preds.csv', index=False)

    print(f"{model_name} training & predictions saved")



# Train ResNet50
build_model(resnet50, model_name='resnet50')


# Train DenseNet201
build_model(densenet201, model_name='densenet201')


# Train EfficientNetB3
build_model(efficientnet_b3, model_name='efficientnetb3', image_size=224)


# Load validation preds
res50_val = pd.read_csv('resnet50_val_preds.csv')
dense201_val = pd.read_csv('densenet201_val_preds.csv')
effb3_val = pd.read_csv('efficientnetb3_val_preds.csv')

# Load test preds
res50_test = pd.read_csv('resnet50_test_preds.csv')
dense201_test = pd.read_csv('densenet201_test_preds.csv')
effb3_test = pd.read_csv('efficientnetb3_test_preds.csv')

# Train stacking dataframe
train_stack = pd.DataFrame({
    'res50_0': res50_val.val_0, 'res50_1': res50_val.val_1,
    'dense201_0': dense201_val.val_0, 'dense201_1': dense201_val.val_1,
    'effb3_0': effb3_val.val_0, 'effb3_1': effb3_val.val_1,
    'y': res50_val.ground_truth_label
})

# Test stacking dataframe
test_stack = pd.DataFrame({
    'res50_0': res50_test.pred_0, 'res50_1': res50_test.pred_1,
    'dense201_0': dense201_test.pred_0, 'dense201_1': dense201_test.pred_1,
    'effb3_0': effb3_test.pred_0, 'effb3_1': effb3_test.pred_1
})



cont_names = list(train_stack.columns[:-1])  # model outputs
dep_var = 'y'

# Make y categorical
train_stack[dep_var] = train_stack[dep_var].astype(str)

# DataLoaders
splits = RandomSplitter(seed=42)(range_of(train_stack))
to = TabularPandas(train_stack, procs=[],
                   cont_names=cont_names,
                   y_names=dep_var,
                   y_block=CategoryBlock(),  
                   splits=splits)
dls = to.dataloaders(bs=64)

# Custom ROC AUC metric
def roc_score(inp, targ):
    # Convert to probabilities
    probs = inp.softmax(dim=1)[:,1].cpu().numpy()
    return roc_auc_score(targ.cpu().numpy(), probs)

# Meta-learner
learn = tabular_learner(dls, layers=[20,10],
                        metrics=[accuracy, roc_score],
                        wd=1e-2)

# Train
learn.fit_one_cycle(20, 1e-3)


# Predict on test set
test_dl = learn.dls.test_dl(test_stack)
preds, _ = learn.get_preds(dl=test_dl)

# Submission (probability for cancer class = column 1)
submission = pd.DataFrame({
    'id': res50_test.id,  # all test sets have same order
    'label': preds[:,1].numpy()
})

submission.to_csv('submission.csv', index=False)



#Extract image width and height info
h,w = img.shape[:2]
IMG_SIZE= (h,w)
#Use ImageDataGenerator to rescale and split images 
train_gen = ImageDataGenerator(
    rescale =1./255,
    validation_split = 0.25
)


# Use flow_from_dataframe to link image path to labels, 
# split data into training and validation samples 
# and generate training batches during training 
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='img_path',
    y_col='label',
    target_size=(96,96),
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    directory=None ,
    subset='training',
    vertical_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

valid_generator = train_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='img_path',
    y_col='label',
    target_size=(96,96),
    class_mode='raw',
    batch_size=32,
    shuffle=False,
    seed=42,
    directory=None,
    subset = 'validation'
)



base_model = EfficientNetB0(
    input_shape=(96, 96, 3), 
    weights='imagenet', 
    include_top=False
)
base_model.trainable = True 

# Build custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
optimizer = Adam(learning_rate=1e-4)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[AUC(name='auc')]
)

# Callbacks
early_stop = EarlyStopping(monitor='val_auc',
                           patience=5, 
                           mode='max', 
                           restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_auc', 
                              factor=0.2, 
                              patience=2, 
                              min_lr=1e-6, 
                              mode='max')

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_df['label']),
                                     y=train_df['label'])

class_weights = dict(enumerate(class_weights))

checkpoint = ModelCheckpoint(
    'efficientnet_best.h5',   # file to save model
    monitor='val_auc',        
    mode='max',
    save_best_only=True,
    save_weights_only=False   # saves full model (weights + optimizer state)
)

callbacks = [early_stop, lr_reduce, checkpoint]
# Train model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=50,
    callbacks=[early_stop, lr_reduce],
    class_weight=class_weights
)


#Extract training auc 
auc = history.history['auc']
#get validation auc
val_auc = history.history['val_auc']

epochs_range = range(len(auc))

#plot training and validation auc curve 
plt.plot(epochs_range, auc, 'r', label="Training AUC")
plt.plot(epochs_range, val_auc, 'b', label="Validation AUC")
plt.title("Training and Validation AUC")
plt.legend()
plt.show()

