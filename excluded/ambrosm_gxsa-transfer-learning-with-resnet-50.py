import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import threading

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.calibration import CalibrationDisplay

import cv2
import os

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.metrics import AUC
import tensorflow.keras.applications.resnet50 as resnet50


def plot_training_history(history):
    """Plot a Keras training history."""
    if len(history['loss']) >= 2:
        _, axs = plt.subplots(2, 1, figsize=(6, 6))
        axs[0].plot(history['loss'], ':', label='train_loss')
        axs[0].plot(history['val_loss'], label='val_loss')
        axs[0].legend()
        axs[1].plot(history['AUC'], ':', label='train_auc')
        axs[1].plot(history['val_AUC'], label='val_auc')
        axs[1].legend()
        plt.tight_layout()
        plt.suptitle('Training history', y=1)
        plt.show()



def evaluate_oof(y_train, oof_pred, baseline_true=None, baseline_oof_pred=None, label='', plot=True):
    """Compute oof scores and visualize them"""
    n_targets = y_train.shape[1]
    n_horizontal = (n_targets + 1) // 2 # diagrams per row
    
    # Process y_train and oof_pred
    if type(oof_pred) is not pd.DataFrame:
        oof_pred = pd.DataFrame(oof_pred, columns=y_train.columns)
    assert y_train.shape == oof_pred.shape, (y_train.shape, oof_pred.shape)
    valid_scores = np.ones(n_targets, dtype=bool)
    scores = np.full(n_targets, 0.5)
    for i, target in enumerate(y_train.columns):
        try:
            scores[i] = roc_auc_score(y_train[target], oof_pred[target])
        except ValueError:
            scores[i] = np.nan
            valid_scores[i] = False
    macro_mean = np.nanmean(scores[:14])

    # Process baseline_true and baseline_oof_pred
    if baseline_oof_pred is not None and type(baseline_oof_pred) is not pd.DataFrame:
        baseline_oof_pred = pd.DataFrame(baseline_oof_pred, columns=y_train.columns)
        assert baseline_true.shape == baseline_oof_pred.shape, (baseline_true.shape, baseline_oof_pred.shape) 
        baseline_scores = np.array(roc_auc_score(baseline_true, baseline_oof_pred, average=None))

    # Print
    print(f"{label}\t{macro_mean:.3f}", end='')
    for i in range(len(scores)):
        print(f"\t{scores[i]:.3f}", end='')
    print()

    if plot:
        # Bar chart
        plt.figure(figsize=(16, 3))
        plt.title(f"Average AUC: {macro_mean:.3f} ({label})")
        if baseline_oof_pred is not None:
            baseline_to_plot = np.where(valid_scores, baseline_scores, np.nan)
            plt.bar(np.arange(len(scores)) - 0.2, baseline_to_plot - 0.5, bottom=0.5, width=0.4, color='lightgray')
            bars = plt.bar(np.arange(len(scores)) + 0.2, scores - 0.5, bottom=0.5, width=0.4, color='darkgreen')
        else:
            bars = plt.bar(np.arange(len(scores)), scores - 0.5, bottom=0.5, width=0.8, color='lightgray')
        plt.bar_label(bars, fmt='%.3f')
        if n_targets > 14:
            plt.axvline(13.5, color='gray')
        plt.xticks(np.arange(len(scores)), y_train.columns, rotation=45, ha='right')
        plt.ylim(0.5, 1)
        plt.show()
    
        # ROC curves
        _, axs = plt.subplots(2, n_horizontal, figsize=(14, 4))
        for i, (target, ax) in enumerate(zip(y_train.columns, axs.ravel())):
            if valid_scores[i]:
                RocCurveDisplay.from_predictions(y_train[target], oof_pred[target], ax=ax)
                ax.get_legend().remove()
                ax.set_title(target)
                ax.set_xlabel(None) # false positive rate
                ax.set_ylabel(None) # true positive rate
                ax.set_aspect('equal')
            else:
                ax.set_visible(False)
        if n_targets < len(axs.ravel()):
            axs[-1, -1].set_visible(False)
        plt.tight_layout()
        plt.suptitle('Receiver operating curves', y=1)
        plt.show()
    
        # Histograms
        _, axs = plt.subplots(2, n_horizontal, figsize=(14, 3))
        for i, (target, ax) in enumerate(zip(y_train.columns, axs.ravel())):
            if valid_scores[i]:
                ax.hist(oof_pred[target], bins=np.linspace(0, 1, 100), color='brown', density=True)
                ax.set_title(target)
                ax.set_xlabel(None) # predicted probability
                ax.set_ylabel(None) # density
            else:
                ax.set_visible(False)
        if n_targets < len(axs.ravel()):
            axs[-1, -1].set_visible(False)
    
        plt.tight_layout()
        plt.suptitle('Predicted probability histograms', y=1.01)
        plt.show()

        # Calibration curves
        _, axs = plt.subplots(2, n_horizontal, figsize=(14, 4))
        for i, (target, ax) in enumerate(zip(y_train.columns, axs.ravel())):
            if valid_scores[i]:
                CalibrationDisplay.from_predictions(y_train[target], oof_pred[target], 
                                                    ax=ax, color='g',
                                                    n_bins=10, strategy='quantile')
                ax.get_legend().remove()
                ax.set_title(target)
                ax.set_xlabel(None) # predicted probability
                ax.set_ylabel(None) # true probability
                ax.set_aspect('equal')
            else:
                ax.set_visible(False)
        if n_targets < len(axs.ravel()):
            axs[-1, -1].set_visible(False)
    
        plt.tight_layout()
        plt.suptitle('Calibration display', y=1)
        plt.show()



train = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
test = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv')[['Image_name']]
targets = list(train.columns[-14:])

print("Targets:", targets)

y_train = train[targets].copy()

# Just for fun, we add three more targets:
y_train['male'] = np.where(train.Sex == 'Male', 1, 0) # must be binary classification
y_train['female'] = np.where(train.Sex == 'Female', 1, 0) # must be binary classification
y_train['frontal'] = (train.ViewCategory == 'Frontal').astype(float)

def engineer_features(df):
    df['Patient_ID'] = df['Image_name'].str.slice(0, 8).astype(int)
    df['Study'] = df['Image_name'].str.slice(9, 12).astype(int)
    df['Patient_Study'] = df['Image_name'].str.slice(0, 12).astype(int)
    df['institute'] = df['Patient_ID'] // 10000000
    return df

engineer_features(train)
engineer_features(test)
display(test.tail(4))



# # Test the notebook with a subset of the data
# train = train.iloc[:1000]
# y_train = y_train.iloc[:1000]
# test = test.iloc[:60]



# %%time
n_threads = 10 # set to 1 for debugging, 10 for production
width = 224 # must be divisible by 32
height = 224 # must be divisible by 32
resnet_embedding_dim = 2048

def load_and_resize_images(n_threads, j, paths, images, width, height):
    """Load and resize a subset of the images.

    This function is to be executed by several threads in parallel.

    The images are resized to width x height pixels. If the original image
    isn't square, the pixels are stretched.
    """
    for im_idx in range(j, len(paths), n_threads):
        img = cv2.imread(paths[im_idx], cv2.IMREAD_GRAYSCALE)
        assert img is not None
        img = cv2.resize(img, (width, height)) # resize expects (width, height)
        # if n_threads == 1 and im_idx < 10:
        #     print(img.shape, img.dtype) # array has shape (height, width) and dtype uint8
        #     plt.imshow(img, vmin=0, vmax=255)
        #     plt.show()
        images[im_idx] = img


def load_and_resize_dataset(directory, df, width, height):
    """Load and resize all images of a dataset.
    
    The images are read by several threads in parallel: Reading the images is i/o-bound, and
    multithreading gives good speedup. Unfortunately, multithreading is incompatible with tqdm.
    The process can take an hour, and you won't see a progress bar.
    """
    assert width % 32 == 0, 'must be divisible by 32' 
    assert height % 32 == 0, 'must be divisible by 32'
    image_dir = f'/kaggle/input/grand-xray-slam-division-a/{directory}/'
    paths = [f'/kaggle/input/grand-xray-slam-division-a/{directory}/{p}' for p in df.Image_name]
    print(f"{directory}: Processing {len(df)} images")

    # Create an array of resized images of shape (n_images, height, width) and dtype uint8
    images = np.zeros((len(df), height, width), dtype=np.uint8)

    if n_threads > 1:
        threads = [threading.Thread(target=load_and_resize_images, args=(n_threads, j, paths, images, width, height)) for j in range(n_threads)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
    else: # debug in the main thread
        load_and_resize_images(1, 0, paths, images, width, height)

    assert (images.sum(axis=2).sum(axis=1) != 0).all()
    return images

    
def embed_with_ResNet(images):
    """Read the uint8 array of images and compute embeddings"""

    embeddings = np.zeros((len(images), resnet_embedding_dim), dtype=np.float32)
    batch_size = 10240
    for batch_start in trange(0, len(images), batch_size):
        
        # Convert grayscale to rgb and center
        # X = img_train # shape (n_samples, HEIGHT, WIDTH) of uint8
        Xb = images[batch_start:batch_start+batch_size]
        XX = np.repeat(Xb, 3).reshape(Xb.shape + (3, )) # shape (n_samples, HEIGHT, WIDTH, 3)
        XX = resnet50.preprocess_input(XX)
        
        # Compute embeddings
        embeddings[batch_start:batch_start+batch_size] = resnet50_model.predict(XX, batch_size=256, verbose=1) # shape (n_samples, 2048)
    return embeddings


def load_resize_embed(directory, df, width, height):
    images = load_and_resize_dataset(directory, df, width, height)
    embeddings = embed_with_ResNet(images)
    print(f"embeddings_{directory[:-1]}.shape = {embeddings.shape}")
    
    embeddings.tofile(f"embeddings_{directory[:-1]}.binary")
    loading = f"np.memmap(f'embeddings_{directory[:-1]}.binary', mode='r', dtype=np.float32, shape={embeddings.shape})"
    embeddings_2 = eval(loading)
    assert (embeddings == embeddings_2).all()
    print(f"Next time, you can load the embeddings with\n\n    embeddings_{directory[:-1]} = {loading}\n")

    return embeddings


# Load the ResNet50 model
resnet50_model = resnet50.ResNet50(include_top=False,
                               weights='imagenet',
                               pooling='avg',
                               input_shape=(height, width, 3))

# Print the model summary
# resnet50_model.summary()

# Do the job
embeddings_train = load_resize_embed('train1', train, width, height)
embeddings_test = load_resize_embed('test1', test, width, height)



# Custom data generator for images

class XRayGeneratorForResNet(Sequence):
    """Sequence of image batches.

    The images are read from disk with fixed width and height. It is possible that
    pixels are no longer square after the transformation.

    X and y have the same index; this index corresponds to the positions of the images in the array.
    """
    def __init__(self, X, y=None, embeddings=None, shuffle=False, batch_size=32):
        super().__init__(workers=1)
        assert embeddings is not None
        if y is not None:
            assert len(X) == len(y)
        self.X = X # dataframe with columns 'Image_name' and 'institute'
        self.y = y # dataframe with one column per target
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Shuffle to avoid all images in a batch being of the same patient
        self.permutation = np.arange(len(self.X))
        if self.shuffle:
            self.rng = np.random.default_rng(1)
            self.rng.shuffle(self.permutation)

    def __len__(self):
        """Number of batches in one epoch, including a smaller batch at the end."""
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        """Return a batch of images (and optionally, labels).

        Returns
        images: float32 array of shape (batch_size, resnet_embedding_dim)
        metadata: float32 array of shape (batch_size, 1)
        labels: float32 array of shape (batch_size, n_targets)
        """
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.X))
        batch_indexes = self.permutation[start_idx:end_idx]
        resnet_embeddings = self.embeddings[self.X.index[batch_indexes]]
        metadata = self.X[['institute']].values[batch_indexes].astype(np.float32)
        features = {'resnet_embeddings': resnet_embeddings, 'metadata': metadata}

        if self.y is not None:
            labels = self.y.values[batch_indexes].astype(np.float32)
            return features, labels
        return features



def create_head_model(num_classes=y_train.shape[1]):
    resnet_embeddings = Input((resnet_embedding_dim, ), name='resnet_embeddings')
    metadata =  Input((1, ), name='metadata')
    x = Concatenate()([resnet_embeddings, metadata])
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs={'resnet_embeddings': resnet_embeddings,
                          'metadata': metadata},
                  outputs=x,
                  name='Resnet50-Head')
    return model    

create_head_model().summary()



# %%time

# Parameters
epochs, n_folds, compute_test_predictions = 50, 5, True # for submission
# epochs, n_folds, compute_test_predictions = 50, 1, False # for validation without submission
batch_size = 64
verbose = 1 if os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive' else 2

if compute_test_predictions:
    test_generator = XRayGeneratorForResNet(test, None, embeddings_test, batch_size=batch_size, shuffle=False)
    test_pred_list = []

gkf = GroupKFold()
for fold, (idx_tr, idx_va) in enumerate(gkf.split(train, groups=train.Patient_ID)):
    if fold == n_folds: break

    # Split into train and test and construct the generators
    X_tr = train.iloc[idx_tr]
    X_va = train.iloc[idx_va]
    y_tr = y_train.iloc[idx_tr]
    y_va = y_train.iloc[idx_va]
    train_generator = XRayGeneratorForResNet(X_tr, y_tr, embeddings_train, batch_size=batch_size, shuffle=True)
    val_generator = XRayGeneratorForResNet(X_va, y_va, embeddings_train, batch_size=batch_size, shuffle=False)
    
    # Construct and compile the model
    model = create_head_model()
    label = f"{model.name}"
    print('Fitting', label)
    label_weights = (y_train.var(axis=0) != 0).astype(float) # exclude constant targets from metric
    label_weights[14:] = 0 # exclude just-for-fun targets from metric
    model.compile(
        # optimizer=Adam(learning_rate=0.001),
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[AUC(multi_label=True, name='AUC', label_weights=label_weights)]
    )
    
    # Train and plot the training history
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[EarlyStopping(patience=3),
                   ReduceLROnPlateau(factor=0.5, patience=0, verbose=1, min_lr=0.000126),
                   TerminateOnNaN()],
        verbose=verbose
    )
    history = history.history
    plot_training_history(history)

    def do_inference(generator, X, y_true=None):
        """Compute predictions grouping by study, and evaluate them."""
        y_pred = model.predict(generator, verbose=verbose)
        if y_true is not None: evaluate_oof(y_true, y_pred, label=label, plot=False)
        
        y_pred_frontal = y_pred[:,-1]
        y_pred = pd.DataFrame(y_pred, index=X.index)
        y_pred = y_pred.groupby(X['Patient_Study'].values).transform(lambda x: x.values.mean()).values
        y_pred[:,-1] = y_pred_frontal
        if y_true is not None: evaluate_oof(y_true, y_pred, label=label+' grouped')
        return y_pred

    # Validate and visualize validation score
    print("\nValidation")
    do_inference(val_generator, X_va, y_va)
    print()

    # Compute test_predictions
    if compute_test_predictions:
        test_pred_list.append(do_inference(test_generator, test))



if len(test_pred_list) > 0:
    test_pred = np.stack(test_pred_list).mean(axis=0)
    submission = pd.DataFrame(test_pred[:,:14],
                              columns=targets,
                              index=test.Image_name)
    display(submission)
    submission.to_csv('submission.csv')
    !head submission.csv





