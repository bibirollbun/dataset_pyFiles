import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from glob import glob
import PIL
from tifffile import imread
from tqdm import tqdm
import math
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, log_loss, accuracy_score, classification_report

from tensorflow import __version__
import keras
import keras.applications.resnet50 as resnet50
from keras.utils import image_dataset_from_directory
from keras.layers import Dense, Activation, Flatten, Dropout, Concatenate
from keras.models import Sequential, Model 
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TerminateOnNaN

print(f"Tensorflow version: {__version__}")

FROM_SCRATCH = False
PREPROCESSED_DIR = "/kaggle/input/forams-lighted-from-front" # lighted from front



if FROM_SCRATCH:
    HEIGHT = 127
    WIDTH = 127
    
    resnet50_model = resnet50.ResNet50(include_top=False,
                                       weights='imagenet',
                                       pooling='avg',
                                       input_shape=(HEIGHT, WIDTH, 3))


%%time
# Read the 210 true labels
labels = pd.read_csv(f'/kaggle/input/forams-classification-2025/labelled.csv', index_col='id')['label'].values # 210 rows, id and label

if FROM_SCRATCH:
    # Read the 3d images and convert every 3d image to
    # 24 2d surface images
    
    linear_f = np.arange(128) + 100
    linear_b = linear_f[::-1]
    
    def make_feature_array(vol_list, labels=None, group0=0, plot=False):
        """Create six surface views for every 3d sample in vol_list."""
        X, more_features, groups, y = [], [], [], []
    
        def add_surface(surface, i, label):
            """Add this surface to the list.
            
            The input surface is a 128*128 array of heights, which will
            be converted into a 127*127 grayscale image."""
            surface = surface.astype(np.float32)
            grad = np.sqrt(np.square(surface[1:,1:] - surface[:-1,1:]) + np.square(surface[1:,1:] - surface[1:,:-1]))
            grad[surface[1:,1:] == 0] = 0
            grad[surface[:-1,1:] == 0] = 0
            grad[surface[1:,:-1] == 0] = 0
            grad = 6 - grad.clip(0, 6)
            grad *= 42
            X.append(np.hstack([grad]))
            more_features.append((scaling_factor, mass))
            groups.append(group0 + i)
            y.append(label)
            
        for i, filename in enumerate(vol_list):
            volume = imread(filename)
    
            # Make black-white for best contrast and so that we can define surfaces
            volume_bw = volume > 110
    
            # Find total mass of object
            mass = volume_bw.sum()
            mass = (mass - 162000) / 56662
    
            # Scaling factor taken from filename
            scaling_factor = float(filename[filename.index('_sc_') + 4 : -4].replace('_', '.'))
    
            # Extract the six surface views
            # A surface is a 128*128 array of heights
            label = labels[i] if labels is not None else None
            add_surface((volume_bw * linear_f.reshape(-1, 1, 1)).max(axis=0), i, label)
            add_surface((volume_bw * linear_b.reshape(-1, 1, 1)).max(axis=0), i, label)
            add_surface((volume_bw * linear_f.reshape(1, -1, 1)).max(axis=1), i, label)
            add_surface((volume_bw * linear_b.reshape(1, -1, 1)).max(axis=1), i, label)
            add_surface((volume_bw * linear_f.reshape(1, 1, -1)).max(axis=2), i, label)
            add_surface((volume_bw * linear_b.reshape(1, 1, -1)).max(axis=2), i, label)
    
            # Plot
            if plot and i == 0:
                _, axs = plt.subplots(1, 6, figsize=(7, 1.25))
                axs = axs.ravel()
                for j in range(len(axs)):
                    axs[j].imshow(X[-1-j], cmap='gray')
                    axs[j].axis('off')
                plt.show()
    
        # Convert grayscale to rgb and center
        X = np.array(X) # shape (n_samples, 127, 127)
        X = np.repeat(X, 3).reshape(X.shape + (3, )) # shape (n_samples, 127, 127, 3)
        X = resnet50.preprocess_input(X)
    
        return (X, 
                np.array(more_features).astype(np.float32),
                np.array(groups),
                np.array(y) if labels is not None else None)
    
    def augment(X, more_features, groups, y):
        """Augment the data 8-fold"""
        # Mirror all images
        X = np.vstack([X, np.flip(X, axis=2)])
        more_features = np.tile(more_features, (2, 1))
        groups = np.tile(groups, 2)
        y = np.tile(y, 2)
        
        # Rotate all images by 180 degrees
        X = np.vstack([X, X[:, ::-1, ::-1, :]])
        more_features = np.tile(more_features, (2, 1))
        groups = np.tile(groups, 2)
        y = np.tile(y, 2)
        
        # Rotate all images by 90 degrees
        X = np.vstack([X, np.rot90(X, axes=(1, 2))])
        more_features = np.tile(more_features, (2, 1))
        groups = np.tile(groups, 2)
        y = np.tile(y, 2)
    
        return X, more_features, groups, y
    
    # Read and augment the labelled data
    vol_list_labelled = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/labelled/*.tif'))
    X_labelled, more_features, groups, y = make_feature_array(vol_list_labelled, labels=labels, plot=True)
    X_labelled, more_features, groups, y = augment(X_labelled, more_features, groups, y)
    
    X_labelled.shape, more_features.shape, groups.shape, y.shape
    # ((10080, 127, 127, 3), (10080, 2), (10080,), (10080))
    # 20 s for all 210 labelled forams
    
    X_labelled = resnet50_model.predict(X_labelled, batch_size=256, verbose=1)
    # 6 minutes for 10080 training samples on cpu
    
    # Save the features
    with open("X_labelled.pickle", "wb") as f:
        pickle.dump((X_labelled, more_features, groups, y), f)

else:
    with open(PREPROCESSED_DIR + "/X_labelled.pickle", "rb") as f:
        X_labelled, more_features, groups, y = pickle.load(f)

groups = - groups - 1

# Binarize the labels
lb = LabelBinarizer()
y_b = lb.fit_transform(y)

X_labelled.shape, more_features.shape, groups.shape, y_b.shape
# ((10080, 127, 127, 3), (10080, 2), (10080,), (10080, 14))



predictions = pd.read_csv('/kaggle/input/forams-predictions/forams-predictions.csv', 
                          sep=';',
                          index_col='id')
predictions['all_equal'] = predictions.var(axis=1) == 0
predictions['pseudolabel'] = np.where(predictions.all_equal, predictions['resnet-v4'], -1)
print(f"Agreement: {predictions.all_equal.mean():.0%}")
predictions


%%time
if FROM_SCRATCH:
    # Read and augment the unlabelled data
    vol_list_ul = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/unlabelled/*.tif'))
    X_ul_list, more_features_ul_list, groups_ul_list, y_ul_list = [], [], [], [] 
    test_batch_size = 256
    for batch_start in tqdm(range(0, len(vol_list_ul), test_batch_size)):
        X_ul, more_features_ul, groups_ul, y_ul = make_feature_array(
            vol_list_ul[batch_start:batch_start+test_batch_size],
            labels=predictions.pseudolabel.values[batch_start:batch_start+test_batch_size],
            group0=batch_start,
            plot=False)
        X_ul, more_features_ul, groups_ul, y_ul = augment(X_ul, more_features_ul, groups_ul, y_ul)
        X_ul = resnet50_model.predict(X_ul, batch_size=256, verbose=0)
        X_ul_list.append(X_ul)
        more_features_ul_list.append(more_features_ul)
        groups_ul_list.append(groups_ul)
        y_ul_list.append(y_ul)
    
    X_ul = np.vstack(X_ul_list)
    more_features_ul = np.vstack(more_features_ul_list)
    groups_ul = np.hstack(groups_ul_list)
    y_ul = np.hstack(y_ul_list)
    
    del X_ul_list, more_features_ul_list, groups_ul_list, y_ul_list

    # Save the features (7 GByte)
    with open("X_pseudolabelled.pickle", "wb") as f:
        pickle.dump((X_ul, more_features_ul, groups_ul, y_ul), f)

else:
    with open(PREPROCESSED_DIR + "/X_pseudolabelled.pickle", "rb") as f:
        X_ul, more_features_ul, groups_ul, y_ul = pickle.load(f) # 7 GByte
    
print('Both:', X_ul.shape, more_features_ul.shape, groups_ul.shape, y_ul.shape)

# Binarize the labels
y_ul_b = lb.transform(y_ul)

# Separate pseudolabelled and unlabelled datasets
X_pl = X_ul[y_ul >= 0]
X_ul = X_ul[y_ul < 0]
more_features_pl = more_features_ul[y_ul >= 0]
more_features_ul = more_features_ul[y_ul < 0]
groups_pl = groups_ul[y_ul >= 0]
groups_ul = groups_ul[y_ul < 0]
y_pl_b = y_ul_b[y_ul >= 0]
y_ul_b = y_ul_b[y_ul < 0]
y_pl = y_ul[y_ul >= 0]
y_ul = y_ul[y_ul < 0] # always -1, must be assigned last
print('Pseudolabelled:', X_pl.shape, more_features_pl.shape, groups_pl.shape, y_pl.shape)
print('Unlabelled:', X_ul.shape, more_features_ul.shape, groups_ul.shape, y_ul.shape)



DROPOUT = 0.5
FC_LAYERS = [256] # [256]
N_CLASSES = 14

def build_model(dropout, fc_layers, n_classes):
    """Some fully connected layers and a final softmax layer"""
    embedding = keras.layers.Input(shape=(2048, ), name='embedding')
    more_features = keras.layers.Input(shape=(2, ), name='more_features')
    x = Concatenate()([embedding, more_features])
    for fc in fc_layers:
        print(fc)
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs={'embedding': embedding, 'more_features': more_features},
                           outputs=outputs)
    return model

# build_model(dropout=DROPOUT, fc_layers=FC_LAYERS, n_classes=N_CLASSES).summary()


%%time
# Cross-validate

BATCH_SIZE = 512
INFERENCE = True
EARLY_STOPPING = True
if EARLY_STOPPING:
    NUM_EPOCHS = 100
else:
    NUM_EPOCHS = 6
VERBOSE = 0

cv = StratifiedGroupKFold(n_splits=15)
oof_prob = np.full((len(y), N_CLASSES), np.nan)
epoch_sum = 0
if INFERENCE:
    ul_probs = []
    
for fold, (idx_tr, idx_va) in enumerate(cv.split(X_labelled, y, groups=groups)):
    # Clean up the memory
    try:
        del history
    except NameError:
        pass
    try:
        del my_model
    except NameError:
        pass

    # Split into training and validation data
    X_tr = X_labelled[idx_tr]
    more_features_tr = more_features[idx_tr]
    y_tr = y_b[idx_tr]
    X_va = X_labelled[idx_va]
    more_features_va = more_features[idx_va]
    y_va = y_b[idx_va]

    # Add the pseudolabelled data for training
    X_tr = np.vstack([X_tr, X_pl])
    more_features_tr = np.vstack([more_features_tr, more_features_pl])
    y_tr = np.vstack([y_tr, y_pl_b])
    
    # Build, compile and fit the model
    my_model = build_model(dropout=DROPOUT,
                           fc_layers=FC_LAYERS,
                           n_classes=N_CLASSES)
    my_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003),
                     loss=keras.losses.CategoricalCrossentropy(),
                     metrics=["categorical_accuracy", "f1_score"]
                    )
    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, 
                           patience=4, verbose=VERBOSE)
    es = EarlyStopping(monitor="val_loss",
                       patience=10, 
                       verbose=1,
                       mode="min", 
                       restore_best_weights=True)
    callbacks = [lr, es, TerminateOnNaN()] if EARLY_STOPPING else [TerminateOnNaN()]
    history = my_model.fit({'embedding': X_tr, 'more_features': more_features_tr},
                           y_tr,
                           batch_size=BATCH_SIZE,
                           epochs=NUM_EPOCHS,
                           # steps_per_epoch=3,
                           validation_data=({'embedding': X_va, 'more_features': more_features_va}, y_va),
                           validation_batch_size=1024,
                           verbose=VERBOSE,
                           callbacks=callbacks)
    epoch_sum += np.argmin(history.history['val_loss'])
    
    # Plot history
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(history.history['loss'])) + 1,
             history.history['loss'],
             label='train loss')
    plt.plot(np.arange(len(history.history['val_loss'])) + 1,
             history.history['val_loss'],
             label='val loss')
    plt.title(f'Training history fold {fold}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0, 2)
    plt.grid(True, axis='y')
    plt.legend()
    plt.show()

    # Clean up the memory
    del lr, es, callbacks, history

    # Compute oof probabilities
    oof_prob[idx_va] = my_model.predict({'embedding': X_va, 'more_features': more_features_va},
                                        verbose=0,
                                        batch_size=1024)

    if INFERENCE:
        ul_prob = my_model.predict({'embedding': X_ul, 'more_features': more_features_ul}, 
                                  verbose=0,
                                  batch_size=1024)
        # ul_prob_grouped = pd.DataFrame(ul_prob).groupby(groups_ul).mean().values # average test-time augmentation
        ul_probs.append(ul_prob) # add to final ensemble



PERCENT_OUTLIERS = 5 # hyperparameter: how many samples do we want to qualify as outliers?

ul_prob = np.stack(ul_probs).mean(axis=0) # mean over predictions of five folds (513168, 14)
ul_prob_grouped = pd.DataFrame(ul_prob).groupby(groups_ul).mean().values
ul_pred = np.argmax(ul_prob_grouped, axis=1)
CLASS_14_THRESHOLD = np.quantile(
    ul_prob_grouped.max(axis=1),
    q=PERCENT_OUTLIERS/100/len(ul_prob_grouped) * 18216
)
ul_pred[ul_prob_grouped.max(axis=1) < CLASS_14_THRESHOLD] = 14 # "unknown" class

# Evaluate the results with test-time augmentation
oof_prob_grouped = pd.DataFrame(oof_prob).groupby(groups).mean().values
y_grouped = pd.Series(y).groupby(groups).mean().values

oof = np.argmax(oof_prob_grouped, axis=1)
oof[oof_prob_grouped.max(axis=1) < CLASS_14_THRESHOLD] = 14 # "unknown" class
print(f"# F1: {f1_score(y_grouped, oof, labels=np.arange(15), average='macro', zero_division=0):.3f}", end='   ')
print(f"Acc: {accuracy_score(y_grouped, oof):.3f}", end='   ')
print(f"Logloss: {log_loss(y_grouped, oof_prob_grouped):.3f} {PERCENT_OUTLIERS}%")

sns.heatmap(confusion_matrix(y_grouped, oof, labels=np.arange(15)), annot=True, fmt='.0f')
plt.title(f'Resnet-50 with TTA{" with early stopping" if EARLY_STOPPING else ""}')
plt.show()
print(classification_report(y_grouped, oof, zero_division=0, labels=np.arange(15)))

# Best epoch
print(f"Best epoch: {epoch_sum / 5:.0f}")

# Save the oof probabilities
with open("oof_prob.pickle", "wb") as f:
    pickle.dump((oof_prob, groups, y), f)

# Save the test probabilities
with open("ul_prob.pickle", "wb") as f:
    pickle.dump(ul_prob, f)




if INFERENCE:
    # Write the submission file
    y_pred = predictions.pseudolabel.values.copy()
    y_pred[y_pred < 0] = ul_pred
    submission = pd.Series(
        y_pred,
        index=pd.RangeIndex(0, len(predictions), name='id'),
        name='label')
    submission.to_csv('submission.csv')
    !head submission.csv

    # Frequency diagram
    plt.figure(figsize=(12, 4))
    plt.title('Predicted classes')
    x, count = np.unique(ul_pred, return_counts=True)
    _, pl_count = np.unique(predictions.pseudolabel.values, return_counts=True)
    pl_count = np.array(list(pl_count[1:]) + [0])
    b = plt.bar(x, count, bottom=pl_count, label='counted')
    plt.bar(np.arange(14), ul_prob_grouped.sum(axis=0), bottom=pl_count[:-1], alpha=0.6, label='expected')
    plt.bar_label(b)
    plt.bar(np.arange(15), pl_count, color='lightgray', label='pseudolabelled')
    plt.xticks(np.arange(15))
    plt.xlabel('class')
    plt.ylabel('count')
    plt.legend()
    plt.show()
    
    # Embedding into two dimensions diagram
    tab20 = matplotlib.colormaps['tab20']
    colors = [tab20.colors[np.round(x).astype(int)] for x in np.linspace(0, 19, 14)] + [(0, 0, 0)]
    colors = np.array(colors) # shape (15, 3)
    
    components = np.vstack([np.cos(np.arange(14) / 14 * 2 * math.pi),
                            np.sin(np.arange(14) / 14 * 2 * math.pi),
                           ])
    tt = oof_prob_grouped @ components.T # train
    uu = ul_prob_grouped @ components.T # test
    plt.figure(figsize=(12, 12))
    plt.title('2d projection of softmax probabilities')
    plt.scatter(tt[:,0], tt[:,1], s=30, c=colors[labels], marker='x', label='train')
    plt.scatter(uu[:,0], uu[:,1], s=3, c=colors[ul_pred], label='test')
    plt.scatter(components[0], components[1], s=100, c='k', alpha=0.3)
    for cl in range(14):
        plt.text(components[0, cl] * 1.06, components[1, cl] * 1.06, cl, ha='center', va='center_baseline')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()

print((oof==14).mean(), (y_pred==14).mean())




