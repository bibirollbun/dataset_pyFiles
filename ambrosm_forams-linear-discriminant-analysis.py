import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from glob import glob
import PIL
from tifffile import imread
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, log_loss, accuracy_score



vis_list = sorted(glob('/kaggle/input/forams-classification-2025/visualizations/visualizations/labelled/*.jpg'))
for j in [0, 5]:
    vis_list_subset = vis_list[j::15]
    plt.figure(figsize=(20, 4))
    for i, filename in enumerate(vis_list_subset):
        # print(filename)
        plt.subplot(1, len(vis_list_subset), i+1)
        img = np.asarray(PIL.Image.open(filename))
        plt.imshow(img)
        plt.title(str(i))
        plt.axis('off')
    plt.show()



def visualize_volume(volume_path, title=None, c=64):
    # Load the volume
    volume = imread(volume_path)
    
    # Get slices in each dimension
    slice_x = volume[c, :, :]
    slice_y = volume[:, c, :]
    slice_z = volume[:, :, c]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    
    # Plot slices
    axes[0].imshow(slice_x, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'X-Slice ({c})')
    
    axes[1].imshow(slice_y, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Y-Slice ({c})')
    
    axes[2].imshow(slice_z, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'Z-Slice ({c})')
    
    plt.tight_layout()
    if title:
        plt.suptitle(title, y=1.05, fontsize=24)
    plt.show()

vol_list = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/labelled/*.tif'))

for i, filename in enumerate(vol_list[::15]):
    visualize_volume(filename, title=f"Class {i}")
    for c in [40, 20]:
        visualize_volume(filename, c=c)



# Read the 210 true labels
labels = pd.read_csv(f'/kaggle/input/forams-classification-2025/labelled.csv', index_col='id')['label'] # 210 rows, id and label



# Squared distance from center for all 128*128*128 pixels
r0 = np.square(np.arange(128) - 63.5)
r = r0.reshape(-1, 1, 1) + r0.reshape(1, -1, 1) + r0.reshape(1, 1, -1)



# Feature engineering

def make_feature_array(vol_list):
    """Compute the features
    
    Parameter
    vol_list: list of filenames with 128*128*128 tiff files

    Return value
    X: array of shape (n_samples, n_features)
    """
    X = np.full((len(vol_list), 14), np.nan)
    for i, filename in enumerate(tqdm(vol_list)):
        volume = imread(filename)
        X[i] = np.array([volume.mean(),
                         volume.ravel()[volume.ravel() > 0].mean(),
                         volume.max(axis=0).mean(),
                         volume.max(axis=1).mean(),
                         volume.max(axis=2).mean(),
                         volume.max(axis=1).max(axis=0).mean(),
                         volume.max(axis=2).max(axis=1).mean(),
                         volume.max(axis=2).max(axis=0).mean(),
                         (r * volume).mean(),
                         (volume != 0).mean(),
                         np.abs(np.diff(volume, axis=0)).mean(),
                         np.abs(np.diff(volume, axis=1)).mean(),
                         np.abs(np.diff(volume, axis=2)).mean(),
                         float(filename[filename.index('_sc_') + 4 : -4].replace('_', '.')),
                        ])
    return X

vol_list = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/labelled/*.tif'))
X = make_feature_array(vol_list)


# Linear Discriminant Analysis

model = LinearDiscriminantAnalysis()

oof = cross_val_predict(model, X, labels)
print(f"# F1:      {f1_score(labels, oof, average='macro'):.3f}", end='   ')
print(f"Accuracy:  {accuracy_score(labels, oof):.3f}")

oof_prob = cross_val_predict(model, X, labels, method='predict_proba')
print(f"# Logloss: {log_loss(labels, oof_prob):.3f}")

sns.heatmap(confusion_matrix(labels, oof), annot=True)
plt.show()
# F1:      0.528   Acc:     0.529
# Logloss: 1.985


%%time
# Fit the model to the full dataset
display(model)
model.fit(X, labels)

# Compute the test predictions
vol_list = sorted(glob('/kaggle/input/forams-classification-2025/volumes/volumes/unlabelled/*.tif'))
# vol_list = np.random.default_rng().choice(vol_list, size=2000, replace=False) # subset to save time
X_test = make_feature_array(vol_list)
y_pred = model.predict(X_test)



x, y = np.unique(y_pred, return_counts=True)
plt.title('Predicted classes')
plt.bar(x, y)
plt.xticks(x)
plt.show()


unlabelled_index = pd.read_csv(f'/kaggle/input/forams-classification-2025/unlabelled.csv', index_col='id').index # 18216 rows, id and label (label is NaN)
pd.Series(y_pred, index=unlabelled_index[:len(y_pred)], name='label').to_csv('submission.csv')
!head submission.csv




