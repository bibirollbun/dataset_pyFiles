!pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop
!pip install git+https://github.com/WildlifeDatasets/wildlife-tools


import numpy as np
from typing import List, Union

def baks_compute(
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        identity_test_only: Union[List, np.ndarray]
    ) -> float:
    """Computes BAKS (balanced accuracy on known samples).
    
    Focuses only on samples with known identities (not in identity_test_only).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        identity_test_only: Labels of unknown identities (only in test set)
        
    Returns:
        Balanced accuracy score for known samples
    """
    # Convert inputs to numpy arrays with object dtype to handle mixed types
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)
    
    # Filter out unknown samples
    mask = ~np.isin(y_true, identity_test_only)
    y_true_known = y_true[mask]
    y_pred_known = y_pred[mask]
    
    if len(y_true_known) == 0:
        return 0.0
    
    # Get unique classes in the filtered true labels
    unique_classes = np.unique(y_true_known)
    n_classes = len(unique_classes)
    
    # Compute per-class accuracy and average
    class_accuracies = []
    for cls in unique_classes:
        cls_mask = (y_true_known == cls)
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(y_pred_known[cls_mask] == cls)
            class_accuracies.append(cls_acc)
    
    # Return the balanced accuracy (mean of per-class accuracies)
    return np.mean(class_accuracies) if class_accuracies else 0.0

def baus_compute(
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        identity_test_only: Union[List, np.ndarray],
        new_class: Union[int, str]
    ) -> float:
    """Computes BAUS (balanced accuracy on unknown samples).
    
    Focuses only on samples with unknown identities (in identity_test_only).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        identity_test_only: Labels of unknown identities (only in test set)
        new_class: Label used for identifying unknown samples
        
    Returns:
        Balanced accuracy score for unknown samples
    """
    # Convert inputs to numpy arrays with object dtype to handle mixed types
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)
    
    # Filter to include only unknown samples
    mask = np.isin(y_true, identity_test_only)
    y_true_unknown = y_true[mask]
    y_pred_unknown = y_pred[mask]
    
    if len(y_true_unknown) == 0:
        return 0.0
    
    # Get unique unknown classes
    unique_unknown_classes = np.unique(y_true_unknown)
    
    # For each unknown class, check if they were correctly predicted as new_class
    class_accuracies = []
    for cls in unique_unknown_classes:
        cls_mask = (y_true_unknown == cls)
        if np.sum(cls_mask) > 0:
            # For unknown samples, correct prediction is new_class
            cls_acc = np.mean(y_pred_unknown[cls_mask] == new_class)
            class_accuracies.append(cls_acc)
    
    # Return the balanced accuracy (mean of per-class accuracies)
    return np.mean(class_accuracies) if class_accuracies else 0.0


def compute_geometric_mean(baks, baus):
    return np.sqrt(baks * baus)


import os
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T
from wildlife_datasets.datasets import AnimalCLEF2025
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)


root = '/kaggle/input/animal-clef-2025'
transform_display = T.Compose([
    T.Resize([384, 384]),
    ])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


from wildlife_datasets import datasets, splits
dataset = AnimalCLEF2025(root, transform=transform_display, load_label=True)
df = dataset.df 
df = df[df['split'] == 'database']
df.species.value_counts()


df.isna().value_counts()


df.fillna('salamander', inplace=True)
df.species.value_counts()


splitter = splits.OpenSetSplit(0.8, 0.1)
for idx_train, idx_test in splitter.split(df):
    splits.analyze_split(df, idx_train, idx_test)


df_train, df_test = df.loc[idx_train], df.loc[idx_test]
len(df_train)
len(df_test)


df_train


training_dataloader = dataset.get_subset(df_train.index)
val_dataloader = dataset.get_subset(df_test.index)


training_dataloader.transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
training_dataloader.transform

val_dataloader.transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
val_dataloader.transform


name = 'hf-hub:BVRA/MegaDescriptor-L-384'
device = 'cuda'
model = timm.create_model(name, num_classes=0, pretrained=True)
extractor = DeepFeatures(model, device=device, batch_size=32, num_workers=0)
features_database = extractor(training_dataloader)
features_query = extractor(val_dataloader)
n_query = len(val_dataloader)


similarity = CosineSimilarity()(features_query, features_database)


pred_idx = similarity.argsort(axis=1)[:,-1]
pred_scores = similarity[range(n_query), pred_idx]


new_individual = 'new_individual'
threshold = 0.6
labels = training_dataloader.labels_string
predictions = labels[pred_idx]
predictions[pred_scores < threshold] = new_individual


unseen_ids = []
all_ids = df['identity'].unique()

for i in all_ids:
    if i in val_dataloader.metadata['identity'].tolist():
        if i not in training_dataloader.metadata['identity'].tolist():
            unseen_ids.append(i)


val_true_labels = val_dataloader.labels_string
baks_score = baks_compute(val_true_labels, predictions, unseen_ids)
baus_score = baus_compute(val_true_labels, predictions, unseen_ids, "new_individual")
geo_mean = compute_geometric_mean(baks_score, baus_score)

print(f"Balanced Accuracy Known Samples (BAKS): {baks_score:.4f}")
print(f"Balanced Accuracy Unknown Samples (BAUS): {baus_score:.4f}")
print(f"Geometric Mean (BAKS & BAUS): {geo_mean:.4f}")


thresholds = np.arange(0.1, 0.9, 0.05)
results = []
labels = np.array(training_dataloader.labels_string)


for threshold in thresholds:
    predictions = labels[pred_idx].copy()
    predictions[pred_scores < threshold] = 'new_individual'

    baks = baks_compute(val_true_labels, predictions, unseen_ids)
    baus = baus_compute(val_true_labels, predictions, unseen_ids, 'new_individual')
    geo_mean = compute_geometric_mean(baks, baus)
    results.append((threshold, baks, baus, geo_mean))

    print(f"Threshold: {threshold:.2f} | BAKS: {baks:.4f} | BAUS: {baus:.4f} | GEO_MEAN: {geo_mean:.4f}")


import matplotlib.pyplot as plt

thresholds, baks_scores, baus_scores, geo = zip(*results)
plt.plot(thresholds, baks_scores, label='BAKS (Known)')
plt.plot(thresholds, baus_scores, label='BAUS (Unknown)')
plt.plot(thresholds, geo, label='Geometrical Mean')
plt.xlabel('Cosine Threshold')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.show()

