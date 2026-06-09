import os
import cv2
import numpy as np
import pandas as pd
import glob
import time
from skimage.feature import hog, local_binary_pattern
from skimage.filters import gabor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from tqdm import tqdm


IMG_SIZE = (192, 144)
HOG_PIXELS_PER_CELL = (8, 8)
HOG_ORIENTATIONS = 12
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_BINS = (24, 24)
EDGE_BINS = 16
GABOR_FREQS = [0.1, 0.3]
GABOR_THETAS = [0, np.pi/4, np.pi/2, 3*np.pi/4]
SEED = 42
MAX_FEATURES = 2500

TRAIN_PATH = '/kaggle/input/state-farm-distracted-driver-detection/imgs/train'
TEST_PATH = '/kaggle/input/state-farm-distracted-driver-detection/imgs/test'
CSV_PATH = '/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv'
CACHE_PATH = '/kaggle/working/features.joblib'


print("Classes found in TRAIN_PATH:", os.listdir(TRAIN_PATH))
classes = sorted([c for c in os.listdir(TRAIN_PATH) if c.startswith('c')])
print("Filtered classes:", classes)
print("Number of classes:", len(classes))


def get_vector_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [COLOR_BINS[0]], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], None, [COLOR_BINS[1]], [0, 256])
    color_feat = np.concatenate([hist_h.flatten(), hist_s.flatten()])
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hog_feat = hog(img_gray, 
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys', 
                   transform_sqrt=True,
                   feature_vector=True)

    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    lbp_hist = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)[0]
    
    # concat
    all_features = [hog_feat, color_feat, lbp_hist]
    shapes = [f.shape for f in all_features]
    
    for f in all_features:
        if np.any(np.isnan(f)) or np.any(np.isinf(f)):
            return None
    
    final_features = np.concatenate(all_features)
    
    if len(final_features) == 0 or np.any(np.isnan(final_features)):
        return None
        
    return final_features


found_classes = sorted(os.listdir(TRAIN_PATH))
print(f"Found folders: {found_classes}")

expected_classes = [f'c{i}' for i in range(10)]
missing = [c for c in expected_classes if c not in found_classes]

if missing:
    print(f"Missing folders: {missing}")
else:
    print("All class folders c0-c9 found.")

if not missing:
    X, y, groups = [], [], []
        
    driver_df = pd.read_csv(CSV_PATH)
    driver_dict = dict(zip(driver_df['img'], driver_df['subject']))
    
    start_time = time.time()
    
    for idx, class_name in enumerate(expected_classes):
        class_dir = os.path.join(TRAIN_PATH, class_name)
        img_files = glob.glob(os.path.join(class_dir, '*.jpg'))
        
        print(f"Processing {class_name}: {len(img_files)} images found.")
        
        if len(img_files) == 0:
            print(f"{class_name} is empty, skipping.")
            continue
    
        for img_path in tqdm(img_files):
            features = get_vector_from_image(img_path)
            
            if features is not None:
                X.append(features)
                y.append(idx)
                
                fname = os.path.basename(img_path)
                groups.append(driver_dict.get(fname, 'unknown'))
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    groups = np.array(groups)
    
    print(f"\nDONE! Final Shape: {X.shape}")



import joblib
import numpy as np

CACHE_PATH = '/kaggle/working/features.joblib'

if len(X) > 0:
    joblib.dump((X, y, groups), CACHE_PATH)
    print(f"Features saved to {CACHE_PATH}")
    print(f"Saved shapes: X={X.shape}, y={y.shape}, groups={groups.shape}")
else:
    print("Error: X is empty.")


import joblib
import os

CACHE_PATH = '/kaggle/working/features.joblib'

if os.path.exists(CACHE_PATH):
    X, y, groups = joblib.load(CACHE_PATH)
    print(f"Loaded features from {CACHE_PATH}")
    print(f"Loaded shapes: X={X.shape}, y={y.shape}, groups={groups.shape}")
else:
    print(f"File {CACHE_PATH} not found. You must run feature extraction first.")


print(f"Original feature dimension: {X.shape[1]}")

# Analyze feature importance before PCA
print('select kbest')
feature_selector = SelectKBest(mutual_info_classif, k=min(1500, X.shape[1]))
X_selected = feature_selector.fit_transform(X, y)
print(f"Selected feature dimension: {X_selected.shape[1]}")

print('scaler')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

print('pca')
if X_selected.shape[1] > 1000:
    pca = PCA(n_components=0.98, random_state=SEED)
    X_final = pca.fit_transform(X_scaled)
    print(f"Final dimension after PCA: {X_final.shape[1]}")
else:
    X_final = X_scaled


from sklearn.ensemble import BaggingClassifier

# Base SVM with better parameters (from grid search suggestion)
base_svm = SVC(kernel='rbf', C=30, gamma='scale', class_weight='balanced', 
               probability=False, random_state=SEED)

# Bagging for robustness (faster than full ensemble)
# clf = BaggingClassifier(
#     base_estimator=base_svm,
#     n_estimators=5,
#     max_samples=0.8,
#     bootstrap=False,  # Use actual data for diversity
#     random_state=SEED,
#     n_jobs=-1
# )

from sklearn.linear_model import SGDClassifier
clf = VotingClassifier([
    ('svm_rbf', SVC(kernel='rbf', C=20, gamma='scale', class_weight='balanced', probability=True)),
    ('svm_linear', SGDClassifier(loss='hinge', class_weight='balanced', max_iter=1000)),
    ('rf', ExtraTreesClassifier(n_estimators=100, max_depth=20, class_weight='balanced'))
], voting='hard', n_jobs=-1)



sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_accuracies = []

print("\n5-Fold Stratified Group Cross-Validation")

fold = 1
for train_idx, val_idx in sgkf.split(X_final, y, groups):
    print(f"\nFOLD {fold}")
    
    X_train, X_val = X_final[train_idx], X_final[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # augmentation: horizontal flip for left/right classes
    # Classes 1,2 (right) and 3,4 (left)
    flip_classes = [1, 2, 3, 4]
    X_aug, y_aug = [], []
    for cls in flip_classes:
        cls_mask = y_train == cls
        if cls_mask.sum() > 0:
            pass
    
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = (time.time() - t0) / 60
    
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    fold_accuracies.append(val_acc)
    
    print(f"Fold Accuracy: {val_acc*100:.2f}% (Train Time: {train_time:.1f} min)")
    fold += 1

print(f"\nMean Accuracy: {np.mean(fold_accuracies)*100:.2f}% (+/- {np.std(fold_accuracies)*100:.2f})")


from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, val_idx = next(gss.split(X_final, y, groups))

X_train_vis, X_val_vis = X_final[train_idx], X_final[val_idx]
y_train_vis, y_val_vis = y[train_idx], y[val_idx]

print("Training Final Model")
clf.fit(X_train_vis, y_train_vis)

y_pred_vis = clf.predict(X_val_vis)
acc = accuracy_score(y_val_vis, y_pred_vis)
print(f"\nValidation Accuracy: {acc*100:.2f}%")

class_names = [
    "c0: Safe", "c1: Text Right", "c2: Talk Right", "c3: Text Left", 
    "c4: Talk Left", "c5: Radio", "c6: Drink", "c7: Reach Behind", 
    "c8: Hair/Makeup", "c9: Talk Pass"
]

print("\nClassification Report")
print(classification_report(y_val_vis, y_pred_vis, target_names=class_names))

# Plot with normalized colors
cm = confusion_matrix(y_val_vis, y_pred_vis)
cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.title('Normalized Confusion Matrix')
plt.show()


import joblib

model_bundle = {
    'model': clf,
    'scaler': scaler,
    'feature_selector': feature_selector,
    'pca': pca if 'pca' in locals() else None,  # Save PCA only if used
    'class_names': class_names
}

output_path = 'driver_distraction_pipeline.joblib'
joblib.dump(model_bundle, output_path)

print(f"Artifacts exported to: {output_path}")

# Load artifacts
# artifacts = joblib.load('driver_distraction_pipeline.joblib')

# # Restore objects
# loaded_model = artifacts['model']
# loaded_scaler = artifacts['scaler']
# loaded_selector = artifacts['feature_selector']

# # Inference pipeline (example on new data 'X_new')
# # 1. Select features
# X_selected = loaded_selector.transform(X_new)
# # 2. Scale
# X_scaled = loaded_scaler.transform(X_selected)
# # 3. PCA (if applicable)
# if artifacts['pca']:
#     X_scaled = artifacts['pca'].transform(X_scaled)
# # 4. Predict
# prediction = loaded_model.predict(X_scaled)

