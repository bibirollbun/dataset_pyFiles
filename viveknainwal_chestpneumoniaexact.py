import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore") # Mute warnings
IMG_SIZE = 256       
PATCH_SIZE = 32      
TRAIN_PATIENT_LIMIT = 800   # Number of patients to TRAIN on
TEST_PATIENT_LIMIT = 200    # Number of patients to TEST on (for Accuracy/Confusion Matrix)

# Install pydicom if missing
try:
    import pydicom
except ImportError:
    os.system('pip install pydicom')
    import pydicom

# Paths
labels_path = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'
images_dir = '/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_images'

# =============================================================================
# 2. FEATURE EXTRACTION ENGINE
# =============================================================================

def get_hog_features(patch):
    """Extracts HOG features (Shape/Texture/Edges)."""
    features = hog(patch, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    return features

def process_image_patches(image, boxes=None, is_training=True):
    """Splits image into 32x32 patches and extracts features."""
    patch_data = []
    steps = IMG_SIZE // PATCH_SIZE
    
    for r in range(steps):
        for c in range(steps):
            y_start, y_end = r*PATCH_SIZE, (r+1)*PATCH_SIZE
            x_start, x_end = c*PATCH_SIZE, (c+1)*PATCH_SIZE
            patch = image[y_start:y_end, x_start:x_end]
            
            # Features
            hog_feats = get_hog_features(patch)
            mean_val = np.mean(patch)
            var_val = np.var(patch)
            y_norm, x_norm = r / steps, c / steps
            full_features = np.concatenate([hog_feats, [mean_val, var_val, y_norm, x_norm]])
            
            # Labeling
            label = 0
            if is_training and boxes is not None:
                p_box = [x_start, y_start, PATCH_SIZE, PATCH_SIZE]
                for box in boxes:
                    gt_x, gt_y, gt_w, gt_h = box
                    # Intersection logic
                    x_left = max(p_box[0], gt_x)
                    y_top = max(p_box[1], gt_y)
                    x_right = min(p_box[0]+p_box[2], gt_x+gt_w)
                    y_bottom = min(p_box[1]+p_box[3], gt_y+gt_h)
                    if x_right > x_left and y_bottom > y_top:
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        if intersection > (PATCH_SIZE*PATCH_SIZE) * 0.15:
                            label = 1; break
            
            patch_data.append((full_features, label))
    return patch_data

# =============================================================================
# 3. DATA PREPARATION (TRAIN/TEST SPLIT)
# =============================================================================

print("Mapping data and splitting into Train/Test sets...")
df = pd.read_csv(labels_path)
box_map = {}
for _, row in df.iterrows():
    pid = row['patientId']
    if pid not in box_map: box_map[pid] = []
    if row['Target'] == 1:
        scale = IMG_SIZE / 1024
        box_map[pid].append([int(row['x']*scale), int(row['y']*scale), 
                             int(row['width']*scale), int(row['height']*scale)])

# Separate Healthy and Sick patients
all_pids = list(box_map.keys())
sick_pids = [pid for pid in all_pids if len(box_map[pid]) > 0]
healthy_pids = [pid for pid in all_pids if len(box_map[pid]) == 0]

# Shuffle
random.shuffle(sick_pids)
random.shuffle(healthy_pids)

# Create Training Lists (Mostly Sick to teach model features + some Healthy)
train_pids = sick_pids[:TRAIN_PATIENT_LIMIT] 

# Create Testing Lists (50% Sick, 50% Healthy for fair evaluation)
test_sick = sick_pids[TRAIN_PATIENT_LIMIT : TRAIN_PATIENT_LIMIT + (TEST_PATIENT_LIMIT//2)]
test_healthy = healthy_pids[: (TEST_PATIENT_LIMIT//2)]
test_pids = test_sick + test_healthy
random.shuffle(test_pids)

print(f"Training on {len(train_pids)} patients.")
print(f"Testing on {len(test_pids)} separated patients.")

# =============================================================================
# 4. TRAINING
# =============================================================================

def load_training_data(pids):
    print("Extracting features from Training Set...")
    positive_patches, negative_patches = [], []
    
    for i, pid in enumerate(pids):
        dcm_path = os.path.join(images_dir, pid + '.dcm')
        if not os.path.exists(dcm_path): continue
        ds = pydicom.dcmread(dcm_path)
        img = cv2.resize(ds.pixel_array, (IMG_SIZE, IMG_SIZE))
        
        patches = process_image_patches(img, box_map[pid], is_training=True)
        for feats, label in patches:
            if label == 1: positive_patches.append(feats)
            else: negative_patches.append(feats)
            
    # Balance Data
    random.shuffle(negative_patches)
    balanced_negatives = negative_patches[:len(positive_patches)]
    X = np.vstack(positive_patches + balanced_negatives)
    y = np.concatenate([np.ones(len(positive_patches)), np.zeros(len(balanced_negatives))])
    return X, y

X_train, y_train = load_training_data(train_pids)
print("\nTraining Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
print("Training Complete!")

# =============================================================================
# 5. EVALUATION (ACCURACY & CONFUSION MATRIX)
# =============================================================================

print(f"\nEvaluating Model on {len(test_pids)} unseen patients...")
y_test_true = []
y_test_pred = []

STRICT_THRESHOLD = 0.80 # Confidence threshold

for i, pid in enumerate(test_pids):
    # 1. Ground Truth
    is_actually_sick = 1 if len(box_map[pid]) > 0 else 0
    y_test_true.append(is_actually_sick)
    
    # 2. Prediction
    dcm_path = os.path.join(images_dir, pid + '.dcm')
    ds = pydicom.dcmread(dcm_path)
    img = cv2.resize(ds.pixel_array, (IMG_SIZE, IMG_SIZE))
    
    patches = process_image_patches(img, is_training=False)
    X_test_feats = np.array([p[0] for p in patches])
    probs = rf_model.predict_proba(X_test_feats)[:, 1]
    
    # Patient Diagnosis Logic
    # If ANY patch in the image is > 80% confident, we mark patient as Sick
    steps = IMG_SIZE // PATCH_SIZE
    heatmap = probs.reshape(steps, steps)
    heatmap_smooth = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
    if np.max(heatmap_smooth) > STRICT_THRESHOLD:
        y_test_pred.append(1) # Predicted Sick
    else:
        y_test_pred.append(0) # Predicted Healthy

# --- CALCULATE METRICS ---
accuracy = accuracy_score(y_test_true, y_test_pred)
cm = confusion_matrix(y_test_true, y_test_pred)

print("\n" + "="*40)
print(f"FINAL MODEL ACCURACY: {accuracy*100:.2f}%")
print("="*40)
print("\nClassification Report:")
print(classification_report(y_test_true, y_test_pred, target_names=['Healthy', 'Pneumonia']))

# --- PLOT CONFUSION MATRIX ---
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Pred: Healthy', 'Pred: Pneumonia'],
            yticklabels=['Actual: Healthy', 'Actual: Pneumonia'])
plt.title('Confusion Matrix (Patient Level)')
plt.show()

# =============================================================================
# 6. VISUALIZATION (Clean Boxes)
# =============================================================================

def visualize_prediction(pid):
    dcm_path = os.path.join(images_dir, pid + '.dcm')
    ds = pydicom.dcmread(dcm_path)
    img = cv2.resize(ds.pixel_array, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Prediction
    patches = process_image_patches(img, is_training=False)
    X_feats = np.array([p[0] for p in patches])
    probs = rf_model.predict_proba(X_feats)[:, 1]
    
    heatmap = cv2.resize(probs.reshape(IMG_SIZE//PATCH_SIZE, IMG_SIZE//PATCH_SIZE), 
                         (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
    max_conf = np.max(heatmap)
    
    # Draw Doctor Label (Red)
    if len(box_map[pid]) > 0:
        for box in box_map[pid]:
            x, y, w, h = box
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img_rgb, "Doctor", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

    # Draw AI Prediction (Blue)
    if max_conf > STRICT_THRESHOLD:
        mask = (heatmap > STRICT_THRESHOLD).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 10 and h > 10:
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img_rgb, f"AI {max_conf:.2f}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    status = "PNEUMONIA" if max_conf > STRICT_THRESHOLD else "HEALTHY"
    plt.title(f"AI Prediction: {status} ({max_conf:.2f})")
    plt.axis('off')
    plt.show()

print("\nVisualizing 3 Test Cases...")
for i in range(3):
    visualize_prediction(test_pids[i])

