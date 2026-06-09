# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os

# List root input directory
base_path = "/kaggle/input/sperm-morphological-quality"
for root, dirs, files in os.walk(base_path):
    print("ğŸ“� Directory:", root)
    for name in files[:3]:  # show only first 3 files in each folder
        print("    ğŸ“„", name)


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths to datasets (adjust these to match your dataset folder names)
sperm_quality_path = "/kaggle/input/sperm-morphological-quality"
evisan_path = "/kaggle/input/evisan-multi-sperm-detection-and-tracking"
visem_path = "/kaggle/input/visem-video-dataset"
fertile_man_path = "/kaggle/input/fertile-man-semen-parameters-2020-who"

# âœ… Update these paths based on your dataset structure
good_path = "/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm"
bad_path = "/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm"


# Preview sample sperm images
def show_sample_images(path, title):
    files = os.listdir(path)[:5]
    plt.figure(figsize=(15, 5))
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"{title} #{i+1}")
        plt.axis('off')
    plt.show()

show_sample_images(good_path, "Good Sperm")
show_sample_images(bad_path, "Bad Sperm")



# VISem dataset sample
videos_df = pd.read_csv(os.path.join(visem_path, "videos.csv"))
semen_df = pd.read_csv(os.path.join(visem_path, "semen_analysis_data.csv"))

print("VISem videos info:")
print(videos_df.head())

print("\nSemen Analysis Data:")
print(semen_df[['video_id', 'sperm_concentration', 'motility_total']].head())

# Fertile Man dataset
fertile_df = pd.read_csv(os.path.join(fertile_man_path, "Fertile_Man_2020.csv"))
print("\nFertile Man dataset info:")
print(fertile_df[['SpermCount', 'Motility', 'Morphology']].head())


import os

visem_path = "/kaggle/input/visem-video-dataset"

for root, dirs, files in os.walk(visem_path):
    print("ğŸ“� Directory:", root)
    for name in files[:3]:  # Show only first 3 files in each folder
        print("    ğŸ“„", name)


print(semen_df.columns)


import pandas as pd
import os

visem_path = "/kaggle/input/visem-video-dataset/visem-dataset"

# Load semen analysis CSV
# Read with the correct separator
semen_df = pd.read_csv(os.path.join(visem_path, "semen_analysis_data.csv"), sep=';')

# Show column names after fixing
print(semen_df.columns)

# Display some useful data
print(semen_df[['ID', 'Sperm concentration (x10â�¶/mL)', 'Progressive motility (%)']].head())



# Load Fertile Man dataset
fertile_man_path = "/kaggle/input/visem-video-dataset/visem-dataset/Fertile_Man_2020.csv"
fertile_df = pd.read_csv(fertile_man_path)

# Show the first few rows of the dataset
print(fertile_df.head())

# Display the column names
print(fertile_df.columns)


import os

# Path to the directory where the Fertile_Man_2020.csv file is supposed to be
fertile_man_dir = "/kaggle/input/visem-video-dataset/visem-dataset"

# List all files in the directory to find the correct file name
files = os.listdir(fertile_man_dir)
print(files)



import pandas as pd

# Load the semen_analysis_data.csv file
semen_analysis_data_path = "/kaggle/input/visem-video-dataset/visem-dataset/semen_analysis_data.csv"
semen_df = pd.read_csv(semen_analysis_data_path)

# Show the first few rows to inspect the data
print("Semen Analysis Data:")
print(semen_df.head())

# Check for any missing values
print("\nMissing values in the dataset:")
print(semen_df.isnull().sum())


# Load the semen_analysis_data.csv with proper delimiter
semen_df = pd.read_csv(semen_analysis_data_path, delimiter=';')

# Display the first few rows
print("Semen Analysis Data (corrected):")
print(semen_df.head())

# Check the column names and missing values
print("\nColumns:")
print(semen_df.columns)

print("\nMissing values:")
print(semen_df.isnull().sum())


# Convert comma decimal numbers to proper float
for col in semen_df.columns:
    if semen_df[col].dtype == 'object':
        semen_df[col] = semen_df[col].str.replace(',', '.', regex=False)
        try:
            semen_df[col] = semen_df[col].astype(float)
        except ValueError:
            pass  # Skip columns that still contain non-numeric text

# Preview cleaned dataset
print("Cleaned Semen Analysis Data:")
print(semen_df.head())

# Check datatypes
print("\nData types:")
print(semen_df.dtypes)


import seaborn as sns
import matplotlib.pyplot as plt

# 1. Basic statistics
print("Summary Statistics:\n")
print(semen_df.describe())

# 2. Correlation matrix
plt.figure(figsize=(14,10))
sns.heatmap(semen_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Semen Parameters")
plt.show()

# 3. Distribution plots for key variables
key_columns = ['Sperm concentration (x10â�¶/mL)', 
               'Total sperm count (x10â�¶)', 
               'Progressive motility (%)', 
               'Normal spermatozoa (%)', 
               'Teratozoospermia index']

for col in key_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(semen_df[col], kde=True, bins=20, color="skyblue")
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Boxplot: progressive motility
plt.figure(figsize=(6,4))
sns.boxplot(y=semen_df['Progressive motility (%)'])
plt.title("Boxplot of Progressive Motility")
plt.show()

# Scatter: Progressive motility vs. DFI
plt.figure(figsize=(6,4))
sns.scatterplot(x=semen_df['DNA fragmentation index, DFI (%)'],
                y=semen_df['Progressive motility (%)'])
plt.title("Motility vs DNA Fragmentation Index")
plt.xlabel("DNA Fragmentation Index (%)")
plt.ylabel("Progressive Motility (%)")
plt.grid(True)
plt.show()

# Scatter: Normal sperm vs Teratozoospermia Index
plt.figure(figsize=(6,4))
sns.scatterplot(x=semen_df['Teratozoospermia index'],
                y=semen_df['Normal spermatozoa (%)'])
plt.title("Normal Sperm vs Teratozoospermia Index")
plt.xlabel("Teratozoospermia Index")
plt.ylabel("Normal Spermatozoa (%)")
plt.grid(True)
plt.show()



# If your DataFrame is called semen_df
semen_df['Label'] = semen_df['Progressive motility (%)'].apply(lambda x: 1 if x >= 32 else 0)

# Check the distribution
print(semen_df['Label'].value_counts())


# Drop ID and Label from features
X = semen_df.drop(['ID', 'Label'], axis=1)
y = semen_df['Label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



import matplotlib.pyplot as plt
import numpy as np

# Get feature importances from RandomForest model
feature_importances = model.feature_importances_

# Sort features by importance
indices = np.argsort(feature_importances)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(X.columns[indices], feature_importances[indices])
plt.xlabel('Importance')
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Split the data with 30% test size
X = df.drop(['Label', 'ID'], axis=1)  # Drop the label and ID columns
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test accuracy
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")



import pandas as pd

# Load your dataset (make sure the file path is in quotes)
df = pd.read_csv('/kaggle/input/visem-video-dataset/visem-dataset/sex_hormones.csv')  # Correct path format

# Clean the dataset if needed (this assumes your dataset is already cleaned)
df['Label'] = df['Progressive motility (%)'].apply(lambda x: 1 if x >= 32 else 0)

# Drop any unnecessary columns or handle missing data if required
df = df.dropna()  # Ensure no missing values remain
df = df.reset_index(drop=True)

# Split the data with 30% test size
X = df.drop(['Label', 'ID'], axis=1)  # Drop the label and ID columns
y = df['Label']

# Continue with splitting and model training
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test accuracy
test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")



import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance from the trained model
importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()


from sklearn.model_selection import GridSearchCV

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters from grid search
print(f"Best Parameters: {grid_search.best_params_}")

# Get the best model
best_model = grid_search.best_estimator_



from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5)  # Use k=5 folds

# Print cross-validation results
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean():.4f}")



# Initialize the Random Forest model with class weights
model_with_weights = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model_with_weights.fit(X_train, y_train)

# Test accuracy
test_accuracy = model_with_weights.score(X_test, y_test)
print(f"Test Accuracy with class weights: {test_accuracy:.4f}")



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Step 2: Create a RandomForest model
rf = RandomForestClassifier(random_state=42)

# Step 3: Apply GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Step 4: Get the best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Step 5: Evaluate the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Step 6: Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Test accuracy
test_accuracy = best_rf.score(X_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")



from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Step 1: Check training accuracy
train_accuracy = best_rf.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# Step 2: Compare with testing accuracy (already printed earlier)
test_accuracy = best_rf.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 3: Plot the ROC curve to evaluate model performance
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # Get the probability for the positive class
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Step 4: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)


# Predict on the test set using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# Print evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()



# Get the best model
best_rf = grid_search.best_estimator_

# Predict on test data
y_pred = best_rf.predict(X_test)

# Evaluate
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



import joblib

# Save the model
joblib.dump(best_rf, 'best_random_forest_model.pkl')



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Evaluation metrics
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("=== Accuracy ===")
print("Accuracy:", accuracy_score(y_test, y_pred))



from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Visual confusion matrix
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()



!pip install ultralytics --quiet


from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# Load the trained model
model = YOLO('yolov8s.pt')  # Use pre-trained YOLOv8 small model

# Load the video
video_path = '/kaggle/input/visem-video-dataset/visem-dataset/videos/11_09.01.23_JMA.avi'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sperm_output.mp4', fourcc, fps, (width, height))

# Tracker ID setup (simple tracker using centroids)
trackers = {}
max_lost = 5  # number of frames to keep 'lost' objects
next_object_id = 0
track_memory = {}

def iou(boxA, boxB):
    # Intersection over Union (IoU)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    new_trackers = {}
    current_centroids = []

    # Loop through detections
    for i, det in enumerate(results.boxes.data):
        x1, y1, x2, y2, score, class_id = det.tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        current_centroids.append(((x1, y1, x2, y2), (cx, cy)))

    used_ids = set()
    for bbox, centroid in current_centroids:
        best_match_id = None
        best_iou = 0.0

        for obj_id, (old_bbox, _) in trackers.items():
            iou_score = iou(bbox, old_bbox)
            if iou_score > best_iou and obj_id not in used_ids:
                best_iou = iou_score
                best_match_id = obj_id

        if best_iou > 0.3:
            new_trackers[best_match_id] = (bbox, centroid)
            used_ids.add(best_match_id)
        else:
            new_trackers[next_object_id] = (bbox, centroid)
            next_object_id += 1

    trackers = new_trackers

    # Draw detections
    for obj_id, (bbox, centroid) in trackers.items():
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = centroid
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Save history for tracking
        if obj_id not in track_memory:
            track_memory[obj_id] = deque(maxlen=30)
        track_memory[obj_id].append((cx, cy))

        # Draw trajectory
        for i in range(1, len(track_memory[obj_id])):
            cv2.line(frame, track_memory[obj_id][i - 1], track_memory[obj_id][i], (0, 0, 255), 2)

    # Display count
    cv2.putText(frame, f"Sperm Count: {len(trackers)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Save to output video
    out.write(frame)

# Cleanup
cap.release()
out.release()
print("âœ… Output video saved as sperm_output.mp4")


pip install ultralytics


from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # Use pre-trained model for transfer learning
model.train(data="path_to_your_data.yaml", epochs=50)  # Train with your labeled data


train: /kaggle/input/evisan-multi-sperm-detection-and-tracking/images  # Path to the training images folder
val: /kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm  # Path to the validation images folder
nc: 1  # Number of classes (for sperm, this is 1 class)
names: ['sperm']  # Class name(s)



import os

# Define base directory for new dataset
base_dir = '/kaggle/working/sperm_dataset'
train_dir = os.path.join(base_dir, 'train/images')
val_dir = os.path.join(base_dir, 'val/images')

# Make the directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("Folder structure created âœ…")



import shutil
import random
from glob import glob

# Source folder
source_folder = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'

# Get all .png image paths
all_images = glob(os.path.join(source_folder, '*.png'))

# Shuffle and split
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Copy to new folders
for img_path in train_images:
    shutil.copy(img_path, os.path.join(train_dir, os.path.basename(img_path)))

for img_path in val_images:
    shutil.copy(img_path, os.path.join(val_dir, os.path.basename(img_path)))

print(f"Copied {len(train_images)} training images and {len(val_images)} validation images âœ…")



import os
import shutil
import random
from pathlib import Path

# Set your source directories for Good and Bad sperm images
source_good = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'
source_bad = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm'

# Define the base directory for the organized dataset
base_dir = '/kaggle/working/sperm_morphology_dataset'

# Define train/val split ratio
train_ratio = 0.8

# Create the folder structure
for split in ['train', 'val']:
    for cls in ['Good', 'Bad']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Helper function to split and copy files
def split_and_copy(source_folder, label):
    files = list(Path(source_folder).glob('*.png'))
    random.shuffle(files)
    
    train_count = int(len(files) * train_ratio)
    train_files = files[:train_count]
    val_files = files[train_count:]

    for f in train_files:
        shutil.copy(f, os.path.join(base_dir, 'train', label))
    for f in val_files:
        shutil.copy(f, os.path.join(base_dir, 'val', label))

# Apply to both classes
split_and_copy(source_good, 'Good')
split_and_copy(source_bad, 'Bad')

print("âœ… Dataset prepared successfully!")



from ultralytics import YOLO

# Load a pre-trained classification model (you can use yolov8n-cls.pt or yolov8s-cls.pt for small models)
model = YOLO("yolov8s-cls.pt")  # pre-trained on ImageNet

# Train the model on your sperm morphology dataset
model.train(data="/kaggle/working/sperm_morphology_dataset", epochs=100, imgsz=224)


results = model.predict(source="/kaggle/working/sperm_morphology_dataset/val/Good", imgsz=224)


%matplotlib inline


!pip install ultralytics


from ultralytics import YOLO
model = YOLO("runs/classify/train/weights/best.pt")  # Load your trained model


from ultralytics import YOLO

# Load a pre-trained model and start training again
model = YOLO("yolov8s-cls.pt")
model.train(data="/kaggle/working/sperm_morphology_dataset", epochs=100, imgsz=224)



import os

# List all folders and files under /kaggle/working
for root, dirs, files in os.walk("/kaggle/working"):
    print(f"\nğŸ“‚ {root}")
    for name in dirs:
        print(f"  ğŸ“� {name}")
    for name in files:
        print(f"  ğŸ“„ {name}")


model.train(data="/kaggle/working/sperm_classifier", epochs=100, imgsz=224)


import os

for root, dirs, files in os.walk("/kaggle/working"):
    print(f"\nğŸ“‚ {root}")
    for d in dirs:
        print(f"  ğŸ“� {d}")
    for f in files:
        print(f"  ğŸ“„ {f}")


import os
import shutil
import pandas as pd

# Path to the CSV file that contains the labels
csv_path = "/kaggle/input/sperm-morphology-labels/labels.csv"

# Path where the images are currently located
images_dir = "/kaggle/input/sperm-morphology-images"

# Path to output classified dataset
output_dir = "/kaggle/working/sperm_morphology_dataset/train"

# Create folders for 'good' and 'bad' if not exist
os.makedirs(os.path.join(output_dir, "good"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "bad"), exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Assuming CSV has columns: 'image_name' and 'label'
for _, row in df.iterrows():
    image_name = row['image_name']
    label = row['label'].lower()  # should be "good" or "bad"

    src = os.path.join(images_dir, image_name)
    dst = os.path.join(output_dir, label, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("âœ… Images sorted into folders successfully!")



from ultralytics import YOLO

# Load a classification model (you can use yolov8s-cls.pt or another size)
model = YOLO("yolov8s-cls.pt")

# Train the model
model.train(
    data="/kaggle/working/sperm_morphology_dataset",  # folder containing train/ and val/
    epochs=100,
    imgsz=224
)


import os

for root, dirs, files in os.walk("/kaggle/working/sperm_morphology_dataset"):
    print(f"\nğŸ“‚ {root}")
    for d in dirs:
        print(f"  ğŸ“� {d}")
    for f in files:
        print(f"  ğŸ“„ {f}")


import os

# Define base directory
base_dir = '/kaggle/working/sperm_morphology_dataset'

# Create folders: train/Good, train/Bad, val/Good, val/Bad
for split in ['train', 'val']:
    for cls in ['Good', 'Bad']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

print("âœ… Folder structure created successfully!")


import shutil
import random
from pathlib import Path

# Source directories for Good and Bad sperm
source_good = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'
source_bad = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm'

# Train/val split ratio
train_ratio = 0.8

# Helper function to copy and split images
def split_and_copy(source_folder, label):
    files = list(Path(source_folder).glob('*.png'))
    random.shuffle(files)
    
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(f, os.path.join(base_dir, 'train', label))
    for f in val_files:
        shutil.copy(f, os.path.join(base_dir, 'val', label))

# Apply to both classes
split_and_copy(source_good, 'Good')
split_and_copy(source_bad, 'Bad')

print("âœ… Dataset re-created and labeled successfully!")


import os

for root, dirs, files in os.walk(base_dir):
    print(f"\nğŸ“‚ {root}")
    for d in dirs:
        print(f"  ğŸ“� {d}")
    for f in files[:3]:  # just show first 3 images for brevity
        print(f"  ğŸ“„ {f}")


from ultralytics import YOLO

model = YOLO("yolov8s-cls.pt")

model.train(
    data=base_dir,
    epochs=100,
    imgsz=224
)


import matplotlib.pyplot as plt
import yaml

# Path to your training run directory (update if you're using train2, train3...)
log_dir = '/kaggle/working/runs/classify/train'

# Load training results
with open(f'{log_dir}/results.yaml', 'r') as f:
    results = yaml.safe_load(f)

# Plot
epochs = list(range(1, len(results['metrics/accuracy_top1']) + 1))
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, results['metrics/accuracy_top1'], label='Top-1 Accuracy')
plt.plot(epochs, results['metrics/accuracy_top5'], label='Top-5 Accuracy')
plt.title('ğŸ“ˆ Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, results['train/loss'], label='Train Loss')
plt.plot(epochs, results['val/loss'], label='Val Loss')
plt.title('ğŸ“‰ Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


import os

base_path = "/kaggle/working/runs/classify"
for folder in os.listdir(base_path):
    path = os.path.join(base_path, folder)
    if os.path.isdir(path):
        print("ğŸ“‚ Found training folder:", path)



import os

for i in range(1, 6):
    folder = f"train{i if i > 1 else ''}"
    path = f"/kaggle/working/runs/classify/{folder}"
    files = os.listdir(path)
    print(f"ğŸ“‚ {folder} contains: {files}")


import pandas as pd
import matplotlib.pyplot as plt

# Load training results
log_dir = '/kaggle/working/runs/classify/train5'  # or train4
df = pd.read_csv(f'{log_dir}/results.csv')

# Plot
epochs = range(1, len(df) + 1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, df['      metrics/accuracy_top1'], label='Top-1 Accuracy')
plt.plot(epochs, df['      metrics/accuracy_top5'], label='Top-5 Accuracy')
plt.title('ğŸ“ˆ Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, df['      train/loss'], label='Train Loss')
plt.plot(epochs, df['      val/loss'], label='Val Loss')
plt.title('ğŸ“‰ Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load training results
log_dir = '/kaggle/working/runs/classify/train5'  # change to train4 if needed
df = pd.read_csv(f'{log_dir}/results.csv')

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Show cleaned column names (optional)
print("ğŸ“‹ Cleaned Columns:", df.columns.tolist())

# Plot
epochs = range(1, len(df) + 1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, df['metrics/accuracy_top1'], label='Top-1 Accuracy')
plt.plot(epochs, df['metrics/accuracy_top5'], label='Top-5 Accuracy')
plt.title('ğŸ“ˆ Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, df['train/loss'], label='Train Loss')
plt.plot(epochs, df['val/loss'], label='Val Loss')
plt.title('ğŸ“‰ Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


from ultralytics import YOLO

# Enable augmentation by setting appropriate args
model = YOLO("yolov8s-cls.pt")
model.train(
    data="/kaggle/working/sperm_morphology_dataset", 
    epochs=50,
    imgsz=224,
    batch=16,
    augment=True  # ğŸª„ Enable augmentation
)


import pandas as pd
import matplotlib.pyplot as plt

# Path to your results.csv (adjust if needed)
csv_path = "/kaggle/working/runs/classify/train5/results.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Plotting loss
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['train/loss'], label='Train Loss', color='blue')
plt.plot(df['epoch'], df['val/loss'], label='Validation Loss', color='red')
plt.title("ğŸ“‰ Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import os

# Set base dataset path
base_path = "/kaggle/working/sperm_morphology_dataset"

# Initialize counters
count_summary = {}

for split in ["train", "val"]:
    split_path = os.path.join(base_path, split)
    count_summary[split] = {}
    
    for label in ["Good", "Bad"]:
        label_path = os.path.join(split_path, label)
        num_images = len([f for f in os.listdir(label_path) if f.endswith(".png")])
        count_summary[split][label] = num_images

# Display counts
for split in count_summary:
    print(f"\nğŸ“� {split.upper()} SET:")
    for label in count_summary[split]:
        print(f"  ğŸ”¹ {label}: {count_summary[split][label]} images")



import os

base_path = "/kaggle/working"

# List all folders and files under base_path
for root, dirs, files in os.walk(base_path):
    print(f"\nğŸ“� Directory: {root}")
    for d in dirs:
        print(f"   ğŸ“‚ Subfolder: {d}")
    for f in files:
        print(f"   ğŸ“„ File: {f}")



import os
import shutil
import random
from pathlib import Path

# Set your source directories for Good and Bad sperm images
source_good = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'
source_bad = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm'

# Define the base directory for the organized dataset
base_dir = '/kaggle/working/sperm_morphology_dataset'

# Define train/val split ratio
train_ratio = 0.8

# Create the folder structure
for split in ['train', 'val']:
    for cls in ['Good', 'Bad']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Helper function to split and copy files
def split_and_copy(source_folder, label):
    files = list(Path(source_folder).glob('*.png'))
    random.shuffle(files)

    train_count = int(len(files) * train_ratio)
    train_files = files[:train_count]
    val_files = files[train_count:]

    for f in train_files:
        shutil.copy(f, os.path.join(base_dir, 'train', label))
    for f in val_files:
        shutil.copy(f, os.path.join(base_dir, 'val', label))

# Apply to both classes
split_and_copy(source_good, 'Good')
split_and_copy(source_bad, 'Bad')

print("âœ… Dataset prepared successfully!")



# Count the number of images per category
count_summary = {}

for split in ["train", "val"]:
    split_path = os.path.join(base_dir, split)
    count_summary[split] = {}

    for label in ["Good", "Bad"]:
        label_path = os.path.join(split_path, label)
        num_images = len([f for f in os.listdir(label_path) if f.endswith(".png")])
        count_summary[split][label] = num_images

# Display results
for split in count_summary:
    print(f"\nğŸ“� {split.upper()} SET:")
    for label in count_summary[split]:
        print(f"  ğŸ”¹ {label}: {count_summary[split][label]} images")



import os
import shutil
import random
from pathlib import Path

# Paths
base_dir = '/kaggle/working/sperm_morphology_balanced_dataset'
source_dir = '/kaggle/working/sperm_morphology_dataset/train'
target_dir = os.path.join(base_dir, 'train')

# Make folders
for cls in ['Good', 'Bad']:
    os.makedirs(os.path.join(target_dir, cls), exist_ok=True)

# Get all Good and Bad image paths
good_images = list(Path(os.path.join(source_dir, 'Good')).glob('*.png'))
bad_images = list(Path(os.path.join(source_dir, 'Bad')).glob('*.png'))

# How many to match?
target_count = len(bad_images)

# Oversample Good images by duplication
oversampled_good = good_images * (target_count // len(good_images)) + random.choices(good_images, k=target_count % len(good_images))

# Shuffle
random.shuffe(oversampled_good)

# Copy images to new balanced dataset
for img in oversampled_good:
    shutil.copy(img, os.path.join(target_dir, 'Good'))

for img in bad_images:
    shutil.copy(img, os.path.join(target_dir, 'Bad'))

print(f"âœ… Balanced dataset created with {len(oversampled_good)} Good and {len(bad_images)} Bad images.")


from pathlib import Path

good_path = Path("/kaggle/working/sperm_morphology_dataset/train/Good")
bad_path = Path("/kaggle/working/sperm_morphology_dataset/train/Bad")

print("ğŸ”� Good images found:", len(list(good_path.glob("*.png"))))
print("ğŸ”� Bad images found:", len(list(bad_path.glob("*.png"))))


import os
import shutil
import random
from pathlib import Path

# Original labeled dataset paths
source_good = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'
source_bad = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm'

# New destination
base_dir = '/kaggle/working/sperm_morphology_dataset'

# Split ratio
train_ratio = 0.8

# Create folder structure
for split in ['train', 'val']:
    for cls in ['Good', 'Bad']:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# Function to split and copy
def split_and_copy(source_folder, label):
    files = list(Path(source_folder).glob('*.png'))
    random.shuffle(files)

    train_count = int(len(files) * train_ratio)
    train_files = files[:train_count]
    val_files = files[train_count:]

    for f in train_files:
        shutil.copy(f, os.path.join(base_dir, 'train', label))
    for f in val_files:
        shutil.copy(f, os.path.join(base_dir, 'val', label))

# Do it for both Good and Bad
split_and_copy(source_good, 'Good')
split_and_copy(source_bad, 'Bad')

print("âœ… Recreated classified dataset with train/val split successfully!")



import os

# Set base dataset path
base_path = "/kaggle/working/sperm_morphology_dataset"

# Initialize counters
count_summary = {}

for split in ["train", "val"]:
    split_path = os.path.join(base_path, split)
    count_summary[split] = {}
    
    for label in ["Good", "Bad"]:
        label_path = os.path.join(split_path, label)
        num_images = len([f for f in os.listdir(label_path) if f.endswith(".png")])
        count_summary[split][label] = num_images

# Display counts
for split in count_summary:
    print(f"\nğŸ“� {split.upper()} SET:")
    for label in count_summary[split]:
        print(f"  ğŸ”¹ {label}: {count_summary[split][label]} images")



import os
import shutil
import random
from pathlib import Path

# Original dataset paths
source_good = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm'
source_bad = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Bad Sperm'

# Base path for balanced dataset
base_dir = '/kaggle/working/sperm_morphology_balanced_dataset'

# Split ratio
train_ratio = 0.8

# Count good sperm images
good_images = list(Path(source_good).glob("*.png"))
num_good = len(good_images)

# Randomly sample bad sperm images to match the count of good ones
bad_images = list(Path(source_bad).glob("*.png"))
sampled_bad = random.sample(bad_images, num_good)

# Split function
def split_and_copy(images, label):
    random.shuffle(images)
    train_cutoff = int(len(images) * train_ratio)
    train_images = images[:train_cutoff]
    val_images = images[train_cutoff:]

    for img in train_images:
        dst = os.path.join(base_dir, 'train', label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(img, dst)

    for img in val_images:
        dst = os.path.join(base_dir, 'val', label)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(img, dst)

# Clear/create directory structure and split/copy
split_and_copy(good_images, 'Good')
split_and_copy(sampled_bad, 'Bad')

print("âœ… Dataset is now balanced and ready!")



import os

# Set base path of the balanced dataset
base_path = "/kaggle/working/sperm_morphology_balanced_dataset"

# Initialize counters
count_summary = {}

for split in ["train", "val"]:
    split_path = os.path.join(base_path, split)
    count_summary[split] = {}
    
    for label in ["Good", "Bad"]:
        label_path = os.path.join(split_path, label)
        if os.path.exists(label_path):
            num_images = len([f for f in os.listdir(label_path) if f.endswith(".png")])
        else:
            num_images = 0
        count_summary[split][label] = num_images

# Display counts
for split in count_summary:
    print(f"\nğŸ“� {split.upper()} SET:")
    for label in count_summary[split]:
        print(f"  ğŸ”¹ {label}: {count_summary[split][label]} images")



!pip install ultralytics


from ultralytics import YOLO

# Load classification model
model = YOLO('yolov8n-cls.pt')

# Train
model.train(
    data="/kaggle/working/sperm_morphology_balanced_dataset",
    epochs=30,
    imgsz=224,
    batch=32,
    name="train_balanced"
)



import pandas as pd
import matplotlib.pyplot as plt

# Path to your training results
results_path = "/kaggle/working/runs/classify/train_balanced/results.csv"

# Load CSV
df = pd.read_csv(results_path)

# Plotting loss
plt.figure(figsize=(8, 5))
plt.plot(df['      train/loss'], label='Train Loss', linewidth=2)
plt.plot(df['         val/loss'], label='Validation Loss', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ğŸ“‰ Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
results_path = "/kaggle/working/runs/classify/train_balanced/results.csv"
df = pd.read_csv(results_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print column names to confirm
print("ğŸ§¾ Available columns:", df.columns.tolist())

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(df['train/loss'], label='Train Loss', linewidth=2)
plt.plot(df['val/loss'], label='Validation Loss', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ğŸ“‰ Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
results_path = "/kaggle/working/runs/classify/train_balanced/results.csv"
df = pd.read_csv(results_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Print column names to confirm
print("ğŸ§¾ Available columns:", df.columns.tolist())

# Find the epoch with the minimum validation loss
best_epoch = df['val/loss'].idxmin()
best_val_loss = df.loc[best_epoch, 'val/loss']
best_train_loss = df.loc[best_epoch, 'train/loss']

print(f"ğŸ“� Best Epoch: {best_epoch}")
print(f"ğŸ�† Best Validation Loss: {best_val_loss}")
print(f"ğŸ“‰ Train Loss at Best Epoch: {best_train_loss}")

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(df['train/loss'], label='Train Loss', linewidth=2)
plt.plot(df['val/loss'], label='Validation Loss', linewidth=2)
plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch: {best_epoch}')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ğŸ“‰ Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import os

# Update this to your original dataset path (before balancing)
original_dataset_path = "/kaggle/working/original_sperm_morphology_dataset"

# Initialize counters
count_summary = {}

for split in ["train", "val"]:
    split_path = os.path.join(original_dataset_path, split)
    count_summary[split] = {}
    
    for label in ["Good", "Bad"]:
        label_path = os.path.join(split_path, label)
        if os.path.exists(label_path):
            num_images = len([f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            count_summary[split][label] = num_images
        else:
            count_summary[split][label] = 0

# Print result
for split in count_summary:
    print(f"\nğŸ“� {split.upper()} SET:")
    for label in count_summary[split]:
        print(f"  ğŸ”¹ {label}: {count_summary[split][label]} images")



import os

# Path to the original dataset
base_path = "/kaggle/working/sperm_morphology_dataset"

# Initialize dictionary to hold image counts
count_summary = {}

for split in ["train", "val"]:
    split_path = os.path.join(base_path, split)
    count_summary[split] = {}
    
    for label in ["Good", "Bad"]:
        label_path = os.path.join(split_path, label)
        if os.path.exists(label_path):
            num_images = len([f for f in os.listdir(label_path) if f.endswith(".png")])
            count_summary[split][label] = num_images
        else:
            count_summary[split][label] = 0

# Display the results
for split in count_summary:
    print(f"\nğŸ“� {split.upper()} SET:")
    for label in count_summary[split]:
        print(f"  ğŸ”¹ {label}: {count_summary[split][label]} images")



import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Define source paths
good_src = "/kaggle/input/good-morphology"
bad_src_1 = "/kaggle/input/bad-morphology"
bad_src_2 = "/kaggle/input/not-sperm"

# Define destination root
base_dest = "/kaggle/working/sperm_morphology_dataset"

# Define target folders
targets = {
    'Good': [good_src],
    'Bad': [bad_src_1, bad_src_2]
}

# Function to copy and split files into train/val
def copy_and_split(source_paths, dest_label):
    all_images = []
    for src in source_paths:
        files = [os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_images.extend(files)

    print(f"ğŸ“¦ Found {len(all_images)} new '{dest_label}' images to add.")
    
    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    # Copy to train
    for img in train_imgs:
        shutil.copy(img, os.path.join(base_dest, "train", dest_label, os.path.basename(img)))
    
    # Copy to val
    for img in val_imgs:
        shutil.copy(img, os.path.join(base_dest, "val", dest_label, os.path.basename(img)))

# Make sure target folders exist
for split in ['train', 'val']:
    for label in ['Good', 'Bad']:
        os.makedirs(os.path.join(base_dest, split, label), exist_ok=True)

# Process and copy
for label, sources in targets.items():
    copy_and_split(sources, label)

print("âœ… All new data has been added and classified successfully.")


import os
import shutil
import random

random.seed(42)

# Source folders
good_src = "/kaggle/input/good-morphology"
bad_src_1 = "/kaggle/input/bad-morphology"
bad_src_2 = "/kaggle/input/not-sperm"

# Destination base path
base_dest = "/kaggle/working/sperm_morphology_dataset"

# Updated function to recursively find images
def get_all_images_recursively(folder):
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

# Function to copy and split files into train/val
def copy_and_split(source_paths, dest_label):
    all_images = []
    for src in source_paths:
        all_images += get_all_images_recursively(src)

    print(f"ğŸ“¦ Found {len(all_images)} new '{dest_label}' images to add.")

    if not all_images:
        return

    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    for img in train_imgs:
        shutil.copy(img, os.path.join(base_dest, "train", dest_label, os.path.basename(img)))

    for img in val_imgs:
        shutil.copy(img, os.path.join(base_dest, "val", dest_label, os.path.basename(img)))

# Ensure folders exist
for split in ['train', 'val']:
    for label in ['Good', 'Bad']:
        os.makedirs(os.path.join(base_dest, split, label), exist_ok=True)

# Process the image folders
copy_and_split([good_src], 'Good')
copy_and_split([bad_src_1, bad_src_2], 'Bad')

print("âœ… All new data has been added and classified successfully.")



import os

folders = {
    "Good": "/kaggle/input/good-morphology",
    "Bad Morphology": "/kaggle/input/bad-morphology",
    "Not Sperm": "/kaggle/input/not-sperm"
}

for label, path in folders.items():
    print(f"\nğŸ“‚ {label.upper()} - Listing files in: {path}\n")
    for root, _, files in os.walk(path):
        for file in files:
            print(" -", os.path.join(root, file))



from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # or yolov8s-cls.pt for a stronger model

# Start training
model.train(
    data='/kaggle/working/sperm_morphology_dataset',
    epochs=30,
    imgsz=224,
    project='runs',
    name='sperm_classification_new',
    val=True
)



import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the results.csv path
base_dir = "runs/classify"
folders = sorted(os.listdir(base_dir), key=lambda x: os.path.getmtime(os.path.join(base_dir, x)))
latest_run = folders[-1]
results_csv_path = os.path.join(base_dir, latest_run, "results.csv")

# Load data
df = pd.read_csv(results_csv_path)

# Show available columns just in case
print("âœ… Columns found:", df.columns)

# Force a simple plot to check if it's working
plt.plot(df['epoch'], df['metrics/accuracy_top1'], color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy (Top-1)')
plt.grid(True)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the latest results.csv file
base_dir = "runs/classify"
folders = sorted(os.listdir(base_dir), key=lambda x: os.path.getmtime(os.path.join(base_dir, x)))
latest_run = folders[-1]
results_csv_path = os.path.join(base_dir, latest_run, "results.csv")
df = pd.read_csv(results_csv_path)

# Plot
plt.figure(figsize=(14, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train/loss'], label='Training Loss', color='blue')
plt.plot(df['epoch'], df['val/loss'], label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['metrics/accuracy_top1'], label='Validation Accuracy (Top-1)', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



from IPython.display import Image, display

# Show training results (loss curves and metrics)
display(Image(filename='runs/classify/train/results.png'))



# Load the trained model (itâ€™s saved automatically after training)
from ultralytics import YOLO

model = YOLO("runs/classify/train/weights/best.pt")  # best weights from training

# Predict on a single image or a folder of images
results = model.predict(source="/kaggle/working/sperm_morphology_dataset/val/Good", imgsz=224)

# View results
for r in results:
    print(f"Predicted: {r.names[r.probs.top1]} | Confidence: {r.probs.data.max().item():.2f}")



from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model (make sure you have the correct path to the weights)
model = YOLO("runs/classify/train/weights/best.pt")

# Path to a single image you want to test
image_path = '/kaggle/input/sperm-morphological-quality/Sperm-Data/High Quality Sperm - Labeled/Good Sperm/2017a_99_99_2450_3741.png'  # Replace this with your image path

# Make prediction on the image
results = model.predict(source=image_path, imgsz=224)  # Adjust imgsz if needed

# Display the image and the prediction result
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image and show results
plt.imshow(image)
plt.title(f"Predicted: {results[0].names[results[0].probs.top1]} | Confidence: {results[0].probs.data.max().item():.2f}")
plt.axis('off')
plt.show()



from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import FileUpload

# Load the trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Function to display uploaded image
def on_upload_change(change):
    # Get the uploaded image
    uploaded_image = list(uploader.value.values())[0]
    
    # Save the uploaded image temporarily
    with open("uploaded_image.png", "wb") as f:
        f.write(uploaded_image['content'])
    
    # Make prediction on the uploaded image
    results = model.predict(source="uploaded_image.png", imgsz=224)
    
    # Display the image and prediction
    image = cv2.imread("uploaded_image.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.title(f"Predicted: {results[0].names[results[0].probs.top1]} | Confidence: {results[0].probs.data.max().item():.2f}")
    plt.axis('off')
    plt.show()

# Create an uploader widget
uploader = FileUpload(accept='image/*', multiple=False)
uploader.observe(on_upload_change, names='value')

# Display the uploader
display(uploader)


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import FileUpload

# Load the trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Function to handle image upload and prediction
def on_upload_change(change):
    # Get the uploaded image
    uploaded_image = list(uploader.value.values())[0]
    
    # Save the uploaded image temporarily
    with open("uploaded_image.png", "wb") as f:
        f.write(uploaded_image['content'])
    
    # Make prediction on the uploaded image
    results = model.predict(source="uploaded_image.png", imgsz=224)
    
    # Display the uploaded image
    image = cv2.imread("uploaded_image.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Display prediction results
    class_name = results[0].names[int(results[0].boxes.cls[0])]
    confidence = results[0].probs[0].item()

    print(f"Predicted: {class_name} | Confidence: {confidence:.2f}")

# Create an uploader widget
uploader = FileUpload(accept='image/*', multiple=False)
uploader.observe(on_upload_change, names='value')

# Display the uploader
display(uploader)



from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from ipywidgets import FileUpload
from PIL import Image
import io

# Load the trained model
model = YOLO("runs/classify/train/weights/best.pt")

# Function to handle image upload and prediction
def on_upload_change(change):
    # Get the uploaded image
    uploaded_image = list(uploader.value.values())[0]
    
    # Convert the uploaded image content to an OpenCV-compatible format
    image_data = uploaded_image['content']
    image = Image.open(io.BytesIO(image_data))
    
    # Save the uploaded image temporarily
    image.save("uploaded_image.png")
    
    # Make prediction on the uploaded image
    results = model.predict(source="uploaded_image.png", imgsz=224)
    
    # Display the uploaded image
    image = cv2.imread("uploaded_image.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Display prediction results
    class_name = results[0].names[int(results[0].boxes.cls[0])]
    confidence = results[0].probs[0].item()

    # Output prediction result in a message
    print(f"Predicted: {class_name} | Confidence: {confidence:.2f}")

# Create an uploader widget
uploader = FileUpload(accept='image/*', multiple=False)
uploader.observe(on_upload_change, names='value')

# Display the uploader
display(uploader)

