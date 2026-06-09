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


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
df.head()



df.shape[0]



df.columns



img_dir = '/kaggle/input/plant-pathology-2020-fgvc7/images'


def get_label(row):
    for label in ['healthy', 'multiple_diseases', 'rust', 'scab']:
        if row[label] == 1:
            return label

df['label'] = df.apply(get_label, axis=1)

samples = 3
labels = df['label'].unique()
fig, axs = plt.subplots(len(labels), samples, figsize=(samples * 3, len(labels) * 3))

for row_idx, label in enumerate(labels):
    subset = df[df['label'] == label].head(samples)

    for col_idx, image_id in enumerate(subset['image_id']):
        img_path = os.path.join(img_dir, image_id + '.jpg')
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axs[row_idx, col_idx].imshow(img_rgb)
        axs[row_idx, col_idx].axis('off')
        if col_idx == 0:
            axs[row_idx, col_idx].set_title(label, fontsize=12)

plt.tight_layout()
plt.show()


X = []
for image_id in df['image_id'].head(10):  # contoh ambil 10 gambar
    img_path = os.path.join(img_dir, image_id + '.jpg')
    img = cv2.imread(img_path)
    X.append(img)

shapes = [img.shape for img in X]

unique_shapes = set(shapes)
if len(unique_shapes) == 1:
    print(f"Semua gambar memiliki dimensi yang sama: {unique_shapes.pop()}")
else:
    print(f"Terdapat {len(unique_shapes)} dimensi gambar berbeda:")
    for s in unique_shapes:
        print(s)



import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder

def extract_features(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (128, 128))
    
    # HSV
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])
    
    # Edge Detection (Canny)
    edges = cv2.Canny(img_resized, 100, 200)
    edge_mean = np.mean(edges)

    # Texture (GLCM)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [h_mean, s_mean, v_mean, edge_mean, contrast, dissimilarity, homogeneity, energy, correlation]

# Contoh dataset
features = []
labels = []

img_dir = '/kaggle/input/plant-pathology-2020-fgvc7/images'

for idx, row in df.iterrows():
    label = row['label']
    img_path = os.path.join(img_dir, row['image_id'] + '.jpg')
    if os.path.exists(img_path):
        feat = extract_features(img_path)
        features.append(feat)
        labels.append(label)

# Simpan ke DataFrame
columns = ['H_mean', 'S_mean', 'V_mean', 'Edge_mean', 'Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
feature_df = pd.DataFrame(features, columns=columns)
feature_df['label'] = labels

# Encode label jadi angka
le = LabelEncoder()
feature_df['label_encoded'] = le.fit_transform(feature_df['label'])

feature_df.head()



feature_df['label_encoded'].nunique()


columns = ['H_mean', 'S_mean', 'V_mean', 'Edge_mean', 'Contrast', 
           'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
feature_df = pd.DataFrame(features, columns=columns)
feature_df['label'] = labels

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
feature_df['label_encoded'] = le.fit_transform(feature_df['label'])



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

fitur = ['H_mean', 'S_mean', 'V_mean', 'Edge_mean', 'Contrast',
         'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']

X = feature_df[fitur]
y = feature_df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Untuk menampilkan hasil normalisasi di semua data
df_normalized = feature_df.copy()
df_normalized[fitur] = scaler.transform(feature_df[fitur])

print("\nFitur (sebelum dinormalisasi):")
print(feature_df.head())

print("\nFitur (setelah normalisasi):")
print(df_normalized.head())



feature_df['label_encoded'].value_counts()



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd

# --- Langkah 1: Undersampling menjadi 91 data per kelas ---
min_count = 91
dfs = []

for label in feature_df['label_encoded'].unique():
    df_kelas = feature_df[feature_df['label_encoded'] == label]
    df_sampled = df_kelas.sample(n=min_count, random_state=42)
    dfs.append(df_sampled)

# Gabungkan semua kelas yang telah diundersampling
feature_df_balanced = pd.concat(dfs).reset_index(drop=True)

# --- Langkah 2: Pisahkan fitur dan label ---
fitur = ['H_mean', 'S_mean', 'V_mean', 'Edge_mean', 'Contrast',
         'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']

X = feature_df_balanced[fitur]
y = feature_df_balanced['label_encoded']

# --- Langkah 3: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify menjaga distribusi seimbang
)

("\nJumlah data per kelas setelah undersampling:")
print(feature_df_balanced['label_encoded'].value_counts())


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# --- Langkah 1: Pisahkan fitur dan label ---
fitur = ['H_mean', 'S_mean', 'V_mean', 'Edge_mean', 'Contrast',
         'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']

X = feature_df[fitur]
y = feature_df['label_encoded']

# --- Langkah 2: Normalisasi ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# --- Langkah 3: SMOTE untuk oversampling kelas minoritas ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# --- Langkah 4: Split data setelah SMOTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# --- Cek hasil ---
print("\nJumlah data per kelas setelah SMOTE:")
print(pd.Series(y_resampled).value_counts())



from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}

svm = SVC()

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)

grid_search.fit(X_train_scaled, y_train)
print("Best parameters found:", grid_search.best_params_)

best_svm = grid_search.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)

print("SVM with Best Parameters:")
print(classification_report(y_test, y_pred_svm))

# Buat confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)

# Visualisasi confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_svm.classes_, yticklabels=best_svm.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - SVM')
plt.show()


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. Buat Model XGBoost =====
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# ===== 2. Latih Model =====
xgb_model.fit(X_train_scaled, y_train)

# ===== 3. Prediksi =====
y_pred = xgb_model.predict(X_test_scaled)

# ===== 4. Evaluasi =====
print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===== 5. Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=xgb_model.classes_, yticklabels=xgb_model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


