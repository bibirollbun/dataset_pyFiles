import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')

df['label'].value_counts().plot(kind='bar', title='Label Distribution')
plt.show()

sample_df = df.sample(n=10000, random_state=3)

sample_ids = sample_df.sample(9, random_state=3)['id'].values
fig, axes = plt.subplots(3, 3, figsize=(8, 5))
for ax, img_id in zip(axes.flatten(), sample_ids):
    img = Image.open(f'/kaggle/input/histopathologic-cancer-detection/train/{img_id}.tif')
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()

def load_and_preprocess_image(img_id, img_size=(32,32)):
    img_path = f'/kaggle/input/histopathologic-cancer-detection/train/{img_id}.tif'
    img = Image.open(img_path)
    img = img.resize(img_size)
    img = np.array(img)
    return img.flatten()

X = []
y = []

for idx, row in sample_df.iterrows():
    img_vector = load_and_preprocess_image(row['id'])
    X.append(img_vector)
    y.append(int(row['label']))

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=3, stratify=y)

print("Train set:", X_train.shape)
print("Test set:", X_test.shape)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)



import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

X_train_std = np.load('X_train.npy')
X_test_std = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

def rescale_for_nb(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_rescaled = (X - X_min) / (X_max - X_min + 1e-8)
    return (X_rescaled * 255).astype(int)

X_train_cat = rescale_for_nb(X_train_std)
X_test_cat = rescale_for_nb(X_test_std)

nb_model = CategoricalNB(min_categories=256)
nb_model.fit(X_train_cat, y_train)

y_train_pred = nb_model.predict(X_train_cat)
y_test_pred = nb_model.predict(X_test_cat)

print("Train Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axes[0])
axes[0].set_title("Train Confusion Matrix")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[1])
axes[1].set_title("Test Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

y_test_proba = nb_model.predict_proba(X_test_cat)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'CategoricalNB (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CategoricalNB')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()





import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# 1. 初始化并训练Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("Train Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axes[0])
axes[0].set_title("Train Set Confusion Matrix")
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[1])
axes[1].set_title("Test Set Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

y_test_proba = rf_model.predict_proba(X_test)[:, 1]  # 预测正类的概率
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # 参考线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()







