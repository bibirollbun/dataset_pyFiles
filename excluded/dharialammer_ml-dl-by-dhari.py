## Make By Dhari LinkedIN:DhariAlamer //>_-??##


import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image


csv_file = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"
img_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/train/"


df = pd.read_csv(csv_file)


if df.empty:
    print(".")
else:
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i in range(5):
        img_name = df.iloc[i, 0]
        label = df.iloc[i, 1]  

        img_path = os.path.join(img_dir, img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"Label: {label}")
        else:
            axes[i].axis('off')
            axes[i].set_title(f"Image not found")

    plt.show()



df.info()


df = df.drop_duplicates().dropna().reset_index(drop=True)



df.info()


###stepppone///>_-??
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
csv_file = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"
img_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/train/"
augmented_dir = "/kaggle/working/augmented_dataset/"
os.makedirs(augmented_dir, exist_ok=True)

df = pd.read_csv(csv_file)
print(df.head(100))



import matplotlib.pyplot as plt


class_distribution = df["label"].value_counts()


plt.figure(figsize=(10, 5))
class_distribution.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()



class_distribution = df['label'].value_counts()
print(class_distribution)



###Data Augmentation//>_-??
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_data = []

for label in df["label"].unique():
    class_images = df[df["label"] == label]
    num_existing = len(class_images)
    num_needed = 85 - num_existing

    existing_images = class_images["filename"].tolist()
    
    for img_name in existing_images:
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)

        augmented_data.append([img_name, label])

        generated = 0
        for batch in datagen.flow(img_array, batch_size=1):
            new_img = (batch[0] * 255).astype(np.uint8)
            new_img = Image.fromarray(new_img)
            
            new_img_name = f"{label}_{generated}.png"
            new_img.save(os.path.join(augmented_dir, new_img_name))
            
            augmented_data.append([new_img_name, label])
            generated += 1
            if generated >= num_needed:
                break

df_augmented = pd.DataFrame(augmented_data, columns=["filename", "label"])
df_augmented.to_csv("/kaggle/working/augmented_train_labels.csv", index=False)

print(" Augmentation Completed ")



df_final = pd.read_csv("/kaggle/working/augmented_train_labels.csv")

X, y = [], []
for img_name, label in zip(df_final["filename"], df_final["label"]):
    img_path = os.path.join(augmented_dir, img_name)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        X.append(np.array(img) / 255.0)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f" Num of photo after Augmentation: {X.shape[0]}, شكل الصور: {X.shape}")



encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_flattened = X.reshape(X.shape[0], -1)
print(f" : X shape = {X_flattened.shape}, y shape = {y_encoded.shape}")



X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
print(f" : Train: {X_train.shape}, Test: {X_test.shape}")



print(" Ater connver:")
for i, label in enumerate(encoder.classes_):
    print(f"{label} → {i}")



svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.4f}")



rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")



knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.4f}")



models = ["SVM", "Random Forest", "KNN"]
accuracies = [accuracy_svm * 100, accuracy_rf * 100, accuracy_knn * 100, ]  

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")  
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)  
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 2, f"{acc:.2f}%", ha='center', fontsize=12, fontweight='bold')  

plt.show()



import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)


svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)


joblib.dump(svm_model, "/kaggle/working/svm_model.pkl")
joblib.dump(encoder, "/kaggle/working/label_encoder.pkl")

print(" ")




svm_model = joblib.load("/kaggle/working/svm_model.pkl")

# تحميل الترميز Label Encoder
encoder = joblib.load("/kaggle/working/label_encoder.pkl")

print("")




image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/03ab6acd.jpg"
img = Image.open(image_path).convert("RGB").resize((128, 128))
img_array = np.array(img) / 255.0  
img_flattened = img_array.reshape(1, -1)  

print(f": {img_flattened.shape}")


pred_svm = svm_model.predict(img_flattened)


predicted_label = encoder.inverse_transform(pred_svm)[0]


print(f": {predicted_label}")


import matplotlib.pyplot as plt
plt.imshow(img)
plt.title(f"SVM Prediction: {predicted_label}")
plt.axis("off")
plt.show()



###ٍstaaart Cnn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

csv_file = "/kaggle/input/open-data-day-2025-dates-types-classification/train_labels.csv"
img_dir = "/kaggle/input/open-data-day-2025-dates-types-classification/train/"

df = pd.read_csv(csv_file)

X, y = [], []
for img_name, label in zip(df["filename"], df["label"]):
    img_path = os.path.join(img_dir, img_name)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB").resize((128, 128))
        X.append(np.array(img) / 255.0)
        y.append(label)

X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

model.save("/kaggle/working/date_classifier_cnn.h5")

loss, acc = model.evaluate(X_test, y_test)
print(f"Accuracy: {acc:.4f}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()



from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import joblib

model = load_model("/kaggle/working/date_classifier_cnn.h5")
encoder = joblib.load("/kaggle/working/label_encoder.pkl")

image_path = "/kaggle/input/open-data-day-2025-dates-types-classification/test/10f11e1e.jpg"

img = Image.open(image_path).convert("RGB").resize((128, 128))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_label = encoder.inverse_transform([np.argmax(prediction)])[0]

plt.imshow(img)
plt.title(f"Predicted: {predicted_label}")
plt.axis("off")
plt.show()

print(f"Predicted Category: {predicted_label}")


