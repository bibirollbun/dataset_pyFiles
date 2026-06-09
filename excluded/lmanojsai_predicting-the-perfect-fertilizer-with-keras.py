import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.head()


df.info()


# Statistical Summary
df.describe()


# Checking for missing values
df.isnull().sum()


plt.figure(figsize=(8, 5))
sns.countplot(x="Soil Type", data=df, order=df["Soil Type"].value_counts().index, palette="muted")
plt.title("Soil Type Distribution")
plt.xlabel("Soil Type")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="Crop Type", data=df, order=df["Crop Type"].value_counts().index, palette="pastel")
plt.title("Crop Type Frequency")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(x="Fertilizer Name", data=df, order=df["Fertilizer Name"].value_counts().index, palette="Set2")
plt.title("Most Common Fertilizers")
plt.xlabel("Fertilizer Name")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


numerical_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

for col in numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    
    sns.histplot(df[col], kde=True, color='skyblue', ax=axes[0])
    plt.title(f'{col} - Histogram')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    sns.boxplot(x=df[col], color='lightcoral', ax=axes[1])
    plt.title(f'{col} - Boxplot')
    plt.xlabel('')
 
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x="Crop Type", y="Nitrogen", data=df)
plt.title("Crop Type vs Nitrogen (Boxplot)")
plt.xlabel("Crop Type")
plt.ylabel("Nitrogen")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x="Fertilizer Name", y="Potassium", data=df)
plt.title("Fertilizer vs Potassium (Boxplot)")
plt.xlabel("Fertilizer Name")
plt.ylabel("Potassium")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x="Soil Type", y="Phosphorous", data=df)
plt.title("Soil Type vs Phosphorous (Boxplot)")
plt.xlabel("Soil Type")
plt.ylabel("Phosphorous")
plt.show()


corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


X = df.drop(columns=["id", "Fertilizer Name"])
y = df["Fertilizer Name"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.head()


ct = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ['Soil Type', 'Crop Type']),
], remainder='passthrough')

X_train_transformed = ct.fit_transform(X_train)
X_val_transformed = ct.transform(X_val)


encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_val_encoded = encoder.transform(y_val)


y_train_dummy = to_categorical(y_train_encoded)
y_val_dummy = to_categorical(y_val_encoded)


input_shape = [X_train.shape[1]]
num_classes = y_train.nunique()


model = Sequential([
    Input(shape=(X_train_transformed.shape[1],)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta = 0.001,
    patience = 20,
    restore_best_weights=True
)


history = model.fit(
    X_train_transformed, y_train_dummy,
    validation_data=(X_val_transformed, y_val_dummy),
    batch_size=256,
    epochs=30,
    callbacks=[early_stopping]
)


plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, dpi=100)


history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Learning Curve")


def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p[:k]:
            score += 1.0 / (list(p[:k]).index(a) + 1)
    return score / len(actual)

y_val_preds = model.predict(X_val_transformed)
top_3 = np.argsort(-y_val_preds, axis=1)[:, :3]
top_3_labels = encoder.inverse_transform(top_3.ravel()).reshape(-1, 3)

map3_score = mapk(y_val.tolist(), top_3_labels)
print("MAP@3 on validation:", map3_score)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

ids = test_df["id"]
X_test = test_df.drop(columns=["id"])
X_test_transformed = ct.transform(X_test)
probs = model.predict(X_test_transformed)

top_3_preds_idx = np.argsort(-probs, axis=1)[:, :3]
top_3_preds_labels = encoder.inverse_transform(top_3_preds_idx.ravel()).reshape(-1, 3)
predictions = [" ".join(row) for row in top_3_preds_labels]

submission_df = pd.DataFrame({
    "id": ids,
    "Fertilizer Name": predictions
})

submission_df.to_csv("submission.csv", index=False)

