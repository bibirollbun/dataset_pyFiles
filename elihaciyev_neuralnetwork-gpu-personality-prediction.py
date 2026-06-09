import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import combinations
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models

warnings.filterwarnings('ignore')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print("Train shape :", train.shape)
print("Test  shape :", test.shape)
display(train.head())
display(test.head())


missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)

missing_test = test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

sns.barplot(x=missing_train.values, y=missing_train.index, palette='mako', ax=ax[0])
ax[0].set_title("Train: Missing Values", fontsize=14)
ax[0].set_xlabel("Number of Missing")
ax[0].grid(axis='x', linestyle='--', alpha=0.5)

sns.barplot(x=missing_test.values, y=missing_test.index, palette='magma', ax=ax[1])
ax[1].set_title("Test: Missing Values", fontsize=14)
ax[1].set_xlabel("Number of Missing")
ax[1].grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


def preprocessing(df):
    df = df.copy()

    df.drop(columns=['id'], inplace=True)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    df['stage_fear'] = df['stage_fear'].fillna('unknown')
    df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')

    num_col = df.select_dtypes(include='number').columns
    for col in num_col:
        df[col] = df[col].fillna(df[col].mean())


    target_encoder = None
    if 'personality' in df.columns:
        target_encoder = LabelEncoder()
        df['personality'] = target_encoder.fit_transform(df['personality'])

    return df, target_encoder


X_train, target_encoder = preprocessing(train)
X_test_df, _ = preprocessing(test)

print("Train shape :", X_train.shape)
print("Test  shape :", X_test_df.shape)
display(X_train.head())
display(X_test_df.head())


X = X_train.drop(columns=['personality'])
y = X_train['personality']


def feature_engineering(df_train, df_test, cols=None, max_comb_len=3):
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')
    df_test.columns = df_test.columns.str.lower().str.replace(' ', '_')

    for r in range(2, max_comb_len + 1):
        for comb in combinations(cols, r):
            new_col_name = '__'.join(comb)
            df_train[new_col_name] = df_train[list(comb)].astype(str).agg('_'.join, axis=1)
            df_test[new_col_name] = df_test[list(comb)].astype(str).agg('_'.join, axis=1)

    ordinal_encoders = {}
    for col in df_train.select_dtypes(include='object').columns:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        df_train[[col]] = enc.fit_transform(df_train[[col]])
        df_test[[col]] = enc.transform(df_test[[col]])

        ordinal_encoders[col] = enc

    scaler = StandardScaler()
    df_train_scaled = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns)
    df_test_scaled = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

    return df_train_scaled, df_test_scaled, ordinal_encoders, scaler


X, X_test_df, _, _ = feature_engineering(X, X_test_df, cols=X.columns, max_comb_len=7)


print("Train shape :", X.shape)
print("Test  shape :", X_test_df.shape)


ensemble = []
accuracy = []
epochs = 30

# EarlyStopping callback
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,  
    restore_best_weights=True,
    verbose=1
)

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print('*'*50)
    print(f"\nðŸ“¦ Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop]
    )

    
    for epoch in range(len(history.history['loss'])):
        train_loss = history.history['loss'][epoch]
        train_acc = history.history['accuracy'][epoch]
        val_loss = history.history['val_loss'][epoch]
        val_acc = history.history['val_accuracy'][epoch]
        print(f"Epoch {epoch+1:2d}: "
              f"Train Loss={train_loss:.4f}, Accuracy={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")

    
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print('-'*50)
    print(f"âœ… Fold {fold + 1} Final Val Accuracy: {acc:.4f}")
    ensemble.append(model)
    accuracy.append(acc)

    
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'Fold {fold+1} Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

print('-'*50)
print(f"\nðŸ“Š Average Accuracy across folds: {np.mean(accuracy):.4f}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

pred_probs = np.zeros(X_test.shape[0])

for model in ensemble:
    pred_probs += model.predict(X_test).reshape(-1) / len(ensemble)


final_preds = (pred_probs > 0.5).astype(int)

class_names = target_encoder.classes_

print("ðŸ“‹ Classification Report:")
print(classification_report(y_test, final_preds, target_names=class_names))

cm = confusion_matrix(y_test, final_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='cividis',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


def submission(X_test, ensemble):
    pred_probs = np.zeros(X_test.shape[0])

    for model in ensemble:
        pred_probs += model.predict(X_test).reshape(-1) / len(ensemble)

    final_preds = (pred_probs > 0.5).astype(int)

    final_labels = target_encoder.inverse_transform(final_preds)

    submission = pd.DataFrame({
        'id': test['id'],
        'personality': final_labels
    })

    submission.to_csv('submission.csv', index=False)
    return submission


submission =  submission(X_test_df, ensemble)
submission.head()




