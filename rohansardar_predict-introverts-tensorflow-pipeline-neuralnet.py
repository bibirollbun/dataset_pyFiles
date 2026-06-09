import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col="id")


train.head()


cat_cols = train.select_dtypes(include=['object']).columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns

train.replace([np.inf, -np.inf], np.nan, inplace=True)
train[num_cols] = train[num_cols].fillna(train[num_cols].mean())

for col in cat_cols:
    if train[col].isnull().any():
        train[col] = train[col].fillna(train[col].mode()[0])


cat_cols = test.select_dtypes(include=['object']).columns
num_cols = test.select_dtypes(include=['int64', 'float64']).columns

test.replace([np.inf, -np.inf], np.nan, inplace=True)
test[num_cols] = test[num_cols].fillna(test[num_cols].mean())

for col in cat_cols:
    if test[col].isnull().any():
        test[col] = test[col].fillna(test[col].mode()[0])


print(f"The categorical value columns are: {cat_cols.values}")


sns.histplot(train['Personality'], color='tomato')
plt.title('Count of Personality')
plt.show()


plt.figure(figsize=(8, 2 * len(cat_cols)))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, len(cat_cols), i)  
    sns.countplot(x=train[col], hue=train['Personality'], palette='Set2')
    plt.title(f"{col} vs Personality count") 
    
plt.tight_layout()
plt.show()


n_plots = len(num_cols)
cols_per_row = math.ceil(n_plots / 2)

plt.figure(figsize=(4 * cols_per_row, 6))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, cols_per_row, i)
    sns.histplot(x=col, hue='Personality', data=train, fill=True, palette='Set2')
    plt.title(f"{col} vs Personality count")

plt.tight_layout()
plt.show()


X = train.drop(columns=['Personality'])
y = train['Personality']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes(include=['float64', 'int64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)
preprocessor


X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)


def build_pipeline():
    return Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

pipeline = build_pipeline()
pipeline


X_train_transformed = pipeline.fit_transform(X_train)
X_val_transformed = pipeline.transform(X_val)
X_test_transformed = pipeline.transform(X_test)


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(8, input_dim=input_dim, activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='swish'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='swish'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    verbose=1, 
    min_lr=1e-6
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True 
)


model = create_model(X_train_transformed.shape[1])
model.summary()


history = model.fit(
    X_train_transformed, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_val_transformed, y_val), 
    callbacks=[lr_scheduler, early_stopping] 
)

test_loss, test_accuracy = model.evaluate(X_test_transformed, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


history_df = pd.DataFrame(history.history)
history_df['epoch'] = history_df.index + 1


plt.figure(figsize=(8, 5))
sns.lineplot(x='epoch', y='loss', data=history_df, label='Training Loss', color='seagreen')
sns.lineplot(x='epoch', y='val_loss', data=history_df, label='Validation Loss', color='orange')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
sns.lineplot(x='epoch', y='accuracy', data=history_df, label='Training Accuracy', color='seagreen')
sns.lineplot(x='epoch', y='val_accuracy', data=history_df, label='Validation Accuracy', color='orange')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


test_transformed = pipeline.transform(X_test)
y_pred = model.predict(test_transformed)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
print(f"Classifier Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}\n")
print(f"Classification Report: {classification_report(y_test, y_pred_classes)}")


test_transformed = pipeline.transform(test).astype('float32')
test_pred_probs = model.predict(test_transformed)
test_pred_classes = (test_pred_probs > 0.5).astype(int).flatten()
test_pred_labels = le.inverse_transform(test_pred_classes)

sub = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission = pd.DataFrame({
    'id': sub['id'],
    'Personality': test_pred_labels
})

submission.to_csv('submission.csv', index=False)




