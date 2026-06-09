import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping


# 2️⃣ Load Data
train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
sub = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

# Separate target
y = train['target']
train.drop(['target'], axis=1, inplace=True)

# Identify categorical columns
cat_cols = [c for c in train.columns if c != 'id']


# ============================
# 3️⃣ EDA - Exploratory Data Analysis (before Label Encoding)
# ============================

# Basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nMissing values in train:\n", train.isnull().sum())
print("\nTarget distribution:\n", y.value_counts(normalize=True))

# Plot target distribution
plt.figure(figsize=(5,3))
sns.countplot(x=y)
plt.title("Target Distribution")
plt.show()

# Column value counts (for first 5 columns)
for c in cat_cols[:5]:
    plt.figure(figsize=(6,3))
    sns.countplot(x=train[c].astype(str))  
    plt.title(f"Distribution of {c}")
    plt.show()

# Unique values per column
for c in cat_cols:
    print(c, "unique values:", train[c].nunique())



# 4️⃣ Feature Engineering
# ============================

# Label Encoding لكل الأعمدة الفئوية
for c in cat_cols:
    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)
    le = LabelEncoder()
    le.fit(list(train[c]) + list(test[c]))
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])

# Feature Engineering - ميزات رقمية
train['unique_count'] = train[cat_cols].nunique(axis=1)
test['unique_count'] = test[cat_cols].nunique(axis=1)
numeric_cols = ['unique_count']

# Columns for embedding
cat_cols_for_embedding = [c for c in cat_cols if c not in numeric_cols]



# 5️⃣ Split Data
# ============================
x_train, x_val, y_train, y_val = train_test_split(train[cat_cols + numeric_cols], y, test_size=0.2, random_state=42)


# 6️⃣ Build Model with Entity Embeddings
# ============================
inputs, embeddings = [], []

# Embeddings for categorical columns
for c in cat_cols_for_embedding:
    inp = Input(shape=(1,))
    dim = min(100, int(np.log2(train[c].nunique()) * 4))
    emb = Embedding(train[c].nunique()+1, dim)(inp)
    emb = Flatten()(emb)
    inputs.append(inp)
    embeddings.append(emb)

# Numeric inputs
num_input = Input(shape=(len(numeric_cols),))
inputs.append(num_input)
embeddings.append(num_input)

# Concatenate all
x = Concatenate()(embeddings)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs, out)
model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=[AUC(name='auc')])


# 7️⃣ Train Model
# ============================
es = EarlyStopping(monitor='val_auc', patience=2, mode='max', restore_best_weights=True)

model.fit([x_train[c] for c in cat_cols_for_embedding] + [x_train[numeric_cols].values],
          y_train,
          validation_data=([x_val[c] for c in cat_cols_for_embedding] + [x_val[numeric_cols].values], y_val),
          epochs=20,
          batch_size=512,
          verbose=1,
          callbacks=[es])



# 8️⃣ Predict & Save Submission
# ============================
preds = model.predict([test[c] for c in cat_cols_for_embedding] + [test[numeric_cols].values])
sub['target'] = preds
sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

