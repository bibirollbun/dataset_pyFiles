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


df = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv")
df.head()


a = df.select_dtypes(include = "object").nunique()
a


cardinality = a / len(df)
cardinality.sort_values(ascending=False)


df.isna().sum() / len(df)


df["engine"].unique()[0:30]


df.info()


import re
import pandas as pd

def parse_engine_info(text):
    hp = re.search(r'(\d+(\.\d+)?)HP', text)
    liters = re.search(r'(\d+(\.\d+)?)L', text)
    cylinders = re.search(r'(\d+)\s+Cylinder', text)
    engine_type = re.search(r'\b(V\d+|Straight \d+)\b', text)
    boost_type = 'Turbo' if 'Turbo' in text else ('Supercharged' if 'Supercharged' in text else 'NA')
    hybrid = int('Hybrid' in text)
    electric = int('Electric' in text)

    return pd.Series({
        'parsed_hp': float(hp.group(1)) if hp else None,
        'parsed_engine_L': float(liters.group(1)) if liters else None,
        'parsed_cylinders': int(cylinders.group(1)) if cylinders else None,
        'parsed_engine_type': engine_type.group(0) if engine_type else 'Unknown',
        'parsed_boost': boost_type,
        'is_hybrid': hybrid,
        'is_electric': electric
    })

# Apply to your dataset
engine_features = df['engine'].apply(parse_engine_info)
df = pd.concat([df, engine_features], axis=1)



df.head()


df["parsed_engine_type"].unique()


df.isna().sum() / len(df)


df["parsed_cylinders"].unique()


df["parsed_boost"].value_counts()


df = df.drop(columns = ["engine"])


df.shape


X = df.drop(columns=["id", "price"])
Y = df["price"]


X.head()


Y.head()


from sklearn.model_selection import train_test_split


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=42, test_size=0.2)


xtrain.head()


ytrain.head()


xtrain.shape


xtest.shape


cat = list(X.columns[X.dtypes == "object"])
con = list(X.columns[X.dtypes != "object"])


cat


con


from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


num_pipe = make_pipeline(
    SimpleImputer(strategy = "median"),
    StandardScaler()
).set_output(transform="pandas")


cat_pipe = make_pipeline(
    SimpleImputer(strategy = "constant", fill_value = "unknown"),
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
).set_output(transform="pandas")


X_con_pre = num_pipe.fit_transform(X[con])
X_con_pre


X_cat_pre = cat_pipe.fit_transform(X[cat])


X_cat_pre


X_cat_pre = X_cat_pre.astype(int)


X_cat_pre


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Convert categorical and continuous preprocessed data to numpy
X_cat_np = X_cat_pre.to_numpy().astype('int32')
X_con_np = X_con_pre.to_numpy().astype('float32')

# Build inputs for each categorical column
cat_inputs = [layers.Input(shape=(1,), dtype='int32', name=f'{col}_in') for col in cat]

# Create embeddings per categorical column
embedding_layers = []
for i, col in enumerate(cat):
    # Estimate vocabulary size (add 2: one for unknown, one for 0-index safety)
    vocab_size = int(np.max(X_cat_np[:, i])) + 2
    embed_dim = min(50, (vocab_size + 1) // 2)  # Heuristic for dimension
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name=f'{col}_embed')(cat_inputs[i])
    x = layers.Reshape((embed_dim,))(x)  # flatten to match dense input
    embedding_layers.append(x)

# Numerical input layer
num_input = layers.Input(shape=(X_con_np.shape[1],), name='num_in')

# Combine embeddings + continuous
x = layers.Concatenate()(embedding_layers + [num_input])

# Fully connected layers
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)

# Output layer (regression)
output = layers.Dense(1, name='output')(x)

# Final model
model = Model(inputs=cat_inputs + [num_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.summary()


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',     # or 'val_mae' if you prefer
    patience=5,             # stop if no improvement after 5 epochs
    restore_best_weights=True
)


# Prepare input dict for training
train_inputs = {f'{col}_in': X_cat_np[:, i] for i, col in enumerate(cat)}
train_inputs['num_in'] = X_con_np

# Target variable
y = df['price'].values

# Train
hist = model.fit(
    train_inputs, y, epochs=100, batch_size=512, validation_split=0.2, callbacks=[early_stop]
)


import matplotlib.pyplot as plt


plt.plot(hist.history["mae"])
plt.plot(hist.history["val_mae"])


xnew = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
xnew.head()


engine_features2 = xnew['engine'].apply(parse_engine_info)


xnew = pd.concat([xnew, engine_features2], axis=1)


xnew.head()


xnew = xnew.drop(columns = ["engine"])
xnew


xnew_cat = xnew[cat]
xnew_con = xnew[con]


xnew_cat_pre = cat_pipe.transform(xnew_cat)
xnew_con_pre = num_pipe.transform(xnew_con)


xnew_cat_pre = xnew_cat_pre.astype(int)
xnew_cat_pre


xnew_cat_pre = xnew_cat_pre.to_numpy()
xnew_con_pre = xnew_con_pre.to_numpy()

new_inputs = {
    f'{col}_in': xnew_cat_pre[:, i] for i, col in enumerate(cat)
}
new_inputs['num_in'] = xnew_con_pre


ypred = model.predict(new_inputs)
ypred


import seaborn as sns
sns.histplot(ypred, kde=True)


res = xnew[["id"]]
res["price"] = ypred


res


res.to_csv("submission.csv", index=False)




