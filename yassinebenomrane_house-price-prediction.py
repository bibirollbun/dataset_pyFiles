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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter, FixedLocator
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer  
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from datetime import date
from statsmodels.tsa.arima.model import ARIMA
import warnings
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore", category=FutureWarning, 
                       message="use_inf_as_na")


train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
train.head()


train["sale_date"] = pd.to_datetime(train["sale_date"], errors='coerce') 
# Sort by date for a cleaner plot
train_sorted = train.sort_values("sale_date")

train["year"] = train["sale_date"].dt.year

# Group by year and calculate average sale price
yearly_avg = train.groupby("year")["sale_price"].mean()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(yearly_avg.index, yearly_avg.values, marker='o')
plt.title("Average Sale Price by Year")
plt.xlabel("Year")
plt.ylabel("Average Sale Price")
plt.grid(True)
plt.tight_layout()
plt.show()



# Drop missing or invalid year_built values
train = train.dropna(subset=["year_built"])

# Create the house_age column
train["house_age"] = 2025 - train["year_built"]

# Drop negative or unrealistic ages (e.g., future builds or very old houses)
train = train[(train["house_age"] >= 0) & (train["house_age"] <= 200)]
# Group by house_age and compute average sale price
age_avg = train.groupby("house_age")["sale_price"].mean()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(age_avg.index, age_avg.values, marker='o', color='green')
plt.title("Average Sale Price by House Age")
plt.xlabel("House Age (in years)")
plt.ylabel("Average Sale Price")
plt.grid(True)
plt.tight_layout()
plt.show()


# Drop rows with missing values
train = train.dropna(subset=["sale_price", "land_val", "imp_val", "sqft"])

# Set up 3 plots side by side
plt.figure(figsize=(18, 5))

# 1. Land Value vs Sale Price
plt.subplot(1, 3, 1)
plt.scatter(train["land_val"], train["sale_price"], alpha=0.3, s=5)
plt.xlabel("Land Value")
plt.ylabel("Sale Price")
plt.title("Land Value vs Sale Price")
plt.grid(True)

# 2. Improvement Value vs Sale Price
plt.subplot(1, 3, 2)
plt.scatter(train["imp_val"], train["sale_price"], alpha=0.3, s=5, color="orange")
plt.xlabel("Improvement Value")
plt.title("Improvement Value vs Sale Price")
plt.grid(True)

# 3. Square Footage vs Sale Price
plt.subplot(1, 3, 3)
plt.scatter(train["sqft"], train["sale_price"], alpha=0.3, s=5, color="green")
plt.xlabel("Square Footage")
plt.title("Square Footage vs Sale Price")
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import MinMaxScaler

def process_time_features(df, year_col='year', built_col='year_built'):
    """
    Processes time-related features:
    - Scales the 'year' column
    - Creates 'year_since_built'
    - Creates binary 'is_recent' (1 if year > 2015, else 0)

    Returns:
        df (pd.DataFrame): with added features
        year_scaled (np.array): scaled version of 'year' for time-branch input
        scaler (MinMaxScaler): fitted scaler for reuse on test data
    """
    df = df.copy()

    # 1. Scale 'year' for time-branch
    scaler = MinMaxScaler()
    # Extract year from sale_date if year_col doesn't exist
    if year_col not in df.columns and 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df[year_col] = df['sale_date'].dt.year

    year_scaled = scaler.fit_transform(df[[year_col]])


    # 2. Add year_since_built
    if built_col in df.columns:
        df['year_since_built'] = 2025 - df[built_col]
        df['year_since_built'] = df['year_since_built'].fillna(df['year_since_built'].median())
    else:
        df['year_since_built'] = 0  # Fallback if missing

    # 3. Add is_recent
    df['is_recent'] = (df[year_col] > 2015).astype(int)

    return df, year_scaled, scaler



import numpy as np
import pandas as pd

def process_temporal_features(df):
    df = df.copy()

    # Ensure sale_date is datetime
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')

    # Extract year and month
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month

    # Cyclical encoding for month
    df['sale_month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
    df['sale_month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)

    # Handle year_reno
    df['was_renovated'] = df['year_reno'].notna().astype(int)

    df['years_since_reno'] = df['sale_year'] - df['year_reno']
    df['years_since_reno'] = df['years_since_reno'].fillna(999)

    df['reno_is_recent'] = (df['year_reno'] > 2010).astype(float)
    df['reno_is_recent'] = df['reno_is_recent'].fillna(0)

    return df



def add_engineered_features(df):
    df = df.copy()

    # 1. House-to-lot ratio (inverted: house/land)
    df['house_to_lot_ratio'] = df['sqft'] / df['sqft_lot']
    df['house_to_lot_ratio'] = df['house_to_lot_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. Bath score
    df['bath_score'] = (
        1.0 * df.get('bath_full', 0) +
        0.75 * df.get('bath_3qtr', 0) +
        0.5 * df.get('bath_half', 0)
    )

    # 3. Bath/bed ratio
    df['bath_bed_ratio'] = df['bath_score'] / df['beds']
    df['bath_bed_ratio'] = df['bath_bed_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 4. Log of land_val + imp_val
    df['log_total_value'] = np.log1p(df.get('land_val', 0) + df.get('imp_val', 0))

    return df




from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_tabular_data(df):
    df = df.copy()

    # Drop features from 'wfnt' to the end
    if 'wfnt' in df.columns:
        wfnt_index = df.columns.get_loc('wfnt')
        df = df.iloc[:, :wfnt_index]  # Keep everything before 'wfnt'

    df = add_engineered_features(df)
    df = process_temporal_features(df)
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Drop the target and time-related features from the lists
    target = 'sale_price'
    if target in numerical_cols:
        numerical_cols.remove(target)
    for col in ['year', 'sale_date', 'year_built', 'year_reno']:
        if col in numerical_cols: numerical_cols.remove(col)
        if col in categorical_cols: categorical_cols.remove(col)

    # Fill missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna('missing')

    # Standardize numerical columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Label encode categorical for embedding layers
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, numerical_cols, categorical_cols, scaler, encoders



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, Dropout, BatchNormalization
import tensorflow.keras.backend as K

# Your preprocessing functions are assumed to be defined above

# === 1. Apply preprocessing functions ===
train_processed, year_scaled, time_scaler = process_time_features(train, year_col='sale_year', built_col='year_built')
train_tabular, num_cols, cat_cols, tabular_scaler, encoders = preprocess_tabular_data(train_processed)

# === 2. Prepare model inputs ===
X_time = year_scaled  # shape: (n_samples, 1)
X_numeric = train_tabular[num_cols].values  # shape: (n_samples, n_numeric)
X_cats = [train_tabular[col].values for col in cat_cols]  # list of arrays

# === 3. Target bounds ===
y_true = train['sale_price'].values
y_bounds = np.stack([y_true * 0.9, y_true * 1.1], axis=1)  # dummy bounds to train on initially



def winkler_loss(alpha=0.1):
    def loss(y_true, y_pred):
        l = y_pred[:, 0]
        u = y_pred[:, 1]
        y = y_true[:, 0]

        in_range = K.cast((y >= l) & (y <= u), 'float32')
        penalty_low = K.cast(y < l, 'float32') * ((l - y) * (2.0 / alpha))
        penalty_high = K.cast(y > u, 'float32') * ((y - u) * (2.0 / alpha))

        base_width = (u - l)
        score = base_width + penalty_low + penalty_high

        return K.mean(score)
    return loss



# === Inputs ===
input_time = Input(shape=(1,), name='time_input')
input_numeric = Input(shape=(len(num_cols),), name='numeric_input')

cat_inputs = []
cat_embeddings = []
for col in cat_cols:
    vocab_size = train_tabular[col].max() + 1
    embed_dim = min(50, vocab_size // 2 + 1)

    inp = Input(shape=(1,), name=f'{col}_input')
    emb = Embedding(input_dim=vocab_size + 1, output_dim=embed_dim)(inp)
    emb = Flatten()(emb)

    cat_inputs.append(inp)
    cat_embeddings.append(emb)

# === Time branch ===
x_time = Dense(16, activation='relu')(input_time)
x_time = Dense(8, activation='relu')(x_time)

# === Tabular branch ===
x_num = Dense(64, activation='relu')(input_numeric)
x_num = BatchNormalization()(x_num)
x_num = Dropout(0.2)(x_num)

x_tab = Concatenate()([x_num] + cat_embeddings)
x_tab = Dense(128, activation='relu')(x_tab)
x_tab = Dropout(0.3)(x_tab)

# === Combine branches ===
x = Concatenate()([x_time, x_tab])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

# === Output both bounds in one layer ===
output_bounds = Dense(2, name='bounds')(x)

model = Model(inputs=[input_time, input_numeric] + cat_inputs, outputs=output_bounds)
model.compile(optimizer='adam', loss=winkler_loss(alpha=0.1))
model.summary()



X_splits = train_test_split(
    X_time,
    X_numeric,
    *X_cats,
    y_bounds,
    test_size=0.2,
    random_state=42
)

# Unpack the splits
X_time_train, X_time_val = X_splits[0], X_splits[1]
X_numeric_train, X_numeric_val = X_splits[2], X_splits[3]
X_cats_train = X_splits[4:4 + len(X_cats)*2:2]
X_cats_val = X_splits[5:5 + len(X_cats)*2:2]
y_train, y_val = X_splits[-2], X_splits[-1]

train_inputs = [X_time_train, X_numeric_train] + list(X_cats_train)
val_inputs = [X_time_val, X_numeric_val] + list(X_cats_val)

history = model.fit(
    x=train_inputs,
    y=y_train,
    validation_data=(val_inputs, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)

# Evaluate on validation
loss = model.evaluate(val_inputs, y_val)
print(f"Validation Winkler loss: {loss:.4f}")

# Predict on new data
# preds = model.predict([X_time_test, X_numeric_test] + list(X_cats_test))




### 1. *Preprocess the test set*

test_processed = test.copy()
test_processed['sale_date'] = pd.to_datetime(test_processed['sale_date'], errors='coerce')
test_processed['sale_year'] = test_processed['sale_date'].dt.year

year_scaled_test = time_scaler.transform(test_processed[['sale_year']])

test_tabular = test_processed.copy()
test_tabular = add_engineered_features(test_tabular)
test_tabular = process_temporal_features(test_tabular)

test_tabular[num_cols] = test_tabular[num_cols].fillna(test_tabular[num_cols].median())
test_tabular[num_cols] = tabular_scaler.transform(test_tabular[num_cols])

for col in cat_cols:
    test_tabular[col] = test_tabular[col].fillna('missing')
    test_tabular[col] = test_tabular[col].map(
        lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else 0
    )


### 2. *Prepare test inputs*

X_time_test = year_scaled_test
X_numeric_test = test_tabular[num_cols].values
X_cats_test = [test_tabular[col].values for col in cat_cols]

test_inputs = [X_time_test, X_numeric_test] + X_cats_test


### 3. *Predict*

y_preds = model.predict(test_inputs)


import pandas as pd
import numpy as np

# Step 1: Extract lower and upper bounds from predictions
pi_lower = y_preds[:, 0]
pi_upper = y_preds[:, 1]

# Step 2: Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],           # Replace 'id' with correct column name if different
    'pi_lower': pi_lower,
    'pi_upper': pi_upper
})

# Step 3: Save to CSV
submission.to_csv('submission.csv', index=False)

