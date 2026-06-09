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


# ğŸ“¦ Imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import gc

# ğŸ“� Load Data Efficiently
train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv", low_memory=True)
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")
sample_submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")

# ğŸ§  Reduce Memory

def reduce_memory_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col != 'sale_price':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                else:
                    df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df

train = reduce_memory_usage(train)
test = reduce_memory_usage(test)
gc.collect()

# ğŸ“Š Interactive EDA
fig = px.histogram(train, x="sale_price", nbins=50, title="ğŸ�¡ House Price Distribution")
fig.show()

fig2 = px.scatter(train, x="sqft", y="sale_price", title="ğŸ“� Square Footage vs Sale Price")
fig2.show()

fig3 = px.box(train, y="sale_price", title="ğŸ“¦ Box Plot of Sale Prices")
fig3.show()

fig4 = px.scatter_matrix(train, dimensions=["sqft", "land_val", "imp_val", "sale_price"], title="ğŸ”� Feature Relationships")
fig4.show()


from sklearn.preprocessing import LabelEncoder

# ğŸ�¯ Set Target Column
target = "sale_price"
excluded_cols = ['id', 'sale_price', 'sale_date', 'sale_warning', 'sale_nbr']
features = [col for col in train.columns if col not in excluded_cols]

# ğŸ› ï¸� Encode Categorical Features
categorical_cols = train[features].select_dtypes(include=["object"]).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined_data = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("missing")
    le.fit(combined_data)
    train[col] = le.transform(train[col].astype(str).fillna("missing"))
    test[col] = le.transform(test[col].astype(str).fillna("missing"))
    label_encoders[col] = le

# ğŸ§ª Split Data
X = train[features]
y = train[target]
X_test = test[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ğŸ§  Train Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ğŸ“ˆ Evaluate Model
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds, squared=False)
print(f"ğŸ“‰ Validation RMSE: {rmse:.2f}")

# ğŸ”® Predict Test Set
test_preds = model.predict(X_test)

# ğŸ“¦ Prepare Submission
submission = sample_submission.copy()
submission["lower"] = test_preds - 25000  # Can be tuned or calculated using std
submission["upper"] = test_preds + 25000
submission["pred"] = test_preds
submission.to_csv("submission.csv", index=False)
submission.head()

