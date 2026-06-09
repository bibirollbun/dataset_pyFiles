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


df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping

int_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in int_cols:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# Encode categorical columns
label_encoders = {}
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df.drop(['id', 'Fertilizer Name'], axis=1)
y = df['Fertilizer Name']

# Sample and split the data
#X_sample, _, y_sample, _ = train_test_split(X, y, train_size=1, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
categorical_cols = ['Soil Type', 'Crop Type']
# Train the model

model = LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[early_stopping(stopping_rounds=50)],
    categorical_feature=categorical_cols,
    #verbose=100
)

# Evaluate
y_pred = model.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, y_pred))

# Save model and encoders
joblib.dump(model, "fertilizer_rf_model.pkl")
for col, le in label_encoders.items():
    joblib.dump(le, f"label_encoder_{col}.pkl")



import pandas as pd
import joblib
import numpy as np

# Load the test set
test_df = test

# Downcast integer columns
int_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in int_cols:
    test_df[col] = pd.to_numeric(test_df[col], downcast='integer')

# Encode categorical columns
label_encoders = {}
for col in ['Soil Type', 'Crop Type']:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])

le_fertilizer = joblib.load("label_encoder_Fertilizer Name.pkl")

# Prepare features
X_test = test_df.drop(['id'], axis=1)

# Predict top 3 probabilities
probs = model.predict_proba(X_test)
top_3_indices = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Top 3 in descending order

# Convert predicted indices to fertilizer names
top_3_labels = le_fertilizer.inverse_transform(top_3_indices.flatten()).reshape(top_3_indices.shape)
top_3_str = [" ".join(row) for row in top_3_labels]


# Create submission file
submission_df = pd.DataFrame({
    "id": test_df['id'],
    "Fertilizer Name": top_3_str
})

submission_df.to_csv("submission.csv", index=False)
print("finished")

