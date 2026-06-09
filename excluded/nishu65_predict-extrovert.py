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
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt


train=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")



# ðŸ”¥ Visualize missing values
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.heatmap(train.isna(), cmap="viridis", cbar=False)
plt.title("Train Missing Values")

plt.subplot(1,2,2)
sns.heatmap(test.isna(), cmap="viridis", cbar=False)
plt.title("Test Missing Values")
plt.show()

# ðŸ§¹ Handle NaNs in train (drop or fill)
# For training, we can drop rows with NaNs
train_clean = train.dropna()

# ðŸ§¹ Handle NaNs in test (DO NOT DROP rows, fill instead)
test_filled = test.copy()
test_filled.fillna(test_filled.median(numeric_only=True), inplace=True)  # Numeric NaNs
test_filled.fillna(test_filled.mode().iloc[0], inplace=True)             # Categorical NaNs

# Encode categorical columns (train and test together to avoid mismatch)
label_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    if col in train_clean.columns:
        train_clean[col] = le.fit_transform(train_clean[col])
    if col in test_filled.columns and col != 'Personality':  # Donâ€™t encode target in test
        test_filled[col] = le.transform(test_filled[col])
    encoders[col] = le

# Split features and target
X_train = train_clean.drop(['id', 'Personality'], axis=1)
y_train = train_clean['Personality']
X_test = test_filled.drop(['id'], axis=1)

# ðŸš€ Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# ðŸ”® Predict on test set
y_pred = model.predict(X_test)

# Decode predictions back to original labels
y_pred_labels = encoders['Personality'].inverse_transform(y_pred)

# ðŸ“„ Add predictions to test DataFrame
submission = test[['id']].copy()  # keep original test IDs
submission['Predicted_Personality'] = y_pred_labels

# Save submission file
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")

# ðŸ“Š Plot feature importance
sns.barplot(x=model.feature_importances_, y=X_train.columns)
plt.title("Feature Importances")
plt.show()





