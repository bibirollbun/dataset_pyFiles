from IPython.display import display, HTML

img_url = "https://www.kaggle.com/competitions/91719/images/header"

display(HTML(f'''
<div style="text-align: center;">
    <img src="{img_url}" width="800">
</div>
'''))




# =========================
# 1. Install required libs
# =========================
!pip install ydata-profiling catboost

# =========================
# 2. Import libraries
# =========================
import pandas as pd
from ydata_profiling import ProfileReport
from catboost import CatBoostClassifier


# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")



# =========================
# 3. Load data
# =========================

data_train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")

data_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# Check structure
print(data_train.shape, data_test.shape)


data_train.head()


data_test.head()


# =========================
# 4. Quick EDA with ydata-profiling
# =========================
profile = ProfileReport(data_train, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")  


import os
print(os.listdir("/kaggle/working"))



profile.to_notebook_iframe()



# =========================
# 5. Prepare data
# =========================
data_train.drop(columns = ['id'], inplace = True)




data_train.head()


# Keep the IDs separately
test_ids = data_test["id"]

# Drop 'id' before prediction
test_features = data_test.drop("id", axis=1)




test_features.head()


# Target column value counts

data_train['y'].value_counts()


# Target column distribution

plt.figure(figsize=(6,4))
sns.countplot(x = 'y', data = data_train, palette = "Paired")
plt.title("Target Column Distribution", fontsize = 14)
plt.xlabel("Target", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


X = data_train.drop(columns = ['y'])
Y = data_train['y']


# Identify categorical columns (object or category dtype)
cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical features:", cat_features)



# =========================
# 6. Train CatBoost model
# =========================
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    eval_metric="AUC",
    random_seed=42,
    verbose=100
)

model.fit(X, Y, cat_features=cat_features)


# Predict probabilities
preds = model.predict_proba(test_features)[:, 1 ]# probability of class 1 

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "y": preds
})





submission


submission.to_csv("submission.csv", index=False)

