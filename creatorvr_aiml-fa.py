# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train




train.info()


train.describe()


train.shape


train.columns



train.isnull().sum()


correlation_matrix = train.drop('id',axis = 1).corr(numeric_only=True)
sns.heatmap(correlation_matrix)


train.info()



train.head()



from scipy.stats import zscore

z_scores = train.select_dtypes(include='number').apply(zscore)
outliers_z = (abs(z_scores) > 3)  # Common cutoff is z > 3
print(outliers_z.sum())



train.dropna(inplace = True)


# test.dropna(inplace = True)


test.info()


test_df = test.copy()
# Option 1: Fill numeric columns with median
numeric_cols = test_df.select_dtypes(include=['float64', 'int64']).columns
test_df[numeric_cols] = test_df[numeric_cols].fillna(test_df[numeric_cols].median())

# Option 2: Fill categorical columns with mode
categorical_cols = test_df.select_dtypes(include=['object', 'category']).columns
test_df[categorical_cols] = test_df[categorical_cols].fillna(test_df[categorical_cols].mode().iloc[0])

# Save cleaned version

print("✅ Null values filled successfully.")



test_df


test_df.isnull().sum()


test.isnull().sum()


train.head()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_cols = ['Stage_fear', "Drained_after_socializing"]

# Create transformer
encoder = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), cat_cols)
    ],
    remainder='passthrough'  # Keep all other columns
)

encoded_array = encoder.fit_transform(test_df)

# Optional: get back a DataFrame
encoded_test = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())



encoded_test.head()


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_cols = ['Stage_fear', "Drained_after_socializing"]

# Create transformer
encoder = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), cat_cols)
    ],
    remainder='passthrough'  # Keep all other columns
)

encoded_array = encoder.fit_transform(train)

# Optional: get back a DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())



train["Personality"].unique()


encoded_df


encoded_df.head()


# encoded_df.to_csv("final.csv")


for i in encoded_df.columns[:-1]:
    encoded_df[i] = encoded_df[i].astype(int)
for i in encoded_test.columns:
    encoded_test[i] = encoded_test[i].astype(int)


from sklearn.preprocessing import MinMaxScaler
import numpy as np
df = encoded_df.copy()
cols_to_normalize = df.select_dtypes(include=np.number).columns.tolist()
cols_to_normalize = [
    col for col in cols_to_normalize
    if col not in ['Unnamed: 0', 'remainder__id']
]
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])



from sklearn.preprocessing import MinMaxScaler
import numpy as np
df_test = encoded_test.copy()
cols_to_normalize = df_test.select_dtypes(include=np.number).columns.tolist()
cols_to_normalize = [
    col for col in cols_to_normalize
    if col not in ['Unnamed: 0', 'remainder__id']
]
scaler = MinMaxScaler()
df_test[cols_to_normalize] = scaler.fit_transform(df_test[cols_to_normalize])



df.head()


from sklearn.model_selection import train_test_split

# Replace this with your actual DataFrame
df = df.copy()

X = df.drop(columns=["remainder__id",'remainder__Personality'],axis = 1)  # <-- target column
y = df['remainder__Personality']


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import class_weight
import numpy as np

# 1. Load Data
X = X
y = y.squeeze() # Target variable as a Series

# 2. Encode the Target Variable (Classifiers require numeric targets)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_ # ['Extrovert' 'Introvert']

# 3. Split Data
# Use stratification to ensure the training and testing sets have the same class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. Handle Class Imbalance with Sample Weights
# Calculate sample weights to give more importance to the minority class ('Introvert')
weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# 5. Initialize and Train Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(
    n_estimators=100,           # Number of trees/stages
    learning_rate=0.1,          # Shrinkage factor
    max_depth=3,                # Depth of individual trees
    random_state=42
)

# Pass sample_weight to the fit method to address class imbalance
gb_model.fit(X_train, y_train, sample_weight=weights)

# 6. Predict and Evaluate
y_pred = gb_model.predict(X_test)
y_proba = gb_model.predict_proba(X_test)[:, 1] # Probability for the positive class (Introvert)

# Classification Report
print(f"Classification Report (Target Mapping: {target_names}):")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC Score
auc_score = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {auc_score:.4f}")


y.to_csv("y.csv")


X.to_csv("X.csv")


X.head()



y.head()


from sklearn.ensemble import RandomForestClassifier
from sklearn.
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



import pandas as pd

feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(ascending=False).head(15).plot(kind='barh')
plt.title("Top Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



ids = df_test["remainder__id"].values



X_final_test = df_test.drop("remainder__id", axis=1)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
y_pred = gb_model.predict(X_final_test)


y_pred_labels = le.inverse_transform(y_pred)


submission = pd.DataFrame({
    'id': ids.astype(int),
    'Personality': y_pred_labels.astype(str)
})
submission.head()



submission.to_csv("submission.csv", index=False)























