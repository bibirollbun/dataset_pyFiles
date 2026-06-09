import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
x_train = df_train.drop(columns=['y'])
y_train = df_train['y']
x_train_numeric = 0


x_train.sample(5)


categorical_cols = x_train.select_dtypes(include=['object']).columns.tolist()


print(x_train['poutcome'].unique())
print(x_train['contact'].unique())


categorical_cols


ohe_columns = ["job", "marital", "month", "contact", "poutcome"]
x_train = pd.get_dummies(x_train, columns=ohe_columns, drop_first=True)



binary_map = {'yes': 1, 'no': 0}
x_train["default"] = x_train["default"].map(binary_map)
x_train["housing"] = x_train["housing"].map(binary_map)
x_train["loan"] = x_train["loan"].map(binary_map)


from sklearn.preprocessing import StandardScaler

# Optional: drop ID if not meaningful
x_train.drop("id", axis=1, inplace=True)

# Fix pdays: replace -1 with a custom value (e.g., 999)
x_train["pdays"] = x_train["pdays"].replace(-1, 999)

# Scale continuous variables
scaler = StandardScaler()
x_train[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]] = scaler.fit_transform(
    x_train[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]]
)



x_train['education'].unique()


edu_order = {
    "unknown": -1,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3
}
x_train["education"] = x_train["education"].map(edu_order)



x_train.sample(2)


df_corr = pd.concat([x_train, y_train], axis=1)
df_corr


corr_matrix = df_corr.corr()

corr_target = corr_matrix[['y']].drop(labels=['y'])

sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
plt.show()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
x_test = df_test

# One-Hot Encode
ohe_columns = ["job", "marital", "month", "contact", "poutcome"]
x_test = pd.get_dummies(x_test, columns=ohe_columns, drop_first=True)

# Binary Encode
binary_map = {'yes': 1, 'no': 0}
for col in ["default", "housing", "loan"]:
    x_test[col] = x_test[col].map(binary_map)

# Handle ID and pdays
x_test.drop("id", axis=1, inplace=True)
x_test["pdays"] = x_test["pdays"].replace(-1, 999)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
x_test[numeric_cols] = scaler.fit_transform(x_test[numeric_cols])

edu_order = {
    "unknown": -1,   # or 0
    "primary": 1,
    "secondary": 2,
    "tertiary": 3
}
x_test["education"] = x_test["education"].map(edu_order)


# Define selected features based on correlation analysis
selected_features = ['duration', 'pdays', 'balance', 'education']

# Subset the training and testing data
x_train_model = x_train[selected_features].copy()
x_test_model = x_test[selected_features].copy()

