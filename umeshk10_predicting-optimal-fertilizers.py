import numpy as np
import pandas as pd


train_data = "/kaggle/input/playground-series-s5e6/train.csv"
test_data = "/kaggle/input/playground-series-s5e6/test.csv"


train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)


## lets randomly see 5 rows
train_df.sample(5)


## drop the 'id' column
train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])


train_df.head()


print(f"Size of training dataset : {train_df.shape[0]}")
print(f"Size of testing dataset : {test_df.shape[0]}")


## statistics of the data
train_df.describe()


## information about the dataset
train_df.info()


soil_type_unq_vals = train_df['Soil Type'].unique()
crop_type_unq_vals = train_df['Crop Type'].unique()
fertilizer_unq_vals = train_df['Fertilizer Name'].unique()

print(f"Unique Number : {soil_type_unq_vals.shape[0]}, Unique Soil Types : {soil_type_unq_vals}")
print(f"Unique Number : {crop_type_unq_vals.shape[0]}, Unique Crop Types : {crop_type_unq_vals}")
print(f"Unique Number : {fertilizer_unq_vals.shape[0]}, Fertlizer unique value : {fertilizer_unq_vals}")


missing_vals = train_df.isna().sum()
print(f"Missing Values :\n{missing_vals}")


missing_vals


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Soil Type', data=train_df)
plt.xticks(rotation=45)
plt.title("Distribution of Soil Type")
plt.show()


sns.countplot(x='Crop Type', data=train_df)
plt.xticks(rotation=45)
plt.show()


sns.countplot(x='Fertilizer Name', data=train_df)
plt.xticks(rotation=45)
plt.show()


sns.displot(x='Humidity', data=train_df, kind='kde')
plt.show()


sns.displot(x='Moisture', data=train_df, kind='kde')
plt.show()


sns.displot(x='Nitrogen', data=train_df, kind='kde')
plt.show()


sns.displot(x='Potassium', data=train_df, kind='kde')


sns.displot(x='Phosphorous', data=train_df, kind='kde')


train_df.columns


# List of 6 feature columns to plot
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Create 2 rows Ã— 3 columns = 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each boxplot
for i, feature in enumerate(features):
    sns.boxplot(data=train_df, x='Crop Type', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} vs Crop Type')
    axes[i].tick_params(axis='x', rotation=60)

# Improve spacing
plt.tight_layout()
plt.show()


# List of 6 feature columns to plot
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Create 2 rows Ã— 3 columns = 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each boxplot
for i, feature in enumerate(features):
    sns.boxplot(data=train_df, x='Soil Type', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} vs Soil Type')
    axes[i].tick_params(axis='x', rotation=45)

# Improve spacing
plt.tight_layout()
plt.show()


# List of 6 feature columns to plot
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Create 2 rows Ã— 3 columns = 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

# Flatten axes for easy iteration
axes = axes.flatten()

# Plot each boxplot
for i, feature in enumerate(features):
    sns.boxplot(data=train_df, x='Fertilizer Name', y=feature, ax=axes[i])
    axes[i].set_title(f'{feature} vs Fertilizer Name')
    axes[i].tick_params(axis='x', rotation=45)

# Improve spacing
plt.tight_layout()
plt.show()


train_df.head()


X = train_df.drop(columns='Fertilizer Name')
y = train_df['Fertilizer Name']


## numerical columns and categorical columns
numerical_columns = X.select_dtypes(exclude=['object']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

print(f"Numerical columns : {numerical_columns}")
print(f"Categorical columns : {categorical_columns}")


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(f"Length of training data : {len(X_train)}")
print(f"Length of testing data : {len(X_val)}")


## Column Transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ct = ColumnTransformer([
    ("Normalization", StandardScaler(), numerical_columns),
    ("OHE", OneHotEncoder(), categorical_columns)
])

lbl_enc = LabelEncoder()

X_train_new = ct.fit_transform(X_train)
y_train_new = lbl_enc.fit_transform(y_train)
X_val_new = ct.transform(X_val)
y_val_new = lbl_enc.transform(y_val)


## MAP@3 function
def mapk(y_true, y_pred, k=3):
    y_pred = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]
    def apk(actual, predicted, k):
        if actual in predicted:
            return 1.0 / (np.where(predicted == actual)[0][0] + 1)
        return 0.0

    return sum(apk(a, p, k) for a, p in zip(y_true, y_pred)) / len(y_true)


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=150, criterion='gini')
rf_model.fit(X_train_new, y_train_new)


## predictions on the validation data
rf_preds = rf_model.predict_proba(X_val_new)
map_score = mapk(y_val_new, rf_preds)
print(f"MAP@3 score for Random Forest Model is : {round(map_score, 2)}")


from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=150)
gb_model.fit(X_train_new, y_train_new)


## predictions and MAP score on validation data
gb_preds = gb_model.predict_proba(X_val_new)
map_score = mapk(y_val_new, gb_preds)
print(f"MAP@3 score for Gradient Boosting Model is : {round(map_score, 2)}")


from xgboost import XGBClassifier
# create model instance
xgb_model = XGBClassifier(n_estimators=150)
# fit model
xgb_model.fit(X_train_new, y_train_new)


xg_preds = xgb_model.predict_proba(X_val_new)
map_score = mapk(y_val_new, xg_preds)
print(f"MAP@3 score for XGBoosting Model is : {round(map_score, 2)}")


test_data = "/kaggle/input/playground-series-s5e6/test.csv"
test_df = pd.read_csv(test_data)
## saving id column
ids = test_df['id']
## removing id column while predictions
X_test = test_df.drop(columns='id')
X_test.head()


## using column transformer to scale the data
X_test_new = ct.transform(X_test)
print(X_test_new[:2])


y_preds = xgb_model.predict_proba(X_test_new)
y_preds.shape


## getting indices of top3 predictions
y_pred = np.argsort(y_preds, axis=1)[:, -3:][:, ::-1]


# Get top 3 predicted class indices for each row
flat = lbl_enc.inverse_transform(y_pred.ravel())
top3_names = flat.reshape(y_pred.shape)


fertilizer_name = [' '.join(row) for row in top3_names]
ids = pd.Series(ids)

submission_df = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': fertilizer_name
})


submission_df.to_csv("Submission.csv", index=False)

