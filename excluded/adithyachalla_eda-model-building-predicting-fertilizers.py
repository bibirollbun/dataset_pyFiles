# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier


train_df= pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train_df


test_df= pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_df


print(train_df.isnull().sum())



plt.figure(figsize=(12, 6))
train_df['Fertilizer Name'].value_counts().plot(kind='bar')
plt.title('Distribution of Fertilizer Labels')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Set seaborn style
sns.set(style="whitegrid")

# Pie Chart: Fertilizer Distribution
fert_counts = train_df['Fertilizer Name'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(fert_counts, labels=fert_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Fertilizer Name Distribution')
plt.axis('equal')
plt.show()


# Boxplots: Numerical Features grouped by Fertilizer
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in numerical_cols:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=train_df, x='Fertilizer Name', y=col)
    plt.title(f'{col} Distribution by Fertilizer Name')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Set style
sns.set(style="whitegrid")

# Heatmap of Soil Type vs Crop Type
soil_crop_ct = pd.crosstab(train_df['Soil Type'], train_df['Crop Type'])

plt.figure(figsize=(12, 6))
sns.heatmap(soil_crop_ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Crop Type Frequency Across Soil Types')
plt.xlabel('Crop Type')
plt.ylabel('Soil Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Stacked Bar Chart
soil_crop_ct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
plt.title('Crop Distribution within Each Soil Type (Stacked)')
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
train_df[numerical_cols].hist(bins=20, figsize=(15, 10), color='skyblue')
plt.suptitle('Distributions of Numerical Features')
plt.show()



fert_crop_ct = pd.crosstab(train_df['Crop Type'], train_df['Fertilizer Name'])

plt.figure(figsize=(14, 8))
sns.heatmap(fert_crop_ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Fertilizer Use by Crop Type')
plt.ylabel('Crop Type')
plt.xlabel('Fertilizer Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



fert_soil_ct = pd.crosstab(train_df['Soil Type'], train_df['Fertilizer Name'])

plt.figure(figsize=(12, 6))
sns.heatmap(fert_soil_ct, annot=True, fmt='d', cmap='Blues')
plt.title('Fertilizer Use by Soil Type')
plt.ylabel('Soil Type')
plt.xlabel('Fertilizer Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Select numerical columns
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Compute correlation and plot
plt.figure(figsize=(10, 7))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


# Encode categorical features
label_encoders = {}
for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le


train_df['NPK_Total'] = train_df['Nitrogen'] + train_df['Potassium'] + train_df['Phosphorous']



train_df['N_to_P_ratio'] = train_df['Nitrogen'] / (train_df['Phosphorous'] + 1)
train_df['K_to_P_ratio'] = train_df['Potassium'] / (train_df['Phosphorous'] + 1)



train_df['Temp_Humidity_Index'] = train_df['Temparature'] * train_df['Humidity']




train_df['Fertility_Need'] = (100 - train_df['Nitrogen']) + (100 - train_df['Phosphorous']) + (100 - train_df['Potassium'])



train_df


# Boxplot of NPK_Total grouped by Fertilizer Name
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fertilizer Name', y='NPK_Total', data=train_df)
plt.title('NPK_Total by Fertilizer Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Boxplot of Fertility_Need grouped by Fertilizer Name
plt.figure(figsize=(10, 6))
sns.boxplot(x='Fertilizer Name', y='Fertility_Need', data=train_df)
plt.title('Fertility Need by Fertilizer Name')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Prepare test set encoders
for col in ['Soil Type', 'Crop Type']:
    test_df[col] = label_encoders[col].transform(test_df[col])


test_df


# Define features and target
X = train_df.drop(['id', 'Fertilizer Name'], axis=1)
y = train_df['Fertilizer Name']
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']


# Train a classifier (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)



# Predict probabilities on test set
probs = clf.predict_proba(X_test)


# Get top 3 predicted class indices for each observation
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]


# Convert class indices back to original fertilizer names
fertilizer_names = label_encoders['Fertilizer Name'].inverse_transform(np.arange(len(label_encoders['Fertilizer Name'].classes_)))
predicted_labels = [
    " ".join([fertilizer_names[i] for i in row])
    for row in top3_preds
]



# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predicted_labels
})


# Save to CSV
submission.to_csv("submission.csv", index=False)


# Convert to DMatrix (optional, but improves speed)
dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test)


# Define parameters for multi-class classification
num_classes = len(np.unique(y))
params = {
    'objective': 'multi:softprob',
    'num_class': num_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 6,
    'eta': 0.1,
    'seed': 42
}


# Train model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)


# Predict probabilities for test set
pred_probs = xgb_model.predict(dtest)


# Get top 3 class indices per row
top3_preds = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]


# Decode predicted indices to actual fertilizer names
fertilizer_names = label_encoders['Fertilizer Name'].inverse_transform(np.arange(num_classes))
predicted_labels = [
    " ".join([fertilizer_names[i] for i in row])
    for row in top3_preds
]


# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predicted_labels
})

# Save to CSV
submission.to_csv('xgboost_submission.csv', index=False)


import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(objective='multiclass', num_class=len(np.unique(y)), random_state=42)
lgb_model.fit(X, y)

probs = lgb_model.predict_proba(X_test)



from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(loss_function='MultiClass', verbose=0, random_seed=42)
cat_model.fit(X, y)

probs = cat_model.predict_proba(X_test)



from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_model.fit(X, y)

probs = lr_model.predict_proba(X_test)



from sklearn.ensemble import VotingClassifier

# Optional: consider using fewer models if needed
reduced_estimators = [
    ('lr', models['Logistic Regression']),
    ('lgb', models['LightGBM']),
    ('cat', models['CatBoost'])
]

# VotingClassifier with soft voting
voting_model = VotingClassifier(
    estimators=reduced_estimators,
    voting='soft',
    n_jobs=1  # Set to 1 to reduce parallel memory load
)

# Train model
voting_model.fit(X, y)

# Get top-3 predicted class indices
top3_preds_idx = np.argsort(voting_model.predict_proba(X_test), axis=1)[:, -3:][:, ::-1]

# Get class labels
fertilizer_names = label_encoders['Fertilizer Name'].inverse_transform(
    np.arange(len(label_encoders['Fertilizer Name'].classes_))
)

# Get top-3 predictions in label form
predicted_labels = [" ".join([fertilizer_names[i] for i in row]) for row in top3_preds_idx]

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predicted_labels
})
submission.to_csv('voting_submission.csv', index=False)


