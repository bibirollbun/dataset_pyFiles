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


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')


train_data.head()


test_data.head()


# Numerical: Temparature Humidity, Moisture, Nitrogen, Potassium, Phosphorous
# Categorial: Soil type, Crop Type
# Target: Fertilizer Name


train_data.describe()


train_data.shape


train_data.info


train_data.isnull().sum()   # no null value 


print("\n--- Unique Value Counts ---")
print(train_data.nunique())


print("\n--- Duplicated Rows ----")
print(train_data.duplicated().sum())


import matplotlib.pyplot as plt
import seaborn as sns


# Target variable Distribution 
plt.figure(figsize=(8, 4))
sns.countplot(data=train_data, x = "Fertilizer Name", order=train_data["Fertilizer Name"].value_counts().index)
plt.title('Target Distribution: Fertilizer Name')
# plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()


# Categorical Feature Distribution 
categorial_cols = ['Soil Type',	'Crop Type']

for col in categorial_cols:
    plt.figure(figsize=(12, 4))
    sns.countplot(data=train_data, x = col, order=train_data[col].value_counts().index)
    plt.title("Target Distribution: "+ col)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Numerical feature distribution
numeric_col = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']

train_data[numeric_col].hist(bins=20, figsize=(14, 10))
plt.suptitle("Numeric Feature Distribution")
plt.show()


# Numerical Distribution with KDE(By PDF)
from scipy.stats import gaussian_kde

numeric_cols = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=train_data, x="Fertilizer Name", y=col)
    plt.title(f'{col} vs fertilizer')
    plt.show()


plt.figure(figsize=(10, 6))
corr = train_data[numeric_col].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()



from sklearn.preprocessing import LabelEncoder
# Encode Categorial variables
le_fert = LabelEncoder()
train_data['Fertilizer Name'] = le_fert.fit_transform(train_data['Fertilizer Name'])
le_soil = LabelEncoder()
train_data['Soil Type'] = le_soil.fit_transform(train_data['Soil Type'])
test_data['Soil Type'] = le_soil.transform(test_data['Soil Type'])
le_crop = LabelEncoder()
train_data['Crop Type'] = le_crop.fit_transform(train_data['Crop Type'])
test_data['Crop Type'] = le_crop.transform(test_data['Crop Type'])


# Soil Crop interaction feature
def create_soil_crop_interaction(df):
    df['Soil_Crop'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)
    df['Soil_Crop'] = LabelEncoder().fit_transform(df['Soil_Crop'])
    return df


train_data = create_soil_crop_interaction(train_data)
test_data = create_soil_crop_interaction(test_data)


# Feature engineering (same for train and test)
def engineer_features(df):
    df['NPK_Total'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    df['N_ratio'] = df['Nitrogen'] / (df['NPK_Total'] + 1e-5)
    df['K_ratio'] = df['Potassium'] / (df['NPK_Total'] + 1e-5)
    df['P_ratio'] = df['Phosphorous'] / (df['NPK_Total'] + 1e-5)

    df['Water_Availabilty'] = df['Humidity'] + df['Moisture']
    # Add interaction terms
    df['Temp_N'] = train_data['Temparature'] * df['Nitrogen']
    df['Moisture_N'] = train_data['Moisture'] * df['Nitrogen']
    return df


train_data = engineer_features(train_data)
test_data = engineer_features(test_data)


# Crop group medium stats
crop_medians = train_data.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium', 'Moisture']].median()
print(crop_medians)


crop_medians = train_data.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium', 'Moisture']].median()
for col in crop_medians.columns:
    train_data[f'{col}_CropMed'] = train_data['Crop Type'].map(crop_medians[col])
    test_data[f'{col}_CropMed'] = test_data['Crop Type'].map(crop_medians[col])


print(train_data)


# Spilt Data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import VotingClassifier

X = train_data.drop(columns=['Fertilizer Name'])
y = train_data['Fertilizer Name']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



# LightGBM model
lgbm = lgb.LGBMClassifier(random_state=42)
xgbm = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
cat = cb.CatBoostClassifier(verbose=0, random_state=42)

# Ensemble with soft voting
ensemble = VotingClassifier(estimators=[
    ('lgbm', lgbm),
    ('xgb', xgbm),
    ('cat', cat)
], voting='soft')

ensemble.fit(X_train, y_train)



y_pred = ensemble.predict(X_val)
y_proba = ensemble.predict_proba(X_val)


print(classification_report(y_val, y_pred, target_names=le_fert.classes_))


# MAP@3 evaluation
from sklearn.metrics import classification_report, label_ranking_average_precision_score
from sklearn.preprocessing import label_binarize

y_val_bin = label_binarize(y_val, classes=np.arange(len(le_fert.classes_)))
map3 =  label_ranking_average_precision_score(y_val_bin, y_proba)
print(f"\n---- MAP@3 Score: {map3: 4f} ----")


# Submission 
X_test = test_data.copy()
y_test_proba = ensemble.predict_proba(X_test)
top_3_preds = np.argsort(y_test_proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = [" ".join(le_fert.inverse_transform(row)) for row in top_3_preds]



test_data.index


submission = pd.DataFrame({'id': test_data.index, 'Fertilizer Name': top_3_labels})
submission.to_csv('submissionv3.csv', index=False)




