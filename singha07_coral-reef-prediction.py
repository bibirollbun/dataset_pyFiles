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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore',category = FutureWarning)


train_df = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/train.csv')
test_df = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/test.csv')


train_df.sample(5)


test_df.sample(4)


train_df.info()


test_df.info()


# Step 1: Identify the dummy columns related to species
dummy_cols = [col for col in train_df.columns if col.startswith('species_')]

# Step 2: Get the original species column back
train_df['species'] = train_df[dummy_cols].idxmax(axis=1).str.replace('species_', '')

# Step 3: Drop the one-hot encoded species columns
train_df = train_df.drop(columns=dummy_cols)



train_df.info()


train_df.head()


train_df.drop(columns = ['id'],inplace = True)


numerical_cols = train_df.select_dtypes(include = ['int','float']).columns
numerical_cols


from sklearn.impute import KNNImputer

# Initialize the imputer only once
kimp = KNNImputer(n_neighbors=5, weights='distance')

# Apply KNN imputer to all numerical columns together (recommended)
train_df[numerical_cols] = kimp.fit_transform(train_df[numerical_cols])



ids = test_df['id'].copy()
test_df.drop(columns = ['id'], inplace = True)


from sklearn.impute import KNNImputer

# Initialize the imputer only once
kimp = KNNImputer(n_neighbors=5, weights='distance')

# Apply KNN imputer to all numerical columns together (recommended)
test_df[numerical_cols] = kimp.fit_transform(test_df[numerical_cols])



train_df.info()


cat_cols = train_df.select_dtypes(include = ['object']).columns
cat_cols


for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)


for col in test_df.select_dtypes(include = ['object']).columns:
    test_df[col].fillna(train_df[col].mode()[0], inplace=True)
    




train_df.info()


test_df.info()


train_df.duplicated().sum()


cat_cols



# Categorical columns to plot
cat_cols = ['region', 'substrate_type', 'light_availability', 
            'marine_protection_status', 'species']

# Create 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# Flatten axes for easy indexing
axes = axes.flatten()

# Loop through the categorical columns
for i, col in enumerate(cat_cols):
    sns.countplot(data=train_df, x=col, ax=axes[i], color='skyblue')
    axes[i].set_title(f'{col} Distribution')
    axes[i].tick_params(axis='x', rotation=45)

# Hide the last empty subplot (if any)
if len(cat_cols) < len(axes):
    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])  # delete extra axes

plt.tight_layout()
plt.show()


train_df['species'].value_counts()


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

cat_cols = ['region', 'substrate_type', 'light_availability', 
            'marine_protection_status']

ord = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_df[cat_cols] = ord.fit_transform(train_df[cat_cols])

test_df[cat_cols] = ord.transform(test_df[cat_cols])

# Label encode the 'species' column
lb = LabelEncoder()
train_df['species'] = lb.fit_transform(train_df['species'])


train_df.select_dtypes(include = ['object']).columns


test_df.select_dtypes(include = ['object']).columns


plt.figure(figsize= (20,20))
sns.heatmap(train_df.corr(),annot = True)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

X = train_df.drop(columns = ['species'],axis = 1)
y= train_df['species']

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


from imlearn.ensemble import BalanceRandomforestClassifier




x_train.shape


x_test.shape


import xgboost as xgb
from sklearn.metrics import hamming_loss, accuracy_score
xg = xgb.XGBClassifier(n_estimators = 300)

xg.fit(x_train, y_train)

y_pred = xg.predict(x_test)

# Predictions
y_pred = xg.predict(x_test)

# Hamming Loss
hloss = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hloss:.4f}")

# Also printing accuracy for reference
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc:.4f}")


from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

Gbr = GradientBoostingClassifier(n_estimators = 300)

Gbr.fit(x_train, y_train)

# Predictions
y_pred = Gbr.predict(x_test)

# Hamming Loss
hloss = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hloss:.4f}")

# Also printing accuracy for reference
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc:.4f}")


bg = BaggingClassifier(estimator = xg, n_estimators = 5)
bg.fit(x_train, y_train)


# Predictions
y_pred = bg.predict(x_test)

# Hamming Loss
hloss = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hloss:.4f}")

# Also printing accuracy for reference
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc:.4f}")


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 400, class_weight = 'balanced_subsample')

rf.fit(x_train, y_train)

# Predictions
y_pred = rf.predict(x_test)

# Hamming Loss
hloss = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hloss:.4f}")

# Also printing accuracy for reference
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc:.4f}")


rf.fit(X,y)


y_pred2 = rf.predict(test_df)


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

X = train_df.drop(columns = ['species'],axis = 1)
y= train_df['species']

Average_loss = []

X = X.values
y = y.values

for fold,(train_idx, Val_idx) in enumerate (skf.split(X,y)):
    x_train, x_val = X[train_idx], X[Val_idx]
    y_train, y_val = y[train_idx], y[Val_idx]


    rf = RandomForestClassifier(n_estimators = 400)

    rf.fit(x_train, y_train)
    
    # Predictions
    y_pred = rf.predict(x_val)
    
    # Hamming Loss
    hloss = hamming_loss(y_val, y_pred)
    Average_loss.append(hloss)
    print(f"Hamming Loss of {fold + 1} is: {hloss:.4f}")
  

print(f'Average loss is: {np.mean(Average_loss)}')  


bgrf = BaggingClassifier(estimator = rf, n_estimators = 10)

bgrf.fit(x_train, y_train)

# Predictions
y_pred = bgrf.predict(x_test)

# Hamming Loss
hloss = hamming_loss(y_test, y_pred)
print(f"Hamming Loss: {hloss:.4f}")

# Also printing accuracy for reference
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc:.4f}")


bgrf.fit(X,y)


y_pred = bgrf.predict(test_df)


submission = pd.DataFrame({
    'id':ids,
    'species': lb.inverse_transform(y_pred)
})


submission2 = pd.DataFrame({
    'id':ids,
    'species': lb.inverse_transform(y_pred2)
})


submission2.head(5)


submission2['species'].value_counts()


submission_encoded = pd.get_dummies(submission, dtype = int)

submission_encoded.head()


submission_encoded2 = pd.get_dummies(submission2, dtype = int)

submission_encoded2.head()


sample_submission = pd.read_csv('/kaggle/input/coral-diversity-at-reef-sites/sample_submission.csv')


sample_submission.drop(columns = ['id'], inplace = True)


submission_encoded2.shape


sample_submission.columns


submission_encoded2.columns


submission_encoded.shape


for col in sample_submission.columns:
    if col not in submission_encoded2.columns:
        submission_encoded2[col] = 0  # Add the missing column with 0s



submission_encoded2.head(5)


submission_encoded2.to_csv('submissionrf2.csv',index = False)

