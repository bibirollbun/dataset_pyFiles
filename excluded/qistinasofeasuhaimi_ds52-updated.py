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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
%matplotlib inline


df_train = pd.read_csv(r"/kaggle/input/playground-series-s4e7/train.csv")
df_train


df_train.shape


df_train.info()


df_train.describe()


print("Number of duplicated rows in dataset : ",df_train.duplicated().sum())


df_train.isna().sum().sort_values(ascending = False)


numerical_features = ['Age', 'Region_Code', 'Annual_Premium', 'Vintage']

categorical_features = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage']



fig, axs = plt.subplots(4, 3, figsize=(16, 20))
axs = axs.flatten()

index = 0

# Loop through each numerical column and plot a histogram
for num_col in numerical_features:
    ax = axs[index]
    ax.hist(df_train[num_col], bins=30, alpha=0.5, color='blue')
    ax.set_title(num_col)
    index += 1

# Loop through each categorical column and plot a bar chart of value counts
for cat_ft in categorical_features:
    ax = axs[index]
    # Get the frequency counts for each category
    counts = df_train[cat_ft].value_counts()
    # Create a bar chart using the category names as labels
    ax.bar(counts.index.astype(str), counts.values, alpha=0.5, color='red')
    ax.set_title(cat_ft)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    index+=1

# Plot response column
ax = axs[index]
# Get the frequency counts for each category
counts = df_train['Response'].value_counts()
# Create a bar chart using the category names as labels
ax.bar(counts.index.astype(str), counts.values, alpha=0.5, color='red')
ax.set_title('Response')
ax.set_xlabel("Category")
ax.set_ylabel("Count")
index+=1

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

# Numerical Features
num_features = numerical_features
n_num = len(num_features)
cols = 2
rows = math.ceil(n_num / cols)

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axs = axs.flatten()

for i, feature in enumerate(num_features):
    sns.boxplot(x='Response', y=feature, data=df_train, ax=axs[i])
    axs[i].set_title(f'{feature} vs Response')

# Hide any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()


#Categorical Features
cat_features = categorical_features
n_cat = len(cat_features)
cols = 2
rows = math.ceil(n_cat / cols)

fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows))
axs = axs.flatten()

for i, feature in enumerate(cat_features):
    sns.countplot(x=feature, hue='Response', data=df_train, ax=axs[i])
    axs[i].set_title(f'{feature} vs Response')
    axs[i].tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()



sample_df = df_train.sample(1000, random_state=42)
col = sample_df.drop(columns='id')
sns.pairplot(col, hue = 'Response',corner=False)
plt.suptitle("Pairplot of Numerical Features", y=1.02)
plt.show()


df_train.info()
print('\n')
df_train.head()


for col in categorical_features:
    print(f"{col} has {df_train[col].nunique()} values : {df_train[col].unique()} \n")


# Binary features
# Display the unique values before encoding 
print("Before encoding:", df_train['Gender'].unique())

le = LabelEncoder()
df_train['Gender'] = le.fit_transform(df_train['Gender'])  # Male = 1, Female = 0

# Verify the transformation 
print("After encoding:", df_train['Gender'].unique())


# View unique values before encoding
print("Before encoding:", df_train['Vehicle_Age'].unique())

# Initialize OrdinalEncoder with explicit category order
ord_enc = OrdinalEncoder(categories=[['< 1 Year', '1-2 Year', '> 2 Years']])

# Apply the encoding and overwrite the existing column
df_train['Vehicle_Age'] = ord_enc.fit_transform(df_train[['Vehicle_Age']])

# Optional: Confirm encoding worked
print("After encoding:", df_train['Vehicle_Age'].unique())


# View unique values before encoding
print("Before encoding:", df_train['Vehicle_Damage'].unique())

# Encode Vehicle_Damage: Yes = 1, No = 0
df_train['Vehicle_Damage'] = le.fit_transform(df_train['Vehicle_Damage'])  

# Optional: Check result
print("After encoding:", df_train['Vehicle_Damage'].unique())


df_train.head()


df_train.info()


# Calculate the Pearson correlation matrix
corr_df = df_train.copy().drop(columns='id')
corr_matrix = corr_df.corr(method='pearson')

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Pearson Correlation Matrix")
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(12, 6))

# numerical features
features = ['Age', 'Annual_Premium', 'Vintage']
for ax, ft in zip(axs, features):
    ax.boxplot(df_train[ft], vert=False)
    ax.set_title(ft)
    ax.grid(True)
    ax.set_xlabel(ft)

plt.tight_layout()
plt.show()



# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])



# Features and target
X = df_train.drop(columns=['id','Driving_License', 'Response'])
y = df_train['Response']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


print(y_train.value_counts())
print()
print(y_val.value_counts())


# Train Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get importances
importances = pd.Series(model.feature_importances_, index=X_train.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 8))
plt.title("Feature Importance from Random Forest")
plt.show()


model_scores = {}


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
print("Logistic Regression AUC:", auc)

model_scores['LogisticRegression'] = auc



model = lgb.LGBMClassifier(
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    n_estimators=200,
    random_state=42
)

# Fit the model on scaled training data
model.fit(X_train, y_train)

# Predict on scaled validation data
y_pred = model.predict_proba(X_val)[:, 1]

# Evaluate with AUC
auc = roc_auc_score(y_val, y_pred)
print("LightGBM AUC:", auc)

# Store
model_scores['LightGBM'] = auc



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=6, 
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
print("Random Forest AUC:", auc)

model_scores['RandomForest'] = auc



from xgboost import XGBClassifier

model = XGBClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=200,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
print("XGBoost AUC:", auc)

model_scores['XGBoost'] = auc



from catboost import CatBoostClassifier

model = CatBoostClassifier(
    learning_rate=0.05,
    depth=6,
    iterations=200,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)
print("CatBoost AUC:", auc)

model_scores['CatBoost'] = auc



results_df = pd.DataFrame.from_dict(model_scores, orient='index', columns=['AUC'])
results_df = results_df.sort_values(by='AUC', ascending=False)

results_df.plot(kind='barh', figsize=(10, 5), legend=False, title="Model AUC Comparison")
plt.xlabel("ROC AUC Score")
plt.show()



from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Step 1: Define parameter grid
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [6, 9],
    'learning_rate': [ 0.05, 0.2],
    'subsample': [0.8],
}

# Step 2: Set up the classifier (with fixed eval_metric to avoid warning)
xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Step 3: GridSearchCV
grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='roc_auc',    
    cv=5,                  
    verbose=2,
    n_jobs=-1            
)

# Step 4: Fit on training data (scaled or unscaled â€” scaling not needed for trees)
grid.fit(X_train, y_train)

# Step 5: Evaluate
print("Best parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred_proba = best_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print("Tuned XGBoost AUC:", auc)



import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

for i in np.arange(0, 1, 0.1):
    print(f"Theshold {i}")
    y_pred_label = (y_pred_proba >= i).astype(int)
    cm = confusion_matrix(y_val,y_pred_label)
    print(cm)
    print(f"net positive = {cm[1][1]-cm[1][0]}")
    print(classification_report(y_val, y_pred_label))
#theshold chosen = 0.1
threshold = 0.1
    


df_test = pd.read_csv(r"/kaggle/input/playground-series-s4e7/test.csv")
df_test


df_test.shape


df_test.info()


print("Number of duplicated rows in dataset : ",df_test.duplicated().sum())


df_test.isna().sum().sort_values(ascending = False)


df_test.info()
print('\n')
df_test.head()


# Binary features
# Display the unique values before encoding 
print("Before encoding:", df_test['Gender'].unique())

le = LabelEncoder()
df_test['Gender'] = le.fit_transform(df_test['Gender'])  # Male = 1, Female = 0

# Verify the transformation 
print("After encoding:", df_test['Gender'].unique())


# View unique values before encoding
print("Before encoding:", df_test['Vehicle_Age'].unique())

# Initialize OrdinalEncoder with explicit category order
ord_enc = OrdinalEncoder(categories=[['< 1 Year', '1-2 Year', '> 2 Years']])

# Apply the encoding and overwrite the existing column
df_test['Vehicle_Age'] = ord_enc.fit_transform(df_test[['Vehicle_Age']])

# Optional: Confirm encoding worked
print("After encoding:", df_test['Vehicle_Age'].unique())


# View unique values before encoding
print("Before encoding:", df_test['Vehicle_Damage'].unique())

# Encode Vehicle_Damage: Yes = 1, No = 0
df_test['Vehicle_Damage'] = le.fit_transform(df_test['Vehicle_Damage'])  

# Optional: Check result
print("After encoding:", df_test['Vehicle_Damage'].unique())


df_test.head()


# Drop unimportant features
df_test = df_test.drop(['id','Driving_License'], axis=1)


df_test.shape


scaler = StandardScaler()
scaler.fit(X_train)
df_test = scaler.transform(df_test)


best_params = grid.best_params_
best_xgb = XGBClassifier(
    **best_params,            
    use_label_encoder=False,  
    eval_metric='logloss',   
    random_state=42  
)

best_xgb.fit(X_train, y_train)


# Predict probability of class 1 for each row in test set
y_test_pred_proba = best_xgb.predict_proba(df_test)[:, 1]
y_test_pred_label = (y_test_pred_proba >=threshold).astype(int)


Submissions= pd.DataFrame()
sample = pd.read_csv('/kaggle/input/playground-series-s4e7/sample_submission.csv')
Submissions["id"] = sample["id"]
Submissions['Response'] = y_test_pred_label
Submissions.to_csv('submission.csv', index=False)
Submissions.head()

