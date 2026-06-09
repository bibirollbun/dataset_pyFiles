import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PREPROCESS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# PERFORMANCE
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

# MODELS
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier



import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
data = df.copy()


data.head()


data.describe()


data.nunique()


data.isnull().sum()


sns.countplot(data=data, x="y")


# NUMERICAL COLUMNS

numerical_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
sns.set(style="whitegrid")
for col in numerical_cols:
    plt.figure(figsize=(14, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x=col, bins=30, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')

    # Boxplot (target 'y')
    plt.subplot(1, 2, 2)
    sns.boxplot(data=data, x='y', y=col, palette='Set2')
    plt.title(f'{col} vs Target (y)')

    plt.tight_layout()
    plt.show()


# HEATMAP

sns.heatmap(data[["age", "balance", "duration", "campaign", "pdays", "previous"]].corr(), annot=True, cmap="coolwarm");


# CATEGORICAL COLUMNS

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
sns.set(style="whitegrid")

for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x=col, hue='y', palette='Set2')
    plt.title(f'{col} distribution and target (y)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# TIME 

month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

plt.figure(figsize=(10, 5))
sns.countplot(data=data, x='month', hue='y', order=month_order, palette='Set2')
plt.title('Distribution of Target Variable (y) by Month')
plt.xlabel('Month')
plt.ylabel('Number of Records')
plt.legend(title='Target (y)', labels=['No (0)', 'Yes (1)'])
plt.tight_layout()
plt.show()


import matplotlib.patches as mpatches
custom_palette = {0: "#FF5733", 1: "#2E86DE"} # idk why but set2 wasnt working on legends so i gave it manually

plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='day', y='duration', hue='y', palette=custom_palette)

plt.title('Call Duration by Day and Target Variable (y)')
plt.xlabel('Day of Month')
plt.ylabel('Call Duration (seconds)')

# Manuel legend
red_patch = mpatches.Patch(color='#FF5733', label='No (0)')
blue_patch = mpatches.Patch(color='#2E86DE', label='Yes (1)')
plt.legend(title='Target (y)', handles=[red_patch, blue_patch])

plt.tight_layout()
plt.show()


def preprocess_data(data):
    data = data.copy()

    
    # it was default -1 so i changed it to 0 to see better describe()
    data["pdays"] = data["pdays"].replace(-1,0)
    
    # check .descrbie(), there were too much different between %75 and max, so i clipped it a a bit
    for col in ['campaign', 'previous', 'duration']:
        q99 = data[col].quantile(0.99)
        data[col] = data[col].clip(upper=q99)

    # onehot
    data = pd.get_dummies(data, columns=['job', 'marital', 'poutcome', 'contact'], drop_first=True)

    
    # i changed unknows with mode so labeling could be better
    mode_edu = data['education'].mode()[0]
    data['education'] = data['education'].replace('unknown', mode_edu)
    le_edu = LabelEncoder()
    data['education'] = le_edu.fit_transform(data['education'])

    
    # binary encoding, there were no null values so its safe to use map here
    data['housing'] = data['housing'].map({'yes': 1, 'no': 0})
    data['loan'] = data['loan'].map({'yes': 1, 'no': 0})
    data['default'] = data['default'].map({'yes': 1, 'no': 0})

    
    # this might look like a bit complicated but, months are not labeled if you think of it, its more like circular(1,2...,11,12,1,2...)
    month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    data['month_num'] = data['month'].map(month_order)
    data['month_sin'] = np.sin(2 * np.pi * data['month_num'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month_num'] / 12)
    data.drop(['month', 'month_num'], axis=1, inplace=True)
    
    # encodes returned bool so we change their type
    data = data.astype({col: int for col in data.select_dtypes(include='bool').columns})

    # FEATURE ENGINEERING
    ###

    
    return data


df = preprocess_data(df)


df.head()


X = df.drop(["id","y"],axis = 1)
y = df["y"]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42,stratify=y)


xgb_model = XGBClassifier(
    colsample_bytree=0.7,
    gamma=0.3,
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    subsample=1,
    use_label_encoder=False,  
    eval_metric='logloss'     
)
lgbm_model = LGBMClassifier(
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=20,
    n_estimators=300,
    num_leaves=50,
    verbose=-1,
    subsample=1.0,
)

ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft' 
)

ensemble_model.fit(X_train, y_train)


y_proba = ensemble_model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_auc:.4f}")


test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


test.head()


test_final = preprocess_data(test.copy())


test_final.head()


ids = test_final["id"]
test_final = test_final.drop(["id"],axis = 1)


test_preds = ensemble_model.predict_proba(test_final)[:, 1]


submission_df = pd.DataFrame({
    'id': ids,
    'prediction': test_preds
})


submission_df.to_csv('/kaggle/working/submission.csv', index=False)

