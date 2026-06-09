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


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.head()


test.head()


train.isnull().sum()


train.shape


test.isnull().sum()


print("\nDataset Information:")
print(train.info())
print("\nStatistical Summary:")
display(train.describe().T)


from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# Encode input features
feature_cols = ['Stage_fear', 'Drained_after_socializing']
ord_enc = OrdinalEncoder()
train[feature_cols] = ord_enc.fit_transform(train[feature_cols].astype(str))
test[feature_cols] = ord_enc.transform(test[feature_cols].astype(str))
# Encode target separately
le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'].astype(str))

print(le.classes_)  # Shows the mapping (e.g., ['Extrovert', 'Introvert'])



from sklearn.experimental import enable_iterative_imputer  # Required first
from sklearn.impute import IterativeImputer



original_columns = train.columns
target_col = 'Personality'



X_train = train.drop(columns=[target_col])
y_train = train[target_col]



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()

# Fit only on X_train
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(test)  # 


X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns)
X_test = pd.DataFrame(X_test_imputed, columns=X_train.columns)

train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)



train.isnull().sum()


X_test.isnull().sum()


X_test.shape


test.shape


import matplotlib.pyplot as plt
import seaborn as sns


col=['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency','Personality']

for i in col:
    plt.figure(figsize=(8, 4))  # Optional: better layout
    sns.histplot(train[i], kde=True, binwidth=1, color='skyblue', edgecolor='black')
    plt.title(f"Distribution of {i} (with KDE)", fontsize=14)
    plt.xlabel(i.replace("_", " ").title())
    plt.ylabel("Number of Individuals")
    plt.tight_layout()
    plt.show()



num_cols = train.select_dtypes(include='number').columns

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=train[col], color='lightgreen')
    plt.title(f"Boxplot of {col.replace('_', ' ').title()}", fontsize=13)
    plt.ylabel(col.replace('_', ' ').title())
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=train[col], palette='pastel')
    plt.title(f"Countplot of {col.replace('_', ' ').title()}")
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()



sns.heatmap(train.corr(), annot=True, cmap='coolwarm') 
plt.title("Correlation Matrix") 
plt.show() 



sns.countplot(x='Stage_fear', hue='Personality', data=train)
plt.title("Stage Fear Distribution by Personality")
plt.show()



sns.countplot(x='Drained_after_socializing', hue='Personality', data=train)
plt.title("Drained_after_socializing Distribution by Personality")
plt.show()






X = train.drop(['id', 'Personality'],axis = 1)
y = train['Personality']


from sklearn.ensemble import BaggingClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier




# Set class weight scale
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]

# Base models
xgb = XGBClassifier(
    max_depth=4, learning_rate=0.01, n_estimators=1000,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

cat = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.1,
    class_weights=[scale_pos_weight, 1], random_seed=42, verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31, learning_rate=0.1, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8,
    class_weight={0: scale_pos_weight, 1: 1}, random_state=42
)

rf = RandomForestClassifier(n_estimators=344, max_depth=11, max_features=None,
    min_samples_split=11, min_samples_leaf=1, random_state=42, n_jobs=-1 )



from sklearn.ensemble import BaggingClassifier

bag_xgb = BaggingClassifier(base_estimator=xgb, n_estimators=10, random_state=42, n_jobs=-1)
bag_cat = BaggingClassifier(base_estimator=cat, n_estimators=10, random_state=42, n_jobs=-1)
bag_lgbm = BaggingClassifier(base_estimator=lgbm, n_estimators=10, random_state=42, n_jobs=-1)
bag_rf = BaggingClassifier(base_estimator=rf, n_estimators=10, random_state=42, n_jobs=-1)



from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', bag_xgb),
        ('cat', bag_cat),
        ('lgbm', bag_lgbm),
        ('rf', bag_rf)
    ],
    voting='soft',  # for probability-based voting
    n_jobs=-1
)



# rfc.fit(X,y)



voting_clf.fit(X,y)


X_test = X_test.drop('id', axis=1)


y_pred = voting_clf.predict(X_test)


# Assuming your predictions are in y_pred
label_map = {
    0: 'Extrovert',
    1: 'Introvert'
}

final_predictions = [label_map[pred] for pred in y_pred]



submission = pd.DataFrame({
    'id': test['id'],                # or sample_submission['id']
    'Personality': final_predictions
})

submission.to_csv("submission.csv", index=False)





