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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter("ignore" , category = FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
print(f"Train set shape: {train.shape}")
print(f"Test set shape: {test.shape}")


train.head(3)


train.isna().sum()


test.isna().sum()


numerical_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
                 "Friends_circle_size" , "Post_frequency"]
categorical_cols = ["Stage_fear", "Drained_after_socializing"]


plt.figure(figsize=(4,4))

colors = sns.color_palette("Set2" , n_colors = 2)

train["Personality"].value_counts().plot.pie(
    autopct = "%1.1f",
    colors = colors
).set_title("Personality (train)")


fig, ax = plt.subplots(5,1,figsize=(10, 15))

for i, col in enumerate(numerical_cols):
    sns.countplot(x=col, data=train, ax=ax[i], palette="Set2", hue="Personality")
    ax[i].set_title(f"Countplot of {col}", fontsize=12)
    ax[i].tick_params(axis="x" , rotation=45)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(1,2 ,figsize=(10, 4))

for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, data=train, ax=ax[i], palette="Set2", hue="Personality")
    ax[i].set_title(f"Countplot of {col}", fontsize=12)
    ax[i].tick_params(axis="x" , rotation=45)

plt.tight_layout()
plt.show()


train_encoded = train.copy()
for col in categorical_cols:
    train_encoded[col] = train_encoded[col].map({"No": 0, "Yes": 1})
    
train_encoded["Personality"] = train_encoded["Personality"].map({"Introvert": 0, "Extrovert": 1})

plt.figure(figsize=(10,8))
sns.heatmap(train_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


y = train["Personality"].copy()
X = train.drop("Personality", axis=1).copy()

data = pd.concat([X, test], axis=0).reset_index(drop = True).reset_index(drop=True)


na_cols = data.columns[data.isna().any()].tolist()

mv = pd.DataFrame(data[na_cols].isna().sum() , columns = ["Number_missing"])
mv["Percentage_missing(%)"] = np.round(100*mv["Number_missing"]/len(data) , 2)
print(mv)


plt.figure(figsize=(15,4))
sns.heatmap(data[na_cols].isna().T,
           cmap = "YlOrRd",
           cbar = False,
           xticklabels = False)
plt.title("Missing Values Heatmap")
plt.xlabel("id")
plt.ylabel("Features with Missing Data")


train["na_count"] = train.isna().sum(axis=1)
plt.figure(figsize=(10,4))
sns.countplot(data=train, x="na_count", hue="Personality", palette="Set2")
plt.title("Number of missing entries by id")
train.drop("na_count" , axis=1, inplace=True)


data_encoded = data.copy()

for col in categorical_cols:
    data_encoded[col] = data_encoded[col].map({"No": 0, "Yes": 1})

fig, ax = plt.subplots(5,1,figsize=(6,15))

for i, col in enumerate(numerical_cols):
    sns.violinplot(
        data=data,
        x="Stage_fear",
        y=col,
        palette="Set2",
        inner="box" ,
        ax=ax[i],
        bw=0.4
    )
    plt.title(f"{col} by Stage Fear (Violin Plot)")
    plt.xlabel("Stage Fear")
    plt.ylabel(f"{col}")

plt.tight_layout()
plt.show()


data_notnull = data_encoded[data_encoded['Stage_fear'].notnull()]
data_null = data_encoded[data_encoded['Stage_fear'].isnull()]

features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency',
    'Drained_after_socializing'
]

X_train = data_notnull[features]
y_train = data_notnull['Stage_fear']
X_pred = data_null[features]


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_stage_fear = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Stage_fear'].isnull(), 'Stage_fear'] = pred_stage_fear


data_encoded.isnull().sum()


data_notnull = data_encoded[data_encoded['Time_spent_Alone'].notnull()]
data_null = data_encoded[data_encoded['Time_spent_Alone'].isnull()]

features = [
    'Stage_fear',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency',
    'Drained_after_socializing'
]

X_train = data_notnull[features]
y_train = data_notnull['Time_spent_Alone']
X_pred = data_null[features]


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_Time_spent_Alone = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Time_spent_Alone'].isnull(), 'Time_spent_Alone'] = pred_Time_spent_Alone

data_encoded.isnull().sum()


data_notnull = data_encoded[data_encoded['Social_event_attendance'].notnull()]
data_null = data_encoded[data_encoded['Social_event_attendance'].isnull()]

features = [
    'Stage_fear',
    'Time_spent_Alone',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency',
    'Drained_after_socializing'
]

X_train = data_notnull[features]
y_train = data_notnull['Social_event_attendance']
X_pred = data_null[features]


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_Social_event_attendance = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Social_event_attendance'].isnull(), 'Social_event_attendance'] \
= pred_Social_event_attendance

data_encoded.isnull().sum()


data_notnull = data_encoded[data_encoded['Going_outside'].notnull()]
data_null = data_encoded[data_encoded['Going_outside'].isnull()]

features = [
    'Stage_fear',
    'Time_spent_Alone',
    'Social_event_attendance',
    'Friends_circle_size',
    'Post_frequency',
    'Drained_after_socializing'
]

X_train = data_notnull[features]
y_train = data_notnull['Going_outside']
X_pred = data_null[features]


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_Going_outside = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Going_outside'].isnull(), 'Going_outside'] \
= pred_Going_outside

data_encoded.isnull().sum()


data_notnull = data_encoded[data_encoded['Drained_after_socializing'].notnull()]
data_null = data_encoded[data_encoded['Drained_after_socializing'].isnull()]

features = [
    'Stage_fear',
    'Time_spent_Alone',
    'Social_event_attendance',
    'Friends_circle_size',
    'Post_frequency',
    'Going_outside'
]

X_train = data_notnull[features]
y_train = data_notnull['Drained_after_socializing']
X_pred = data_null[features]


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_Drained_after_socializing = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Drained_after_socializing'].isnull(), 'Drained_after_socializing'] \
= pred_Drained_after_socializing

data_encoded.isnull().sum()


data_notnull = data_encoded[data_encoded['Friends_circle_size'].notnull()]
data_null = data_encoded[data_encoded['Friends_circle_size'].isnull()]

features = [
    'Stage_fear',
    'Time_spent_Alone',
    'Social_event_attendance',
    'Drained_after_socializing',
    'Post_frequency',
    'Going_outside'
]

X_train = data_notnull[features]
y_train = data_notnull['Friends_circle_size']
X_pred = data_null[features]


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_Friends_circle_size = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Friends_circle_size'].isnull(), 'Friends_circle_size'] \
= pred_Friends_circle_size

data_encoded.isnull().sum()


data_notnull = data_encoded[data_encoded['Post_frequency'].notnull()]
data_null = data_encoded[data_encoded['Post_frequency'].isnull()]

features = [
    'Stage_fear',
    'Time_spent_Alone',
    'Social_event_attendance',
    'Drained_after_socializing',
    'Friends_circle_size',
    'Going_outside'
]

X_train = data_notnull[features]
y_train = data_notnull['Post_frequency']
X_pred = data_null[features]


imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_pred_imp = imputer.transform(X_pred)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_imp, y_train)
pred_Post_frequency = model.predict(X_pred_imp)

data_encoded.loc[data_encoded['Post_frequency'].isnull(), 'Post_frequency'] \
= pred_Post_frequency

data_encoded.isnull().sum()


data = data_encoded.copy()

X=data[data["id"].isin(train["id"].values)].copy()
X_test = data[data["id"].isin(test["id"].values)].copy()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.8, random_state=0)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

classifiers = {
    "LogisticRegression" : LogisticRegression(random_state = 0),
    "RandomForest" : RandomForestClassifier(random_state = 0),
    "GradientBoosting" : GradientBoostingClassifier(random_state = 0),
    "LightGBM" : LGBMClassifier(random_state = 0, verbose=-1),
    "XGBoost" : XGBClassifier(random_state = 0, use_label_encoder=False, eval_metric="logloss")
}

LR_grid = {"penalty" : ['l1', 'l2'],
          'C' : [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2],
          'max_iter' : [50, 100, 150, 200],
          'solver' : ['liblinear']}

RF_grid = {'n_estimators' : [50, 100, 150, 200, 250, 300],
           'max_depth' : [4,6,8,10,12]}

GB_grid = {'n_estimators' : [50, 100, 150],
          'learning_rate' : [0.05, 0.1, 0.2],
           'max_depth' : [3, 5, 7] }

LGB_grid = {'n_estimators' : [100, 200],
           'learning_rate' : [0.05, 0.1],
           'max_depth' : [3, 5, 7],
           'num_leaves' : [15, 31, 63]},

XGB_grid = {'n_estimators' : [100, 200],
           'max_depth' : [3, 5, 7],
           'learning_rate' : [0.05, 0.1, 0.2],
           'subsample' : [0.8, 1.0],
           'colsample_bytree' : [0.8, 1.0]}

grid = {
    "LogisticRegression" : LR_grid,
    "RandomForest" : RF_grid,
    "GradientBoosting" : GB_grid,
    "LightGBM" : LGB_grid,
    "XGBoost" : XGB_grid
}


from sklearn.model_selection import GridSearchCV
import time

y_train = y_train.map({"Introvert": 0, "Extrovert": 1})
y_valid = y_valid.map({"Introvert": 0, "Extrovert": 1})

valid_scores = []
best_estimators = {}

for i, (name, clf) in enumerate(classifiers.items()):
    print(f'Tuning model: {name}')
    start = time.time()

    grid_search = GridSearchCV(estimator = clf,
                              param_grid = grid[name],
                              n_jobs = -1,
                               cv = 5
                              )
    grid_search.fit(X_train, y_train)
    end = time.time()

    score = grid_search.score(X_valid, y_valid)
    duration = round((end-start)/60, 2)
    best_params = grid_search.best_params_

    valid_scores.append({
        "Classifier": name,
        "Validation accuracy": score,
        "Training time (min)" : duration
    })
    best_estimators[name] = grid_search.best_estimator_

    print(f"Done: {name}")
    print(f"Best params: {best_params}")
    print(f"Validation accuracy: {score: .4f}")
    print(f"Time: {duration} min\n")

valid_scores_df = pd.DataFrame(valid_scores)


valid_scores


best_estimators


from sklearn.ensemble import VotingClassifier

model1 = GradientBoostingClassifier(learning_rate=0.05, min_samples_leaf=2,
                           n_estimators=250, random_state=0, subsample=0.8)

model2 = LGBMClassifier(learning_rate=0.05, max_depth=3, num_leaves=15, random_state=0, verbose=-1)

model3 = XGBClassifier(colsample_bytree = 0.8, 
                       learning_rate = 0.05, 
                       max_depth = 3, 
                       n_estimators = 100, 
                       subsample = 1.0 )
    
voting_clf = VotingClassifier(estimators = [
    ('gb', model1),
    ('lgbm', model2),
    ('xgb', model3)
], voting='soft')

voting_clf.fit(X_train, y_train)

val_acc = voting_clf.score(X_valid, y_valid)
print(f"VotingClassifier Validation Accuracy: {val_acc:.6f}")


#from sklearn.model_selection import RandomizedSearchCV
#import time

#GB_param_dist = {
#    'n_estimators': [100, 150, 200, 250, 300],           # 少し上限上げる
#    'learning_rate': [0.01, 0.03, 0.05, 0.075, 0.1, 0.15],
#    'max_depth': [3, 4, 5, 6, 7, 8],                     # 7までから8までに拡大
#    'subsample': [0.6, 0.8, 1.0],                       
#    'min_samples_split': [2, 5, 10],                     
#    'min_samples_leaf': [1, 2, 4]
#}

#gb_model = GradientBoostingClassifier(random_state=0)

#random_search = RandomizedSearchCV(
#    estimator=gb_model,
#    param_distributions=GB_param_dist,
#    n_iter=100,                # 50から100に増やす
#    cv=5,
#    scoring="accuracy",
#    n_jobs=-1,
#    random_state=0,
#    verbose=2                  # 詳細出力で進行状況わかりやすく
#)

#start = time.time()
#random_search.fit(X_train, y_train)
#end = time.time()

#best_model = random_search.best_estimator_
#val_score = best_model.score(X_valid, y_valid)

#print("\nBest Parameters:")
#print(random_search.best_params_)
#print(f"\nValidation Accuracy: {val_score:.4f}")
#print(f"Training Time: {round((end - start) / 60, 2)} min")



#best_model


clf = voting_clf
clf.fit(X, y)


y_test_preds = clf.predict(X_test)


submission = pd.DataFrame({
    'id': test['id'],
    'Personality': y_test_preds  
})


submission.to_csv('submission.csv', index=False)







