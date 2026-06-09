%%time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from collections import Counter
from tqdm.notebook import tqdm as tqdm2
import seaborn as sns
import matplotlib

matplotlib.style.use('tableau-colorblind10')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%%time

# reading the contents of the competitions' datasets

train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col = 'id')
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col = 'id')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# printing the dataframe
train


%%time

test


%%time

sample_submission


%%time

original1 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv") .rename(columns={'Personality': 'match_p'}).drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'])

original1


%%time

original2 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")

original2


%%time

def summary(df):
    
    #to retrieve the datatype of each column
    summary_df = pd.DataFrame(df.dtypes, columns=['dtypes'])

     #to get number of rows with data for each column
    summary_df['Count'] = df.count().values
    
    #to get the number of missing rows for each column
    summary_df['Nbr_Missing'] = df.isna().sum()
    
    #to get the percentage of missing data of each column
    summary_df['%_Missing'] = (df.isna().sum())/len(df)
       
    #to get the number of unique values in the column
    summary_df['Unique_Values'] = df.nunique().values
    
    return summary_df

summary(train).style.background_gradient(cmap='Blues')


%%time

summary(test).style.background_gradient(cmap='Blues')


%%time

summary(original1).style.background_gradient(cmap='Blues')


%%time

summary(original2).style.background_gradient(cmap='Blues')


%%time

import matplotlib.pyplot as plt
import math

#univariate distribution of the features.
col = ["Time_spent_Alone", "Stage_fear", "Social_event_attendance", "Going_outside", "Drained_after_socializing", "Friends_circle_size", "Post_frequency"]
num_col = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]
cat_col = ["Stage_fear", "Drained_after_socializing", "match_p"]

n_cols = 3
n_num_col = len(num_col)
n_cat_col = len(cat_col)
total_plots = n_num_col + n_cat_col

n_rows = math.ceil(total_plots / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))  
axes = axes.flatten()
colors = sns.color_palette("colorblind", len(num_col))

for i in range(n_num_col):
    sns.histplot(train[num_col[i]], kde=True, ax=axes[i], color=colors[i])
    axes[i].set_ylabel("")

palette = sns.color_palette("colorblind", len(train['Stage_fear'].dropna().unique()))
sns.countplot(x='Stage_fear', data=train, ax=axes[n_num_col], palette=palette, hue='Stage_fear')
axes[n_num_col].set_ylabel("")
axes[n_num_col].tick_params(axis='x', rotation=45)

palette2 = sns.color_palette("colorblind", len(train['Drained_after_socializing'].dropna().unique()))
sns.countplot(x='Drained_after_socializing', data=train, ax=axes[n_num_col +1], palette=palette2, hue='Drained_after_socializing')
axes[n_num_col + 1].set_ylabel("")
axes[n_num_col + 1].tick_params(axis='x', rotation=45)

for j in range(total_plots + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


%%time

for col in ('Stage_fear', 'Drained_after_socializing'):
    print(original2[col].value_counts(normalize=True))
    print("")
    print(20*'-')
    print("")
    print(train[col].value_counts(normalize=True))
    print("")
    print(20*'-')


%%time

name_counts = train['Personality'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(name_counts, labels=[f'{name} ({count})' for name, count in name_counts.items()], autopct='%1.1f%%')
plt.title('Distribution of Target Variable : Personality')
plt.axis('equal')
plt.show()



%%time

print(original2['Personality'].value_counts(normalize=True))
print("")
print(20*'-')
print("")
print(train['Personality'].value_counts(normalize=True))
print("")
print(20*'-')


%%time

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder

ss = StandardScaler()
rs = RobustScaler()
oe = OrdinalEncoder()
le = LabelEncoder()
mms = MinMaxScaler()
mms2 = MinMaxScaler()

train = train.merge(original1, how='left')
test = test.merge(original1, how='left')

train


%%time

train["Stage_fear"] = train["Stage_fear"].map({"No": 0, "Yes": 1})
train["Drained_after_socializing"] = train["Drained_after_socializing"].map({"No": 0, "Yes": 1})

test["Stage_fear"] = test["Stage_fear"].map({"No": 0, "Yes": 1})
test["Drained_after_socializing"] = test["Drained_after_socializing"].map({"No": 0, "Yes": 1})

# Impute numerical columns
for col in num_col:
    train[col] = train[col].fillna(train[col].mean()).astype("category")
    test[col] = test[col].fillna(test[col].mean()).astype("category")

# Impute categorical columns
for col in cat_col:
    train[col] = train[col].fillna(-1).astype("category")
    test[col] = test[col].fillna(-1).astype("category")
    
train


%%time

X = train.drop(['Personality', ], axis=1) 

X


%%time

y = train['Personality']

y_encoded = y.map({"Extrovert": 0, "Introvert": 1})

y_encoded


%%time

import datetime

def print_ts(message):
    now = datetime.datetime.now()
    timestamp = now.strftime("%H:%M:%S")
    print(f"{timestamp} {message}")
    


%%time

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

mean_acs = []

params = {'learning_rate': 0.04273291169377194, 
          'gamma': 0.037579100594478866, 
          'max_depth': 10, 
          'subsample': 0.8470097534510113, 
          'colsample_bytree': 0.553823331416667, 
          'colsample_bylevel': 0.8119082515452634, 
          'colsample_bynode': 0.6308173677360011, 
          'max_delta_step': 1, 
          'reg_alpha': 2.5297942600804135, 
          'reg_lambda': 4.64156378717834, 
          'n_estimators': 3891, 
          'random_state': 26,  
          'n_jobs': -1,
          'verbosity': 0,
          'objective': 'binary:logistic',  
          'eval_metric': 'logloss',  # Evaluation metric
          'enable_categorical': True,
          'missing': np.inf
          }

i=1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=53)

for train_index,test_index in tqdm2(skf.split(X, y_encoded), desc='Stratified Cross-Validation on XGBClassifier'):
    print('\n{} of kfold {}'.format(i,skf.n_splits))
    print("")
    print_ts(" *** KFold split into training and validation sets *** ")
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    print("   X completed")
    ytr,yvl = y_encoded[train_index],y_encoded[test_index]
    print("   y completed")
   
    print(" *** Model Creation *** ")
    xgb_model = XGBClassifier(**params) 

    print(" *** Model Fitting *** ")
    xgb_model.fit(xtr, ytr)
    print(" *** Predicting class *** ")
    pred_test = xgb_model.predict(xvl)
    print(" *** Scoring Accuracy on validation set *** ")
    score = accuracy_score(yvl,pred_test)
    mean_acs.append(score)
    print("")
    print('Accuracy Score: ',score)
    print("")
    print_ts(60*'-')
    i+=1
    
print("\nMean validation Accuracy Score : ", sum(mean_acs)/len(mean_acs))
print("")



%%time

from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(9,5))
plot_importance(xgb_model, ax=ax)
plt.show()


%%time

from sklearn.metrics import make_scorer, classification_report, confusion_matrix, ConfusionMatrixDisplay

y_pred = xgb_model.predict(xvl)

cm = confusion_matrix(yvl, y_pred) #, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Row-wise)")
plt.show()


%%time

print(classification_report(yvl, y_pred))


%%time

import optuna
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, stratify=y, random_state=26)

# Optuna objective
def objective(trial):
    
    params = {
    'objective': 'binary:logistic',  # For multi-class classification
    'eval_metric': 'logloss',  # Evaluation metric
    'enable_categorical': True,
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06),
    "gamma": trial.suggest_float("gamma", 0.01, 0.30),
    "max_depth": trial.suggest_int("max_depth", 5, 10),
    "subsample": trial.suggest_float("subsample", 0.7, 0.9),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.7, 0.9),
    "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 0.8),
    "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
    "reg_alpha": trial.suggest_float("reg_alpha", 2.0, 6.0),
    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
    "n_estimators": trial.suggest_int("n_estimators", 500, 10000),
    "verbosity": 0,
    "random_state": trial.suggest_int("random_state", 1, 99),
    "missing": np.inf
    ##"early_stopping_rounds": 50,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              verbose=False)
    y_preds = model.predict(X_valid)
    score = accuracy_score(y_valid, y_preds)
    print("")
    print('Accuracy Score: ', score)
    print("")
    print_ts(60*'-')
    return score

# Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("âœ… Best Accuracy Score:", study.best_value)
print("ðŸ”§ Best params:", study.best_params)


#parameters: {'learning_rate': 0.02300298102005577, 'gamma': 0.2592116290004392, 'max_depth': 7, 'subsample': 0.7531685790459205, 'colsample_bytree': 0.8649179249332986, 'colsample_bylevel': 0.8534560246309426, 'colsample_bynode': 0.7248293934661602, 'max_delta_step': 3, 'reg_alpha': 4.276497985824039, 'reg_lambda': 2.897615815915758, 'n_estimators': 9289, 'random_state': 14}


%%time

best_params = study.best_params

best_params2 = {'learning_rate': 0.02300298102005577, 
                'gamma': 0.2592116290004392, 
                'max_depth': 7, 
                'subsample': 0.7531685790459205, 
                'colsample_bytree': 0.8649179249332986, 
                'colsample_bylevel': 0.8534560246309426, 
                'colsample_bynode': 0.7248293934661602, 
                'max_delta_step': 3, 
                'reg_alpha': 4.276497985824039, 
                'reg_lambda': 2.897615815915758, 
                'n_estimators': 9289, 
                'random_state': 14}

final_model = XGBClassifier(**best_params, missing=np.inf, enable_categorical=True,)

final_model.fit(X, y_encoded)


%%time

X_final = test.copy()

X_final


%%time

final_y = final_model.predict(X_final)

sample_submission['Personality'] = final_y

sample_submission['Personality'] = sample_submission['Personality'].map({0: "Extrovert", 1: "Introvert"})

sample_submission


%%time

#y_final = xgb_model.predict(X_final)

#sample_submission['Personality'] = y_final
#sample_submission['Personality'] = sample_submission['Personality'].map({0: "Extrovert", 1: "Introvert"}) #le.inverse_transform(y_final)

#sample_submission


%%time

sample_submission.to_csv('submission.csv', index=None)

