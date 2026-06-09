import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df.head()


df.drop('id', inplace=True, axis=1)
df.shape


df.info()


df.describe()


plt.figure(figsize=(6, 6))
temp = df["Personality"].value_counts()
plt.pie(
    temp, labels=temp.index,
    autopct="%.0f%%",
    explode=[0.03, 0.03],
)
plt.title("Target Variable Distribution", fontsize=14)
plt.show()


df = pd.concat([df, df[df['Personality']=='Introvert']], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.shape


plt.figure(figsize=(6, 6))
temp = df["Personality"].value_counts()
plt.pie(
    temp, labels=temp.index,
    autopct="%.0f%%",
    explode=[0.03, 0.03],
)
plt.title("Target Variable Distribution", fontsize=14)
plt.show()


cat_cols = df.columns[:-1]
cat_cols


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))

axes = axes.flatten()
for ax, col in zip(axes, cat_cols):
    df[col]=df[col].round()
    sb.countplot(data=df, x=col, ax=ax, hue='Personality')
    ax.set_title(col)

plt.tight_layout()
plt.show()


df.isnull().sum()


df['null_count'] = df.isnull().sum(axis=1)
df[df['null_count']>0].head()


df['null_count'].unique()


def fill_null(df):
    for idx, row in tqdm(df.iterrows()):
        
        if row.isnull().sum()==0: continue
        
        poss = []
        if row.Time_spent_Alone>4: poss.append(1)
        if row.Stage_fear=='Yes': poss.append(1)
        if row.Social_event_attendance<4: poss.append(1)
        if row.Going_outside<3: poss.append(1)
        if row.Drained_after_socializing=='Yes': poss.append(1)
        if row.Friends_circle_size<6: poss.append(1)
        if row.Post_frequency<3: poss.append(1)

        cnt=row.isnull().sum()
    
        pred='Extrovert'
        if sum(poss)>cnt//2: pred='Introvert'
    
        if pred=='Extrovert':
            if pd.isnull(row.Time_spent_Alone): df.loc[idx, 'Time_spent_Alone']= 4
            if pd.isnull(row.Stage_fear): df.loc[idx, 'Stage_fear']='No'
            if pd.isnull(row.Social_event_attendance): df.loc[idx, 'Social_event_attendance']=4
            if pd.isnull(row.Going_outside): df.loc[idx, 'Going_outside']=3
            if pd.isnull(row.Drained_after_socializing): df.loc[idx, 'Drained_after_socializing']='No'
            if pd.isnull(row.Friends_circle_size): df.loc[idx, 'Friends_circle_size']=6
            if pd.isnull(row.Post_frequency): df.loc[idx, 'Post_frequency']=3
        else:
            if pd.isnull(row.Time_spent_Alone): df.loc[idx, 'Time_spent_Alone']= 3
            if pd.isnull(row.Stage_fear): df.loc[idx, 'Stage_fear']='Yes'
            if pd.isnull(row.Social_event_attendance): df.loc[idx, 'Social_event_attendance']=3
            if pd.isnull(row.Going_outside): df.loc[idx, 'Going_outside']=2
            if pd.isnull(row.Drained_after_socializing): df.loc[idx, 'Drained_after_socializing']='Yes'
            if pd.isnull(row.Friends_circle_size): df.loc[idx, 'Friends_circle_size']=5
            if pd.isnull(row.Post_frequency): df.loc[idx, 'Post_frequency']=2
    
    return df
df = fill_null(df)


df = df.replace({"Introvert": 1, "Extrovert": 0})
df = df.replace({"Yes": 0, "No": 1})
df.head()


sb.heatmap(df.corr()>0.8, annot=True, cbar=False)
plt.show()


from sklearn.model_selection import train_test_split

X = df.drop('Personality', axis=1)
y = df['Personality']
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()


models = [RandomForestClassifier(), XGBClassifier(), \
          CatBoostClassifier(verbose=False), LogisticRegression(), \
          LGBMClassifier(objective='binary', metric='binary_error', )]

for i in range(len(models)):   
    
    models[i].fit(X_tr, y_tr)
    y_tr_pred = models[i].predict(X_tr)
    y_val_pred = models[i].predict(X_val)

    print(models[i])
    print("Train Accuracy:", accuracy_score(y_tr, y_tr_pred))
    print("Val Accuracy:", accuracy_score(y_val, y_val_pred))
    plot_confusion(y_val, y_val_pred, "Validation Confusion Matrix")
    
    print()


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.head()


test.drop('id', axis=1, inplace=True)
test['null_count'] = test.isnull().sum(axis=1)
test = fill_null(test)
test.replace({"Yes": 0, "No": 1}, inplace=True)
test.head()


ss = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
ss['Personality'] = models[0].predict(test)
ss.replace({1: "Introvert", 0: "Extrovert"}, inplace=True)
ss.head()


ss.to_csv('Submission.csv', index=False)


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test.drop('id', axis=1, inplace=True)
test.head()


preds = []
for idx, row in test.iterrows():
    poss = []
    if row.Time_spent_Alone>4: poss.append(1)
    if row.Stage_fear=='Yes': poss.append(1)
    if row.Social_event_attendance<4: poss.append(1)
    if row.Going_outside<3: poss.append(1)
    if row.Drained_after_socializing=='Yes': poss.append(1)
    if row.Friends_circle_size<6: poss.append(1)
    if row.Post_frequency<3: poss.append(1)

    cnt=row.isnull().sum()
    if sum(poss)>cnt//2: preds.append('Introvert')
    else: preds.append('Extrovert')


ss1 = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
ss1['Personality'] = preds
ss1.to_csv('Submission_noml.csv', index=False)
ss1.head()

