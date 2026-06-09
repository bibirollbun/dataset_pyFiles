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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.info()


test_df.info()


train_df.duplicated().sum()


test_df.duplicated().sum()


train_df.columns


train_df.sample(5)


fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 

# Plot 1
sns.histplot(data=train_df, x='Stage_fear', ax=axes[0], kde=True, color='skyblue')
axes[0].set_title('Stage Fear')

# Plot 2
sns.histplot(data=train_df, x='Drained_after_socializing', ax=axes[1], kde=True, color='salmon')
axes[1].set_title('Drained After Socializing')

# Plot 3
sns.histplot(data=train_df, x='Personality', ax=axes[2], kde=True, color='lightgreen')
axes[2].set_title('Personality')

plt.tight_layout()
plt.show()


train_df.drop(columns = ['id'],inplace = True)
id = test_df['id'].copy()
test_df.drop(columns = ['id'],inplace = True)


#Handling missing Values
for col in train_df.select_dtypes('object').columns:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    
    

for col in train_df.select_dtypes(include = ['int','float']).columns:
    train_df[col] = train_df[col].fillna(train_df[col].median())


for col in test_df.select_dtypes('object').columns:
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])
    
    

for col in test_df.select_dtypes(include = ['int','float']).columns:
    test_df[col] = test_df[col].fillna(test_df[col].median())
    


train_df.info()


test_df.info()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df['Stage_fear'] = le.fit_transform(train_df['Stage_fear'])
train_df['Drained_after_socializing'] = le.fit_transform(train_df['Drained_after_socializing'])

le2 = LabelEncoder()
train_df['Personality'] = le2.fit_transform(train_df['Personality'])


#Feature Engineering

# Alone × Drained → Introversion proxy
train_df['Alone_Drained'] = train_df['Time_spent_Alone'] * train_df['Drained_after_socializing']

# Stage Fear × Social Events → Social anxiety score
train_df['Fear_vs_Event'] = train_df['Stage_fear'] * (1 / (train_df['Social_event_attendance'] + 1))

# Going out × Friends Circle → social activity index
train_df['Social_activity'] = train_df['Going_outside'] * train_df['Friends_circle_size']


# Time spent alone bucketed
train_df['Alone_high'] = (train_df['Time_spent_Alone'] > train_df['Time_spent_Alone'].median()).astype(int)

# Low vs high posting frequency
train_df['Post_active'] = (train_df['Post_frequency'] > 3).astype(int)

# Has large friend circle
train_df['Extrovert_friend_circle'] = (train_df['Friends_circle_size'] > 5).astype(int)



train_df.info()


for col in test_df.select_dtypes('object').columns:
    test_df[col] = le.fit_transform(test_df[col])


# Alone × Drained → Introversion proxy
test_df['Alone_Drained'] = test_df['Time_spent_Alone'] * test_df['Drained_after_socializing']

# Stage Fear × Social Events → Social anxiety score
test_df['Fear_vs_Event'] = test_df['Stage_fear'] * (1 / (test_df['Social_event_attendance'] + 1))

# Going out × Friends Circle → social activity index
test_df['Social_activity'] = test_df['Going_outside'] * test_df['Friends_circle_size']


# Time spent alone bucketed
test_df['Alone_high'] = (test_df['Time_spent_Alone'] > test_df['Time_spent_Alone'].median()).astype(int)

# Low vs high posting frequency
test_df['Post_active'] = (test_df['Post_frequency'] > 3).astype(int)

# Has large friend circle
test_df['Extrovert_friend_circle'] = (test_df['Friends_circle_size'] > 5).astype(int)



plt.figure(figsize = (10,8))
sns.heatmap(train_df.corr(),annot = True)
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


X = train_df.drop(columns = ['Personality'])
y = train_df['Personality']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


x_train.shape


x_test.shape


import xgboost as xgb
xg = xgb.XGBClassifier(n_estimators = 90)
xg.fit(x_train,y_train)
y_pred = xg.predict(x_test)

accuracy_score(y_pred,y_test)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 300)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)

accuracy_score(y_test,y_pred)


from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300)
gb.fit(x_train,y_train)
y_pred = gb.predict(x_test)

accuracy_score(y_test,y_pred)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 1000)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

accuracy_score(y_test,y_pred)


from sklearn.ensemble import VotingClassifier,StackingClassifier,BaggingClassifier
vc = VotingClassifier(estimators = [('xgb',xg),('rf',rf),('gb',gb),('lr',lr)],voting = 'hard')

vc.fit(x_train,y_train)
y_pred = vc.predict(x_test)

accuracy_score(y_test,y_pred)


bg = BaggingClassifier(estimator= xg, n_estimators = 5,random_state = 42)
bg.fit(x_train,y_train)
bg.fit(x_train,y_train)
y_pred = bg.predict(x_test)

accuracy_score(y_test,y_pred)


!pip install pytorch-tabnet


import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier


# Split features and labels
X = train_df.drop(columns=['Personality'])
y = train_df['Personality']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define model
model = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5,
    gamma=1.5, lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    verbose=10
)

# Fit model — pass arrays, not keyword args
model.fit(
    x_train.values, y_train.values,
    eval_set=[(x_test.values, y_test.values)],
    eval_name=['val'],
    eval_metric=['accuracy'],
    max_epochs=100,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Predict and evaluate
y_pred = model.predict(x_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))



y_pred2 = model.predict(test_df.values)


submission = pd.DataFrame({
    'id': id,
    'personality': le2.inverse_transform(y_pred2)
})


submission.head()


submission.to_csv('submissiont2.csv', index = False)

