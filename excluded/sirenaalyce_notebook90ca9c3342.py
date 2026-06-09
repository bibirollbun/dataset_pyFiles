# conda install seaborn


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv') 
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')


train.head()


train['Stage_fear'].unique()


train['Drained_after_socializing'].unique()


train.describe()


train.info()


test.info()


train.isnull().sum()


test.isnull().sum()


train.isnull().sum()/train.shape[0]  * 100


original.isnull().sum()


original.isnull().sum()/original.shape[0] * 100


fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=train, x='Personality', ax=ax, palette='viridis', hue='Personality')
ax.set_title('Distribution of Personality Types in Training Set')


fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=original, x='Personality', ax=ax, palette='viridis', hue='Personality')
ax.set_title('Distribution of Personality Types in Original Set')


# # df['groupmean_sales'] = df.groupby('region')['sales'].transform(lambda x: x.fillna(x.mean()))
# # df

# stage_mode = train.groupby('Personality')['Stage_fear'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
# train['Stage_fear'].fillna(stage_mode, inplace=True)

# # test_stage_mode = train.groupby('Personality')['Stage_fear'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
# # test['Stage_fear'].fillna(test.mean(), inplace=True)
# test['Stage_fear'].fillna(test['Stage_fear'].mode()[0], inplace=True)

# o_stage_mode = original.groupby('Personality')['Stage_fear'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))   
# original['Stage_fear'].fillna(o_stage_mode, inplace=True)

# drained_mode = train.groupby('Personality')['Drained_after_socializing'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))   
# train['Drained_after_socializing'].fillna(drained_mode, inplace=True) 

# o_drained_mode = original.groupby('Personality')['Drained_after_socializing'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))  
# original['Drained_after_socializing'].fillna(o_drained_mode, inplace=True) 

# # test_drained_mode = train.groupby('Personality')['Drained_after_socializing'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
# test['Drained_after_socializing'].fillna(test['Stage_fear'].mode()[0], inplace=True)

# alone_mean = train.groupby('Personality')['Time_spent_Alone'].transform(lambda x: x.fillna(x.mean()))
# train['Time_spent_Alone'].fillna(alone_mean, inplace=True)

# original_alone_mean = original.groupby('Personality')['Time_spent_Alone'].transform(lambda x: x.fillna(x.mean()))
# original['Time_spent_Alone'].fillna(original_alone_mean, inplace=True) 

# # test_alone_mean = train.groupby('Personality')['Time_spent_Alone'].transform(lambda x: x.fillna(x.mean()))
# test['Time_spent_Alone'].fillna(test['Time_spent_Alone'].mean(), inplace=True)

# social_event_mean = train.groupby('Personality')['Social_event_attendance'].transform(lambda x: x.fillna(x.mean()))
# train['Social_event_attendance'].fillna(social_event_mean, inplace=True)

# original_social_event_mean = original.groupby('Personality')['Social_event_attendance'].transform(lambda x: x.fillna(x.mean()))
# original['Social_event_attendance'].fillna(original_social_event_mean, inplace=True)

# # test_social_event_mean = train.groupby('Personality')['Social_event_attendance'].transform(lambda x: x.fillna(x.mean()))
# test['Social_event_attendance'].fillna(test['Social_event_attendance'].mean(), inplace=True)

# outside_mean = train.groupby('Personality')['Going_outside'].transform(lambda x: x.fillna(x.mean()))
# train['Going_outside'].fillna(outside_mean, inplace=True)

# original_outside_mean = original.groupby('Personality')['Going_outside'].transform(lambda x: x.fillna(x.mean()))
# original['Going_outside'].fillna(original_outside_mean, inplace=True)   

# # test_outside_mean = train.groupby('Personality')['Going_outside'].transform(lambda x: x.fillna(x.mean()))
# test['Going_outside'].fillna(test['Going_outside'].mean(), inplace=True)

# friends_mean = train.groupby('Personality')['Friends_circle_size'].transform(lambda x: x.fillna(x.mean()))
# train['Friends_circle_size'].fillna(friends_mean, inplace=True)

# original_friends_mean = original.groupby('Personality')['Friends_circle_size'].transform(lambda x: x.fillna(x.mean()))
# original['Friends_circle_size'].fillna(original_friends_mean, inplace=True)

# # test_friends_mean = train.groupby('Personality')['Friends_circle_size'].transform(lambda x: x.fillna(x.mean()))
# test['Friends_circle_size'].fillna(test['Friends_circle_size'].mean(), inplace=True)

# posts_mean = train.groupby('Personality')['Post_frequency'].transform(lambda x: x.fillna(x.mean()))
# train['Post_frequency'].fillna(posts_mean, inplace=True)

# original_posts_mean = original.groupby('Personality')['Post_frequency'].transform(lambda x: x.fillna(x.mean()))
# original['Post_frequency'].fillna(original_posts_mean, inplace=True)

# # test_posts_mean = train.groupby('Personality')['Post_frequency'].transform(lambda x: x.fillna(x.mean()))
# test['Post_frequency'].fillna(test['Post_frequency'].mean(), inplace=True)


import seaborn.objects as so


(
    so.Plot(train, x="Personality", color="Stage_fear")
    .add(so.Bar(), so.Count(), so.Stack())
).show()




(
    so.Plot(train, x="Personality", color="Drained_after_socializing")
    .add(so.Bar(), so.Count(), so.Stack())
).show()


le = LabelEncoder()
les = LabelEncoder()
led = LabelEncoder()

train['Stage_fear'] = les.fit_transform(train['Stage_fear'])
train['Drained_after_socializing'] = led.fit_transform(train['Drained_after_socializing'])
train['Personality'] = le.fit_transform(train['Personality'])

original['Stage_fear'] = les.fit_transform(original['Stage_fear'])
original['Drained_after_socializing'] = led.fit_transform(original['Drained_after_socializing'])
original['Personality'] = le.fit_transform(original['Personality'])

test['Stage_fear'] = les.fit_transform(test['Stage_fear'])
test['Drained_after_socializing'] = led.fit_transform(test['Drained_after_socializing'])
# test['Personality'] = le.fit_transform(test['Personality'])


train.head()


test.info()


X = train.drop(columns=['id','Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

# model = LogisticRegression(solver='liblinear', max_iter=1000)

# model = XGBClassifier()

model = LGBMClassifier()

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print(f"Cross-validated accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f}")



model.fit(X, y)


y_pred = model.predict(X_test)


decoded  = le.inverse_transform(y_pred)
decoded


submission = test[['id']]
submission['Personality'] = decoded
submission.info()


test.info()


submission.head()


submission.info()


submission.to_csv('submission.csv', index=False)

