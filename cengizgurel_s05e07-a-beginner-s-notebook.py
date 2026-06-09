import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


#creating the df
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

#checking shape's
print(f'train_df shape: {train_df.shape}')
print(f'test_df shape: {test_df.shape}')


#viewing the training data
train_df.head()


#checking missing values
print(train_df.isna().sum())
print('\n')
print(test_df.isna().sum())


#number of unique values
train_df.nunique()


#checking the data type's
train_df.dtypes


#figure size
plt.figure(figsize = (6, 6))

#pie plot 
train_df['Personality'].value_counts().plot.pie(autopct='%1.2f%%', textprops={'fontsize':16}).set_title("Target distribution")


#categorical features
cat_features = ["Stage_fear", "Drained_after_socializing"]

fig = plt.figure(figsize=(10, 20))
for i, var_name in enumerate(cat_features):
    ax=fig.add_subplot(4,1,i+1)
    sns.countplot(data=train_df, x=var_name, ax=ax, hue="Personality")
    ax.set_title(var_name)


#numerical features
num_features = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]

fig = plt.figure(figsize=(10, 20))
for i, var_name in enumerate(num_features):
    x = fig.add_subplot(len(num_features), 1, i + 1)
    sns.histplot(data=train_df, x=var_name, hue="Personality", kde=True, binwidth=1)
    ax.set_title(var_name)


#heatmap
train_corr = train_df.corr(numeric_only=True)
sns.heatmap(data=train_corr, annot=True)


train_id = train_df["id"]
test_id = test_df["id"]

#drop the id column
train_df.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)

ntrain = train_df.shape[0]
ntest = test_df.shape[0]

y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values 

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)


all_data.head()


print(f"Time_spent_Alone nan values before: {all_data['Time_spent_Alone'].isna().sum()}")

#define masks
mask1 = (all_data['Going_outside'] <= 3) & (all_data['Time_spent_Alone'].isna())
mask2 = (all_data['Going_outside'] > 3) & (all_data['Time_spent_Alone'].isna())

#compute group-wise medians 
median_low = all_data.loc[(all_data['Going_outside'] <= 3) & (all_data['Time_spent_Alone'].notna()), 'Time_spent_Alone'].median()
median_high = all_data.loc[(all_data['Going_outside'] > 3) & (all_data['Time_spent_Alone'].notna()), 'Time_spent_Alone'].median()

#fill nans with the median of each group
all_data.loc[mask1, 'Time_spent_Alone'] = median_low
all_data.loc[mask2, 'Time_spent_Alone'] = median_high

print(f"Time_spent_Alone nan values after: {all_data['Time_spent_Alone'].isna().sum()}")


print(f"Time_spent_Alone nan values before: {all_data['Time_spent_Alone'].isna().sum()}")

mask1 = (all_data['Social_event_attendance'] <= 4) & (all_data['Time_spent_Alone'].isna())
mask2 = (all_data['Social_event_attendance'] > 4) & (all_data['Time_spent_Alone'].isna())

median_low = all_data.loc[(all_data['Social_event_attendance'] <= 4) & (all_data['Time_spent_Alone'].notna()), 'Time_spent_Alone'].median()
median_high = all_data.loc[(all_data['Social_event_attendance'] > 4) & (all_data['Time_spent_Alone'].notna()), 'Time_spent_Alone'].median()

all_data.loc[mask1, 'Time_spent_Alone'] = median_low
all_data.loc[mask2, 'Time_spent_Alone'] = median_high

print(f"Time_spent_Alone nan values after: {all_data['Time_spent_Alone'].isna().sum()}")


print(f"Social_event_attendance nan values before: {all_data['Social_event_attendance'].isna().sum()}")

mask1 = (all_data['Going_outside'] <= 3) & (all_data['Social_event_attendance'].isna())
mask2 = (all_data['Going_outside'] > 3) & (all_data['Social_event_attendance'].isna())

median_low = all_data.loc[(all_data['Going_outside'] <= 3) & (all_data['Social_event_attendance'].notna()), 'Social_event_attendance'].median()
median_high = all_data.loc[(all_data['Going_outside'] > 3) & (all_data['Social_event_attendance'].notna()), 'Social_event_attendance'].median()

all_data.loc[mask1, 'Social_event_attendance'] = median_low
all_data.loc[mask2, 'Social_event_attendance'] = median_high

print(f"Social_event_attendance nan values after: {all_data['Social_event_attendance'].isna().sum()}")


print(f"Social_event_attendance nan values before: {all_data['Social_event_attendance'].isna().sum()}")

mask1 = (all_data['Post_frequency'] <= 3) & (all_data['Social_event_attendance'].isna())
mask2 = (all_data['Post_frequency'] > 3) & (all_data['Social_event_attendance'].isna())

median_low = all_data.loc[(all_data['Post_frequency'] <= 3) & (all_data['Social_event_attendance'].notna()), 'Social_event_attendance'].median()
median_high = all_data.loc[(all_data['Post_frequency'] > 3) & (all_data['Social_event_attendance'].notna()), 'Social_event_attendance'].median()

all_data.loc[mask1, 'Social_event_attendance'] = median_low
all_data.loc[mask2, 'Social_event_attendance'] = median_high

print(f"Social_event_attendance nan values after: {all_data['Social_event_attendance'].isna().sum()}")


print(f"Social_event_attendance nan values before: {all_data['Social_event_attendance'].isna().sum()}")

mask1 = (all_data['Friends_circle_size'] <= 6) & (all_data['Social_event_attendance'].isna())
mask2 = (all_data['Friends_circle_size'] > 6) & (all_data['Social_event_attendance'].isna())

median_low = all_data.loc[(all_data['Friends_circle_size'] <= 6) & (all_data['Social_event_attendance'].notna()), 'Social_event_attendance'].median()
median_high = all_data.loc[(all_data['Friends_circle_size'] > 6) & (all_data['Social_event_attendance'].notna()), 'Social_event_attendance'].median()

all_data.loc[mask1, 'Social_event_attendance'] = median_low
all_data.loc[mask2, 'Social_event_attendance'] = median_high

print(f"Social_event_attendance nan values after: {all_data['Social_event_attendance'].isna().sum()}")


print(f"Going_outside nan values before: {all_data['Going_outside'].isna().sum()}")

mask1 = (all_data['Social_event_attendance'] <= 4) & (all_data['Going_outside'].isna())
mask2 = (all_data['Social_event_attendance'] > 4) & (all_data['Going_outside'].isna())

median_low = all_data.loc[(all_data['Social_event_attendance'] <= 4) & (all_data['Going_outside'].notna()), 'Going_outside'].median()
median_high = all_data.loc[(all_data['Social_event_attendance'] > 4) & (all_data['Going_outside'].notna()), 'Going_outside'].median()

all_data.loc[mask1, 'Going_outside'] = median_low
all_data.loc[mask2, 'Going_outside'] = median_high

print(f"Going_outside nan values after: {all_data['Going_outside'].isna().sum()}")


print(f"Friends_circle_size nan values before: {all_data['Friends_circle_size'].isna().sum()}")

mask1 = (all_data['Going_outside'] <= 3) & (all_data['Friends_circle_size'].isna())
mask2 = (all_data['Going_outside'] > 3) & (all_data['Friends_circle_size'].isna())

median_low = all_data.loc[(all_data['Going_outside'] <= 3) & (all_data['Friends_circle_size'].notna()), 'Friends_circle_size'].median()
median_high = all_data.loc[(all_data['Going_outside'] > 3) & (all_data['Friends_circle_size'].notna()), 'Friends_circle_size'].median()

all_data.loc[mask1, 'Friends_circle_size'] = median_low
all_data.loc[mask2, 'Friends_circle_size'] = median_high

print(f"Friends_circle_size nan values after: {all_data['Friends_circle_size'].isna().sum()}")


print(f"Post_frequency nan values before: {all_data['Post_frequency'].isna().sum()}")

mask1 = (all_data['Time_spent_Alone'] > 4) & (all_data['Post_frequency'].isna())
mask2 = (all_data['Time_spent_Alone'] <= 4) & (all_data['Post_frequency'].isna())

median_high = all_data.loc[(all_data['Time_spent_Alone'] > 4) & (all_data['Post_frequency'].notna()), 'Post_frequency'].median()
median_low = all_data.loc[(all_data['Time_spent_Alone'] <= 4) & (all_data['Post_frequency'].notna()), 'Post_frequency'].median()

all_data.loc[mask1, 'Post_frequency'] = median_high
all_data.loc[mask2, 'Post_frequency'] = median_low

print(f"Post_frequency nan values after: {all_data['Post_frequency'].isna().sum()}")


all_data.fillna({'Stage_fear':'Unknown', 'Drained_after_socializing':'Unknown'}, inplace=True)


#applying one-hot encoding to Stage_fear and Drained_after_socializing
all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing'])


#splitting back the data
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]

X = X_train
y = y_train


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=12)


#setting scale positive weight
class_1 = y_train.sum()             
class_0 = len(y_train) - class_1     
scale_pos_weight = class_0 / class_1

print(f'scale_pos_weight: {scale_pos_weight}')


xgb = XGBClassifier(
    n_estimators=300,               
    max_depth=4,                
    subsample=0.8,            
    colsample_bytree=0.8,      
    random_state=12,
    learning_rate=0.02, 
    use_label_encoder=False,    
    eval_metric='logloss',          
)

cbc = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=12,
    verbose=0
)


rfc = RandomForestClassifier(
    n_estimators=200,        
    max_depth=10,           
    min_samples_split=5,      
    min_samples_leaf=2,      
    max_features='sqrt',      
    random_state=12,          
    class_weight={0: scale_pos_weight, 1: 1},
)



vc = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cbc', cbc),
        ('rfc', rfc)
    ],
    voting='soft'
)


vc.fit(X_train, y_train)


#optimizing threshold
valid_probs = vc.predict_proba(X_valid)[:, 1]
best_threshold = 0.5
best_score = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (valid_probs >= threshold).astype(int)
    score = accuracy_score(y_valid, preds)  

    if score > best_score:
        best_score = score
        best_threshold = threshold

print(f"Best threshold: {best_threshold}, Best accuracy: {best_score}")

#predicting test set using best_threshold
test_probs = vc.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)



#creating submission
submission = pd.DataFrame({
    'id': test_id,
    'Personality': test_preds
})

submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)
print("Submitted successfully")


