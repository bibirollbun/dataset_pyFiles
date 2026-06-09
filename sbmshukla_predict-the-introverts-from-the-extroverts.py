import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df =  pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df.head(3)


train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)


train_df.shape


train_df.isna().sum()


train_df['Personality'].replace(
    {
        'Extrovert':1,
        'Introvert':0
    }, inplace=True
)


train_df.groupby(by='Stage_fear')['Time_spent_Alone'].mean()


train_df.replace(
    {
        'Yes':1,
        'No':0
    }, inplace=True
)
test_df.replace(
    {
        'Yes':1,
        'No':0
    }, inplace=True
)


# Target distribution
sns.countplot(x='Personality', data=train_df)
plt.title("Target Class Distribution")
plt.show()


train_df.head()


# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Numerical Feature Correlation")
plt.show()


train_df.isnull().sum()


train_df.head()


mask = train_df['Time_spent_Alone'] == 0
train_df.loc[mask, ['Stage_fear', 'Drained_after_socializing']] = train_df.loc[mask, ['Stage_fear', 'Drained_after_socializing']].fillna('No')

mask2 = test_df['Time_spent_Alone'] == 0
test_df.loc[mask2, ['Stage_fear', 'Drained_after_socializing']] = test_df.loc[mask2, ['Stage_fear', 'Drained_after_socializing']].fillna('No')


train_df.isna().sum()


train_df['Time_spent_Alone'] = train_df['Time_spent_Alone'].fillna(
    train_df.groupby('Stage_fear')['Time_spent_Alone'].transform('mean')
)

test_df['Time_spent_Alone'] = test_df['Time_spent_Alone'].fillna(
    test_df.groupby('Stage_fear')['Time_spent_Alone'].transform('mean')
)


train_df.isna().sum()


mask = train_df['Stage_fear'].isnull()
train_df.loc[mask, 'Stage_fear'] = train_df.loc[mask, 'Drained_after_socializing']

mask2 = test_df['Stage_fear'].isnull()
test_df.loc[mask2, 'Stage_fear'] = test_df.loc[mask2, 'Drained_after_socializing']


train_df.isna().sum()


train_df['Social_event_attendance'] = train_df['Social_event_attendance'].fillna(
    train_df.groupby('Stage_fear')['Social_event_attendance'].transform('mean')
)

test_df['Social_event_attendance'] = test_df['Social_event_attendance'].fillna(
    test_df.groupby('Stage_fear')['Social_event_attendance'].transform('mean')
)


train_df.isna().sum()


train_df['Going_outside'] = train_df['Going_outside'].fillna(
    train_df.groupby('Drained_after_socializing')['Going_outside'].transform('mean')
)
test_df['Going_outside'] = test_df['Going_outside'].fillna(
    test_df.groupby('Drained_after_socializing')['Going_outside'].transform('mean')
)


train_df.isna().sum()


mask = train_df['Drained_after_socializing'].isnull()
train_df.loc[mask, 'Drained_after_socializing'] = train_df.loc[mask, 'Stage_fear']

mask2 = test_df['Drained_after_socializing'].isnull()
test_df.loc[mask2, 'Drained_after_socializing'] = test_df.loc[mask2, 'Stage_fear']


train_df.isna().sum()


train_df['Friends_circle_size'] = train_df['Friends_circle_size'].fillna(
    train_df.groupby('Drained_after_socializing')['Friends_circle_size'].transform('mean')
)

test_df['Friends_circle_size'] = test_df['Friends_circle_size'].fillna(
    test_df.groupby('Drained_after_socializing')['Friends_circle_size'].transform('mean')
)


train_df.isna().sum()


train_df['Post_frequency'] = train_df['Post_frequency'].fillna(
    train_df.groupby('Drained_after_socializing')['Post_frequency'].transform('mean')
)
test_df['Post_frequency'] = test_df['Post_frequency'].fillna(
    test_df.groupby('Drained_after_socializing')['Post_frequency'].transform('mean')
)


train_df.isna().sum()


train_df.head(2)


num_cols = ['Time_spent_Alone', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Fill numeric columns with mean
for col in num_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mean())
    test_df[col] = test_df[col].fillna(test_df[col].mean())

# Fill categorical columns with mode
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
    test_df[col] = test_df[col].fillna(test_df[col].mode()[0])



test_df.isna().sum()


train_df.dtypes


train_df.replace(
    {
        'Yes':1,
        'No':0
    }, inplace=True
)

test_df.replace(
    {
        'Yes':1,
        'No':0
    }, inplace=True
)


train_df_clean = train_df.drop(columns=['id'])

test_df_clean = test_df.drop(columns=['id'])


X = train_df_clean.drop(columns=['Personality'])
y = train_df_clean['Personality']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


from sklearn.linear_model import LogisticRegression


lr =  LogisticRegression()


lr.fit(X_train,y_train)


y_pred = lr.predict(X_test)


from sklearn.metrics import accuracy_score, f1_score


accuracy_score(y_test,y_pred)


f1_score(y_test,y_pred)


y_lr_pred = lr.predict(test_df_clean)


# ðŸŒ² Random Forest
submission_lr = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_lr_pred
})


submission_lr['Personality'].replace(
    {
        1: 'Extrovert',
        0: 'Introvert'
    }, inplace=True
)


# ðŸŒ² Random Forest
submission_lr.to_csv('submission_lr_predict.csv', index=False)


submission_lr.shape


test_df.shape





test_df_clean.shape








from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


xgb_clf = XGBClassifier()


param_grid_xgb = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


grid_search_xgb = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)


grid_search_xgb.fit(X,y)


y_xgb_pred = grid_search_xgb.predict(test_df_clean)


# ðŸŒ² Random Forest
submission_xgb = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_xgb_pred
})


submission_xgb['Personality'].replace(
    {
        1: 'Extrovert',
        0: 'Introvert'
    }, inplace=True
)


# ðŸŒ² Random Forest
submission_xgb.to_csv('submission_xgb_predict.csv', index=False)


print("âœ… Best Parameters:")
print(grid_search_xgb.best_params_)


print("âœ… Best Score:")
print(grid_search_xgb.best_score_)





from sklearn.ensemble import RandomForestClassifier


rf_clf = RandomForestClassifier(random_state=42)


param_grid_rfr = {
    'n_estimators': [100, 150],         # Number of trees
    'max_depth': [5, 10],               # Tree depth
    'min_samples_split': [2, 5],        # Min samples to split
    'min_samples_leaf': [1, 2],         # Min samples at leaf
    'criterion': ['gini', 'entropy']    # Split criterion
}


grid_search_rfr = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid_rfr,
    cv=5,                     # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,                # Use all CPU cores
    verbose=1
)


grid_search_rfr.fit(X,y)


y_rfc_pred = grid_search_rfr.predict(test_df_clean)


print("âœ… Best Parameters:")
print(grid_search_rfr.best_params_)


print("âœ… Best Score:")
print(grid_search_rfr.best_score_)


# ðŸŒ² Random Forest
submission_rfc = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_rfc_pred
})


submission_rfc['Personality'].replace(
    {
        1: 'Extrovert',
        0: 'Introvert'
    }, inplace=True
)


# ðŸŒ² Random Forest
submission_rfc.to_csv('submission_rfc_predict.csv', index=False)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


base_learners = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, 
                          colsample_bytree= 0.8, 
                          learning_rate= 0.1, 
                          max_depth= 3, 
                          subsample= 0.8, 
                          random_state=42)),
    ('rf', RandomForestClassifier(criterion= 'entropy', 
                                   max_depth= 10, 
                                   min_samples_leaf= 2, 
                                   min_samples_split= 5, 
                                   n_estimators= 150))
]


final_estimator = LogisticRegression()


stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=final_estimator,
    cv=5,           # Cross-validation for base learners
    n_jobs=-1
)


stacking_model.fit(X,y)


y_stack_pred = stacking_model.predict(test_df_clean)


# ðŸŒ² Random Forest
submission_stack = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_stack_pred
})


submission_stack['Personality'].replace(
    {
        1: 'Extrovert',
        0: 'Introvert'
    }, inplace=True
)



submission_stack.to_csv('submission_stack_predict.csv', index=False)













