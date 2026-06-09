import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
train_df.drop(['id'],axis = 1 ,inplace = True)
test_id = test_df['id']
test_df.drop(['id'],axis = 1, inplace = True)


train_df.head()


train_df.info()


train_df.isna().sum()


test_df.isna().sum()


train_df.info()


#eda
#target distribution
sns.countplot(x='Personality', data=train_df)
plt.title("Target Distribution - Personality")
plt.show()


#features vs target
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Personality', y=col, data=train_df)
    plt.title(f"{col} vs Personality")
    plt.show()



#violin plot
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='Personality', y=col, data=train_df, inner='quartile')
    plt.title(f"{col} vs Personality (Violin Plot)")
    plt.show()



#handelling the nan values
num_cols = test_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = test_df.select_dtypes(include=['object']).columns.tolist()
target_col= ['Personality']


#handeling the nan values
#replacing values
train_df[num_cols]=train_df[num_cols].fillna(train_df[num_cols].median())
test_df[num_cols]=test_df[num_cols].fillna(test_df[num_cols].median())
train_df[cat_cols]=train_df[cat_cols].fillna(train_df[cat_cols].mode())
test_df[cat_cols]=test_df[cat_cols].fillna(test_df[cat_cols].mode())


num_cols


cat_cols


#label encoding the cat cols
from sklearn.preprocessing import LabelEncoder
enc_cols = ['Stage_fear', 'Drained_after_socializing']
target_col = 'Personality'
for col in enc_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
target_le = LabelEncoder()
train_df[target_col] = target_le.fit_transform(train_df[target_col])



#cor heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


train_df.head()


train_df.isna().sum()


test_df.isna().sum()


X = train_df.drop(['Personality'],axis=1)
y = train_df['Personality']


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size = 0.2)


from xgboost import XGBClassifier
model = XGBClassifier(enable_categorical=True, verbosity=0)
from sklearn.model_selection import RandomizedSearchCV


param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [1, 1.5, 2, 3]
}
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,            # number of random combinations to try
    scoring='accuracy',   # or 'roc_auc', 'f1', etc.
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Get best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)



best_params = random_search.best_params_
xgb_best = XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_best.fit(X_train, y_train)
y_pred = xgb_best.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


test_predictions = xgb_best.predict(test_df)
decoded_predictions = target_le.inverse_transform(test_predictions)

submission = pd.DataFrame({
    'id': test_id,
    'Personality': decoded_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")


