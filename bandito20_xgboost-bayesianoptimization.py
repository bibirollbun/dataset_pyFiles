# prompt: install basyian optimizer

!pip install bayesian-optimization


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, average_precision_score
from bayes_opt import BayesianOptimization


df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


df.drop(['id'], inplace=True, axis=1)


df.head()


df.info()


df.isna().sum()


df.describe()


len(df.loc[df['Personality'] == 'Extrovert']) / (len(df.loc[df['Personality'] == 'Introvert']))


num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns.difference(['Personality'])


fig, ax = plt.subplots(5, 1)
for i, col in enumerate(num_cols):
  sns.boxplot(data=df, x=col, ax=ax[i])
  ax[i].set_title(col)


Q1 = df['Time_spent_Alone'].quantile(0.25)
Q3 = df['Time_spent_Alone'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Time_spent_Alone'] = df['Time_spent_Alone'].clip(lower=lower_bound, upper=upper_bound)


df['Time_spent_Alone'].describe()


df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})


X = df.drop(['Personality'], axis=1)
for col in cat_cols:
  X[col] = X[col].astype('category')

y = df[['Personality']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(
    eta=0.1,#0.1
    n_estimators=100,#100
    enable_categorical=True,
    scale_pos_weight=2.839,
    gamma=1,
    max_depth=6,
    min_child_weight=2,#2
    reg_lambda=2,
    reg_alpha=2,#2
    subsample=1,#
    colsample_bytree=0.82#
)
model.fit(X_train, y_train)
print("model score: %.3f" % model.score(X_train, y_train))


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))


y_pred_prob = model.predict_proba(X_test)


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
ROC_AUC = roc_auc_score(y_test, y_pred_prob[:, 1])

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr)
plt.title(f"ROC curve for XGBoost Classifier with AUC: {ROC_AUC}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.show()


from xgboost import plot_importance

plot_importance(model)
plt.show()


# Convert to DMatrix format for xgboost (better for bayesian optimization)
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

def xgb_eval(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda, scale_pos_weight):
    """
    XGBoost evaluation function for Bayesian optimization
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',  # AUC-PR is better for imbalanced rain prediction
        'learning_rate': max(min(learning_rate, 0.3), 0.01),  # Clipped between 0.01-0.3
        'max_depth': int(max_depth),
        'min_child_weight': int(min_child_weight),
        'subsample': max(min(subsample, 1), 0.5),
        'colsample_bytree': max(min(colsample_bytree, 1), 0.5),
        'gamma': max(gamma, 0),
        'reg_alpha': max(reg_alpha, 0),
        'reg_lambda': max(reg_lambda, 0),
        'scale_pos_weight': max(scale_pos_weight, 1),
        'tree_method': 'hist',
        'seed': 42
    }

    # Train with early stopping
    model = xgb.XGBClassifier(**params, enable_categorical=True,)
    model.fit(
        X_train, y_train,

        eval_set=[(X_test, y_test)],
        verbose=0
    )

    # Get predictions and calculate multiple metrics
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_pr = average_precision_score(y_test, y_pred)

    # You can return other metrics if needed
    return auc_pr

# Define the parameter bounds
pbounds = {
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1),
    'gamma': (0, 5),
    'reg_alpha': (0, 10),
    'reg_lambda': (0, 10),
    'scale_pos_weight': (3, 4)
}

# Initialize Bayesian optimizer
optimizer = BayesianOptimization(
    f=xgb_eval,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Run optimization
optimizer.maximize(
    init_points=10,
    n_iter=30,
)




# Print the best parameters found
print("Best parameters found:")
print(optimizer.max)

best_params = optimizer.max['params']


# ensuring that max_depth and min_child_weight are integers
best_params['max_depth'] = int(best_params['max_depth'])
best_params['min_child_weight'] = int(best_params['min_child_weight'])

#
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    enable_categorical=True,
    **best_params
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=0
)


y_pred_proba = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.53).astype(int)

print("\nFinal Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-PR: {average_precision_score(y_test, y_pred_proba):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print(f"ROC_AUC Score: {roc_auc_score(y_test, y_pred_prob[:, 1])}")


test_df.head()


test_data = test_df.drop('id', axis=1)


cat_col = test_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    test_data[col] = test_data[col].astype('category')
predections = model.predict(test_data)


predections


submission = pd.DataFrame({
    'id': test_df['id'],                
    'Personality': predections          
})


submission['Personality'] = submission['Personality'].map({0: 'Introvert', 1: 'Extrovert'})


submission.to_csv('submission.csv', index=False)  


sub_df = pd.read_csv('/kaggle/working/submission.csv')
sub_df.head()




