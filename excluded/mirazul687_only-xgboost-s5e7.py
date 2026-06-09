import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train_df = train_df.drop('id', axis=1)


def clean_data(df):
    # Replace missing values with the mean of each column in: 'Time_spent_Alone'
    df = df.fillna({'Time_spent_Alone': df['Time_spent_Alone'].mean()})
    # Round column 'Time_spent_Alone' (Number of decimals: 1)
    df = df.round({'Time_spent_Alone': 0})

    # Replace missing values with the most common value of each column in: 'Stage_fear'
    df = df.fillna({'Stage_fear': df['Stage_fear'].mode()[0]})

    # Replace missing values with the mean of each column in: 'Social_event_attendance'
    df = df.fillna({'Social_event_attendance': df['Social_event_attendance'].mean()})
    df = df.round({'Social_event_attendance': 0})

    # Replace missing values with the mean of each column in: 'Going_outside'
    df = df.fillna({'Going_outside': df['Going_outside'].mean()})
    df = df.round({'Going_outside': 0})

    # Replace missing values with the most common value of each column in: 'Drained_after_socializing'
    df = df.fillna({'Drained_after_socializing': df['Drained_after_socializing'].mode()[0]})

    # Replace missing values with the mean of each column in: 'Friends_circle_size'
    df = df.fillna({'Friends_circle_size': df['Friends_circle_size'].mean()})
    df = df.round({'Friends_circle_size': 0})

    # Replace missing values with the mean of each column in: 'Post_frequency'
    df = df.fillna({'Post_frequency': df['Post_frequency'].mean()})
    df = df.round({'Post_frequency': 0})
    return df




df = clean_data(train_df)
test_df = clean_data(test_df)


def feature_engineering(df):
    df['social_activity_score'] = (
        0.5 * df['Social_event_attendance'].fillna(0) +
        0.3 * df['Going_outside'].fillna(0) +
        0.2 * df['Post_frequency'].fillna(0)
    )
    df['friend_post_density'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1e-5)
    df['introversion_index'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    df['interactiveness'] = (
        df['Friends_circle_size'].fillna(0) +
        df['Social_event_attendance'].fillna(0) +
        df['Post_frequency'].fillna(0)
    )
    
    return df


df = feature_engineering(df)
test_df = feature_engineering(test_df)


df.shape


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# Encode categorical and target columns
for col in ['Stage_fear', 'Drained_after_socializing']:
    df[col] = LabelEncoder().fit_transform(df[col])
    
# 1. Fit once during training
target_le = LabelEncoder()
df['Personality'] = target_le.fit_transform(train_df['Personality'])

# Split X, y
X = df.drop(['Personality'], axis=1)
y = df['Personality']


final_xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'gpu_hist',               # Use 'hist' if you don't have GPU
    'predictor': 'gpu_predictor',
    'random_state': 42,
    'n_estimators': 5000,                    # High value + early stopping
    'learning_rate': 0.007,
    'max_depth': 10,
    'subsample': 0.76,
    'colsample_bytree': 0.51,
    'reg_lambda': 6.51,
    'reg_alpha': 5.56,
    'min_child_weight': 5,
    'gamma': 4.97,

    # Additional regularizing params (optional)
    'max_delta_step': 1,
    'scale_pos_weight': 1,                   # Use class balancing if necessary
    'grow_policy': 'depthwise',
    'sampling_method': 'uniform',
    'verbosity': 0
}



kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    model = XGBClassifier(**final_xgb_params)  # ✅ Correct
    model.fit(X.iloc[train_idx], y[train_idx],
              eval_set=[(X.iloc[val_idx], y[val_idx])],
              verbose=False)

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    val_preds = model.predict(X.iloc[val_idx])
    oof_preds[val_idx] = val_preds
    test_preds.append(model.predict(X_train))

    acc = accuracy_score(y[val_idx], val_preds)
    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

final_acc = accuracy_score(y, oof_preds)
print(f"\nFinal OOF Accuracy: {final_acc:.4f}")


# Encode categorical and target columns
for col in ['Stage_fear', 'Drained_after_socializing']:
    test_df[col] = LabelEncoder().fit_transform(test_df[col])

X_test = test_df.drop(['id'], axis=1)

# Predict
y_pred = model.predict(X_test)
y_pred_labels = target_le.inverse_transform(y_pred)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': y_pred_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ Submission saved as 'submission.csv'")

