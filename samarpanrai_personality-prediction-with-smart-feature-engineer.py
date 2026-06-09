import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


def basic_preprocessing(df):
    df = df.copy()
    df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    return df

train_df = basic_preprocessing(train_df)
test_df = basic_preprocessing(test_df)

num_cols = ['Social_event_attendance', 'Going_outside', 'Friends_circle_size',
            'Post_frequency', 'Time_spent_Alone']

train_df[num_cols] = train_df[num_cols].fillna(0)
test_df[num_cols] = test_df[num_cols].fillna(0)


def create_interaction_features(df):
    df['Social_Energy'] = df['Social_event_attendance'] * df['Going_outside']
    df['Social_Fatigue'] = df['Social_event_attendance'] * (df['Drained_after_socializing'] == 1).astype(int)
    df['Social_Engagement'] = df['Friends_circle_size'] * df['Post_frequency']
    df['Balance_Score'] = df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1e-5)
    df['Social_Recovery'] = ((df['Time_spent_Alone'] > 5) & (df['Drained_after_socializing'] == 1)).astype(int)
    return df

def create_derived_features(df):
    df['Social_Activity_Level'] = pd.cut(df['Social_event_attendance'],
                                         bins=[-1, 3, 6, 11],
                                         labels=['Low', 'Medium', 'High'])
    df['Friend_Engagement_Ratio'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)
    df['Alone_Time_Category'] = pd.cut(df['Time_spent_Alone'],
                                       bins=[-1, 2, 5, 8, 12],
                                       labels=['Low', 'Medium', 'High', 'Very High'])

    # Added some new features
    df['Alone_Ratio'] = df['Time_spent_Alone'] / (df['Time_spent_Alone'] + df['Social_event_attendance'] + 1)
    df['Friends_per_Event'] = df['Friends_circle_size'] / (df['Social_event_attendance'] + 1)
    df['Posts_per_Outside'] = df['Post_frequency'] / (df['Going_outside'] + 1)
    df['Alone_Time_Squared'] = df['Time_spent_Alone'] ** 2
    df['Social_Attendance_Squared'] = df['Social_event_attendance'] ** 2
    df['Heavy_Socializer'] = ((df['Social_event_attendance'] > 8) & (df['Going_outside'] > 8)).astype(int)
    df['Is_Balanced'] = ((df['Time_spent_Alone'] > 2) & (df['Time_spent_Alone'] < 6) &
                         (df['Social_event_attendance'] > 3) & (df['Social_event_attendance'] < 7)).astype(int)
    return df

def create_personality_signatures(df):
    df['Introvert_Signature'] = (
        (df['Time_spent_Alone'] > 5) & 
        (df['Social_event_attendance'] < 3) & 
        (df['Stage_fear'] == 1)
    ).astype(int)
    
    df['Extrovert_Signature'] = (
        (df['Time_spent_Alone'] < 3) & 
        (df['Social_event_attendance'] > 5) & 
        (df['Friends_circle_size'] > 8)
    ).astype(int)
    
    df['Social_Battery'] = np.where(
        (df['Drained_after_socializing'] == 1) & (df['Social_event_attendance'] > 5),
        'Low',
        np.where(
            (df['Drained_after_socializing'] == 0) & (df['Social_event_attendance'] > 5),
            'High',
            'Medium'
        )
    )
    return df


train_df_processed = create_interaction_features(train_df)
train_df_processed = create_derived_features(train_df_processed)
train_df_processed = create_personality_signatures(train_df_processed)

test_df_processed = create_interaction_features(test_df)
test_df_processed = create_derived_features(test_df_processed)
test_df_processed = create_personality_signatures(test_df_processed)


category_mappings = {
    'Social_Activity_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Alone_Time_Category': {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3},
    'Social_Battery': {'Low': 0, 'Medium': 1, 'High': 2}
}

for col, mapping in category_mappings.items():
    for df in [train_df_processed, test_df_processed]:
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float).fillna(-1)


numeric_cols_train = train_df_processed.select_dtypes(include=['number']).columns
train_df_processed[numeric_cols_train] = SimpleImputer(strategy='median').fit_transform(train_df_processed[numeric_cols_train])

numeric_cols_test = test_df_processed.select_dtypes(include=['number']).columns
test_df_processed[numeric_cols_test] = SimpleImputer(strategy='median').fit_transform(test_df_processed[numeric_cols_test])


le = LabelEncoder()
train_df_processed['Personality'] = le.fit_transform(train_df_processed['Personality'])

X = train_df_processed.drop(['id', 'Personality'], axis=1)
y = train_df_processed['Personality']

# making sure that the test data matches train columns
for col in X.columns:
    if col not in test_df_processed.columns:
        test_df_processed[col] = 0
X_test = test_df_processed[X.columns]
test_ids = test_df['id']


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_acc = accuracy_score(y_val, rf_preds)
print(f"Random Forest Validation Accuracy: {rf_acc:.4f}")


rf_feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importances)
plt.title('Random Forest Feature Importances')
plt.tight_layout()
plt.show()


xgb_model = XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)
xgb_acc = accuracy_score(y_val, xgb_preds)
print(f"XGBoost Validation Accuracy: {xgb_acc:.4f}")

# Retraining on full data
xgb_model.fit(X, y)
xgb_test_preds = xgb_model.predict(X_test)
xgb_test_labels = le.inverse_transform(xgb_test_preds)

xgb_model.fit(X, y)
xgb_test_preds = xgb_model.predict(X_test)
xgb_test_labels = le.inverse_transform(xgb_test_preds)
xgb_submission = pd.DataFrame({'id': test_ids, 'Personality': xgb_test_labels})
xgb_submission.to_csv('/kaggle/working/submission.csv', index=False)


xgb_feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=xgb_feature_importances)
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.show()

