# ğŸ“š Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# ğŸ“¥ Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')


# ğŸ§¹ Combine Train + Test for unified preprocessing
train['is_train'] = 1
test['is_train'] = 0
test['Personality'] = np.nan
df = pd.concat([train, test])

# ğŸ§¼ Handle Missing Values
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])




df['social_score'] = df['Social_event_attendance'] + df['Going_outside'] - df['Time_spent_Alone']
df['online_ratio'] = df['Post_frequency'] / (df['Friends_circle_size'] + 1)


# ğŸ”¤ Encode Categorical Variables
df = pd.get_dummies(df, columns=cat_cols)

# ğŸ�¯ Encode Target Variable
le = LabelEncoder()
df.loc[df['is_train'] == 1, 'Personality'] = le.fit_transform(df.loc[df['is_train'] == 1, 'Personality'])

# ğŸ”„ Split Back
train = df[df['is_train'] == 1].drop(columns=['is_train'])
test = df[df['is_train'] == 0].drop(columns=['is_train', 'Personality'])

X = train.drop(columns='Personality')
y = train['Personality'].astype(int)


# ğŸ§ª Model Setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# ğŸ”� GridSearchCV for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X, y)
best_rf = rf_grid.best_estimator_

# ğŸ”� GridSearchCV for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 6]
}

xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    xgb_param_grid, cv=cv, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X, y)
best_xgb = xgb_grid.best_estimator_




# ğŸ“Š Cross-validation comparison
rf_score = cross_val_score(best_rf, X, y, cv=cv, scoring='accuracy').mean()
xgb_score = cross_val_score(best_xgb, X, y, cv=cv, scoring='accuracy').mean()

print(f"\nâœ… CV Accuracy: Random Forest = {rf_score:.4f}, XGBoost = {xgb_score:.4f}")

# ğŸ“Œ Select Best Model
best_model = best_xgb if xgb_score > rf_score else best_rf
best_model_name = "XGBoost" if best_model == best_xgb else "Random Forest"
print(f"ğŸ“Œ Selected Model: {best_model_name}")

# ğŸ”¥ Train Best Model
best_model.fit(X, y)


# ğŸ“ˆ Feature Importance
def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = X.columns
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title(f"Top 15 Feature Importances ({model_name})")
        plt.tight_layout()
        plt.show()

plot_feature_importance(best_model, best_model_name)


# ğŸ”® Predict on Test
test_preds = best_model.predict(test)
test_preds_label = le.inverse_transform(test_preds.astype(int))

# ğŸ“¤ Create Submission
submission = pd.DataFrame({'id': test.index, 'Personality': test_preds_label})
submission.to_csv("submission.csv", index=False)
submission.head(5)

