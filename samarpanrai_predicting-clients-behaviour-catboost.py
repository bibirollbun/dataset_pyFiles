import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.info()


def preprocess(df):
    df = df.copy()
    df['was_contacted'] = (df['pdays'] != -1).astype(int)
    
    # New feature
    bins = [0, 30, 50, 70, 100]
    labels = ['young', 'middle_aged', 'senior', 'elderly']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    # New feature
    season_map = {
        'jan': 'winter', 'feb': 'winter', 'mar': 'spring',
        'apr': 'spring', 'may': 'spring', 'jun': 'summer',
        'jul': 'summer', 'aug': 'summer', 'sep': 'fall',
        'oct': 'fall', 'nov': 'fall', 'dec': 'winter'
    }
    df['season'] = df['month'].map(season_map)

    # New features
    df['has_debt'] = ((df['housing'] == 'yes') | (df['loan'] == 'yes')).astype(int)
    df['total_contacts'] = df['campaign'] + df['previous']
    df['duration_per_contact'] = df['duration'] / (df['campaign'] + 1)  # +1 to avoid division by zero
    
    return df

train = preprocess(train_df)
test = preprocess(test_df)


X = train.drop(['id', 'y'], axis=1)
y = train['y'].astype(int)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'poutcome', 'age_group', 'season']
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'previous', 
            'was_contacted', 'has_debt', 'total_contacts', 'duration_per_contact']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

# XGBoost Model
xgb = Pipeline([
    ('preprocess', preprocessor),
    ('model', XGBClassifier(random_state=42))
])
xgb.fit(X_train, y_train)

# catboost Model
catboost = Pipeline([
    ('preprocess', preprocessor),
    ('model', CatBoostClassifier(random_state=42, verbose=0))  # verbose=0 to silence training output
])
catboost.fit(X_train, y_train)

catboost_acc = accuracy_score(y_val, catboost.predict(X_val))
xgb_acc = accuracy_score(y_val, xgb.predict(X_val))

print(f"XGBoost Accuracy: {xgb_acc:.4f}")
print(f"CatBoost Accuracy: {catboost_acc:.4f}")


# feature names
encoder = preprocessor.named_transformers_['cat']
feature_names = list(encoder.get_feature_names_out(cat_cols)) + num_cols

def plot_importance(importance, title):
    plt.figure(figsize=(10, 6))
    pd.Series(importance, index=feature_names).nlargest(15).plot(kind='barh')
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_importance(xgb.named_steps['model'].feature_importances_, 'XGBoost Feature Importance')
plot_importance(catboost.named_steps['model'].feature_importances_, 'Catboost Feature Importance')


# Predictions
catboost_preds = catboost.predict_proba(test.drop('id', axis=1))[:, 1]

# submission file
catboost_submission = pd.DataFrame({'id': test['id'], 'y': catboost_preds})
catboost_submission.to_csv('/kaggle/working/submission.csv', index=False)


catboost_submission.head()

