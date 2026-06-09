import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
ext_data = (
    pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
       'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
       'Post_frequency']))


train = train.merge(ext_data, how = "left")


train.head()


train.info()


# Percentage of nulls for each column
null_percent = train.isnull().mean() * 100
print(null_percent.sort_values(ascending=False))


sns.countplot(x="Stage_fear", data=train)


sns.countplot(x="Personality", data=train)


train["Drained_after_socializing"].value_counts()


def create_personality_features(df):
    """
    Adds engineered features relevant to personality prediction.
    Assumes all missing values have already been imputed.
    """

    # Social Activity Score
    df['social_activity_score'] = (
        df['Social_event_attendance'] +
        df['Going_outside'] +
        df['Post_frequency']
    )

    # Isolation Index
    df['isolation_index'] = (
        df['Time_spent_Alone'] / (df['Social_event_attendance'] + 1)
    )

    # Friend Ratio
    df['friend_ratio'] = (
        df['Friends_circle_size'] / (df['Social_event_attendance'] + 1)
    )

    # Activity Variability (std dev of time/event-related features)
    df['activity_std'] = df[[
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Post_frequency'
    ]].std(axis=1)

    return df


X = train.drop(columns=["id", "Personality"])
y = train["Personality"]


from sklearn.preprocessing import LabelEncoder

stage_fear_encoder = LabelEncoder()
drained_after_socializing_encoder = LabelEncoder()
y_encoder = LabelEncoder()

X["Stage_fear"] = stage_fear_encoder.fit_transform(X["Stage_fear"])
X["Drained_after_socializing"] = drained_after_socializing_encoder.fit_transform(X["Drained_after_socializing"])
y = pd.Series(y_encoder.fit_transform(y))


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# imputer = IterativeImputer(max_iter=200)
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame
X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
X = create_personality_features(X)


X.info()


df = pd.concat([X, y], axis=1)
sns.heatmap(df.corr())


test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

test = test.drop(columns="id")

test["Stage_fear"] = stage_fear_encoder.transform(test["Stage_fear"])
test["Drained_after_socializing"] = drained_after_socializing_encoder.transform(test["Drained_after_socializing"])

test_imputed = imputer.transform(test)

# Convert back to DataFrame
test = pd.DataFrame(test_imputed, columns=test.columns, index=test.index)


test = create_personality_features(test)


from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

best_model = None
best_score = 0
all_preds = []
all_true = []

rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=2, random_state=42)

for fold, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        device='gpu',  # Enable GPU
        boosting_type='gbdt',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=10,
        objective='binary',
        metric='binary_logloss',
        verbose=-1,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        # verbose=False
    )

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    all_preds.extend(y_pred)
    all_true.extend(y_val)

    if acc > best_score:
        best_score = acc
        best_model = model

# Final evaluation using best model
print(f"\nâœ… Best Fold Accuracy: {best_score}\n")
print("ðŸ“Š Classification Report:\n", classification_report(all_true, all_preds))
print("ðŸ§© Confusion Matrix:\n", confusion_matrix(all_true, all_preds))

# Predict on test set using best model
test_preds = best_model.predict(test)
test_preds = y_encoder.inverse_transform(test_preds)

# Create submission
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission["Personality"] = test_preds
submission.to_csv("submission.csv", index=False)
submission.head()

