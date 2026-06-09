!pip install imbalanced-learn
!pip install xgboost==3.0.2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


traindf = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
testdf = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

traindf.head()


traindf.info()


for col in traindf.select_dtypes(include = "object").columns.tolist():
    print(traindf[col].value_counts())


traindf["Stage_fear"] = traindf["Stage_fear"].map({"No" : 0, "Yes" : 1})
traindf["Drained_after_socializing"] = traindf["Drained_after_socializing"].map({"No" : 0, "Yes" : 1})
traindf = traindf.drop(["id"], axis=1)


testdf["Stage_fear"] = testdf["Stage_fear"].map({"No" : 0, "Yes" : 1})
testdf["Drained_after_socializing"] = testdf["Drained_after_socializing"].map({"No" : 0, "Yes" : 1})
testid = testdf["id"]
testdf = testdf.drop(["id"], axis=1)


import seaborn as sns

sns.heatmap(traindf.corr(numeric_only=True), cmap="coolwarm", annot=True)


def plot_categorical_vs_binary(df, cat_col, binary_col='Stage_fear'):
   ct = pd.crosstab(df[cat_col], df[binary_col], normalize='index') * 100
   
   plt.figure(figsize=(10, 6))
   ax = ct.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'])
   
   # Add percentage labels on bars
   for i, (idx, row) in enumerate(ct.iterrows()):
       cumulative = 0
       for j, (col, value) in enumerate(row.items()):
           if value >= 10:  # Only show if >= 10%
               y_pos = cumulative + value/2
               ax.text(i, y_pos, f'{value:.0f}%', 
                      ha='center', va='center', 
                      fontweight='bold', 
                      color='white' if value > 25 else 'black',
                      rotation=90)
           cumulative += value
   
   plt.title(f'{cat_col} vs Stage Fear', fontsize=14, fontweight='bold')
   plt.xlabel(cat_col, fontsize=12)
   plt.ylabel('Percentage', fontsize=12)
   plt.legend(['No Fear', 'Fear'], loc='upper right')
   plt.xticks(rotation=45, ha='right')
   plt.ylim(0, 100)
   plt.tight_layout()
   plt.show()

columns = ['Time_spent_Alone',
 'Social_event_attendance',
 'Going_outside',
 'Drained_after_socializing',
 'Friends_circle_size',
 'Post_frequency',
 'Personality']

for col in columns:
    plot_categorical_vs_binary(traindf, col)


def fear_based_na_filler(traindf : pd.DataFrame, isTrain = True):
    # Masks
    is_afraid = traindf["Stage_fear"].isna()

    is_drained = traindf["Drained_after_socializing"] == 1 
    # Almost all of the people who are drained after socializing 
    # also have stage fear based on the graph
    # similar comment applies to other values
    
    is_alone_more = traindf["Time_spent_Alone"] > 4
    is_not_social = traindf["Social_event_attendance"] < 3
    is_not_outside = traindf["Going_outside"] > 3
    is_not_famous = traindf["Friends_circle_size"] < 4

    if isTrain:
        is_introvert = traindf["Personality"] == "Introvert"
        fears = [is_drained, is_alone_more, is_not_social, is_not_outside, is_not_famous, is_introvert]
    else:
        fears = [is_drained, is_alone_more, is_not_social, is_not_outside, is_not_famous]

    for fear in fears:
        traindf.loc[is_afraid & fear, "Stage_fear"] = 1
        is_afraid = traindf["Stage_fear"].isna()

    traindf.loc[is_afraid, "Stage_fear"] = 0

    other_columns = ['Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Drained_after_socializing',
    'Friends_circle_size', 
    'Post_frequency'
    ]

    for column in other_columns:
        traindf[column] = traindf[column].fillna(
            traindf.groupby("Stage_fear")[column].transform("mean")
        )

    return traindf

traindf = fear_based_na_filler(traindf)
testdf = fear_based_na_filler(testdf, False)


# Encode all object columns
from sklearn.preprocessing import OrdinalEncoder

y = traindf['Personality']
X = traindf.drop(["Personality", "Drained_after_socializing",], axis=1)
X_test = testdf.drop(["Drained_after_socializing",], axis=1)


from sklearn.calibration import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)


from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

# SMOTE with editing
smote = SMOTE(random_state=42)
enn = EditedNearestNeighbours()
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled, y_resampled = enn.fit_resample(X_resampled, y_resampled)


from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Stacking ensemble
base_models = [
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier(verbose=-1)),
    ('rf', RandomForestClassifier()),
    ('cb', CatBoostClassifier(verbose=False))
]

stacking_classifier = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv = 5,
    stack_method='predict_proba'
)



import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sgk = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accs, aucs, f1s = [], [], []

for fold, (train_idx, val_idx) in enumerate(sgk.split(X_resampled, y_resampled)):
    X_tr = X_resampled.loc[train_idx]
    y_tr = y_resampled[train_idx]

    X_val = X_resampled.loc[val_idx]
    y_val = y_resampled[val_idx]

    stacking_classifier.fit(X_tr, y_tr)
    pred = stacking_classifier.predict(X_val)
    pred_prob = stacking_classifier.predict_proba(X_val)[:, 1]

    accs.append(accuracy_score(y_val, pred))
    aucs.append(roc_auc_score(y_val, pred_prob))
    f1s.append(f1_score(y_val, pred))

    print(f"========= Fold {fold} =========")
    print(f"Accuracy: {accs[-1]:.4f}, AUC: {aucs[-1]:.4f}, F1: {f1s[-1]:.4f}")

print(f"\nMean Accuracy: {np.mean(accs):.4f}")
print(f"Mean AUC: {np.mean(aucs):.4f}")
print(f"Mean F1: {np.mean(f1s):.4f}")


# Generate predictions
test_preds = stacking_classifier.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'id': testid,
    'Personality': le.inverse_transform(test_preds)
})

submission.to_csv('submission.csv', index=False)

