import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def cross_val_predict(model, X_train, y_train, X_test, n_splits=5, random_state=42):
    print(f"Model: {model.__class__.__name__}")

    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_score = 0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Fold {fold + 1}")
        
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        cur_val_score = f1_score(y_val, val_preds)
        print(f"Current F1 score: {cur_val_score}")
        
        val_score += cur_val_score / n_splits

        # Predict for test set and take average
        test_fold_preds = model.predict(X_test)
        test_preds += test_fold_preds / n_splits 

    print(f"Average F1 score: {val_score}")

    # Uśrednione predykcje dla testu mogą być floatami – progowanie do 0/1:
    test_preds_binary = (test_preds >= 0.5).astype(int)

    return oof_preds, test_preds_binary, val_score


def preprocess_data(df):
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['overtime_status'] = df['overtime_status'].map({'no': 0, 'yes': 1})
    
    df.drop(columns=['is_adult'], inplace=True)

    df.drop(columns=['Unnamed: 0'], inplace=True)

    df['travel_freq'] = df['travel_freq'].map({
        'no_travel': 0,
        'rare_travel': 1,
        'frequent_travel': 2
    })

    df = pd.get_dummies(df, columns=[
        'work_division',
        'degree_field',
        'job_title',
        'marital_state'
    ], drop_first=True) 

    return df


train = pd.read_csv('/kaggle/input/machine-learning-4-sbu/train.csv')
test = pd.read_csv('/kaggle/input/machine-learning-4-sbu/test.csv')


train.head(3)


# Count the occurrences of each unique value in the 'left_company' column
left_company_counts = train['left_company'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(left_company_counts, 
        labels=left_company_counts.index, 
        autopct='%1.1f%%',  # Show percentages with one decimal place
        startangle=90,      # Start the first slice at the top
        colors=['#66b3ff','#ff9999'])  # Optional: colors for slices

plt.title('Distribution of the left_company Column')
plt.show()


len(train)


len(test)


for col in train.select_dtypes(include='object').columns:
        print(f"Column: {col}")
        print(train[col].unique())
        print("-" * 40)


train = preprocess_data(train)
test = preprocess_data(test)


NUNIQUE1=[c for c in train.columns if train[c].nunique()==1]
NUNIQUE1


target = 'left_company'
features = [col for col in train.columns if col != target and col not in NUNIQUE1]

X_train = train[features]
y_train = train[target]
X_test = test[features]


models = [
    LGBMClassifier(class_weight='balanced'),
    # XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6),
    # CatBoostClassifier(verbose=0)
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train, X_test)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


def plot_feature_importance(model, feature_names, top_n=50):
    
    # Frequency importance (default attribute)
    split_importance = model.feature_importances_
    # Gain importance (using the booster)
    gain_importance = model.booster_.feature_importance(importance_type="gain")
    
    # Create a DataFrame combining both importances
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "frequency": split_importance,
        "gain": gain_importance
    })
    
    # Sort features based on gain importance and select top_n
    importance_df = importance_df.sort_values("gain", ascending=False).head(top_n)
    
    # Create two side-by-side bar plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot frequency importance
    axes[0].barh(importance_df["feature"][::-1], importance_df["frequency"][::-1])
    axes[0].set_title("Feature Importance (Frequency)")
    axes[0].set_xlabel("Frequency")
    
    # Plot gain importance
    axes[1].barh(importance_df["feature"][::-1], importance_df["gain"][::-1])
    axes[1].set_title("Feature Importance (Gain)")
    axes[1].set_xlabel("Gain")
    
    plt.tight_layout()
    plt.show()


cur_model = models[0]
feature_names = X_train.columns
plot_feature_importance(cur_model, feature_names)


y_pred = results['LGBMClassifier']['test']


sub = pd.read_csv('/kaggle/input/machine-learning-4-sbu/sample_submission.csv')
sub[target] = y_pred.astype(int)
sub.to_csv('submission.csv', index = False)
sub

