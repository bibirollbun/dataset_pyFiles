!pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv',index_col = 0)
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col=0)


sample_submission.shape


train.shape


test.shape


train.info()


train.describe()


test.info()


test.describe()


X = train.drop('Personality', axis=1)
y = train['Personality']


target_encoder = LabelEncoder()
y = pd.Series(target_encoder.fit_transform(y))


def preprocess_fold(X_train, X_val):
    for df in [X_train, X_val]:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.drop(columns=['id'], inplace=True, errors='ignore')
        df['stage_fear'] = df['stage_fear'].fillna('unknown')
        df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
        #df['personality'] = df['personality'].fillna('unknown')
        for col in df.select_dtypes(include='number').columns:
            df[col] = df[col].fillna(df[col].mean())
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = encoder.transform(X_val[cat_cols])
    return X_train, X_val, encoder


smote = SMOTE(random_state=42)


classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Classifier": RidgeClassifier(),
    "SGD Classifier": SGDClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "HistGradient Boosting": HistGradientBoostingClassifier(),
    "SVC": SVC(),
    "Linear SVC": LinearSVC(max_iter=10000),
    "Gaussian NB": GaussianNB(),
    "Multinomial NB": MultinomialNB(),
    "KNN": KNeighborsClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "MLP": MLPClassifier(max_iter=1000)
}


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


all_models = {} 
all_encoders = {}

for name, clf in classifiers.items():
    print(f"\n[INFO] Training and validating {name}...")
    fold_scores = []
    fold_models = []
    fold_encoders = [] 
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        X_train_prep, X_val_prep, encoder = preprocess_fold(X_train, X_val)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_prep, y_train)
        model = clone(clf)
        model.fit(X_train_resampled, y_train_resampled)
        y_val_pred = model.predict(X_val_prep)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"[INFO] Fold {fold + 1} Accuracy: {acc:.6f}")
        fold_scores.append(acc)
        fold_models.append(model)
        fold_encoders.append(encoder)  
    mean_acc = np.mean(fold_scores)
    print(f"[INFO] Mean CV Accuracy for {name}: {mean_acc:.6f}")    
    all_models[name] = fold_models
    all_encoders[name] = fold_encoders  


def preprocess_the_test_set(df, encoder):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    ids = df['id'] if 'id' in df.columns else pd.Series(range(len(df)))    
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['stage_fear'] = df['stage_fear'].fillna('unknown')
    df['drained_after_socializing'] = df['drained_after_socializing'].fillna('unknown')
    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = encoder.transform(df[cat_cols])
    return df, ids


results_df = pd.DataFrame()
_, test_ids = preprocess_the_test_set(test.copy(), all_encoders[list(all_encoders.keys())[0]][0])
results_df['id'] = test_ids

for name, models in all_models.items():
    fold_encoders = all_encoders[name]
    if hasattr(models[0], "predict_proba"):
        probas = np.zeros((len(test), len(target_encoder.classes_)))
        for model, encoder in zip(models, fold_encoders):
            X_test_prep, _ = preprocess_the_test_set(test.copy(), encoder)
            probas += model.predict_proba(X_test_prep)
        probas /= len(models)
        preds = target_encoder.inverse_transform(np.argmax(probas, axis=1))
    else:
        preds_folds = []
        for model, encoder in zip(models, fold_encoders):
            X_test_prep, _ = preprocess_the_test_set(test.copy(), encoder)
            preds_folds.append(model.predict(X_test_prep))
        preds_folds = np.array(preds_folds)
        preds_mode, _ = mode(preds_folds, axis=0)
        preds = target_encoder.inverse_transform(preds_mode.flatten())

    col_name = f"Personality_{name.replace(' ', '_').lower()}"
    results_df[col_name] = preds


results_df


true_labels = sample_submission['Personality']

# Dictionary to store accuracy per model
model_accuracies = {}

for col in results_df.columns:
    if col == 'id':
        continue  # skip ID column
    preds = results_df[col]
    acc = accuracy_score(true_labels, preds)
    model_accuracies[col] = acc
    print(f"Accuracy for {col}: {acc:.4f}")


test = test.reset_index()

submission = test[['id']].copy()
submission['Personality'] = results_df['Personality_histgradient_boosting'].values


submission


submission.to_csv('submission.csv', index=False)

print("Submission file saved successfully!")

