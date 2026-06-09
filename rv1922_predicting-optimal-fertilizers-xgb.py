import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train.head()


train.info()


# Drop the 'id' column
X = train.drop(columns=['id', 'Fertilizer Name'])

# Extract the target column
y = train['Fertilizer Name']

X.head()


# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Output labels are now numbers
y_encoded[:10]


orig_copy = original.copy()

# Number of copies
n = 9
for i in range(n):
    original = pd.concat([original, orig_copy], axis=0, ignore_index=True)
    
original.info()


# Store scores
f1_scores = []
map3_scores = []
models = []

# Collect predictions and true labels across all folds
all_y_true = []
all_y_pred = []

# Prepare K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n***** Fold {fold + 1} *****")

    # Make full copies to avoid warnings
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y_encoded[train_idx]
    y_val = y_encoded[val_idx]

    # Combine original with train data 
    X_train = pd.concat([X_train, original], ignore_index=True)
    y_train = np.concatenate([y_train, le.transform(original['Fertilizer Name'])])

    # Drop target column from training data
    X_train.drop(columns=['Fertilizer Name'], inplace=True)

    # Convert all features to categorical (except target, which is already separated)
    for col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        
    for col in X_val.columns:
        X_val[col] = X_val[col].astype('category')
    
    cat_features = X_train.columns.tolist()   # capture all input columns

    # For debugging purposes
    # print(cat_features)
    # print(X_train.info())
    # print(X_val.info())

    model = XGBClassifier(
                max_depth=7,
                colsample_bytree=0.4,
                subsample=0.8,
                n_estimators=20000,
                learning_rate=0.01,
                gamma=0.26,
                max_delta_step=4,
                reg_alpha=2.7,
                reg_lambda=1.4,
                objective='multi:softprob',
                random_state=13,
                enable_categorical=True,
                tree_method='hist',     
                device='cuda'  
            )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train),(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=1000,
    )
    
    # Predict class labels and probabilities
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)

    # Store predictions and true labels
    all_y_true.extend(y_val)
    all_y_pred.extend(y_pred)

    # F1 Score
    report = classification_report(y_val, y_pred, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]
    f1_scores.append(f1_macro)
    
    # MAP@3
    top3_preds = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]
    
    def mapk(actual, predicted, k=3):
        def apk(a, p, k):
            if a in p[:k]:
                return 1.0 / (p[:k].index(a) + 1)
            return 0.0
        return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

    map3 = mapk(y_val.tolist(), top3_preds.tolist(), k=3)
    map3_scores.append(map3)
    models.append(model)

    print(f"F1 (macro): {f1_macro:.4f} | MAP@3: {map3:.4f}")

# Final Results
print("\n***** Final CV Results *****")
print(f"Avg F1: {np.mean(f1_scores):.4f}")
print(f"Avg MAP@3: {np.mean(map3_scores):.4f}")


# Loss curves for the last fold
results = model.evals_result()
plt.plot(results['validation_0']['mlogloss'], label='Train')
plt.plot(results['validation_1']['mlogloss'], label='Val')
plt.legend()
plt.show()


from sklearn.metrics import classification_report

print(classification_report(y_val, y_pred, digits=4))


X_test = test.drop(columns='id')

# Convert test data
for col in X_test.columns:
    X_test[col] = X_test[col].astype('category')

# Accumulate prediction probabilities
all_preds = np.zeros((test.shape[0], len(le.classes_)))

for model in models:
    probs = model.predict_proba(X_test)
    all_preds += probs

# Average over folds
avg_preds = all_preds / len(models)

# Get top 3 indices like before
top3_preds = np.argsort(avg_preds, axis=1)[:, -3:][:, ::-1]  # Top 3 class indices, descending order

# Convert class indices back to original label strings
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

submission = pd.DataFrame({
    'id': test['id'],  # Replace with actual ID column name
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

submission.to_csv('submission.csv', index=False)
print("File Saved!")


submission.head()

