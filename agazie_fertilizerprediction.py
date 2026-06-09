import numpy as np
import pandas as pd
import shap
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



train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


train.head()


train.info()


test.info()


for column in train.select_dtypes(include='object'):
    counts = train[column].value_counts()
    print(counts)


    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribution of {column}")
    plt.axis("equal")
    plt.show()



numerical_features = train.select_dtypes("number")
feature_names = numerical_features.columns
num_features = len(feature_names)

# Define grid size (adjust as needed)
rows, columns = 2, 3

fig, ax = plt.subplots(rows, columns, figsize=(15, 8))
ax = np.atleast_2d(ax)

# Plot each histogram
for idx, column in enumerate(feature_names):
    i, j = divmod(idx, columns)
    ax[i, j].hist(numerical_features[column], bins=50, alpha=0.8)
    ax[i, j].set_title(f'Histogram of {column}')
    ax[i, j].set_xlabel(column)
    ax[i, j].set_ylabel('Count')

# Hide any unused subplots
for k in range(num_features, rows * columns):
    i, j = divmod(k, columns)
    fig.delaxes(ax[i, j])

plt.tight_layout()
plt.show()



corr = numerical_features.corr(method = "spearman")

corr_heatmap = sns.heatmap(corr, cmap="crest", annot = True)


for col in ['Soil Type', "Crop Type"]:
    plt.figure(figsize=(10,5))
    sns.countplot(x=col, hue = "Fertilizer Name", data = train)
    plt.title(f"Fertilizer Name by {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


X = train.drop(columns=["id", "Fertilizer Name"])
y = train["Fertilizer Name"]


# Encode labels 

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_encoded


categorial_features = X.select_dtypes("object").columns.to_list()


from catboost import CatBoostRegressor
# Store scores
f1_scores = []
map3_scores = []
models = []

# Collect predictions and true labels across all 
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

    model = CatBoostRegressor(posterior_sampling = True,
                             iterations = 1000)

    model.fit(
        X_train,
        y_train,
        cat_features = cat_features,
        eval_set=[(X_train, y_train),(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=500,
    )

    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap.plots.bar(shap_values)
    # Predict class labels and probabilities
    y_pred = model.predict(X_val)
    y_probs = model.virtual_ensembles_predict(X_val, prediction_type = "TotalUncertainty")[0]

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


for model in models:
    explainer = shap.Explainer(model)
    shap_values = explainer()


# Store scores
f1_scores = []
map3_scores = []
models = []

# Collect predictions and true labels across all 
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
                n_estimators=10000,
                learning_rate=0.01,
                gamma=0.26,
                max_delta_step=4,
                reg_alpha=2.7,
                reg_lambda=1.4,
                objective='multi:softprob',
                random_state=13,
                enable_categorical=True,
                tree_method='hist',     
                device='acuda'  
            )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train),(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=500,
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


for col in cat_features:
    test[col] = test[col].astype('category')

# Accumulate prediction probabilities
all_preds = np.zeros((test.shape[0], len(le.classes_)))

X_test = test.drop(columns='id')
cat_features = X_test.columns.tolist()   # capture all input columns

for model in models:
    probs = model.predict_proba(X_test)
    all_preds += probs

# Average over folds
avg_preds = all_preds / len(models)

# Get top 3 indices like before
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Top 3 class indices, descending order

# Convert class indices back to original label strings
top3_labels = le.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

submission = pd.DataFrame({
    'id': test['id'],  # Replace with actual ID column name
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

submission.to_csv('submission.csv', index=False)
print("Done!")

