from graphviz import Digraph
from IPython.display import Image, display

pipeline_graph = Digraph()
# larger canvas
pipeline_graph.attr(rankdir='LR', size='140,140')
# colorful filled nodes
pipeline_graph.attr('node', style='filled', fillcolor='lightgoldenrod1', color='darkorange', fontcolor='black')
pipeline_graph.attr('edge', color='gray')

pipeline_graph.node('A', 'Data Loading')
pipeline_graph.node('B', 'Preprocessing')
pipeline_graph.node('C', 'Feature Engineering')
pipeline_graph.node('D', 'Stratified K-Fold Split')
pipeline_graph.node('E', 'Model Training (LightGBM)')
pipeline_graph.node('F', 'Evaluation')
pipeline_graph.node('G', 'Feature Importance')
pipeline_graph.node('H', 'Inference')
pipeline_graph.node('I', 'Submission')

pipeline_graph.edges([
    ('A','B'), ('B','C'), ('C','D'), ('D','E'),
    ('E','F'), ('E','G'), ('E','H'), ('H','I')
])

# Render without cleanup, print filename, and display the image
output_path = pipeline_graph.render('ml_pipeline_diagram', format='png', cleanup=False)
print(f"Diagram saved to {output_path}")
display(Image(filename=output_path))



# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
print("All import done ")

warnings.filterwarnings("ignore")




path = '/kaggle/input/playground-series-s5e8/'
train = pd.read_csv(path + 'train.csv', index_col='id')
test = pd.read_csv(path + 'test.csv', index_col='id')
submission = pd.read_csv(path + 'sample_submission.csv', index_col='id')


#print(train.columns)
sns.countplot(x='y', data=train)
train = train.rename(columns={'y': 'target'})

print(train.info())
print("\nMissing values:")
print(train.isnull().sum())

# Check class distribution
sns.countplot(x='target', data=train)
plt.title("Target Variable Distribution")
plt.show()



# Drop ID column if present
train.drop(columns=["id"], inplace=True, errors='ignore')
test.drop(columns=["id"], inplace=True, errors='ignore')

# Convert object columns to categorical dtype
categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

# Separate target variable
X = train.drop(columns=["target"])
y = train["target"]

# Capture feature names for reuse
features = X.columns.tolist()


# You can add domain-specific or statistical features here
# For simplicity, we're using raw features in this notebook

# Statistical features on numerical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
X['row_sum'] = X[numeric_cols].sum(axis=1)
test['row_sum'] = test[numeric_cols].sum(axis=1)
X['row_mean'] = X[numeric_cols].mean(axis=1)
test['row_mean'] = test[numeric_cols].mean(axis=1)
X['row_std'] = X[numeric_cols].std(axis=1)
test['row_std'] = test[numeric_cols].std(axis=1)
# Interaction feature between first two numeric columns
f1, f2 = numeric_cols[:2]
X['interaction'] = X[f1] * X[f2]
test['interaction'] = test[f1] * test[f2]
# Refresh feature list for model input
features = X.columns.tolist()
print("Applied feature engineering: statistical summaries and interaction.")


# Initialize Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Placeholder for predictions and feature importance
test_preds = np.zeros((test.shape[0], len(np.unique(y))))
oof_preds = np.zeros((X.shape[0], len(np.unique(y))))
feature_importance_df = pd.DataFrame()

# LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y)),
    'learning_rate': 0.05,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'random_state': 42,
    'categorical_feature': categorical_cols
}



for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training fold {fold + 1}...")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    # Predict validation and test
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test) / skf.n_splits

    # Record feature importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

print("Training complete.")


# Compute Out-of-Fold log loss
score = log_loss(y, oof_preds)
print(f"OOF Log Loss: {score:.5f}")


# Average feature importance across folds
avg_importance = feature_importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_importance.values, y=avg_importance.index)
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.show()




# Create submission DataFrame with explicit id and positive-class probability
submission = pd.DataFrame({'id': test.index, 'y': test_preds[:, 1]})
print("Submission data!", submission.head())

print("Verification done for data!!!")

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

