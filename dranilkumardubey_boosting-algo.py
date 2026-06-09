pip install optuna-integration[sklearn]


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from optuna.integration import OptunaSearchCV
from skopt import BayesSearchCV
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

print(train.shape, test.shape)
train.head()



target = 'Personality'  # Assuming target column is named this
id_col = 'id' if 'id' in train.columns else None

# Encode target
le = LabelEncoder()
train[target] = le.fit_transform(train[target])  # 0=Introvert, 1=Extrovert

# Separate features
X = train.drop([target, id_col], axis=1) if id_col else train.drop(target, axis=1)
y = train[target]
X_test = test.drop([id_col], axis=1) if id_col else test.copy()

# Identify columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

# Column transformer
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])


df_viz = train.copy()

# Label encode categorical columns
for col in cat_cols:
    df_viz[col] = LabelEncoder().fit_transform(df_viz[col].astype(str))

# Correlation matrix
corr_matrix = df_viz.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Including Personality)', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# Option 1: Simple imputation (mean for numeric, most frequent for categorical)
from sklearn.impute import SimpleImputer

# Define imputers
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Impute numeric
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Impute categorical
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])



X_transformed = preprocessor.fit_transform(X)
X_test_transformed = preprocessor.transform(X_test)

from sklearn.decomposition import PCA

# PCA for 95% variance
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_transformed)
X_test_pca = pca.transform(X_test_transformed)

# Optional: Check how many components were retained
print(f"PCA reduced features from {X_transformed.shape[1]} to {X_pca.shape[1]}")


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lgb = LGBMClassifier(random_state=42)
catb = CatBoostClassifier(verbose=0, random_state=42)

# Stacking
stack = StackingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('cat', catb)],
    final_estimator=LGBMClassifier(),
    passthrough=True,
    cv=5
)


search_space = {
    'xgb__max_depth': (3, 10),
    'xgb__learning_rate': (0.01, 0.3, 'log-uniform'),
    'lgb__num_leaves': (20, 100),
    'cat__depth': (4, 10),
}

bayes_search = BayesSearchCV(
    estimator=stack,
    search_spaces=search_space,
    cv=StratifiedKFold(n_splits=5),
    n_iter=20,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
    random_state=42
)



X_train, X_val, y_train, y_val = train_test_split(X_pca, y, test_size=0.2, random_state=42)
bayes_search.fit(X_train, y_train)

# Best model
best_model = bayes_search.best_estimator_
y_pred = best_model.predict(X_val)



def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("RMSLE:", rmsle(y_val, y_pred))


final_preds = best_model.predict(X_test_pca)
submission[target] = le.inverse_transform(final_preds)
submission.to_csv('submission.csv', index=False)




