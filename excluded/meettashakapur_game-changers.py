# 1. Setup & Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import shap
from lightgbm import early_stopping

# 2. Load Data
train = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/train.csv', engine='python')
test = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/test.csv', engine='python')
sub = pd.read_csv('/kaggle/input/burnout-datathon-ieeecsmuj/sample_submission.csv', engine='python')

# ğŸ§½ Clean Column Names
train.columns = train.columns.str.strip().str.lower().str.replace(" ", "_")
test.columns = test.columns.str.strip().str.lower().str.replace(" ", "_")

# âœ… Define target
target = 'lap_time_seconds'

# 3. EDA
sns.histplot(train[target], kde=True)
plt.title("Lap Time Distribution")
plt.show()

# 4. Preprocessing
data = pd.concat([train.drop(columns=[target]), test], sort=False)

numeric = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat = data.select_dtypes(include=['object']).columns.tolist()

# Impute missing values
imp_num = SimpleImputer(strategy='median')
data[numeric] = imp_num.fit_transform(data[numeric])

imp_cat = SimpleImputer(strategy='most_frequent')
data[cat] = imp_cat.fit_transform(data[cat])

# Convert numeric to float32 to save memory
for col in numeric:
    data[col] = data[col].astype(np.float32)

# Convert categoricals to category dtype
for col in cat:
    data[col] = data[col].astype('category')

# Final feature set
features = numeric + cat

# Split back to train/test
X_train = data.iloc[:len(train)][features]
y_train = train[target]
X_test = data.iloc[len(train):][features]

# 5. Modeling with KFold
oof = np.zeros(len(X_train))
preds = np.zeros(len(X_test))

kf = KFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"ğŸ”¥ Fold {fold+1}")
    
    X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
    X_tr, y_tr,
    eval_set=[(X_v, y_v)],
    eval_metric='rmse',
    callbacks=[early_stopping(50)],
    categorical_feature=cat,
)

    
    oof[val_idx] = model.predict(X_v)
    preds += model.predict(X_test) / kf.n_splits
    
    # Free up memory
    del X_tr, X_v, y_tr, y_v
    gc.collect()

print('âœ… OOF RMSE:', np.sqrt(mean_squared_error(y_train, oof)))

# 6. Feature Importance / SHAP (Optional on Sample)
print("ğŸ”� Generating SHAP values on a sample")
sample_X = X_train.sample(1000, random_state=42)
explainer = shap.Explainer(model)
shap_values = explainer(sample_X)
shap.summary_plot(shap_values, sample_X, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# 7. Submission
sub['lap_time_seconds'] = preds
sub.to_csv('submission.csv', index=False)
print("ğŸš€ Submission ready! Lap time predicted like a boss.")


