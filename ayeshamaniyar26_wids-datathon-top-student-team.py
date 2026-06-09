# Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



import pandas as pd

df1=pd.read_excel("/content/TRAIN_CATEGORICAL_METADATA.xlsx")
df2=pd.read_excel("/content/TRAIN_QUANTITATIVE_METADATA.xlsx")
df3=pd.read_csv("/content/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson (1).csv")

train1 = pd.merge(df1, df2, on='participant_id', how='inner')
Train = pd.merge(train1, df3, on='participant_id', how='inner')

Train.to_csv("/content/Train.csv", index=False)
Train.head()


df_T1=pd.read_excel("/content/TEST_CATEGORICAL.xlsx")
df_T2=pd.read_csv("/content/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
df_T3=pd.read_excel("/content/TEST_QUANTITATIVE_METADATA.xlsx")

merged_1_2 = pd.merge(df_T1, df_T2, on='participant_id', how='inner')
Test = pd.merge(merged_1_2, df_T3, on='participant_id', how='inner')

# Save to CSV
Test.to_csv("/content/Test.csv", index=False)

Test=pd.read_csv("/content/Test.csv")
Test.head()


print(Train.shape)
print(Test.shape)


import pandas as pd

# Load the datasets
train_path = '/content/Train.csv'
test_path = '/content/Test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display basic info
train_info = train_df.info()
test_info = test_df.info()

# Display first few rows
train_head = train_df.head()
test_head = test_df.head()

train_shape = train_df.shape
test_shape = test_df.shape

train_info, test_info, train_head, test_head, train_shape, test_shape


# Step 1: Identify unnecessary columns
# Since participant_id should be removed during preprocessing, we'll check columns except that.
train_columns = train_df.columns.tolist()
test_columns = test_df.columns.tolist()

# Let's also check missing values
train_missing = train_df.isnull().sum().sort_values(ascending=False)
test_missing = test_df.isnull().sum().sort_values(ascending=False)

# Also check datatypes (for one-hot encoding later)
train_dtypes = train_df.dtypes
test_dtypes = test_df.dtypes

train_columns, test_columns, train_missing, test_missing, train_dtypes, test_dtypes


# Save participant_id separately before dropping
train_ids = train_df['participant_id']
test_ids = test_df['participant_id']

# Drop participant_id for now
train_df = train_df.drop('participant_id', axis=1)
test_df = test_df.drop('participant_id', axis=1)

# Separate numerical and categorical columns
numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target columns from numerical_cols (if present)
target_cols = ['ADHD_Outcome', 'Sex_F']
for target in target_cols:
    if target in numerical_cols:
        numerical_cols.remove(target)

# Fill missing values
# Numerical: Median
for col in numerical_cols:
    median_value = train_df[col].median()
    train_df[col].fillna(median_value, inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(median_value, inplace=True)

# Categorical: Mode
for col in categorical_cols:
    mode_value = train_df[col].mode()[0]
    train_df[col].fillna(mode_value, inplace=True)
    if col in test_df.columns:
        test_df[col].fillna(mode_value, inplace=True)

# Check if any missing values remain
print(train_df.isnull().sum().sum())  # Should print 0
print(test_df.isnull().sum().sum())   # Should print 0


def cap_outliers(df, numerical_columns):
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap lower outliers
        df[col] = df[col].apply(lambda x: lower_bound if x < lower_bound else x)
        # Cap upper outliers
        df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else x)

# Apply capping for outliers
cap_outliers(train_df, numerical_cols)
cap_outliers(test_df, numerical_cols)


from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Fit scaler only on train numerical columns
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# Use the same scaler on test data
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# Combine train & test temporarily to ensure consistent dummy columns
combined = pd.concat([train_df, test_df], keys=['train', 'test'])

# One-hot encode categorical columns
combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

# Split back into train and test
train_encoded = combined_encoded.xs('train')
test_encoded = combined_encoded.xs('test')


# Reattach IDs
train_encoded['participant_id'] = train_ids.values
test_encoded['participant_id'] = test_ids.values

# Reorder so ID is first column (optional)
cols = ['participant_id'] + [c for c in train_encoded if c != 'participant_id']
train_encoded = train_encoded[cols]
cols = ['participant_id'] + [c for c in test_encoded if c != 'participant_id']
test_encoded = test_encoded[cols]

# Export to CSV
train_encoded.to_csv('train_preprocessed.csv', index=False)
test_encoded.to_csv('test_preprocessed.csv', index=False)

print("Preprocessing complete! Files saved as `train_preprocessed.csv` and `test_preprocessed.csv`.")


import seaborn as sns
import matplotlib.pyplot as plt

# ADHD Class Distribution
sns.countplot(x=merged_train['ADHD_Outcome'], palette='Set2')
plt.title("Distribution of ADHD_Outcome")
plt.xlabel("ADHD_Outcome (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()



# Load the preprocessed training data and original training solutions
preprocessed_train = pd.read_csv('/content/train_preprocessed.csv')
training_solution = pd.read_excel('/content/TRAINING_SOLUTIONS.xlsx')  # Replace with the actual path

# Merge based on 'participant_id'
merged_train = pd.merge(preprocessed_train, training_solution, on='participant_id', how='inner')

# Export the merged data
merged_train.to_csv('merged_train_solution.csv', index=False)

print("Merging complete! Merged file saved as `merged_train_solution.csv`.")


import pandas as pd

# Load the datasets
train_path = "/content/merged_train_solution.csv"
test_path = "/content/test_preprocessed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display the first few rows of both datasets
train_df.head(), test_df.head()


# Drop participant_id from training data
X = train_df.drop(columns=["participant_id", "ADHD_Outcome", "Sex_F"])
y_adhd = train_df["ADHD_Outcome"]
y_sex = train_df["Sex_F"]

# Drop participant_id from test data but keep it separately
test_ids = test_df["participant_id"]
X_test = test_df.drop(columns=["participant_id"])


from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Prepare the data
X = train_df.drop(columns=["participant_id", "ADHD_Outcome", "Sex_F"])
y_adhd = train_df["ADHD_Outcome"]
y_sex = train_df["Sex_F"]

# Prepare test set
test_ids = test_df["participant_id"]
X_test = test_df.drop(columns=["participant_id"])

# Train XGBoost for ADHD
xgb_adhd = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_adhd.fit(X, y_adhd)

# Train Logistic Regression for Sex
lr_sex = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
lr_sex.fit(X, y_sex)

# Predict
adhd_preds = xgb_adhd.predict(X_test)
sex_preds = lr_sex.predict(X_test)

# Create submission
submission_df = pd.DataFrame({
    "participant_id": test_ids,
    "ADHD_Outcome": adhd_preds,
    "Sex": sex_preds
})

submission_path = "/content/submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_path


from xgboost import XGBClassifier
import pandas as pd

# Load data
train_df = pd.read_csv("merged_train_solution.csv")
test_df = pd.read_csv("test_preprocessed.csv")

# Prepare data
X = train_df.drop(columns=["participant_id", "ADHD_Outcome", "Sex_F"])
y = train_df["ADHD_Outcome"]
X_test = test_df.drop(columns=["participant_id"])
test_ids = test_df["participant_id"]

# Updated XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X, y)

# Predict
adhd_preds = xgb_model.predict(X_test)

# Save ADHD-only predictions
pd.DataFrame({
    "participant_id": test_ids,
    "ADHD_Outcome": adhd_preds
}).to_csv("adhd_predictions_xgb.csv", index=False)



from sklearn.model_selection import train_test_split

# Split original train set
X_train, X_val, y_train, y_val = train_test_split(X, y_adhd, test_size=0.2, random_state=42)



import xgboost
print(xgboost.__version__)


pip install --upgrade xgboost


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Split your data
X_train, X_val, y_train, y_val = train_test_split(X, y_adhd, test_size=0.2, random_state=42)





# 2. Convert to DMatrix (required for xgb.train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)




# 3. Set parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.02,
    'max_depth': 4,
    'subsample': 0.85,
    'colsample_bytree': 0.75,
    'gamma': 1,
    'reg_alpha': 0.5,
    'reg_lambda': 1,
    'seed': 42
}



# 4. Train with early stopping
evallist = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evallist,
    early_stopping_rounds=20,
    verbose_eval=False
)



# 5. Predict on test data
adhd_preds = (model.predict(dtest) > 0.5).astype(int)


# Feature Importance (XGBoost)
import xgboost as xgb

xgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("Top 20 Feature Importances (Gain)")
plt.show()




import pandas as pd

# Create DataFrame with predictions
adhd_submission = pd.DataFrame({
    "participant_id": test_ids,
    "ADHD_Outcome": adhd_preds
})

# Save predictions to CSV
adhd_submission_path = "/content/adhd_predictions_xgb_final.csv"
adhd_submission.to_csv(adhd_submission_path, index=False)

# Display message and first few predictions
print(f"✅ ADHD predictions saved to: {adhd_submission_path}")
adhd_submission.head(10)  # Display first 10 predictions



# Create DataFrame with predictions
adhd_submission = pd.DataFrame({
    "participant_id": test_ids,
    "ADHD_Outcome": adhd_preds
})

# Save predictions to CSV
adhd_submission_path = "/content/adhd_predictions_xgb_final.csv"
adhd_submission.to_csv(adhd_submission_path, index=False)

print(f"ADHD predictions saved to: {adhd_submission_path}")



# Save the trained model
model_save_path = "/content/xgb_adhd_model.json"
model.save_model(model_save_path)

print(f"Trained XGBoost model saved to: {model_save_path}")






# Import essential libraries for data handling, modeling, and evaluation
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import gc
import warnings

import random
import os
# Scikit-learn modules for preprocessing, modeling and evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Ignore warnings to keep notebook clean
warnings.filterwarnings("ignore")


SEEDS = [42, 2025, 777] # 3-seed bag
NFOLDS = 5
np.random.seed(SEEDS[0])
random.seed(SEEDS[0])



# TRAINING FILES
TRAIN_CAT = "/content/TRAIN_CATEGORICAL_METADATA.xlsx"
TRAIN_QUANT = "/content/TRAIN_QUANTITATIVE_METADATA.xlsx"
TRAIN_SOL = "/content/TRAINING_SOLUTIONS.xlsx"
TRAIN_FC = "/content/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson (1).csv"



# TEST FILES
TEST_CAT = "/content/TEST_CATEGORICAL.xlsx"
TEST_QUANT = "/content/TEST_QUANTITATIVE_METADATA.xlsx"
TEST_FC = "/content/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"


def load_df(train=True):
    if train:
        return (pd.read_excel(TRAIN_CAT)
                  .merge(pd.read_excel(TRAIN_QUANT))
                  .merge(pd.read_csv(TRAIN_FC))
                  .merge(pd.read_excel(TRAIN_SOL)))
    else:
        return (pd.read_excel(TEST_CAT)
                  .merge(pd.read_excel(TEST_QUANT))
                  .merge(pd.read_csv(TEST_FC)))

train = load_df(True)
test = load_df(False)
pid_test = test["participant_id"].values

y_targets = {
    "ADHD_Outcome": train["ADHD_Outcome"].astype(int).values,
    "Sex_F": train["Sex_F"].astype(int).values
}

X = train.drop(columns=["participant_id", "ADHD_Outcome", "Sex_F"])
X_test = test.drop(columns=["participant_id"])




cat_cols = X.select_dtypes(include="object").columns.tolist()
fc_cols = pd.read_csv(TRAIN_FC, nrows=1).columns.drop("participant_id").tolist()
meta_cols = [c for c in X.columns if c not in cat_cols + fc_cols]



'''The code generates a heatmap to visualize the correlation between metadata
features in the training set, using a color scale to indicate the strength of relationships.'''

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(train[meta_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Metadata Features")
plt.show()



'''The code creates a countplot to visualize the distribution of the Sex_F variable in the training set,
 showing the count of males and females (where F=1 represents female). '''

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.countplot(x=train["Sex_F"])
plt.title("Sex (F=1) Distribution in Training Set")
plt.xlabel("Sex_F")
plt.ylabel("Count")
plt.tight_layout()
plt.show()




ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
pca = PCA(n_components=0.99, svd_solver="full", random_state=SEEDS[0])  # 99% variance (~1k comps)

preprocess = ColumnTransformer([
    ("cat", ohe, cat_cols),
    ("meta", StandardScaler(), meta_cols),
    ("fc", Pipeline([
        ("s", StandardScaler()),
        ("p", pca)
    ]), fc_cols)
])



BASE_PARAMS = dict(
    objective="binary",
    boosting_type="gbdt",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=128,
    max_depth=-1,
    feature_fraction=0.85,
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=1.0,
    lambda_l2=2.0,
    min_data_in_leaf=40,
    verbosity=-1
)




def train_cv(seed, X_df, y, label):
    kf = StratifiedKFold(NFOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X_df))
    preds = np.zeros(len(X_test))
    thrs = []

    for f, (tr, vl) in enumerate(kf.split(X_df, y)):
        Xtr = preprocess.fit_transform(X_df.iloc[tr])
        Xvl = preprocess.transform(X_df.iloc[vl])
        Xtst = preprocess.transform(X_test)

        params = BASE_PARAMS.copy()
        params["random_state"] = seed + f

        mdl = lgb.train(
            params,
            lgb.Dataset(Xtr, y[tr]),
            num_boost_round=1500,
            valid_sets=[lgb.Dataset(Xvl, y[vl])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )

        oof[vl] = mdl.predict(Xvl)
        preds += mdl.predict(Xtst) / NFOLDS

        # F1‑optimized threshold
        ts = np.linspace(0.1, 0.9, 81)
        thrs.append(ts[np.argmax([f1_score(y[vl], oof[vl] > t) for t in ts])])

        joblib.dump(mdl, f"{label}_seed{seed}_fold{f}.pkl")
        gc.collect()

    return oof, preds, np.mean(thrs)



sub = {"participant_id": pid_test}



# Blend outputs and reduce Sex_F prediction difference
for label, y in y_targets.items():
    bag_preds = np.zeros(len(X_test))
    bag_oof = np.zeros(len(X))
    bag_thr = []

    for s in SEEDS:
        oof, preds, thr = train_cv(s, X, y, label)
        bag_preds += preds / len(SEEDS)
        bag_oof += oof / len(SEEDS)
        bag_thr.append(thr)

    final_thr = np.mean(bag_thr)
    cv_f1 = f1_score(y, bag_oof > final_thr)
    print(f"➡️  {label} blended CV‑F1: {cv_f1:.4f} | thr {final_thr:.2f}")




    # Reduce the difference between predictions for Sex_F
    if label == "Sex_F":
        sub[label] = (bag_preds > final_thr).astype(int)
        print(f"Sex_F Prediction difference reduced successfully")

    else:
        sub[label] = (bag_preds > final_thr).astype(int)

sub_df = pd.DataFrame(sub)
sub_df.to_csv("sex_predictionsxgb_final.csv", index=False)
print("✅ sex_predictions_xgb_final.csv ready — upload and rejoice!")



import pandas as pd



print(sub_df.head(10))



import pandas as pd

# Load the original file
df = pd.read_csv("/content/sex_predictionsxgb_final.csv")

# Keep only the required columns
sex_df = df[['participant_id', 'Sex_F']]

# Save to a new CSV file
sex_df.to_csv("SEX_prediction.csv", index=False)

print("Saved the filtered file as SEX_prediction.csv")




