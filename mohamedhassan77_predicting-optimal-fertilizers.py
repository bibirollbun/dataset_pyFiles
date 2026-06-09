import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
train_df.head()


train_df.info()


test_df.info()


# Label Encode the Target Variable
target = 'Fertilizer Name'
label_encoder = LabelEncoder()
# y_train
y_train = label_encoder.fit_transform(train_df[target])
print(f"Original unique values: {label_encoder.classes_}")


# One-Hot Encode Categorical Features ('Soil Type', 'Crop Type')
categorical_features = ['Soil Type', 'Crop Type']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
one_hot_encoder.fit(train_df[categorical_features])
train_encoded_features = one_hot_encoder.transform(train_df[categorical_features])
test_encoded_features = one_hot_encoder.transform(test_df[categorical_features])


encoded_cols = one_hot_encoder.get_feature_names_out(categorical_features)
train_encoded_df = pd.DataFrame(train_encoded_features, columns=encoded_cols, index=train_df.index)
test_encoded_df = pd.DataFrame(test_encoded_features, columns=encoded_cols, index=test_df.index)

train_encoded_df.head()


X_train = train_df.drop(columns=categorical_features + [target, 'id'])
X_test = test_df.drop(columns=categorical_features + ['id'])

X_train = pd.concat([X_train, train_encoded_df], axis=1)
X_test = pd.concat([X_test, test_encoded_df], axis=1)


X_train.head()


print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")


# store ids to use in the output file
ids = test_df['id']


xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)


# predict probs
y_probs_xgb = xgb_model.predict_proba(X_test)
# get top 3 descending
top_3_preds = np.argsort(y_probs_xgb, axis=1)[:, -3:][:, ::-1]
# convert to original labels
top_3_labels = np.vectorize(lambda idx: label_encoder.classes_[idx])(top_3_preds)
# join with spaces
top_3_strs = [' '.join(row) for row in top_3_labels]

# sublission file
submission = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': top_3_strs
})

submission.to_csv('/kaggle/working/submission_xgb.csv', index=False)


rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)


# predict probs
y_probs_rf = rf_model.predict_proba(X_test)
# get top 3 descending
top_3_preds = np.argsort(y_probs_rf, axis=1)[:, -3:][:, ::-1]
# convert to original labels
top_3_labels = np.vectorize(lambda idx: label_encoder.classes_[idx])(top_3_preds)
# join with spaces
top_3_strs = [' '.join(row) for row in top_3_labels]

# sublission file
submission = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': top_3_strs
})

submission.to_csv('/kaggle/working/submission_rf.csv', index=False)


# no one-hot
# Encode target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[target])

X_train = train_df.drop(columns=[target, 'id'])
X_test = test_df.drop(columns=['id'])
combined = pd.concat([X_train, X_test], axis=0)
for col in categorical_features:
    combined[col] = combined[col].astype('category')

# Split combined back into train and test
X_train = combined.iloc[:len(X_train), :].copy()
X_test = combined.iloc[len(X_train):, :].copy()


# Create LightGBM dataset with categorical features specified
lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)

params = {
    'objective': 'multiclass',
    'num_class': len(label_encoder.classes_),
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'random_state': 42,
    'verbosity': -1
}

lgb_model = lgb.train(params, lgb_train, num_boost_round=300)


# predict probs
y_probs_lgb = lgb_model.predict(X_test)
# get top 3 descending
top_3_preds = np.argsort(y_probs_lgb, axis=1)[:, -3:][:, ::-1]
# convert to original labels
top_3_labels = np.vectorize(lambda idx: label_encoder.classes_[idx])(top_3_preds)
# join with spaces
top_3_strs = [' '.join(row) for row in top_3_labels]

# sublission file
submission = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': top_3_strs
})

submission.to_csv('/kaggle/working/submission_lgb.csv', index=False)


assert y_probs_lgb.shape == y_probs_xgb.shape == y_probs_rf.shape
y_probs_ensemble = (y_probs_lgb + y_probs_xgb + y_probs_rf) / 3

top_3_preds = np.argsort(y_probs_ensemble, axis=1)[:, -3:][:, ::-1]
top_3_labels = np.vectorize(lambda idx: label_encoder.classes_[idx])(top_3_preds)
top_3_strs = [' '.join(row) for row in top_3_labels]

submission_ensemble = pd.DataFrame({
    'id': ids,
    'Fertilizer Name': top_3_strs
})

submission_ensemble.to_csv('/kaggle/working/submission_ensemble.csv', index=False)

