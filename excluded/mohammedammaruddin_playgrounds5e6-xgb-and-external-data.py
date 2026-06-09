# !pip install --quiet catboost xgboost



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from catboost import CatBoostClassifier



train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
original_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')
original_df.head()



le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

# Encode main dataset
train_df['Soil Type'] = le_soil.fit_transform(train_df['Soil Type'])
train_df['Crop Type'] = le_crop.fit_transform(train_df['Crop Type'])
train_df['Fertilizer Name'] = le_fert.fit_transform(train_df['Fertilizer Name'])

# Encode test dataset
test_df['Soil Type'] = le_soil.transform(test_df['Soil Type'])
test_df['Crop Type'] = le_crop.transform(test_df['Crop Type'])

# Encode original dataset (align columns)
original_df.rename(columns={'Temperature': 'Temparature'}, inplace=True)
original_df['Soil Type'] = le_soil.transform(original_df['Soil Type'])
original_df['Crop Type'] = le_crop.transform(original_df['Crop Type'])
original_df['Fertilizer Name'] = le_fert.transform(original_df['Fertilizer Name'])



# Ensure both datasets have the same columns
used_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
             'Nitrogen', 'Phosphorous', 'Potassium', 'Fertilizer Name']

combined_df = pd.concat([train_df[used_cols], original_df[used_cols]], ignore_index=True)

X = combined_df.drop('Fertilizer Name', axis=1)
y = combined_df['Fertilizer Name']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def add_interaction_features(df):
    df = df.copy()
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
    df['N_P_K_total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['NP_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
    df['PK_diff'] = df['Phosphorous'] - df['Potassium']
    return df

X_train_fe = add_interaction_features(X_train)
X_val_fe = add_interaction_features(X_val)
test_fe = add_interaction_features(test_df.drop(['id'], axis=1))

common_cols = X_train_fe.columns.tolist()
X_train_final = X_train_fe[common_cols]
X_val_final = X_val_fe[common_cols]
test_final = test_fe[common_cols]



def map3(actual, probs):
    top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    score = 0.0
    for i, pred in enumerate(top_3):
        if actual[i] in pred:
            index = np.where(pred == actual[i])[0][0]
            score += 1 / (index + 1)
    return score / len(actual)



xgb_gpu = XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(y.unique()),
    use_label_encoder=False,
    random_state=42
)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=xgb_gpu,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_log_loss',
    cv=3,
    verbose=2,
    n_jobs=1  # Keep GPU safe
)

search.fit(X_train_final, y_train)

print("âœ… Best Params:", search.best_params_)



best_xgb = search.best_estimator_
val_probs = best_xgb.predict_proba(X_val_final)
print("ðŸ“ˆ MAP@3 (tuned XGBoost + original data):", map3(y_val.values, val_probs))


test_probs = best_xgb.predict_proba(test_final)
top_3 = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top_3_labels = le_fert.inverse_transform(top_3.flatten()).reshape(top_3.shape)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv('submission.csv', index=False)
print("âœ… Submission file 'submission.csv' created and ready.")





