!pip install -U scikit-learn imbalanced-learn



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data proces


# Sklearn imports
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, LogisticRegression

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Skorch (PyTorch wrapper)
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

# SHAP for explainability
import shap

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Gradient boosting frameworks
import lightgbm as lgb
import xgboost as xgb

# Misc
import gc


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train_df.head(20)


train_df.info()


train_df.describe()


cols_with_nans = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                  'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']

for col in cols_with_nans:
    if train_df[col].dtype == 'float64':
        print(f"\nAverage {col} by Personality:")
        print(train_df.groupby('Personality')[col].mean())
    else:
        print(f"\nValue counts of {col} by Personality:")
        print(train_df.groupby('Personality')[col].value_counts(dropna=False))


cols_with_nans = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                  'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
for i in cols_with_nans:
    print(train_df[i].unique())


train_df['Personality'] = train_df['Personality'].str.strip().str.lower()
introvert_avg = {
    "Time_spent_Alone": 7.051937,
    "Social_event_attendance": 1.934202,
    "Going_outside": 1.534136,
    "Friends_circle_size": 3.263923,
    "Post_frequency": 1.611342
}

extrovert_avg = {
    "Time_spent_Alone": 1.747146,
    "Social_event_attendance": 6.389450,
    "Going_outside": 4.895894,
    "Friends_circle_size": 9.624587,
    "Post_frequency": 6.113682
}
for col in introvert_avg:
    mask_intro = (train_df['Personality'] == 'introvert') & (train_df[col].isna())
    mask_extro = (train_df['Personality'] == 'extrovert') & (train_df[col].isna())

    filled_intro = mask_intro.sum()
    filled_extro = mask_extro.sum()

    train_df.loc[mask_intro, col] = introvert_avg[col]
    train_df.loc[mask_extro, col] = extrovert_avg[col]




train_df['Personality'] = train_df['Personality'].str.lower().str.strip()
obj_cols = train_df.select_dtypes(include='object').columns.drop('Personality')
for col in obj_cols:
    intro_mask = train_df['Personality'] == 'introvert'
    extro_mask = train_df['Personality'] == 'extrovert'
    intro_data = train_df.loc[intro_mask, col]
    extro_data = train_df.loc[extro_mask, col]
    if intro_data.notna().any():
        mode_intro = intro_data.mode(dropna=True)[0]
        train_df.loc[intro_mask & train_df[col].isna(), col] = mode_intro
    if extro_data.notna().any():
        mode_extro = extro_data.mode(dropna=True)[0]
        train_df.loc[extro_mask & train_df[col].isna(), col] = mode_extro

# Step 4: Final check
print("\n Updated DataFrame info:")
print(train_df.info())


categorical_features = ['Stage_fear', 'Drained_after_socializing']

encoded_train = pd.get_dummies(train_df,columns= ['Stage_fear', 'Drained_after_socializing'], dtype="int64")
encoded_test = pd.get_dummies(test_df,columns=['Stage_fear', 'Drained_after_socializing'], dtype="int64")
encoded_test.info()


def corr(train_encoded):
    if 'Personality' in train_encoded.columns:
        train_corr = train_encoded.drop(columns=['Personality'])
    else:
        train_corr = train_encoded.copy()
    corr_matrix = train_corr.corr()
    
    plt.figure(figsize=(60, 40))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", linewidths=0.5, fmt=".2f")
    
    plt.title("Blue Correlation Heatmap")
    plt.show()
print(corr(encoded_train))





X_train,y_train = encoded_train.drop(columns=['Personality']),encoded_train['Personality']
le = LabelEncoder()
y_train = le.fit_transform(y_train)


standard = StandardScaler()
X_train_sc = standard.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_sc,columns=X_train.columns)


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
imputer = SimpleImputer(strategy='mean')
X_test = encoded_test.copy()
X_test[num_cols] = imputer.fit_transform(X_test[num_cols])
X_test_scaled = standard.fit_transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_test_scaled_df.info()



sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)


X_base = X_res.copy()
y_base = y_res.copy()
numerical_cols = X_base.select_dtypes(include=['float64', 'int64']).columns.tolist()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_base[numerical_cols])
feature_names = poly.get_feature_names_out(numerical_cols)
X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly_df)
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y_base)  
selected_features = np.array(feature_names)[lasso.coef_ != 0]
X_selected_df = X_poly_df[selected_features]



class SimpleNNModule(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNNModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameter tuning for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5]
}

grid_search_xgb = GridSearchCV(
    estimator=xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
        random_state=42
    ),
    param_grid=xgb_param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)

# L1: Neural Network
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_nn = np.zeros(X_res.shape[0])
test_preds_all_folds = []

X_tensor = torch.tensor(X_res.values, dtype=torch.float32)
y_tensor = torch.tensor(y_res.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled_df.values, dtype=torch.float32)

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_res, y_res)):
    print(f"\n Fold {fold+1}/5")

    X_train_tensor = X_tensor[train_idx]
    y_train_tensor = y_tensor[train_idx]
    X_val_tensor = X_tensor[valid_idx]
    y_val_tensor = y_tensor[valid_idx]

    model = SimpleNNModule(X_res.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).numpy().flatten()
        test_preds = model(X_test_tensor).numpy().flatten()

    oof_preds_nn[valid_idx] = val_preds
    test_preds_all_folds.append(test_preds)

    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    gc.collect()

print("\n L1 Neural Network training complete!")

# Test-Time Ensembling: Averaging predictions
ensemble_test_preds = np.mean(test_preds_all_folds, axis=0)

# L2: XGBoost on NN predictions
X_l2_train = pd.DataFrame({'nn_pred': oof_preds_nn})
X_l2_test = pd.DataFrame({'nn_pred': ensemble_test_preds})

grid_search_xgb.fit(X_l2_train, y_res)
print("\n Best XGBoost Parameters:", grid_search_xgb.best_params_)

best_xgb_model = grid_search_xgb.best_estimator_
final_preds = best_xgb_model.predict(X_l2_test)

print("\n XGBoost L2 model trained and predicted!")



submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
TARGET_COL = 'Personality'
label_map = {
    'introvert': 'Introvert',
    'extrovert': 'Extrovert'
}
test_preds = final_preds
decoded_preds = le.inverse_transform(test_preds)  
final_preds = [label_map[label] for label in decoded_preds] 

submission[TARGET_COL] = final_preds
submission.to_csv('submission.csv', index=False)
submission.head(25)


submission.head(25)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
sample_submission.head(20)

