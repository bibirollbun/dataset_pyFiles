import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    expected_cols = ['Brand', 'Material', 'Size', 'Compartments', 
                    'Laptop Compartment', 'Waterproof', 'Style', 'Color', 
                    'Weight Capacity (kg)', 'Price']
    assert train.columns[1:].tolist() == expected_cols, "Train schema mismatch!"
    
    return train, test

train, test = load_data('/kaggle/input/playground-series-s5e2/train.csv', 
                        '/kaggle/input/playground-series-s5e2/test.csv')


print(f"Train shape: {train.shape}, Test shape: {test.shape}")


train.head()


test.head()


fig = px.histogram(train, x='Price', nbins=50, 
                   title='Price Distribution with Outlier Detection',
                   marginal='box', color_discrete_sequence=['#2A3132'])
fig.update_layout(bargap=0.1)
fig.show()


brand_stats = train.groupby('Brand')['Price'].agg(['mean', 'count'])
brand_stats = brand_stats[brand_stats['count'] > 10].sort_values('mean')
fig = px.bar(brand_stats, x='mean', y=brand_stats.index, orientation='h',
             title='Average Price by Brand (Minimum 10 Samples)')
fig.show()


def create_features(df):
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes':1, 'No':0})
    df['Waterproof'] = df['Waterproof'].map({'Yes':1, 'No':0})
    
    size_order = {'Small':1, 'Medium':2, 'Large':3}
    df['Size_encoded'] = df['Size'].map(size_order)
    
    df['Brand_Material'] = df['Brand'].fillna('Unknown') + '_' + df['Material'].fillna('Unknown')
    
    df['Compartment_per_kg'] = df['Compartments'] / (df['Weight Capacity (kg)'] + 1e-6)
    df['Volume_efficiency'] = df['Size_encoded'] * df['Compartments']
    
    df.fillna({'Brand': 'Unknown'}, inplace=True)
    
    return df


train = create_features(train)
test = create_features(test)


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Compartments', 'Weight Capacity (kg)', 'Compartment_per_kg', 'Size_encoded']),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Style', 'Color', 'Brand_Material']),
    ('drop', 'drop', ['Brand', 'Material', 'Size'])
])


xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.08,
    'subsample': 0.5,
    'colsample_bytree': 0.9,
    'gamma': 1.2,
    'tree_method': 'hist',
    'verbosity': 1
}


lgbm_params = {
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.85,  
    'n_jobs': -1,       
    'n_estimators': 100,
    'force_row_wise': True,
    'colsample_bytree': None  # Explicitly set to None
}



def ensure_numeric(df):
    return df.apply(pd.to_numeric, errors='coerce')


estimators = [
    ('xgb', XGBRegressor(**xgb_params)),
    ('lgbm', LGBMRegressor(**lgbm_params))
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=XGBRegressor(n_estimators=400, max_depth=6),
    passthrough=True
)


brands = train['Brand'].fillna('Unknown')
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

fold_scores = []
for train_idx, val_idx in kf.split(train, groups=brands):
    X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
    y_train, y_val = X_train['Price'], X_val['Price']
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    X_train_processed = ensure_numeric(pd.DataFrame(X_train_processed))
    X_val_processed = ensure_numeric(pd.DataFrame(X_val_processed))
    
    stack.fit(X_train_processed, y_train)
    
    preds = stack.predict(X_val_processed)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    fold_scores.append(rmse)
    print(f'Fold RMSE: {rmse:.4f}')

print(f'Mean CV RMSE: {np.mean(fold_scores):.4f}')


X_full = preprocessor.fit_transform(train)
y_full = train['Price']
test_processed = preprocessor.transform(test)

stack.fit(X_full, y_full)
final_preds = stack.predict(test_processed)

submission = pd.DataFrame({
    'id': test['id'],
    'price': final_preds
})
submission.to_csv('submission.csv', index=False)

