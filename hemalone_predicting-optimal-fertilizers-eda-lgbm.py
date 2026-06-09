import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


test_ids = df_test['id']


df_train.drop('id', inplace=True, axis=1)
df_test.drop('id', inplace=True, axis=1)
print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")


df_train.head()


df_test.head()


plt.figure(figsize = (10, 4))
sns.heatmap(df_train.isnull(), cbar = False, cmap = 'viridis')
plt.title('Missing Values Heatmap')
plt.show()


plt.figure(figsize = (10, 4))
sns.heatmap(df_test.isnull(), cbar = False, cmap = 'viridis')
plt.title('Missing Values Heatmap')
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(df_train.select_dtypes('number'), 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df_train[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


df_train.describe()


df_test.describe()


plt.figure(figsize=(12, 8))
corr = df_train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()


import plotly.express as px

nutrient_means = df_train.groupby('Crop Type')[['Nitrogen', 'Phosphorous', 'Potassium']].mean().reset_index()

df_melted = nutrient_means.melt(id_vars='Crop Type', var_name='Nutrient', value_name='Value')

fig = px.bar(
    df_melted,
    x='Crop Type',
    y='Value',
    color='Nutrient',
    color_discrete_sequence=['#4CAF50', '#2196F3', '#FFC107'],  
    title='Absolute NPK Values by Crop Type',
    labels={'Value': 'Absolute Value'},
    height=500
)

fig.update_layout(
    font=dict(size=12),
    hovermode='x unified',
    showlegend=True,
    plot_bgcolor='white'
)
fig.update_xaxes(tickangle=45)
fig.show()


from plotly.express import bar

top_fertilizers = df_train.groupby('Crop Type')['Fertilizer Name'].agg(lambda x: x.mode()[0]).reset_index()

fig = bar(
    top_fertilizers, 
    x='Crop Type', 
    y=[1]*len(top_fertilizers),  
    color='Fertilizer Name',
    title='Fertilizer by Crop Type',
    hover_data={'Fertilizer Name': True}
)
fig.update_layout(showlegend=False)
fig.show()


fig_3d = px.scatter_3d(
    df_train.sample(200),  
    x='Nitrogen',
    y='Phosphorous',
    z='Potassium',
    color='Fertilizer Name',
    symbol='Soil Type',
    hover_name='Crop Type',
    opacity=0.7,
    title='NPK vs. Soil/Crop'
)
fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig_3d.show()


temprature_bins = [
    df_train['Temparature'].min(),
    df_train['Temparature'].quantile(0.33),
    df_train['Temparature'].quantile(0.66),
    df_train['Temparature'].max()
]
df_train['Season'] = pd.cut(df_train['Temparature'], bins=temprature_bins, labels=['Cool', 'Moderate', 'Hot'], include_lowest=True)
df_test['Season'] = pd.cut(df_test['Temparature'], bins=temprature_bins, labels=['Cool', 'Moderate', 'Hot'], include_lowest=True)

moisture_bins = [
    df_train['Moisture'].min(),
    df_train['Moisture'].quantile(0.5),
    df_train['Moisture'].max()
]
df_train['Moisture_Level'] = pd.cut(df_train['Moisture'], bins=moisture_bins, labels=['Dry', 'Wet'], include_lowest=True)
df_test['Moisture_Level'] = pd.cut(df_test['Moisture'], bins=moisture_bins, labels=['Dry', 'Wet'], include_lowest=True)


le_target = LabelEncoder()
df_train['label'] = le_target.fit_transform(df_train['Fertilizer Name'])
num_classes = len(le_target.classes_)
print(f"Number of unique fertilizer classes: {num_classes}")


features = [col for col in df_train.columns if col not in ['Fertilizer Name', 'label']]
cat_cols = ['Soil Type', 'Crop Type', 'Season', 'Moisture_Level']

for col in cat_cols:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')

X = df_train[features]
X_test = df_test[features].copy()
y = df_train['label']


params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': num_classes,
    'n_estimators': 2000,         
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}



N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
test_preds_proba = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n----------- FOLD {fold + 1} -----------")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**params)
    
    early_stopping_callback = lgb.early_stopping(
        stopping_rounds=100,
        verbose=False
    )
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='multi_logloss',
              callbacks=[early_stopping_callback],
              categorical_feature=cat_cols) 
    
    fold_preds = model.predict_proba(X_test)
    test_preds_proba += fold_preds / N_SPLITS
    
    val_score = model.best_score_['valid_0']['multi_logloss']
    print(f"Fold {fold + 1} Validation LogLoss: {val_score}")


top3_indices = np.argsort(-test_preds_proba, axis=1)[:, :3]
top3_labels = [le_target.inverse_transform(idx) for idx in top3_indices]


submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(preds) for preds in top3_labels]
})



submission.to_csv('submission.csv', index=False)

