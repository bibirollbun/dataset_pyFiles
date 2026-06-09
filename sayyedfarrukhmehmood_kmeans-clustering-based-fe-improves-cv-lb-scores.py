import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm



# Load training, test, and submission files
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").drop(['id'], axis=1)
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv").drop(['id'], axis=1)
train.head()




kf=KFold(n_splits=5,shuffle=True,random_state=42)

# Target Encoding
for i,(train_index, test_index) in enumerate(kf.split(train)):
    X_train=train.loc[train_index]
    means=X_train.groupby('Sex')['Calories'].agg('mean')
    train.loc[test_index,'TE_Sex']=train.loc[test_index,'Sex'].map(means)

full_means=train.groupby('Sex')['Calories'].agg('mean')
test['TE_Sex']=test['Sex'].map(full_means)



def feature_engineer(df):
      # BMR
    df['BMR'] =  np.where(df['Sex'] == 'Male',
                    10 * df['Weight'] + 6.25 * df['Height']/100 - 5 * df['Age'] + 5,
                    10 * df['Weight'] + 6.25 * df['Height']/100 - 5 * df['Age'] - 161)

    ## BMI
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    # Duration / Heart Rate
    df["Duration_per_HeartRate"] = df["Duration"] / (df["Heart_Rate"] + 1e-5)
    # Intensity
    df['Intensity'] = df["Heart_Rate"] / (df["Duration"] + 1e-5)

    # Other
    df["Duration_x_HeartRate"] = df["Duration"] * df["Heart_Rate"]

    # Label Encode
    df['Sex']=df['Sex'].map({"male":1,"female":0})

    return df
    
train=feature_engineer(train)
test=feature_engineer(test)
train.head()


clustering_features=['Age','Height','Duration','Heart_Rate','Body_Temp']
k=5
for feature in clustering_features:
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train[feature].to_frame())
    train[f'{feature}_clusters']=kmeans.predict(train[feature].to_frame())
    test[f'{feature}_clusters']=kmeans.predict(test[feature].to_frame())

train.head()


plt.figure(figsize=(20, 12))
for i, feature in enumerate(clustering_features, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(
        data=train,
        x=feature,
        hue=f'{feature}_clusters',
        common_norm=False,
        fill=True,
        palette='tab10',
        alpha=0.6,
        linewidth=1.5
    )
    plt.title(f'{feature} Cluster Distribution')
    plt.xlabel(feature)
    plt.ylabel('Density')
plt.tight_layout()
plt.show()



plt.figure(figsize=(20, 12))
for i, feature in enumerate(clustering_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(
        data=train,
        x=feature,
        y='Calories',
        hue=f'{feature}_clusters',
        palette='tab10',
        alpha=0.6,
        linewidth=0.5
    )
    plt.title(f'Calories vs {feature} (Clustered)')
    plt.xlabel(feature)
    plt.ylabel('Calories')
    plt.legend(title='Cluster', loc='upper right', fontsize='small')
plt.tight_layout()
plt.show()


%%time

features=train.drop(['Calories'],axis=1).columns.to_list()
# Feature matrix and log target
X = train[features]
X_test = test[features]
y_log = np.log1p(train['Calories'])

# CV setup
kf = KFold(n_splits=20, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(train))
xgb_preds = np.zeros(len(test))
xgb_scores = []

# XGBoost parameters
xgb_params = {
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': False,
    'random_state': 42,
    'early_stopping_rounds': 100,
    'tree_method': 'gpu_hist'
}

best_model = None
best_score = float('inf')

# Training loop with progress bar
for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=kf.get_n_splits()), 1):
    model = XGBRegressor(**xgb_params)
    
    model.fit(
        X.iloc[train_idx], y_log.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y_log.iloc[val_idx])],
        verbose=False
    )
    
    preds = model.predict(X.iloc[val_idx])
    xgb_oof[val_idx] = preds
    
    score = np.sqrt(mean_squared_log_error(np.expm1(y_log.iloc[val_idx]), np.expm1(preds)))
    xgb_scores.append(score)
    
    if score < best_score:
        best_score = score
        best_model = model

# Final evaluation
mean_oof_score = np.mean(xgb_scores)
print(f"\nâœ… XGBoost Mean RMSLE (OOF): {mean_oof_score:.5f}")
print(f"ğŸ�† Best Fold RMSLE: {best_score:.5f}")


best_model.best_iteration


# Take the best model parameters for final training on full dataset
xgb_params = {
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': False,
    'random_state': 42,
    'tree_method': 'gpu_hist'
}
xgb_params['n_estimators'] = best_model.best_iteration




final_model = XGBRegressor(**xgb_params)
final_model.fit(X, y_log)


sub=pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")

test_predictions=np.expm1(final_model.predict(test[features]))
sub['Calories']=test_predictions
sub.to_csv(f"submission.csv",index=False)

sub.head()

