import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df.head()


train_df.info()


train_df.describe()


train_df.shape


train_df.isna().sum()


cat_cols = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day','holiday', 'school_season']
num_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']


plt.figure(figsize=(15, 12))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)   
    sns.countplot(data=train_df, x=col)
    plt.xticks(rotation=45)
    plt.title(f"Countplot of {col}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)   
    sns.histplot(data=train_df, x=col, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)   
    sns.boxplot(train_df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


corr_matrix = train_df[num_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


bool_cols = train_df.select_dtypes(include='bool').columns
train_df[bool_cols] = train_df[bool_cols].astype(int)


train_df.head()


def create_features(df):
    """Advanced feature engineering"""
    df = df.copy()
    
    # 1. Polynomial features for key numerical variables
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    
    # 2. Binned features
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 1.0], labels=[0, 1, 2])
    df['speed_category'] = pd.cut(df['speed_limit'], bins=[0, 30, 50, 100], labels=[0, 1, 2])
    
    # 3. Interaction features
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['lanes_curvature'] = df['num_lanes'] * df['curvature']
    df['speed_lanes'] = df['speed_limit'] * df['num_lanes']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    
    # 4. Risk score combinations
    df['high_risk_combo'] = ((df['curvature'] > 0.5) & (df['speed_limit'] >= 60)).astype(int)
    df['weather_lighting_risk'] = ((df['weather'] == 'foggy') | (df['weather'] == 'rainy')) & \
                                   ((df['lighting'] == 'dim') | (df['lighting'] == 'night'))
    df['weather_lighting_risk'] = df['weather_lighting_risk'].astype(int)
    
    # 5. Categorical aggregations (target encoding will be done in CV)
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    
    # 6. Time-based features
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)  # Using holiday as proxy
    
    # 7. Safety features
    df['safety_score'] = df['road_signs_present'].astype(int) * 2 + \
                         (df['lighting'] == 'daylight').astype(int) + \
                         (df['weather'] == 'clear').astype(int)
    
    df['danger_score'] = (df['curvature'] > 0.6).astype(int) + \
                         (df['speed_limit'] >= 60).astype(int) + \
                         df['is_bad_weather'] + df['is_night'] + \
                         (df['num_reported_accidents'] >= 2).astype(int)
    
    # 8. Ratio features
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['risk_intensity'] = df['curvature'] * df['speed_limit'] / 50
    
    return df


train_df = create_features(train_df)
test_df = create_features(test_df)


X_train = train_df.drop('accident_risk', axis=1) 
y_train = train_df['accident_risk']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']  
num_cols = ['num_lanes', 'curvature', 'speed_limit', 
            'num_reported_accidents', 
            'road_signs_present', 'public_road', 'holiday', 'school_season']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),         
        ('cat', OneHotEncoder(drop='first'), cat_cols)  
    ]
)


xgb = XGBRegressor(verbosity=0,
                  max_depth=8,
        learning_rate=0.01,
        n_estimators=1000,
        subsample=0.9,
        colsample_bytree=0.9)
cgb = CatBoostRegressor(verbose=0)
params = {
    'subsample': 0.9,
    'eta': 0.1,
    'n_estimators': 300,
    'random_state': 0,
    'n_jobs': 4,
    'verbose':-1
}
lgb = LGBMRegressor(**params)
base_models = [('xgb', xgb), ('cgb', cgb), ('lgb', lgb)]
from sklearn.linear_model import RidgeCV
final_model = RidgeCV()



kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds_xgb = np.zeros(len(test_df))
test_preds_cgb = np.zeros(len(test_df))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_trn, y_val = y_train[trn_idx], y_train[val_idx]

    X_trn = preprocessor.fit_transform(X_trn)  
    X_val = preprocessor.transform(X_val)  

    stack_model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model,
        passthrough=True,   
        n_jobs=-1
    )
    stack_model.fit(X_trn, y_trn)
    # xgb.fit(X_trn, y_trn)
    # cgb.fit(X_trn, y_trn)

    # val_pred_xgb = xgb.predict(X_val)
    # val_pred_cgb = cgb.predict(X_val)

    # val_pred = (val_pred_xgb + val_pred_cgb)/2
    val_pred = stack_model.predict(X_val)

    oof_preds[val_idx] = val_pred

    print(f"Fold {fold+1} RMSE: {mean_squared_error(y_val, oof_preds[val_idx], squared=False)}")

print(f"Overall OOF RMSE: {mean_squared_error(y_train, oof_preds, squared=False):.5f}")


X_train = preprocessor.fit_transform(X_train) 
# xgb.fit(X_train, y_train)
# cgb.fit(X_train, y_train)


stack_model = StackingRegressor(
        estimators=base_models,
        final_estimator=final_model,
        passthrough=False,   
        n_jobs=-1
)

stack_model.fit(X_train,y_train)


test_id = test_df['id']
X_test_transformed = preprocessor.transform(test_df)


# final_lgb_pred = lgb.predict(X_test_transformed)
# final_xgb_pred = xgb.predict(X_test_transformed)
# final_cgb_pred = cgb.predict(X_test_transformed)
# final_pred = (final_xgb_pred + final_cgb_pred )/2


submission = pd.DataFrame({
    'id': test_id,
    'accident_risk': stack_model.predict(X_test_transformed)
})

submission["accident_risk"] = submission["accident_risk"].round(3)

submission.to_csv("submission.csv", index=False)

