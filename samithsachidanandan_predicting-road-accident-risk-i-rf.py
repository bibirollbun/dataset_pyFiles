import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform



import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


train.head()


train.shape


test.shape


train.info()


train.dtypes


print("Target column statistics (accident_risk):")

train['accident_risk'].describe()


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())


train_numeric = train.select_dtypes(include=['int64', 'float64']).drop(columns=['accident_risk'], errors='ignore')


test_numeric = test.select_dtypes(include=['int64', 'float64']).drop(columns=['accident_risk'], errors='ignore')


train_numeric



numeric_cols = [col for col in train_numeric.columns if col != 'id']

num_cols = 4  
num_rows = (len(numeric_cols) + num_cols - 1) // num_cols

plt.figure(figsize=(5*num_cols, 4*num_rows))

for i, col in enumerate(numeric_cols):
    plt.subplot(num_rows, num_cols, i+1)
    plt.hist(train_numeric[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(col)
    plt.xlabel('Value')
    plt.ylabel('Count')

plt.tight_layout()
plt.show()



plt.figure(figsize=(5*num_cols, 4*num_rows))

for i, col in enumerate(numeric_cols):
    plt.subplot(num_rows, num_cols, i+1)
    plt.boxplot(train_numeric[col], vert=False)
    plt.title(col)

plt.tight_layout()
plt.show()


numeric_cols_test = [col for col in test_numeric.columns if col != 'id']
plt.figure(figsize=(5*num_cols, 4*num_rows))

for i, col in enumerate(numeric_cols_test):
    plt.subplot(num_rows, num_cols, i+1)
    plt.boxplot(test_numeric[col], vert=False)
    plt.title(col)

plt.tight_layout()
plt.show()


from scipy import stats
z_train = np.abs(stats.zscore(train_numeric))


threshold = 3
outliers = np.where(z_train > threshold)
print("Number of outliers detected:", len(outliers[0]))


train = train[(z_train < threshold).all(axis=1)]



train.shape



numeric_cols = train.select_dtypes(include=['number']).drop(columns=['id'])

corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='rocket', linewidths=0.5)
plt.show()


def create_features(df):
 
    df = df.copy()
    
    # 1. Polynomial features for key numerical variables
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    
    # 2. Binned features
    df['curvature_bin'] = pd.cut(df['curvature'], bins=[-np.inf, 0.3, 0.6, np.inf], labels=[0, 1, 2])
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
    
    # 5. Categorical aggregations 
    df['is_night'] = (df['lighting'] == 'night').astype(int)
    df['is_bad_weather'] = df['weather'].isin(['foggy', 'rainy']).astype(int)
    df['is_highway'] = (df['road_type'] == 'highway').astype(int)
    df['is_urban'] = (df['road_type'] == 'urban').astype(int)
    
    # 6. Time-based features
    df['is_peak_time'] = df['time_of_day'].isin(['morning', 'evening']).astype(int)
    df['is_weekend'] = df['holiday'].astype(int)  
    
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

train = create_features(train)
test = create_features(test)



numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()


categorical_cols = train.select_dtypes(include=['object']).columns.tolist()


bool_cols = train.select_dtypes(include=['bool']).columns.tolist()

print("Numeric:", numeric_cols)

print("Categorical:", categorical_cols)

print("Boolean:", bool_cols)


train_encoded = pd.get_dummies(train, columns=categorical_cols, drop_first=True, dtype=int)
test_encoded = pd.get_dummies(test, columns=categorical_cols, drop_first=True, dtype=int)


train_encoded.head()


from sklearn.preprocessing import MinMaxScaler

params = { 'bootstrap': True,
          'max_depth': 9, 
          'max_features': 0.672894435115225,
          'max_samples': 0.902144564127061, 
          'min_samples_leaf': 2,
          'min_samples_split': 7, 
          'n_estimators': 753, 
          'random_state': 42, 
          'n_jobs': -1 }

features = train_encoded.columns.drop(['id', 'accident_risk'])
X = train_encoded[features]
y = train_encoded['accident_risk']


X_test = test_encoded[features]  

print("Random Forest Cross-Validation with Best Parameters")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_rf = np.zeros(len(X))
rf_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*40}")
    print(f"Training Fold {fold + 1}/5")
    print(f"{'='*40}")
    
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    
  
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_val_scaled = scaler.transform(X_val_fold)
    
    
    rf_model = RandomForestRegressor(**params)
    rf_model.fit(X_train_scaled, y_train_fold)
    rf_models.append(rf_model)
    

    oof_rf[val_idx] = rf_model.predict(X_val_scaled)
    
  
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, oof_rf[val_idx]))
    fold_r2 = r2_score(y_val_fold, oof_rf[val_idx])
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.6f}, R2: {fold_r2:.6f}")


rf_cv_rmse = np.sqrt(mean_squared_error(y, oof_rf))
rf_cv_r2 = r2_score(y, oof_rf)
print(f"\n{'='*40}")
print("Overall Random Forest Cross-Validation Results:")
print(f"Random Forest - CV RMSE: {rf_cv_rmse:.6f}, R2: {rf_cv_r2:.6f}")






scaler_full = MinMaxScaler()
X_scaled = scaler_full.fit_transform(X)
X_test_scaled = scaler_full.transform(X_test)  # for final prediction



print("\nTraining final Random Forest model on full dataset...")
final_rf = RandomForestRegressor(**params)
final_rf.fit(X, y)


X_test = test_encoded.drop('id', axis=1)


test_preds = final_rf.predict(X_test)

print(f"Final model trained successfully!")
print(f"Number of features: {len(features)}")
print(f"Test predictions range: [{test_preds.min():.6f}, {test_preds.max():.6f}]")


print("Submission shape:", submission.shape)
print("Test predictions shape:", len(test_preds))


submission = submission.copy()
submission['accident_risk'] = test_preds

submission.to_csv('submission.csv', index=False)
print("\n Submission saved to 'submission.csv'")





