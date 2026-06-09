# Imports and environment checks
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
try:
    import xgboost as xgb
    has_xgb = True
except:
    has_xgb = False

# Paths (Kaggle competition layout)
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

# Load if available; otherwise create a small demo dataset so the notebook runs
if os.path.exists(train_path) and os.path.exists(test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print('Loaded competition files.')
else:
    print('Competition files not found. Generating demo dataset (for local runs).')
    rng = np.random.default_rng(42)
    n_train = 5000; n_test = 1500
    road_types = ['urban','rural','highway']
    light = ['daylight','dim','night']
    weather = ['clear','rainy','foggy','snowy']
    time_of_day = ['morning','afternoon','evening','night']
    def make_df(n, with_target=True):
        df = pd.DataFrame({
            'id': np.arange(n),
            'road_type': rng.choice(road_types, size=n),
            'num_lanes': rng.integers(1,5,size=n),
            'curvature': rng.random(size=n),
            'speed_limit': rng.choice([30,40,50,60,70,80], size=n),
            'lighting': rng.choice(light, size=n),
            'weather': rng.choice(weather, size=n),
            'road_signs_present': rng.choice([True, False], size=n, p=[0.7,0.3]),
            'public_road': rng.choice([True, False], size=n, p=[0.85,0.15]),
            'time_of_day': rng.choice(time_of_day, size=n),
            'holiday': rng.choice([True, False], size=n, p=[0.1,0.9]),
            'school_season': rng.choice([True, False], size=n, p=[0.3,0.7]),
            'num_reported_accidents': rng.integers(0,5,size=n)
        })
        if with_target:
            base = 0.12 + 0.35*(df['road_type']=='rural').astype(float) + 0.15*(df['weather']=='rainy').astype(float)
            base += 0.02*df['num_reported_accidents'] + 0.08*(~df['road_signs_present']).astype(int)
            noise = rng.normal(scale=0.08, size=n)
            df['accident_risk'] = np.clip(base + noise, 0, 1)
        return df
    train = make_df(n_train, with_target=True)
    test = make_df(n_test, with_target=False)

print('Train shape:', train.shape, 'Test shape:', test.shape)
train.head()


# Types and sample stats
display(train.dtypes.to_frame('dtype').T)
display(train.describe(include='all').T)

plt.figure(figsize=(8,4))
sns.histplot(train['accident_risk'], bins=40, kde=True)
plt.title('Accident Risk Distribution')
plt.xlabel('accident_risk')
plt.show()


# Missing values and quick cleaning (if any)
missing = train.isnull().sum().sort_values(ascending=False)
if missing.sum() == 0:
    print('No missing values detected in training set.')
else:
    display(missing[missing>0])


# Basic exploration plots and an interactive Plotly bar
import plotly.express as px
for col in ['road_type','lighting','weather','time_of_day']:
    plt.figure(figsize=(6,3))
    sns.countplot(data=train, x=col, order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Interactive bar: mean accident_risk by road_type
agg_rt = train.groupby('road_type')['accident_risk'].mean().reset_index().sort_values('accident_risk', ascending=False)
fig = px.bar(agg_rt, x='road_type', y='accident_risk', title='Mean Accident Risk by Road Type', text_auto='.3f')
fig.show()

# Boxplot: num_reported_accidents vs risk
plt.figure(figsize=(6,4))
sns.boxplot(data=train, x='num_reported_accidents', y='accident_risk')
plt.title('Accident Risk by Number of Reported Accidents')
plt.show()

# Violin: speed_limit vs risk
plt.figure(figsize=(8,4))
sns.violinplot(data=train, x='speed_limit', y='accident_risk')
plt.title('Accident Risk distribution across speed limits')
plt.show()


# Feature engineering
df = train.copy()
df['high_speed'] = (df['speed_limit'] >= 60).astype(int)
df['curvature_sq'] = df['curvature']**2
df['road_signs_present'] = df['road_signs_present'].astype(int)
df['public_road'] = df['public_road'].astype(int)
df['holiday'] = df['holiday'].astype(int)
df['school_season'] = df['school_season'].astype(int)

display(df[['num_lanes','speed_limit','high_speed','curvature','curvature_sq','num_reported_accidents']].describe().T)


# Prepare features and modeling
features = ['road_type','num_lanes','curvature','speed_limit','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season','num_reported_accidents','high_speed','curvature_sq']
target = 'accident_risk'
X = df[features].copy(); y = df[target].values
cat_cols = ['road_type','lighting','weather','time_of_day']
num_cols = [c for c in features if c not in cat_cols]

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
    ('num', StandardScaler(), num_cols)
])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

rf_pipeline = Pipeline([('pre', preprocessor), ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
rf_pipeline.fit(X_train, y_train)
rf_val_preds = rf_pipeline.predict(X_valid)
rf_val_rmse = mean_squared_error(y_valid, rf_val_preds, squared=False)
print('RandomForest validation RMSE:', round(rf_val_rmse,5))

if has_xgb:
    xgb_pipeline = Pipeline([('pre', preprocessor), ('xgb', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0))])
    xgb_pipeline.fit(X_train, y_train)
    xgb_val_preds = xgb_pipeline.predict(X_valid)
    xgb_val_rmse = mean_squared_error(y_valid, xgb_val_preds, squared=False)
    print('XGBoost validation RMSE:', round(xgb_val_rmse,5))
else:
    print('XGBoost not available in this environment.')


# Ensemble if XGBoost present
if has_xgb:
    ensemble_preds = (rf_val_preds + xgb_val_preds) / 2
    ensemble_rmse = mean_squared_error(y_valid, ensemble_preds, squared=False)
    print('Ensemble RMSE:', round(ensemble_rmse,5))
else:
    print('Ensemble not created (XGBoost not available).')

# Cross-validated RF score
cv_scores = cross_val_score(rf_pipeline, X, y, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
print('RF CV RMSE (3-fold avg):', round(-cv_scores.mean(),5))


from sklearn.inspection import permutation_importance

print("Calculating permutation importance (this may take ~1â€“2 minutes)...")

# âœ… Use n_jobs=1 to avoid joblib memory/pickling issues on Kaggle
try:
    perm = permutation_importance(
        rf_pipeline,
        X_valid,
        y_valid,
        n_repeats=5,
        random_state=42,
        n_jobs=1  # <--- changed from -1 to 1
    )
    
    # Get one-hot encoded feature names
    ohe = rf_pipeline.named_steps['pre'].named_transformers_['cat']
    cat_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = cat_names + num_cols

    # Importance dataframe
    imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

    print("Top features contributing to accident risk:")
    display(imp.head(20))

    plt.figure(figsize=(8,5))
    imp.head(12).sort_values().plot(kind='barh', color="#48cae4")
    plt.title('Top 12 Permutation Importances (RandomForest)')
    plt.xlabel('Importance (mean decrease in RMSE)')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("âš ï¸� Permutation importance failed â€” using RandomForest native feature importances instead.")
    rf_model = rf_pipeline.named_steps['rf']
    imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    display(imp.head(20))

    plt.figure(figsize=(8,5))
    imp.head(12).sort_values().plot(kind='barh', color="#48cae4")
    plt.title('Top 12 Feature Importances (RF built-in)')
    plt.tight_layout()
    plt.show()


# Prepare test and create submission
test_df = test.copy()
# Feature engineering same as train
test_df['high_speed'] = (test_df['speed_limit'] >= 60).astype(int)
test_df['curvature_sq'] = test_df['curvature']**2
test_df['road_signs_present'] = test_df['road_signs_present'].astype(int)
test_df['public_road'] = test_df['public_road'].astype(int)
test_df['holiday'] = test_df['holiday'].astype(int)
test_df['school_season'] = test_df['school_season'].astype(int)

test_X = test_df[features].copy()

# Predict using pipelines
try:
    test_preds_rf = rf_pipeline.predict(test_X)
except Exception as e:
    # fallback: align categories using preprocessor fit_transform
    combined = pd.concat([X, test_X], ignore_index=True)
    preprocessor.fit(combined)
    test_trans = preprocessor.transform(test_X)
    test_preds_rf = rf_pipeline.named_steps['rf'].predict(test_trans)

if has_xgb:
    try:
        test_preds_xgb = xgb_pipeline.predict(test_X)
    except:
        # transform properly
        test_preds_xgb = xgb_pipeline.named_steps['xgb'].predict(preprocessor.transform(test_X))
    test_preds = (test_preds_rf + test_preds_xgb) / 2
else:
    test_preds = test_preds_rf

# Clip and save
import numpy as np
test_preds = np.clip(test_preds, 0, 1)
submission = pd.DataFrame({'id': test_df['id'], 'accident_risk': test_preds})
submission.to_csv('submission.csv', index=False)
print('Saved submission.csv â€” rows:', submission.shape[0])
submission.head()

