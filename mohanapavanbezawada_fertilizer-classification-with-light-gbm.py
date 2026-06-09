import warnings
warnings.simplefilter('ignore')

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from lightgbm import LGBMClassifier, early_stopping
from tqdm import tqdm
import pandas as pd


PATH = '../input/playground-series-s5e6/'
train = pl.read_csv(PATH + 'train.csv')
test = pl.read_csv(PATH + 'test.csv')


def enhanced_feature_engineering(input_df, is_train=True):
    """Enhanced feature engineering with domain knowledge and interactions"""
    
    # Base features
    base_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    
    out_df = input_df.select(base_features + ['Soil Type', 'Crop Type'])
    
    # 1. Nutrient ratios (important for fertilizer recommendations)
    out_df = out_df.with_columns([
        (pl.col('Nitrogen') / (pl.col('Potassium') + 1e-8)).alias('N_K_ratio'),
        (pl.col('Nitrogen') / (pl.col('Phosphorous') + 1e-8)).alias('N_P_ratio'),
        (pl.col('Potassium') / (pl.col('Phosphorous') + 1e-8)).alias('K_P_ratio'),
        (pl.col('Nitrogen') + pl.col('Potassium') + pl.col('Phosphorous')).alias('total_nutrients'),
    ])
    
    # 2. Environmental interaction features
    out_df = out_df.with_columns([
        (pl.col('Temparature') * pl.col('Humidity')).alias('temp_humidity_interaction'),
        (pl.col('Temparature') * pl.col('Moisture')).alias('temp_moisture_interaction'),
        (pl.col('Humidity') * pl.col('Moisture')).alias('humidity_moisture_interaction'),
    ])
    out_df = out_df.with_columns([
        pl.col('Temparature').qcut(5, labels=['very_cold', 'cold', 'moderate', 'warm', 'hot']).alias('temp_bin'),
        pl.col('Humidity').qcut(5, labels=['very_dry', 'dry', 'moderate', 'humid', 'very_humid']).alias('humidity_bin'),
        pl.col('Nitrogen').qcut(5, labels=['very_low', 'low', 'medium', 'high', 'very_high']).alias('nitrogen_bin'),
    ])
    
    # 4. Nutrient deficiency indicators
    out_df = out_df.with_columns([
        (pl.col('Nitrogen') < pl.col('Nitrogen').quantile(0.3)).alias('low_nitrogen'),
        (pl.col('Potassium') < pl.col('Potassium').quantile(0.3)).alias('low_potassium'),
        (pl.col('Phosphorous') < pl.col('Phosphorous').quantile(0.3)).alias('low_phosphorous'),
    ])
    
    # 5. Polynomial features for key nutrients
    out_df = out_df.with_columns([
        (pl.col('Nitrogen') ** 2).alias('nitrogen_sq'),
        (pl.col('Potassium') ** 2).alias('potassium_sq'),
        (pl.col('Phosphorous') ** 2).alias('phosphorous_sq'),
    ])
    
    # Convert categorical columns
    categorical_cols = ['Soil Type', 'Crop Type', 'temp_bin', 'humidity_bin', 'nitrogen_bin']
    out_df = out_df.with_columns([
        pl.col(col).cast(pl.Categorical) for col in categorical_cols
    ])
    
    return out_df


def scale_features(x_train, x_test, numerical_indices):
    """Scale numerical features using RobustScaler"""
    scaler = RobustScaler()
    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()
    
    x_train_scaled[:, numerical_indices] = scaler.fit_transform(x_train[:, numerical_indices])
    x_test_scaled[:, numerical_indices] = scaler.transform(x_test[:, numerical_indices])
    
    return x_train_scaled, x_test_scaled, scaler


def single_apk(y, oof):
    """Calculate MAP@3 score"""
    sorted_oof = np.argsort(oof, axis=1)[:, ::-1][:, :3]
    score = 0
    for i in range(3):
        score += accuracy_score(y, sorted_oof[:, i]) / (i+1)
    return score


x0 = enhanced_feature_engineering(train, is_train=True)
test_x0 = enhanced_feature_engineering(test, is_train=False)


le = LabelEncoder()
y = le.fit_transform(train['Fertilizer Name'])
N_CLASS = train['Fertilizer Name'].n_unique()

# Get feature names and categorical columns
feature_names = x0.columns
cat_columns = [name for name, dtype in x0.schema.items() if dtype == pl.Categorical]

# Convert to numerical format
print("Converting features to numerical format...")
all_x = pl.concat([x0, test_x0], how='vertical').with_columns([
    pl.col(c).to_physical() for c in cat_columns
])

x = all_x[:len(train)].to_numpy().astype(np.float32)
test_x = all_x[len(train):].to_numpy().astype(np.float32)


numerical_indices = []
for i, (name, dtype) in enumerate(x0.schema.items()):
    if dtype not in [pl.Categorical]:
        numerical_indices.append(i)

# Scale numerical features
print("Scaling numerical features...")
x_scaled, test_x_scaled, scaler = scale_features(x, test_x, numerical_indices)


N_FOLDS = 7  # Increased folds for better validation
oof = np.zeros((len(train), N_CLASS))
pred = np.zeros((len(test), N_CLASS))
logloss = []
map3 = []
iterations = []
fi_df = pl.DataFrame()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


model = LGBMClassifier(
    objective='multiclass',
    n_estimators=15000,  # Increased for better performance
    max_depth=6,         # Slightly deeper
    num_leaves=63,       # 2^6 - 1
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    learning_rate=0.05,  # Lower learning rate for better convergence
    importance_type='gain',
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    force_col_wise=True  # Faster training
)


print(f"Starting {N_FOLDS}-fold cross-validation...")


for i, (train_idx, valid_idx) in enumerate(tqdm(skf.split(x_scaled, y), desc="CV Folds")):
    x_train, y_train = x_scaled[train_idx], y[train_idx]
    x_valid, y_valid = x_scaled[valid_idx], y[valid_idx]
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        feature_name=feature_names,
        categorical_feature=cat_columns,
        callbacks=[early_stopping(stopping_rounds=200, verbose=False)]
    )
    
    oof[valid_idx, :] = model.predict_proba(x_valid)
    pred += model.predict_proba(test_x_scaled) / N_FOLDS
    
    # Store feature importance
    fi_df = pl.concat([
        fi_df,
        pl.DataFrame({
            'feature': feature_names, 
            'importance': model.feature_importances_, 
            'fold': i
        })
    ])
    
    logloss.append(log_loss(y_valid, oof[valid_idx, :]))
    map3.append(single_apk(y_valid, oof[valid_idx, :]))
    iterations.append(model.best_iteration_)


fold_df = pl.DataFrame({
    'fold': range(N_FOLDS),
    'Logloss': logloss,
    'MAP@3': map3,
    'iterations': iterations
})


total_logloss = log_loss(y, oof)
total_map3 = single_apk(y, oof)


feature_importance = (fi_df.group_by('feature')
                     .agg(pl.col('importance').mean().alias('mean_importance'))
                     .sort('mean_importance', descending=True))


sorted_pred = np.argsort(pred, axis=1)[:, ::-1]
submission = pl.DataFrame({
    'id': test['id'],
    'pred1': le.inverse_transform(sorted_pred[:, 0]),
    'pred2': le.inverse_transform(sorted_pred[:, 1]),
    'pred3': le.inverse_transform(sorted_pred[:, 2])
}).with_columns(
    Fertilizer=pl.col('pred1') + ' ' + pl.col('pred2') + ' ' + pl.col('pred3')
).select(['id', 'Fertilizer']).rename({'Fertilizer': 'Fertilizer Name'})
submission.write_csv('enhanced_submission.csv')
print("\nSubmission saved to 'enhanced_submission.csv'")


_order = feature_importance['feature'].to_list()[:15]  # Top 15 features
fig, ax = plt.subplots(figsize=(10, 8))
fi_plot_df = fi_df.filter(pl.col('feature').is_in(_order))
sns.boxenplot(
    y='feature', x='importance', 
    data=fi_plot_df.to_pandas(), 
    orient='h', order=_order, ax=ax
)
plt.title('Top 15 Feature Importance Distribution Across Folds')
plt.tight_layout()
plt.show()


print(f"\nModel trained with {len(feature_names)} features")
print(f"Training completed in {N_FOLDS} folds")

