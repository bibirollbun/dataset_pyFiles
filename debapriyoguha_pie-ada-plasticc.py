# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
from scipy import stats
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("Libraries imported successfully")


# Cell 2: Load Data
print("Loading PLAsTiCC dataset...")
train_meta = pd.read_csv('../input/PLAsTiCC-2018/training_set_metadata.csv')
train = pd.read_csv('../input/PLAsTiCC-2018/training_set.csv')

SAMPLE_SIZE = None

if SAMPLE_SIZE:
    train_ids = train['object_id'].unique()[:SAMPLE_SIZE//100]
    train = train[train['object_id'].isin(train_ids)]
    train_meta = train_meta[train_meta['object_id'].isin(train_ids)]

print(f"Train shape: {train.shape}")
print(f"Train meta shape: {train_meta.shape}")
print(f"Unique objects: {train['object_id'].nunique()}")
print(f"Number of classes: {train_meta['target'].nunique()}")


# Cell 3: Feature Engineering Functions
def extract_features(df):
    print("Extracting statistical features per passband...")
    
    features = df.groupby(['object_id', 'passband']).agg({
        'flux': [
            'mean', 'std', 'min', 'max', 'skew',
            lambda x: stats.kurtosis(x),
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75),
            lambda x: np.percentile(x, 90),
            lambda x: x.max() - x.min(),
            lambda x: np.median(np.abs(x - np.median(x))),
            lambda x: np.sum(x > 0) / len(x) if len(x) > 0 else 0,
            lambda x: np.sum(x > x.mean()) / len(x) if len(x) > 0 else 0,
        ],
        'flux_err': [
            'mean', 'std', 'min', 'max',
            lambda x: np.mean(x) / (np.std(x) + 1e-8) if np.std(x) > 0 else 0
        ],
        'detected': ['sum', 'mean'],
        'mjd': [
            'min', 'max', 'count',
            lambda x: np.std(np.diff(sorted(x))) if len(x) > 1 else 0
        ]
    })
    
    new_columns = []
    for col in features.columns:
        if col[1] == '<lambda>':
            if col[0] == 'flux':
                lambda_names = ['kurtosis', 'q1', 'q3', 'p90', 'range', 'mad', 'pos_frac', 'above_mean_frac']
                lambda_index = len([c for c in new_columns if c.startswith(f'{col[0]}_')])
                if lambda_index < len(lambda_names):
                    new_columns.append(f'{col[0]}_{lambda_names[lambda_index]}')
                else:
                    new_columns.append(f'{col[0]}_lambda{lambda_index}')
            elif col[0] == 'flux_err':
                new_columns.append(f'{col[0]}_snr')
            elif col[0] == 'mjd':
                new_columns.append(f'{col[0]}_gap_std')
        else:
            new_columns.append(f'{col[0]}_{col[1]}')
    
    features.columns = new_columns
    features = features.reset_index()
    
    features_pivot = features.pivot_table(
        index='object_id',
        columns='passband',
        values=[col for col in features.columns if col not in ['object_id', 'passband']]
    )
    
    features_pivot.columns = ['_'.join(map(str, col)) for col in features_pivot.columns]
    features_pivot = features_pivot.reset_index()
    
    return features_pivot

def add_time_features(df):
    print("Extracting time-based features...")
    
    time_features = df.groupby('object_id').agg({
        'mjd': lambda x: x.max() - x.min(),
        'flux': [
            lambda x: np.sum(np.abs(np.diff(x))),
            lambda x: len(x[x > np.median(x)]) / len(x) if len(x) > 0 else 0,
            lambda x: np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0
        ],
        'detected': lambda x: x.sum() / len(x)
    })
    
    time_features.columns = ['duration', 'total_variation', 'above_median_frac',
                             'mean_flux_change', 'detection_rate']
    time_features = time_features.reset_index()
    
    return time_features

def add_peak_features(df):
    print("Extracting peak and rise/fall features...")
    peak_features = []
    
    for obj_id in tqdm(df['object_id'].unique(), desc="Peak features"):
        obj_data = df[df['object_id'] == obj_id]
        features_dict = {'object_id': obj_id}
        
        for pb in range(6):
            pb_data = obj_data[obj_data['passband'] == pb]
            
            if len(pb_data) > 3:
                peak_idx = pb_data['flux'].idxmax()
                features_dict[f'peak_flux_{pb}'] = pb_data.loc[peak_idx, 'flux']
                features_dict[f'peak_mjd_{pb}'] = pb_data.loc[peak_idx, 'mjd']
                features_dict[f'time_to_peak_{pb}'] = (
                    pb_data.loc[peak_idx, 'mjd'] - pb_data['mjd'].min()
                )
                
                peak_time = pb_data.loc[peak_idx, 'mjd']
                before_peak = pb_data[pb_data['mjd'] < peak_time]
                after_peak = pb_data[pb_data['mjd'] > peak_time]
                
                if len(before_peak) > 1:
                    rise_rate = (pb_data.loc[peak_idx, 'flux'] - before_peak['flux'].min()) / (
                        peak_time - before_peak['mjd'].min() + 1e-8)
                    features_dict[f'rise_rate_{pb}'] = rise_rate
                else:
                    features_dict[f'rise_rate_{pb}'] = 0
                    
                if len(after_peak) > 1:
                    fall_rate = (pb_data.loc[peak_idx, 'flux'] - after_peak['flux'].min()) / (
                        after_peak['mjd'].max() - peak_time + 1e-8)
                    features_dict[f'fall_rate_{pb}'] = fall_rate
                else:
                    features_dict[f'fall_rate_{pb}'] = 0
            else:
                features_dict[f'peak_flux_{pb}'] = np.nan
                features_dict[f'peak_mjd_{pb}'] = np.nan
                features_dict[f'time_to_peak_{pb}'] = np.nan
                features_dict[f'rise_rate_{pb}'] = np.nan
                features_dict[f'fall_rate_{pb}'] = np.nan
        
        peak_features.append(features_dict)
    
    return pd.DataFrame(peak_features)

def add_color_features(df):
    print("Extracting color features...")
    color_features = []
    
    for obj_id in tqdm(df['object_id'].unique(), desc="Color features"):
        obj_data = df[df['object_id'] == obj_id]
        features_dict = {'object_id': obj_id}
        
        for pb1 in range(5):
            for pb2 in range(pb1+1, 6):
                pb1_data = obj_data[obj_data['passband'] == pb1]
                pb2_data = obj_data[obj_data['passband'] == pb2]
                
                if len(pb1_data) > 0 and len(pb2_data) > 0:
                    pb1_flux = pb1_data['flux'].mean()
                    pb2_flux = pb2_data['flux'].mean()
                    
                    if pb1_flux > 0 and pb2_flux > 0:
                        color = -2.5 * np.log10(pb1_flux / pb2_flux)
                        features_dict[f'color_{pb1}_{pb2}'] = color
                    else:
                        features_dict[f'color_{pb1}_{pb2}'] = 0
                else:
                    features_dict[f'color_{pb1}_{pb2}'] = np.nan
                    
        color_features.append(features_dict)
    
    return pd.DataFrame(color_features)

def add_advanced_features(df):
    print("Extracting advanced features...")
    adv_features = []
    
    for obj_id in tqdm(df['object_id'].unique(), desc="Advanced features"):
        obj_data = df[df['object_id'] == obj_id]
        features_dict = {'object_id': obj_id}
        
        for pb in range(6):
            pb_data = obj_data[obj_data['passband'] == pb].sort_values('mjd')
            if len(pb_data) > 10:
                flux = pb_data['flux'].values
                flux = flux - np.mean(flux)
                fft = np.fft.fft(flux)
                freqs = np.fft.fftfreq(len(flux))
                
                pos_mask = freqs > 0
                fft_pos = np.abs(fft[pos_mask])
                
                if len(fft_pos) > 0:
                    features_dict[f'fft_max_freq_{pb}'] = fft_pos.max()
                    features_dict[f'fft_mean_amp_{pb}'] = fft_pos.mean()
                    features_dict[f'fft_std_amp_{pb}'] = fft_pos.std()
                else:
                    features_dict[f'fft_max_freq_{pb}'] = 0
                    features_dict[f'fft_mean_amp_{pb}'] = 0
                    features_dict[f'fft_std_amp_{pb}'] = 0
            else:
                features_dict[f'fft_max_freq_{pb}'] = np.nan
                features_dict[f'fft_mean_amp_{pb}'] = np.nan
                features_dict[f'fft_std_amp_{pb}'] = np.nan
        
        for pb1 in range(6):
            for pb2 in range(pb1+1, 6):
                pb1_flux = obj_data[obj_data['passband'] == pb1]['flux'].mean()
                pb2_flux = obj_data[obj_data['passband'] == pb2]['flux'].mean()
                
                if pb2_flux != 0 and not np.isnan(pb2_flux):
                    features_dict[f'flux_ratio_{pb1}_{pb2}'] = pb1_flux / pb2_flux
                else:
                    features_dict[f'flux_ratio_{pb1}_{pb2}'] = 0
        
        total_flux = obj_data['flux'].sum()
        if total_flux > 0:
            features_dict['weighted_mean_time'] = (obj_data['flux'] * obj_data['mjd']).sum() / total_flux
        else:
            features_dict['weighted_mean_time'] = obj_data['mjd'].mean()
            
        for pb in range(6):
            pb_data = obj_data[obj_data['passband'] == pb]['flux']
            if len(pb_data) > 0:
                features_dict[f'flux_5th_percentile_{pb}'] = np.percentile(pb_data, 5)
                features_dict[f'flux_95th_percentile_{pb}'] = np.percentile(pb_data, 95)
                features_dict[f'flux_iqr_{pb}'] = np.percentile(pb_data, 75) - np.percentile(pb_data, 25)
            else:
                features_dict[f'flux_5th_percentile_{pb}'] = np.nan
                features_dict[f'flux_95th_percentile_{pb}'] = np.nan
                features_dict[f'flux_iqr_{pb}'] = np.nan
        
        for pb in range(6):
            pb_data = obj_data[obj_data['passband'] == pb].sort_values('mjd')
            if len(pb_data) > 2:
                flux_diff = np.diff(pb_data['flux'].values)
                time_diff = np.diff(pb_data['mjd'].values)
                
                time_diff[time_diff == 0] = 1e-8
                flux_derivative = flux_diff / time_diff
                
                features_dict[f'flux_deriv_max_{pb}'] = flux_derivative.max()
                features_dict[f'flux_deriv_min_{pb}'] = flux_derivative.min()
                features_dict[f'flux_deriv_std_{pb}'] = flux_derivative.std()
            else:
                features_dict[f'flux_deriv_max_{pb}'] = np.nan
                features_dict[f'flux_deriv_min_{pb}'] = np.nan
                features_dict[f'flux_deriv_std_{pb}'] = np.nan
                
        adv_features.append(features_dict)
    
    return pd.DataFrame(adv_features)


# Cell 4: Data Augmentation for Rare Classes
print("Performing data augmentation for rare classes...")

train_original = train.copy()
train_meta_original = train_meta.copy()

class_counts = train_meta['target'].value_counts()
rare_classes = class_counts[class_counts < 100].index.tolist()

print(f"Rare classes to augment: {rare_classes}")
print(f"Class distribution before augmentation:")
for cls in rare_classes:
    print(f"  Class {cls}: {class_counts[cls]} samples")

def advanced_augment_light_curves(df, meta_df, target_classes, n_augment=10):
    augmented_data = []
    augmented_meta = []
    
    target_objects = meta_df[meta_df['target'].isin(target_classes)]['object_id'].values
    
    print(f"Augmenting {len(target_objects)} objects")
    
    for obj_id in tqdm(target_objects, desc="Augmenting"):
        obj_data = df[df['object_id'] == obj_id].copy()
        obj_meta = meta_df[meta_df['object_id'] == obj_id].copy()
        
        peak_flux_dict = {}
        for pb in range(6):
            pb_data = obj_data[obj_data['passband'] == pb]
            if len(pb_data) > 0:
                peak_flux_dict[pb] = pb_data['flux'].max()
        
        for i in range(n_augment):
            aug_data = obj_data.copy()
            aug_meta = obj_meta.copy()
            
            for pb in range(6):
                pb_mask = aug_data['passband'] == pb
                pb_data = aug_data[pb_mask].copy()
                
                if len(pb_data) > 3:
                    pb_data = pb_data.sort_values('mjd')
                    
                    noise = np.random.randn(len(pb_data))
                    for j in range(1, len(noise)):
                        noise[j] = 0.6 * noise[j-1] + 0.4 * noise[j]
                    
                    noise_scaled = noise * pb_data['flux_err'].values * 0.2
                    pb_data['flux'] += noise_scaled
                    
                    aug_data.loc[pb_mask, 'flux'] = pb_data['flux'].values
            
            time_factor = np.random.uniform(0.95, 1.05)
            min_mjd = aug_data['mjd'].min()
            aug_data['mjd'] = min_mjd + (aug_data['mjd'] - min_mjd) * time_factor
            
            extinction = np.random.uniform(0, 0.2)
            passband_extinction = {
                0: 1.0 - extinction * 0.4,
                1: 1.0 - extinction * 0.3,
                2: 1.0 - extinction * 0.2,
                3: 1.0 - extinction * 0.15,
                4: 1.0 - extinction * 0.1,
                5: 1.0 - extinction * 0.05
            }
            
            for pb, factor in passband_extinction.items():
                mask = aug_data['passband'] == pb
                aug_data.loc[mask, 'flux'] *= factor
                aug_data.loc[mask, 'flux_err'] *= factor
            
            phase_shift = np.random.uniform(-3, 3)
            aug_data['mjd'] += phase_shift
            
            for pb in peak_flux_dict:
                pb_mask = aug_data['passband'] == pb
                min_allowed = -0.2 * peak_flux_dict[pb]
                aug_data.loc[pb_mask & (aug_data['flux'] < min_allowed), 'flux'] = min_allowed
            
            new_obj_id = obj_id * 10000 + (i + 1)
            aug_data['object_id'] = new_obj_id
            aug_meta['object_id'] = new_obj_id
            
            if 'hostgal_photoz' in aug_meta.columns:
                aug_meta['hostgal_photoz'] += np.random.normal(0, 0.02)
                aug_meta['hostgal_photoz'] = aug_meta['hostgal_photoz'].clip(0, 3)
            
            augmented_data.append(aug_data)
            augmented_meta.append(aug_meta)
    
    if len(augmented_data) > 0:
        return pd.concat(augmented_data), pd.concat(augmented_meta)
    else:
        return pd.DataFrame(), pd.DataFrame()

if len(rare_classes) > 0:
    aug_train, aug_meta = advanced_augment_light_curves(
        train, train_meta, rare_classes, n_augment=10
    )
    
    if len(aug_train) > 0:
        train = pd.concat([train, aug_train], ignore_index=True)
        train_meta = pd.concat([train_meta, aug_meta], ignore_index=True)
        
        print(f"\nOriginal samples: {len(train_original)}")
        print(f"After augmentation: {len(train)}")
        print(f"Increase: +{len(train) - len(train_original)} samples")


# Cell 5: Label Encoding and Feature Extraction
print("\nLabel Encoding...")
print(f"Original unique classes: {sorted(train_meta['target'].unique())}")

class_mapping = {c: i for i, c in enumerate(sorted(train_meta['target'].unique()))}
reverse_mapping = {i: c for c, i in class_mapping.items()}

print("\nClass mapping:")
for original, encoded in class_mapping.items():
    print(f"Class {original} -> {encoded}")

train_meta['target_original'] = train_meta['target'].copy()
train_meta['target'] = train_meta['target'].map(class_mapping)

print("\nFeature Extraction...")

stat_features = extract_features(train)
print(f"Statistical features: {stat_features.shape}")

time_features = add_time_features(train)
print(f"Time features: {time_features.shape}")

peak_features = add_peak_features(train)
print(f"Peak features: {peak_features.shape}")

color_features = add_color_features(train)
print(f"Color features: {color_features.shape}")

advanced_features = add_advanced_features(train)
print(f"Advanced features: {advanced_features.shape}")

print("\nMerging all features...")
features = stat_features
features = features.merge(time_features, on='object_id', how='left')
features = features.merge(peak_features, on='object_id', how='left')
features = features.merge(color_features, on='object_id', how='left')
features = features.merge(advanced_features, on='object_id', how='left')
features = features.merge(train_meta, on='object_id', how='left')

duplicate_cols = features.columns[features.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"Warning: Found duplicate columns: {duplicate_cols}")
    features = features.loc[:, ~features.columns.duplicated()]

print(f"\nMissing values before filling: {features.isnull().sum().sum()}")
features = features.fillna(0)

print(f"Final features shape: {features.shape}")
print(f"Target range: {features['target'].min()} to {features['target'].max()}")


# Cell 6: Prepare Training Data
feature_cols = [col for col in features.columns if col not in [
    'object_id', 'target', 'target_original', 'hostgal_specz', 'ra', 'decl'
]]

X = features[feature_cols].values
y = features['target'].values

print(f"\nTraining data shape: X={X.shape}, y={y.shape}")
print(f"Number of features: {len(feature_cols)}")

classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, class_weights))

print("\nClass distribution:")
print(pd.Series(y).value_counts().sort_index())


# Cell 7: LightGBM Training with Cross-Validation
print("\nLightGBM Training with Weighted Log Loss")

class_weights_plasticc = {6: 1, 15: 1, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1,
                          64: 1, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1, 99: 2}

mapped_weights = {}
for c, w in class_weights_plasticc.items():
    if c in class_mapping:
        mapped_weights[class_mapping[c]] = w
    else:
        mapped_weights[c] = 1

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(class_mapping),
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'num_leaves': 300,
    'max_depth': 15,
    'min_data_in_leaf': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.05,
    'feature_fraction_bynode': 0.8,
    'verbose': -1,
    'seed': 42,
    'num_threads': -1,
    'extra_trees': True,
    'path_smooth': 0.1,
    'max_bin': 255,
    'min_sum_hessian_in_leaf': 0.1,
    'subsample_for_bin': 200000,
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

cv_scores = []
feature_importance_list = []
predictions = np.zeros((len(X), params['num_class']))
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    sample_weights = np.array([mapped_weights.get(label, 1) for label in y_train])
    val_weights = np.array([mapped_weights.get(label, 1) for label in y_val])
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, weight=val_weights, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200)
        ]
    )
    
    models.append(model)
    
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    predictions[val_idx] = val_preds
    
    val_score = 0
    for i in range(len(y_val)):
        prob = val_preds[i][y_val[i]]
        val_score += -mapped_weights.get(y_val[i], 1) * np.log(max(prob, 1e-15))
    val_score /= sum(val_weights)
    cv_scores.append(val_score)
    
    print(f"Fold {fold + 1} Weighted Log Loss: {val_score:.4f}")
    
    fold_accuracy = (y_val == np.argmax(val_preds, axis=1)).mean()
    print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
    importance = model.feature_importance(importance_type='gain')
    for i, feat in enumerate(feature_cols):
        feature_importance_list.append({
            'feature': feat,
            'importance': importance[i],
            'fold': fold + 1
        })

feature_importance = pd.DataFrame(feature_importance_list)

print(f"\nLightGBM Mean Weighted Log Loss: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
print(f"Overall Accuracy: {(y == np.argmax(predictions, axis=1)).mean():.4f}")


# Cell 8: Feature Importance Analysis
print("\nFeature Importance Analysis")

importance_summary = feature_importance.groupby('feature').agg({
    'importance': ['mean', 'std', 'count']
}).round(2)

importance_summary.columns = ['mean_importance', 'std_importance', 'count']
importance_summary = importance_summary.sort_values('mean_importance', ascending=False)

top_20_features = importance_summary.head(20)

# IEEE ডাবল কলাম - অনেক বড় ফন্ট
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
})

plt.figure(figsize=(16, 14))
y_pos = np.arange(len(top_20_features))

bars = plt.barh(y_pos, top_20_features['mean_importance'],
                xerr=top_20_features['std_importance'],
                align='center', alpha=0.85, capsize=8,
                color='#2196F3', edgecolor='black', linewidth=1.5,
                error_kw={'elinewidth': 3, 'capthick': 3})

plt.yticks(y_pos, top_20_features.index, fontsize=18)
plt.xticks(fontsize=20)
plt.xlabel('Feature Importance (Gain)', fontsize=24, fontweight='bold', labelpad=15)
plt.title(f'Top 20 Feature Importances\nAveraged across {n_folds} folds', 
          fontsize=26, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.4, linewidth=1.5)

# X-axis এর রেঞ্জ বাড়ানো যাতে টেক্সট ফিট হয়
max_val = top_20_features['mean_importance'].max() + top_20_features['std_importance'].max()
plt.xlim(0, max_val * 1.30)

# প্রতিটি বারে ভ্যালু দেখানো - অনেক বড় ফন্ট
for i, (bar, val, std) in enumerate(zip(bars, top_20_features['mean_importance'], 
                                         top_20_features['std_importance'])):
    plt.text(val + std + max_val * 0.04, 
             bar.get_y() + bar.get_height()/2,
             f'{val:.1f}', va='center', 
             fontsize=20, fontweight='bold', color='black')

plt.tick_params(axis='both', width=2, length=8)
plt.tight_layout()
# আগে ছিল:
# plt.savefig('feature_importance.png', dpi=600, bbox_inches='tight')

# এখন এটা করো:
plt.savefig('feature_importance.pdf', format='pdf', bbox_inches='tight')
plt.savefig('feature_importance.eps', format='eps', bbox_inches='tight')
plt.savefig('feature_importance.png', dpi=600, bbox_inches='tight')  # backup PNG ও রাখো
plt.show()

mean_importance = importance_summary['mean_importance']

print("\nTop 10 Most Important Features:")
for i, (feat, row) in enumerate(top_20_features.head(10).iterrows(), 1):
    print(f"{i:2d}. {feat:30s}: {row['mean_importance']:6.2f} (±{row['std_importance']:5.2f})")


# Cell 9: Confusion Matrix
print("\nConfusion Matrix")

y_pred = np.argmax(predictions, axis=1)

y_original = features['target_original'].values
y_pred_original = [reverse_mapping[pred] for pred in y_pred]

cm = confusion_matrix(y_original, y_pred_original, labels=sorted(train_meta['target_original'].unique()))

# IEEE ডাবল কলাম - অনেক বড় ফন্ট
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
})

plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(train_meta['target_original'].unique()),
            yticklabels=sorted(train_meta['target_original'].unique()),
            annot_kws={'size': 22, 'fontweight': 'bold'},
            linewidths=1.5,
            linecolor='white',
            cbar_kws={'shrink': 0.8})

plt.xlabel('Predicted Class', fontsize=24, fontweight='bold', labelpad=20)
plt.ylabel('Actual Class', fontsize=24, fontweight='bold', labelpad=20)
plt.title('Confusion Matrix - PLAsTiCC Classification', fontsize=26, fontweight='bold', pad=25)
plt.xticks(fontsize=18, rotation=45, ha='right')
plt.yticks(fontsize=18, rotation=0)

# Colorbar ফন্ট বড় করা
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
# আগে ছিল:
# plt.savefig('confusion_matrix.png', dpi=600, bbox_inches='tight')

# এখন এটা করো:
plt.savefig('confusion_matrix.pdf', format='pdf', bbox_inches='tight')
plt.savefig('confusion_matrix.eps', format='eps', bbox_inches='tight')
plt.savefig('confusion_matrix.png', dpi=600, bbox_inches='tight')  # backup
plt.show()

print("\nClassification Report:")
print(classification_report(y_original, y_pred_original,
                          target_names=[str(c) for c in sorted(train_meta['target_original'].unique())]))


# Cell 10: LightGBM Results Summary
print("\nLightGBM Results Summary")

lgb_results = pd.DataFrame({
    'Metric': ['Log Loss', 'Accuracy', 'Macro F1', 'Weighted F1'],
    'Score': [
        np.mean(cv_scores),
        (y == y_pred).mean(),
        classification_report(y, y_pred, output_dict=True)['macro avg']['f1-score'],
        classification_report(y, y_pred, output_dict=True)['weighted avg']['f1-score']
    ]
})

print(lgb_results.to_string(index=False))


# Cell 11: Multiple Algorithms Comparison
print("\nMultiple Algorithms Comparison")

n_classes = len(np.unique(y))
results_comparison = []

print("\n1. Training Random Forest...")
start_time = time.time()

rf_scores = []
rf_predictions = np.zeros((len(X), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"RF Fold {fold + 1}/{n_folds}", end=' ')
    
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
        criterion='entropy',
        bootstrap=True,
        oob_score=True
    )
    
    rf.fit(X[train_idx], y[train_idx])
    
    rf_pred_raw = rf.predict_proba(X[val_idx])
    
    if rf_pred_raw.shape[1] < n_classes:
        rf_pred = np.zeros((len(val_idx), n_classes))
        for i, class_label in enumerate(rf.classes_):
            rf_pred[:, class_label] = rf_pred_raw[:, i]
    else:
        rf_pred = rf_pred_raw
    
    rf_predictions[val_idx] = rf_pred
    
    scores = []
    val_weights = np.array([mapped_weights.get(label, 1) for label in y[val_idx]])
    for i in range(len(val_idx)):
        prob = rf_pred[i, y[val_idx][i]]
        scores.append(-mapped_weights.get(y[val_idx][i], 1) * np.log(max(prob, 1e-15)))
    score = np.sum(scores) / np.sum(val_weights)
    rf_scores.append(score)
    print(f"Weighted Loss: {score:.4f}")

rf_time = time.time() - start_time
print(f"\nRF Mean Weighted Log Loss: {np.mean(rf_scores):.4f} (Time: {rf_time:.1f}s)")
print(f"RF Overall Accuracy: {(y == np.argmax(rf_predictions, axis=1)).mean():.4f}")

results_comparison.append({
    'Algorithm': 'Random Forest',
    'Log Loss': np.mean(rf_scores),
    'Std': np.std(rf_scores),
    'Accuracy': (y == np.argmax(rf_predictions, axis=1)).mean(),
    'Time (s)': rf_time
})

print("\n2. Training XGBoost...")
try:
    from xgboost import XGBClassifier
    start_time = time.time()
    
    xgb_scores = []
    xgb_predictions = np.zeros((len(X), n_classes))
    successful_folds = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"XGB Fold {fold + 1}/{n_folds}", end=' ')
        
        try:
            xgb = XGBClassifier(
                n_estimators=1000,
                max_depth=15,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                colsample_bynode=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                min_child_weight=1,
                gamma=0.05,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                tree_method='hist'
            )
            
            sample_weights = np.array([mapped_weights.get(y[idx], 1) for idx in train_idx])
            
            xgb.fit(
                X[train_idx], y[train_idx],
                sample_weight=sample_weights,
                eval_set=[(X[val_idx], y[val_idx])],
                early_stopping_rounds=150,
                verbose=False
            )
            
            xgb_pred = xgb.predict_proba(X[val_idx])
            xgb_predictions[val_idx] = xgb_pred
            
            scores = []
            val_weights = np.array([mapped_weights.get(label, 1) for label in y[val_idx]])
            for i in range(len(val_idx)):
                prob = xgb_pred[i, y[val_idx][i]]
                scores.append(-mapped_weights.get(y[val_idx][i], 1) * np.log(max(prob, 1e-15)))
            score = np.sum(scores) / np.sum(val_weights)
            xgb_scores.append(score)
            successful_folds += 1
            print(f"Weighted Loss: {score:.4f}")
                
        except Exception as e:
            print(f"Error: {str(e)[:50]}...")
            continue
    
    if successful_folds > 0:
        xgb_time = time.time() - start_time
        print(f"\nXGB Mean Weighted Log Loss: {np.mean(xgb_scores):.4f} (Time: {xgb_time:.1f}s)")
        print(f"XGB Overall Accuracy: {(y == np.argmax(xgb_predictions, axis=1)).mean():.4f}")
        
        results_comparison.append({
            'Algorithm': 'XGBoost',
            'Log Loss': np.mean(xgb_scores),
            'Std': np.std(xgb_scores) if len(xgb_scores) > 1 else 0,
            'Accuracy': (y == np.argmax(xgb_predictions, axis=1)).mean(),
            'Time (s)': xgb_time
        })
    else:
        print("\nXGBoost failed on all folds")
        
except ImportError:
    print("XGBoost not installed")

print("\n3. Training Extra Trees...")
start_time = time.time()

et_scores = []
et_predictions = np.zeros((len(X), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"ET Fold {fold + 1}/{n_folds}", end=' ')
    
    et = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample',
        criterion='entropy',
        bootstrap=False,
        max_features='sqrt'
    )
    
    et.fit(X[train_idx], y[train_idx])
    
    et_pred_raw = et.predict_proba(X[val_idx])
    
    if et_pred_raw.shape[1] < n_classes:
        et_pred = np.zeros((len(val_idx), n_classes))
        for i, class_label in enumerate(et.classes_):
            et_pred[:, class_label] = et_pred_raw[:, i]
    else:
        et_pred = et_pred_raw
    
    et_predictions[val_idx] = et_pred
    
    scores = []
    val_weights = np.array([mapped_weights.get(label, 1) for label in y[val_idx]])
    for i in range(len(val_idx)):
        prob = et_pred[i, y[val_idx][i]]
        scores.append(-mapped_weights.get(y[val_idx][i], 1) * np.log(max(prob, 1e-15)))
    score = np.sum(scores) / np.sum(val_weights)
    et_scores.append(score)
    print(f"Weighted Loss: {score:.4f}")

et_time = time.time() - start_time
print(f"\nET Mean Weighted Log Loss: {np.mean(et_scores):.4f} (Time: {et_time:.1f}s)")
print(f"ET Overall Accuracy: {(y == np.argmax(et_predictions, axis=1)).mean():.4f}")

results_comparison.append({
    'Algorithm': 'Extra Trees',
    'Log Loss': np.mean(et_scores),
    'Std': np.std(et_scores),
    'Accuracy': (y == np.argmax(et_predictions, axis=1)).mean(),
    'Time (s)': et_time
})

results_comparison.append({
    'Algorithm': 'LightGBM',
    'Log Loss': np.mean(cv_scores),
    'Std': np.std(cv_scores),
    'Accuracy': (y == np.argmax(predictions, axis=1)).mean(),
    'Time (s)': 0
})

print("\nAlgorithms trained successfully")

if len(results_comparison) > 0:
    results_df = pd.DataFrame(results_comparison)
    print("\nResults Summary:")
    print(results_df.to_string(index=False))


# Cell 12: Neural Network Training
print("\n4. Training Neural Network...")
start_time = time.time()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp_scores = []
mlp_predictions = np.zeros((len(X), len(class_mapping)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"MLP Fold {fold + 1}/{n_folds}", end=' ')
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        verbose=False,
        alpha=0.001,
        learning_rate_init=0.001
    )
    
    mlp.fit(X_scaled[train_idx], y[train_idx])
    
    mlp_pred = mlp.predict_proba(X_scaled[val_idx])
    
    if mlp_pred.shape[1] < len(class_mapping):
        mlp_pred_full = np.zeros((len(val_idx), len(class_mapping)))
        for i, class_label in enumerate(mlp.classes_):
            mlp_pred_full[:, class_label] = mlp_pred[:, i]
        mlp_pred = mlp_pred_full
    
    mlp_predictions[val_idx] = mlp_pred
    
    scores = []
    val_weights = np.array([mapped_weights.get(label, 1) for label in y[val_idx]])
    for i in range(len(val_idx)):
        prob = mlp_pred[i, y[val_idx][i]]
        scores.append(-mapped_weights.get(y[val_idx][i], 1) * np.log(max(prob, 1e-15)))
    score = np.sum(scores) / np.sum(val_weights)
    mlp_scores.append(score)
    print(f"Weighted Loss: {score:.4f}")

mlp_time = time.time() - start_time
print(f"\nMLP Mean Weighted Log Loss: {np.mean(mlp_scores):.4f} (Time: {mlp_time:.1f}s)")
print(f"MLP Overall Accuracy: {(y == np.argmax(mlp_predictions, axis=1)).mean():.4f}")

results_comparison.append({
    'Algorithm': 'Neural Network',
    'Log Loss': np.mean(mlp_scores),
    'Std': np.std(mlp_scores),
    'Accuracy': (y == np.argmax(mlp_predictions, axis=1)).mean(),
    'Time (s)': mlp_time
})

print("\nNeural Network training completed")


# Cell 13: Results Comparison Visualization
results_df = pd.DataFrame(results_comparison).sort_values('Log Loss')
print("\nAlgorithm Comparison Results:")
print(results_df.to_string(index=False))

best_base_model = results_df.iloc[0]
for idx, row in results_df.iterrows():
    improvement = (best_base_model['Log Loss'] - row['Log Loss']) / best_base_model['Log Loss'] * 100
    print(f"{row['Algorithm']:20s}: {improvement:+6.2f}% vs best model")

results_df.to_csv('algorithm_comparison_final.csv', index=False)

# IEEE ডাবল কলাম - অনেক বড় ফন্ট
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.0,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

algorithms = results_df['Algorithm']
log_losses = results_df['Log Loss']
errors = results_df['Std']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars1 = ax1.bar(algorithms, log_losses, yerr=errors, capsize=8, 
                color=colors[:len(algorithms)], edgecolor='black', linewidth=2)

ax1.set_xlabel('Algorithm', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_ylabel('Log Loss', fontsize=22, fontweight='bold', labelpad=15)
ax1.set_title('Log Loss Comparison', fontsize=24, fontweight='bold', pad=20)
ax1.set_xticklabels(algorithms, rotation=30, ha='right', fontsize=18)
ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(axis='both', width=2, length=8)
ax1.grid(axis='y', alpha=0.4, linewidth=1.5)

# বারের উপরে টেক্সট - অনেক বড় ফন্ট
max_height1 = max(log_losses) + max(errors)
ax1.set_ylim(0, max_height1 * 1.22)
for bar, loss, err in zip(bars1, log_losses, errors):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.025,
             f'{loss:.3f}', ha='center', va='bottom', 
             fontsize=22, fontweight='bold', color='black')

accuracies = results_df['Accuracy']
bars2 = ax2.bar(algorithms, accuracies, color=colors[:len(algorithms)], 
                edgecolor='black', linewidth=2)

ax2.set_xlabel('Algorithm', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_ylabel('Accuracy', fontsize=22, fontweight='bold', labelpad=15)
ax2.set_title('Accuracy Comparison', fontsize=24, fontweight='bold', pad=20)
ax2.set_xticklabels(algorithms, rotation=30, ha='right', fontsize=18)
ax2.tick_params(axis='y', labelsize=18)
ax2.tick_params(axis='both', width=2, length=8)
ax2.grid(axis='y', alpha=0.4, linewidth=1.5)

# বারের উপরে টেক্সট - অনেক বড় ফন্ট
max_height2 = max(accuracies)
ax2.set_ylim(0, max_height2 * 1.15)
for bar, acc in zip(bars2, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
             f'{acc:.3f}', ha='center', va='bottom', 
             fontsize=22, fontweight='bold', color='black')

plt.suptitle('Algorithm Performance Comparison on PLAsTiCC Dataset', 
             fontsize=26, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(top=0.88, bottom=0.18, wspace=0.3)
plt.savefig('algorithm_comparison.png', dpi=600, bbox_inches='tight')
# আগে ছিল:
# plt.savefig('algorithm_comparison.png', dpi=600, bbox_inches='tight')

# এখন এটা করো:
plt.savefig('algorithm_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('algorithm_comparison.eps', format='eps', bbox_inches='tight')
plt.savefig('algorithm_comparison.png', dpi=600, bbox_inches='tight')  # backup
plt.show()


# Cell 14: Ensemble Model
print("\nAdvanced Ensemble Model")

class LGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params, num_rounds=2000):
        self.params = params
        self.num_rounds = num_rounds
        self.model = None
        
    def fit(self, X, y, sample_weight=None):
        train_data = lgb.Dataset(X, label=y, weight=sample_weight)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_rounds,
            callbacks=[lgb.log_evaluation(0)]
        )
        return self
    
    def predict_proba(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

lgb_model = LGBWrapper(params)

rf_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced_subsample',
    criterion='entropy',
    random_state=42,
    n_jobs=-1
)

try:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    use_xgb = True
except:
    use_xgb = False

et_model = ExtraTreesClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

if use_xgb:
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('et', et_model)
        ],
        voting='soft',
        weights=[0.4, 0.3, 0.2, 0.1]
    )
else:
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('et', et_model)
        ],
        voting='soft',
        weights=[0.5, 0.3, 0.2]
    )

ensemble_scores = []
ensemble_predictions = np.zeros((len(X), len(class_mapping)))

print("Training ensemble with cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nEnsemble Fold {fold + 1}/{n_folds}")
    
    ensemble.fit(X[train_idx], y[train_idx])
    
    ensemble_pred = ensemble.predict_proba(X[val_idx])
    ensemble_predictions[val_idx] = ensemble_pred
    
    val_score = 0
    val_weights = np.array([mapped_weights.get(label, 1) for label in y[val_idx]])
    for i in range(len(val_idx)):
        prob = ensemble_pred[i, y[val_idx][i]]
        val_score += -mapped_weights.get(y[val_idx][i], 1) * np.log(max(prob, 1e-15))
    val_score /= sum(val_weights)
    ensemble_scores.append(val_score)
    
    fold_accuracy = (y[val_idx] == np.argmax(ensemble_pred, axis=1)).mean()
    print(f"Fold {fold + 1} Weighted Log Loss: {val_score:.4f}")
    print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

ensemble_mean_loss = np.mean(ensemble_scores)
ensemble_accuracy = (y == np.argmax(ensemble_predictions, axis=1)).mean()

print(f"\nEnsemble Mean Weighted Log Loss: {ensemble_mean_loss:.4f} (+/- {np.std(ensemble_scores):.4f})")
print(f"Ensemble Overall Accuracy: {ensemble_accuracy:.4f}")

best_single = min(results_comparison, key=lambda x: x['Log Loss'])
print(f"\nBest Single Model: {best_single['Algorithm']}")
print(f"Best Single Log Loss: {best_single['Log Loss']:.4f}")
print(f"Best Single Accuracy: {best_single['Accuracy']:.4f}")

print(f"\nEnsemble Improvement:")
print(f"Log Loss: {(best_single['Log Loss'] - ensemble_mean_loss)/best_single['Log Loss']*100:+.2f}%")
print(f"Accuracy: {(ensemble_accuracy - best_single['Accuracy'])*100:+.2f}%")

print("\nAdvanced ensemble training completed")


# Cell 15: Test Set Evaluation
print("\nTest Set Evaluation")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train set size: {X_train.shape[0]} objects")
print(f"Test set size: {X_test.shape[0]} objects")

print("\nTraining models on train set...")

print("Training LightGBM...")
train_data = lgb.Dataset(
    X_train, label=y_train,
    weight=[mapped_weights.get(label, 1) for label in y_train]
)

lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    callbacks=[lgb.log_evaluation(0)]
)
lgb_pred_test = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
lgb_pred_train = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)

print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=25, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred_test = rf_model.predict_proba(X_test)
rf_pred_train = rf_model.predict_proba(X_train)

print("Training Extra Trees...")
et_model = ExtraTreesClassifier(
    n_estimators=500, max_depth=25, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1,
    class_weight='balanced'
)
et_model.fit(X_train, y_train)
et_pred_test = et_model.predict_proba(X_test)
et_pred_train = et_model.predict_proba(X_train)

try:
    from xgboost import XGBClassifier
    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=500, max_depth=12, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=[mapped_weights.get(label, 1) for label in y_train]
    )
    xgb_pred_test = xgb_model.predict_proba(X_test)
    xgb_pred_train = xgb_model.predict_proba(X_train)
    use_xgb = True
except:
    print("XGBoost not available, skipping...")
    xgb_pred_test = None
    xgb_pred_train = None
    use_xgb = False

print("Training Neural Network...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), max_iter=100, random_state=42,
    early_stopping=True, validation_fraction=0.2
)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred_test = mlp_model.predict_proba(X_test_scaled)
mlp_pred_train = mlp_model.predict_proba(X_train_scaled)

print("\nCreating ensemble predictions...")

if use_xgb:
    ensemble_weights = {
        'LightGBM': 0.35,
        'RandomForest': 0.20,
        'ExtraTrees': 0.15,
        'XGBoost': 0.25,
        'NeuralNetwork': 0.05
    }
    predictions_test = [lgb_pred_test, rf_pred_test, et_pred_test, xgb_pred_test, mlp_pred_test]
    predictions_train = [lgb_pred_train, rf_pred_train, et_pred_train, xgb_pred_train, mlp_pred_train]
    model_names = ['LightGBM', 'RandomForest', 'ExtraTrees', 'XGBoost', 'NeuralNetwork']
else:
    ensemble_weights = {
        'LightGBM': 0.40,
        'RandomForest': 0.30,
        'ExtraTrees': 0.20,
        'NeuralNetwork': 0.10
    }
    predictions_test = [lgb_pred_test, rf_pred_test, et_pred_test, mlp_pred_test]
    predictions_train = [lgb_pred_train, rf_pred_train, et_pred_train, mlp_pred_train]
    model_names = ['LightGBM', 'RandomForest', 'ExtraTrees', 'NeuralNetwork']

ensemble_pred_test = np.zeros_like(lgb_pred_test)
ensemble_pred_train = np.zeros_like(lgb_pred_train)

for name, pred_test, pred_train in zip(model_names, predictions_test, predictions_train):
    weight = ensemble_weights[name]
    ensemble_pred_test += weight * pred_test
    ensemble_pred_train += weight * pred_train

print("\nIndividual Model Performance on Test Set:")

for name, pred_test in zip(model_names, predictions_test):
    test_loss = 0
    test_weights_sum = 0
    for i in range(len(y_test)):
        prob = pred_test[i, y_test[i]]
        weight = mapped_weights.get(y_test[i], 1)
        test_loss += -weight * np.log(max(prob, 1e-15))
        test_weights_sum += weight
    test_loss /= test_weights_sum
    
    test_accuracy = (y_test == np.argmax(pred_test, axis=1)).mean()
    
    print(f"{name:15s}: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.4f}")

print("\nEnsemble Performance on Test Set:")

test_weighted_loss = 0
test_weights = np.array([mapped_weights.get(label, 1) for label in y_test])

for i in range(len(y_test)):
    prob = ensemble_pred_test[i, y_test[i]]
    weight = mapped_weights.get(y_test[i], 1)
    test_weighted_loss += -weight * np.log(max(prob, 1e-15))

test_weighted_loss /= sum(test_weights)

test_preds = np.argmax(ensemble_pred_test, axis=1)
test_accuracy = (y_test == test_preds).mean()

test_f1_macro = f1_score(y_test, test_preds, average='macro')
test_f1_weighted = f1_score(y_test, test_preds, average='weighted')

print(f"Weighted Log Loss: {test_weighted_loss:.4f}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Macro F1-score: {test_f1_macro:.4f}")
print(f"Weighted F1-score: {test_f1_weighted:.4f}")

print(f"\nComparison to Kaggle 1st Place (Private LB: 0.51):")
print(f"Difference: {test_weighted_loss - 0.51:.4f} ({(test_weighted_loss - 0.51)/0.51 * 100:.1f}% higher)")

print("\nPer-class Test Set Performance:")
print(classification_report(y_test, test_preds, target_names=[f"Class {i}" for i in range(n_classes)]))

print("\nTest set evaluation completed")


# Cell 16: Save Results
print("\nSaving results...")

submission_df = pd.DataFrame({
    'object_id': features['object_id'],
    'true_target': y,
    'predicted_target': y_pred,
    'true_class': features['target_original'],
    'predicted_class': [reverse_mapping[p] for p in y_pred]
})

submission_df.to_csv('final_predictions.csv', index=False)
print("Predictions saved to 'final_predictions.csv'")

importance_summary.to_csv('feature_importance_final.csv')
print("Feature importance saved to 'feature_importance_final.csv'")

results_df.to_csv('model_comparison_final.csv', index=False)
print("Model comparison saved to 'model_comparison_final.csv'")

models[-1].save_model('best_lgb_model.txt')
print("Best LightGBM model saved to 'best_lgb_model.txt'")


# Cell 17: Key Results Summary
print("\nKey Results Summary")

print("\n1. Dataset:")
print(f"   Total objects: {len(features):,}")
print(f"   Number of features: {len(feature_cols)}")
print(f"   Number of classes: {len(class_mapping)}")

print("\n2. Model Performance:")
for idx, row in results_df.iterrows():
    print(f"   {row['Algorithm']:15s}: Log Loss = {row['Log Loss']:.3f}, Accuracy = {row['Accuracy']:.3f}")

print("\n3. Top Features:")
for i, (feat, imp) in enumerate(mean_importance.head(5).items(), 1):
    print(f"   {i}. {feat}: {imp:.2f}")

print("\n4. Computational Efficiency:")
total_time = sum([r['Time (s)'] for r in results_comparison if r['Time (s)'] > 0])
print(f"   Total training time: {total_time:.1f} seconds")


# Cell 18: Quick Prediction Demo
print("\nQuick Prediction Demo")

sample_indices = np.random.choice(len(X), 5, replace=False)

print("\nSample predictions:")
for idx in sample_indices:
    true_class = reverse_mapping[y[idx]]
    pred_probs = predictions[idx]
    pred_class = reverse_mapping[np.argmax(pred_probs)]
    confidence = pred_probs[np.argmax(pred_probs)]
    
    print(f"Object {features.iloc[idx]['object_id']}:")
    print(f"  True class: {true_class}")
    print(f"  Predicted: {pred_class} (confidence: {confidence:.3f})")
    print(f"  Correct: {'Yes' if true_class == pred_class else 'No'}")
    print()

print("\nAll processing completed successfully")

