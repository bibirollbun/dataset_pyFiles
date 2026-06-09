#Base
import os
import pandas as pd
import numpy as np

#Error preprocessing
import warnings
warnings.filterwarnings("ignore")

#Graphics, pictures
from IPython.display import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import seaborn as sns

#Preprocessing
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, KFold
from sklearn.inspection import permutation_importance

#Optimization heyperparameters of models
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

#ML models
#Boosting models
from xgboost import XGBRegressor, plot_importance
import lightgbm as lgb
from catboost import CatBoostRegressor
#Linear models with regularization
from sklearn.linear_model import LinearRegression, Lasso, Ridge

#NN
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD

from keras import initializers
from keras import regularizers

#Statistic tests of regression models
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import skew
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant

#Other
import pickle #for save Optuna study file
import time
import shap


df_train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv").drop(columns = ['id']).drop_duplicates()
df_test  = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv").drop(columns = ['id'])


df_train.info()

'''
                                                                    Data description:
                                                                    

1. id (int64) - delete
Type: Identifier
Description: The unique identifier of the record
Purpose: An official feature for identifying observations

2. RhythmScore (float64)
Type: Audio characteristic
Description: Assessment of rhythmic complexity or rhythm quality
Range: From 0 to 1
Meaning: High values indicate a complex/high-quality rhythm

3. AudioLoudness (float64)
Type: Audio characteristic
Description: Audio Recording Volume Level
Units: LUFS (Loudness Units Full Scale)
Value: Determines the perceived volume of the track

4. VocalContent (float64)
Type: Audio characteristic
Description: A measure of vocal content or presence of vocals
Range: From 0 (instrumental) to 1 (fully vocal)
Value: Shows the proportion of the vocal component in the track

5. AcousticQuality (float64)
Type: Audio characteristic
Description: Evaluation of acoustic quality or "purity" of sound
Value: High values indicate better acoustic quality

6. InstrumentalScore (float64)
Type: Audio characteristic
Description: Assessment of the instrumental content or complexity of the arrangement
Value: High values indicate a rich instrumental palette

7. LivePerformanceLikelihood (float64)
Type: Performance Attribute
Description: The probability that the recording was made from a live performance
Range: From 0 (studio) to 1 (live performance)
Meaning: Increases the "liveliness" of the sound

8. MoodScore (float64)
Type: Emotional characteristic
Description: Assessment of the mood or emotional coloring of the track
Range: From 0 (sad) to +1 (cheerful)
Meaning: Reflects the emotional impact of music

9. TrackDurationMs (float64)
Type: Meta Information
Description: Track Duration in Milliseconds
Units: Milliseconds
Value: Audio recording duration

10. Energy (float64)
Type: Audio characteristic
Description: The energy level or intensity of the track
Range: From 0 (calm) to 1 (energetic)
Value: Determines the "drive" and dynamics of the composition

11. BeatsPerMinute (float64) - TARGET
Type: Target Variable
Description: Music tempo in beats per minute
Units: BPM (beats per minute)
Value: The basic metric of musical tempo
''';


df_train.head()


round(df_train.describe(include = 'float64').T, 2)


fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# Hist
axes[0, 0].hist(df_train['BeatsPerMinute'], bins = 50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distibution BeatsPerMinute')
axes[0, 0].set_xlabel('BPM')
axes[0, 0].set_ylabel('Frequency')

# Boxplot
axes[0, 1].boxplot(df_train['BeatsPerMinute'])
axes[0, 1].set_title('Boxplot BeatsPerMinute')
axes[0, 1].set_ylabel('BPM')

# Q-Q plot for check normality
stats.probplot(df_train['BeatsPerMinute'], dist="norm", plot = axes[1, 0])
axes[1, 0].set_title('Q-Q Plot BeatsPerMinute')

# Density distribution
sns.kdeplot(df_train['BeatsPerMinute'], ax = axes[1, 1], fill=True)
axes[1, 1].set_title('Density distribution of BeatsPerMinute')
axes[1, 1].set_xlabel('BPM')

plt.tight_layout()
plt.show()

# Outliers
Q1 = df_train['BeatsPerMinute'].quantile(0.25)
Q3 = df_train['BeatsPerMinute'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_train[(df_train['BeatsPerMinute'] < lower_bound) | (df_train['BeatsPerMinute'] > upper_bound)]
print(f"Counts of outliers in BeatsPerMinute: {len(outliers)} ({len(outliers)/len(df_train)*100:.2f}%)")


features = df_train.columns.drop(['BeatsPerMinute'])

fig, axes = plt.subplots(3, 3, figsize = (15, 12))
axes = axes.ravel()

for i, feature in enumerate(features):
    sns.histplot(df_train[feature], ax = axes[i], kde=True, bins = 30)
    axes[i].set_title(f'Distribution {feature}')
    axes[i].set_xlabel('')
    
plt.tight_layout()
plt.show()


corr_matrix = df_train.corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype = bool), k = 1)
sns.heatmap(corr_matrix, mask = mask, annot = True, cmap = 'seismic', center=0, fmt='.3f')
plt.title('Pirson correlation matrix')
plt.show()

# Print correlation
corr_with_target = corr_matrix['BeatsPerMinute'].sort_values(ascending=False)
print("Feature's correlation with BeatsPerMinute:")
print(corr_with_target)


top_features = corr_with_target.index[1:4]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(top_features):
    sample_df = df_train.sample(2000, random_state=42)
    axes[i].scatter(sample_df[feature], sample_df['BeatsPerMinute'], alpha = 0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('BeatsPerMinute')
    axes[i].set_title(f'{feature} vs BeatsPerMinute')
    
    # Trend line
    z = np.polyfit(sample_df[feature], sample_df['BeatsPerMinute'], 1)
    p = np.poly1d(z)
    axes[i].plot(sample_df[feature], p(sample_df[feature]), "r--", alpha=0.8)

plt.tight_layout()
plt.show()


top_4_features = corr_with_target.index[1:5].tolist() + ['BeatsPerMinute']
sns.pairplot(df_train[top_4_features].sample(1000, random_state = 42), diag_kind = 'kde')
plt.suptitle('Paired relationships of the top 4 features with the target variable', y = 1.02)
plt.show()


X = df_train.drop(['BeatsPerMinute'], axis = 1)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("VIF factors (multicollinearity):")
print(vif_data.sort_values('VIF', ascending = False))


# Convert TrackDurationMs to seconds for better interpretation
df_train['TrackDurationSec'] = df_train['TrackDurationMs'] / 1000

plt.figure(figsize=(12, 6))
sns.scatterplot(x = df_train['TrackDurationSec'], y = df_train['BeatsPerMinute'], alpha = 0.3)
plt.title('The dependence of BPM on the length of the track')
plt.xlabel('Track length (seconds)')
plt.ylabel('BeatsPerMinute')

# Adding a trend line
z = np.polyfit(df_train['TrackDurationSec'], df_train['BeatsPerMinute'], 1)
p = np.poly1d(z)
plt.plot(df_train['TrackDurationSec'], p(df_train['TrackDurationSec']), "r--", alpha=0.8)

plt.show()


Image("/kaggle/input/lgbm-v3/Zonning.png")


df_train = df_train.drop(columns = ['TrackDurationSec'])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters = 5, random_state = 42, n_init = 10)
clusters = kmeans.fit_predict(X_scaled)
df_train['cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.boxplot(x = 'cluster', y = 'BeatsPerMinute', data = df_train)
plt.title('Distribution of BPM across clusters')
plt.show()

cluster_means = df_train.groupby('cluster').mean()
print("Average values of features by cluster:")
round(cluster_means, 2)


df_train = df_train.drop(columns = ['cluster'])


# Boosting models generally treat histogram tails well, but it is better to take their influence into account through skew value and quantification to achieve the best score.
start_numerical_features = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality', 'InstrumentalScore', 
                            'LivePerformanceLikelihood', 'MoodScore', 'TrackDurationMs', 'Energy']

skew_features = df_train[start_numerical_features].select_dtypes(exclude = ['object']).skew().sort_values(ascending = False)
skew_features = pd.DataFrame({'Skew' : skew_features})
skew_features.style.background_gradient('seismic')


Image("/kaggle/input/lgbm-v3/Skew.png")


target_col = 'BeatsPerMinute'

'''
Feature's correlation with BeatsPerMinute:
MoodScore                    0.007059
TrackDurationMs              0.006637
RhythmScore                  0.005440
VocalContent                 0.004876
LivePerformanceLikelihood    0.003471
InstrumentalScore            0.001900
AcousticQuality             -0.000820
AudioLoudness               -0.003327
Energy                      -0.004375
''';


'''
def make_features(df1, df2, n_components_pca = 3, n_clusters = 5, random_state = 42):
    #Feature engineering pipeline for train and test datasets.
    #Steps:
    #1. QuantileTransformer (robust ranking) for numerical features
    #2. Binning (qcut) + Binary indicators (Energy, Mood)
    #3. Logarithmization / interactions / polynomials
    #4. PCA (3 components)
    #5. KMeans (cluster + distances to centroids)
    
    # Copy datasets
    train = df1.copy()
    test  = df2.copy()

    # 1. QuantileTransformer
    # The features have a strong skewness (on hist and skew value)
    # For example, you have a feature with a long right tail (Duration, Loudness)
    # QuantileTransformer "stretches" the data so that the distribution becomes uniform or normal.
    # This removes the bias and makes the signs more "balanced".

    numeric_cols = ['RhythmScore','AudioLoudness',
                    'VocalContent','AcousticQuality',
                    'InstrumentalScore','LivePerformanceLikelihood',
                    'MoodScore','TrackDurationMs', 'Energy']

    numeric_cols_qt = ['InstrumentalScore', 'VocalContent', 'AcousticQuality', 'LivePerformanceLikelihood',
                       'TrackDurationMs', 'MoodScore', 'AudioLoudness'] # After skew's value analysis


    qt = QuantileTransformer(n_quantiles = 256, output_distribution = 'uniform', subsample = int(2e5), random_state = random_state)
    train_qt = qt.fit_transform(train[numeric_cols_qt])
    test_qt  = qt.transform(test[numeric_cols_qt])
    
    train_qt = pd.DataFrame(train_qt, columns=[f"{c}_qt" for c in numeric_cols_qt], index = train.index)
    test_qt  = pd.DataFrame(test_qt,  columns=[f"{c}_qt" for c in numeric_cols_qt], index = test.index)
    
    train = pd.concat([train, train_qt], axis = 1)
    test  = pd.concat([test,  test_qt], axis = 1)

    
    # 2. Binning + indicators (manual binning)
    # Binning (partitioning into intervals/quantiles) allows:
    # --- remove the impact of emissions (all extreme values fall into one "basket");
    # --- to make the distribution more "stepwise", which sometimes helps trees to find thresholds better.;
    # --- to identify non-linear dependencies (for example, "average values of a feature work better than low and high values").
    # Binary indicators (high/medium/low):
    # --- they help to clearly identify the extreme states of a feature (for example, tracks with very high energy);
    # --- they allow trees and linear models to capture threshold effects that are difficult to express in a single numeric variable.

    # For qt1 take 5 quantiles, because the count of anomal values < 100 000
    # For qt2 take 10 quantiles, because the count of anomal values > 100 000 (150k and 200k count of values)
    numeric_cols_qt_1 = ['AcousticQuality', 'LivePerformanceLikelihood',
                         'TrackDurationMs', 'MoodScore', 'AudioLoudness']
    
    numeric_cols_qt_2 = ['VocalContent', 'InstrumentalScore']
    
    for col in numeric_cols_qt_1:
        #train immediately by quantiles
        train[f"{col}_bin"] = pd.qcut(train[col], q = 5, labels = False, duplicates = "drop")
        # to make test match the borders â†’ we take the intervals from train
        _, bins = pd.qcut(train[col], q = 5, retbins = True, duplicates = "drop")
        test[f"{col}_bin"] = pd.cut(test[col], bins = bins, labels = False, include_lowest = True)

    for col in numeric_cols_qt_2:
        train[f"{col}_bin"] = pd.qcut(train[col], q = 10, labels = False, duplicates = "drop")
        _, bins = pd.qcut(train[col], q = 10, retbins = True, duplicates = "drop")
        test[f"{col}_bin"] = pd.cut(test[col], bins = bins, labels = False, include_lowest = True)

    
    q25, q75 = train['Energy'].quantile([0.25, 0.75])
    train['High_Energy']   = (train['Energy'] > q75).astype(int)
    train['Medium_Energy'] = ((train['Energy'] > q25) & (train['Energy'] < q75)).astype(int)
    train['Low_Energy']    = (train['Energy'] < q25).astype(int)
    
    test['High_Energy']   = (test['Energy'] > q75).astype(int)
    test['Medium_Energy'] = ((test['Energy'] > q25) & (test['Energy'] < q75)).astype(int)
    test['Low_Energy']    = (test['Energy'] < q25).astype(int)

    q25, q75 = train['MoodScore'].quantile([0.25, 0.75])
    train['High_Mood']   = (train['MoodScore'] > q75).astype(int)
    train['Medium_Mood'] = ((train['MoodScore'] > q25) & (train['MoodScore'] < q75)).astype(int)
    train['Low_Mood']    = (train['MoodScore'] < q25).astype(int)
    
    test['High_Mood']   = (test['MoodScore'] > q75).astype(int)
    test['Medium_Mood'] = ((test['MoodScore'] > q25) & (test['MoodScore'] < q75)).astype(int)
    test['Low_Mood']    = (test['MoodScore'] < q25).astype(int)
    
    # 3. Logarithms / interactions / polynomials / composite audio metrics
    for col in numeric_cols_qt:
        train[f'log_{col}'] = np.log1p(train[col])
        test[f'log_{col}']  = np.log1p(test[col])
    
    # Interactions
    train['Energy_to_Rhythm_ratio'] = train['Energy'] / (train['RhythmScore'] + 1e-6)
    test['Energy_to_Rhythm_ratio']  = test['Energy']  / (test['RhythmScore'] + 1e-6)
    
    train['Mood_to_Energy_ratio'] = train['MoodScore'] / (train['Energy'] + 1e-6)
    test['Mood_to_Energy_ratio']  = test['MoodScore']  / (test['Energy'] + 1e-6)

    # composite audio metrics
    train['audio_complexity']         = (train['RhythmScore'] + train['InstrumentalScore'] + train['AcousticQuality']) / 3
    train['energy_mood_balance']      = train['Energy'] - train['MoodScore']
    train['vocal_instrumental_ratio'] = train['VocalContent'] / (train['InstrumentalScore'] + 1e-6)
    
    test['audio_complexity']         = (test['RhythmScore'] + test['InstrumentalScore'] + test['AcousticQuality']) / 3
    test['energy_mood_balance']      = test['Energy'] - test['MoodScore']
    test['vocal_instrumental_ratio'] = test['VocalContent'] / (test['InstrumentalScore'] + 1e-6)
    
    # Polynomials
    poly = PolynomialFeatures(degree = 2, include_bias = False)
    poly_train = poly.fit_transform(train[numeric_cols])
    poly_test  = poly.transform(test[numeric_cols])
    
    poly_cols = poly.get_feature_names_out(numeric_cols)
    poly_train = pd.DataFrame(poly_train, columns=[f"Poly_{c}" for c in poly_cols], index = train.index)
    poly_test  = pd.DataFrame(poly_test,  columns=[f"Poly_{c}" for c in poly_cols], index = test.index)
    
    train = pd.concat([train, poly_train], axis = 1)
    test  = pd.concat([test,  poly_test], axis = 1)
    
    # 4. PCA
    # What does PCA do?
    # Compresses correlated features into fewer new variables (components).
    # Each component = a linear combination of the initial features.
    # The components are ordered by decreasing variance: the first component explains the most variation, the second explains the next, and so on.

    scaler = StandardScaler()
    pca_cols = ['InstrumentalScore_qt', 'VocalContent_qt', 'AcousticQuality_qt', 'LivePerformanceLikelihood_qt', 'TrackDurationMs_qt', 'MoodScore_qt', 'AudioLoudness_qt',
                'RhythmScore', 'Energy']

    X_train_scaled = scaler.fit_transform(train[pca_cols])
    X_test_scaled  = scaler.transform(test[pca_cols])
    
    pca = PCA(n_components = n_components_pca, random_state = random_state)
    train_pca = pca.fit_transform(X_train_scaled)
    test_pca  = pca.transform(X_test_scaled)
    
    for i in range(n_components_pca):
        train[f'pca_component_{i+1}'] = train_pca[:, i]
        test[f'pca_component_{i+1}']  = test_pca[:, i]
    
    # 5. KMeans
    # What does KMeans do ?
    # The algorithm groups the data into k clusters, minimizing the sum of the squares of the distances from the points to the nearest centroid.
    # Result:
    # --- cluster label (.predict â†’ 0,1,...,k-1),
    # --- the coordinates of the centroids (.cluster_centers_),
    # --- distances to the centers (.transform â†’ distance matrix).
        
    kmeans = KMeans(n_clusters = n_clusters, random_state = random_state, n_init = 10)
    kmeans.fit(X_train_scaled)
    
    train['audio_cluster'] = kmeans.predict(X_train_scaled)
    test['audio_cluster']  = kmeans.predict(X_test_scaled)
    
    distances_train = kmeans.transform(X_train_scaled)
    distances_test  = kmeans.transform(X_test_scaled)
    
    for i in range(n_clusters):
        train[f'distance_to_cluster_{i}'] = distances_train[:, i]
        test[f'distance_to_cluster_{i}']  = distances_test[:, i]
    
    return train, test
''';


#df_train, df_test = make_features(df_train, df_test)


#df_train.info()


#df_test.info()


def create_features(df):
    df = df.copy()
    
    # 1. Rhythm and Energy interactions
    df['RhythmEnergyProduct'] = df['RhythmScore'] * df['Energy']
    df['RhythmEnergyRatio'] = df['RhythmScore'] / (df['Energy'] + 1e-8)
    
    # 2. Audio characteristics
    df['LoudnessEnergyProduct'] = df['AudioLoudness'] * df['Energy']
    df['VocalInstrumentalRatio'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-8)
    
    # 3. Track duration features
    df['TrackDurationMin'] = df['TrackDurationMs'] / 60000
    df['DurationMoodProduct'] = df['TrackDurationMin'] * df['MoodScore']
    
    # 4. Performance and quality features
    df['QualityPerformanceProduct'] = df['AcousticQuality'] * df['LivePerformanceLikelihood']
    
    # 5. Polynomial features for top correlated features
    corr_without_target = corr_with_target.drop('BeatsPerMinute')
    top_3_features = corr_without_target.head(3).index.tolist()
    for feature in top_3_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))
    
    # 6. Binned features (some of the data is very well separated initially, so we will not use these features in future models)
    df['EnergyBin'] = pd.cut(df['Energy'], bins = 5, labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    df['RhythmBin'] = pd.cut(df['RhythmScore'], bins = 5, labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    # 7. Interaction between rhythm and tempo-related features
    df['RhythmDurationInteraction'] = df['RhythmScore'] * df['TrackDurationMin']
    
    return df


df_train = create_features(df_train)
df_test  = create_features(df_test)


df_train.info()


df_test.info()


# Which speakers are only in train, but not in test
missing_in_test = set(df_train.columns) - set(df_test.columns)
print("Features that are not present in df_test:", missing_in_test)

# Which columns are only in test but not in train
missing_in_train = set(df_test.columns) - set(df_train.columns)
print("Features that are not present in df_train:", missing_in_train)

print(df_train.columns[df_train.columns.duplicated()])


numerical_features = df_train.select_dtypes(include = [np.number]).columns
numerical_features = [col for col in numerical_features if col not in ['id']]

categorical_features = df_train.select_dtypes(exclude = [np.number]).columns
categorical_features = [col for col in categorical_features if col not in ['id']]

feature_columns = [col for col in numerical_features if col != target_col]

X_final = df_train[feature_columns]
y_final = df_train[target_col]
X_test = df_test


# Proba model's parameter
params = {'n_estimators': 2300,
          'max_depth': 5,
          'learning_rate': 0.01,
          'colsample_bytree': 0.30,
          'min_child_weight': 8,
          'objective': 'reg:squarederror',
          'eval_metric': 'rmse',
          'tree_method': 'gpu_hist',
          'device': 'gpu',
          'seed': 42}

model = XGBRegressor(**params)
model.fit(X_final[feature_columns], y_final)

result = permutation_importance(model,
                                X_final[feature_columns].values, y_final.values,
                                n_repeats = 10,
                                scoring='neg_mean_squared_error')

importance = result.importances_mean


features = X_final[feature_columns].columns if hasattr(X_final[feature_columns], 'columns') else range(len(importance))
mask = importance > 0.01
features_filtered = np.array(features)[mask]
importance_filtered = importance[mask]

plt.figure(figsize=(10, 6))
plt.bar(features_filtered, importance_filtered, color="steelblue")
plt.axhline(0, color='k', linestyle='--')

plt.title('Permutation Importance ( > 0.01 )')
plt.ylabel('Increase in MSE')
plt.ylim([-0.5, 3])

plt.xticks(rotation = 90, fontsize=9)
plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

plt.show()


importance_df = pd.DataFrame({"feature": features,
                              "importance": importance})

importance_df = importance_df.sort_values("importance", ascending = False)

threshold = 0.01
selected_features = importance_df[importance_df["importance"] > threshold]["feature"].tolist()

print(f"Selected {len(selected_features)} features from {len(features)}")
print("Top 10 features:\n", selected_features[:10])


def evaluate_model(model, X_train, y_train, X_valid, y_valid, model_name):
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {'model_name': model_name,
            'train_rmse': train_rmse, 'val_rmse': val_rmse,
            'cv_rmse': cv_rmse, 'cv_std': cv_std,
            'model': model}


# Define models to test
models = {'XGBoost': XGBRegressor(objective = 'reg:squarederror', eval_metric = 'rmse', tree_method = 'gpu_hist', device = 'gpu', seed = 42),
          'LightGBM': lgb.LGBMRegressor(random_state = 42, device = 'gpu', verbose = -1),
          'CatBoost': CatBoostRegressor(bootstrap_type = 'Bayesian', task_type = 'GPU', random_seed = 42, verbose = 0),
          'Ridge': Ridge(alpha = 1.0),
          'Lasso': Lasso(alpha = 1.0)}


scaler = RobustScaler().set_output(transform = "pandas")
X_train, X_valid, y_train, y_valid = train_test_split(X_final[feature_columns], y_final, test_size = 0.2, random_state = 42)
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled  = scaler.transform(X_test[feature_columns])


# Evaluate all models
results = []
print("ğŸš€ Training and evaluating models...")

for name, model in models.items():
    print(f"Training {name}...")
    if name in ['Ridge', 'Lasso']:
        result = evaluate_model(model, X_train_scaled, y_train, X_valid_scaled, y_valid, name)
    else:
        result = evaluate_model(model, X_train, y_train, X_valid, y_valid, name)
    results.append(result)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_rmse')

print("\nğŸ�† Model performance comparison")
print("="*70)
for _, row in results_df.iterrows():
    print(f"{row['model_name']:<15} | Val RMSE: {row['val_rmse']:.4f} | "
          f"CV RMSE: {row['cv_rmse']:.4f} Â± {row['cv_std']:.4f}")


def objective(trial):
    y_pred = np.zeros(len(X_valid))
    xgb_params = {'n_estimators': trial.suggest_int("n_estimators", 1000, 3000, step = 100),
                  'max_depth': trial.suggest_int("max_depth", 3, 12, step = 2),
                  'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
                  'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 1e-1, log = True),
                  'subsample': trial.suggest_float("subsample", 0.5, 0.95),
                  'colsample_bytree': trial.suggest_float("colsample_bytree", 0.3, 0.95),
                  'min_child_weight': trial.suggest_int("min_child_weight", 1, 10),
                  'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 1e-1, log = True),
                  'enable_categorical': True,
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'tree_method': 'gpu_hist',
                        'device': 'gpu',
                        'seed': 42
                }
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              verbose = 500)
    y_pred = model.predict(X_valid)
    score = np.sqrt(mean_squared_error(y_valid,  y_pred))
    return score

#sampler = TPESampler(seed = 42)
#study_1 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_1.optimize(objective, n_trials = 100)


# Optuna 100 iterations
# 26.43886 on validation data
XGB_params_1 = {'n_estimators': 1200, 
                'max_depth': 5, 
                'learning_rate': 0.0015489833537889902, 
                'reg_alpha': 0.03406635287133327, 
                'subsample': 0.5526805942440346, 
                'colsample_bytree': 0.7606342905595761, 
                'min_child_weight': 6, 
                'reg_lambda': 0.012987769990997834,
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'tree_method': 'gpu_hist',
                        'device': 'gpu',
                        'seed': 42}

y_pred_val_xgb = np.zeros(len(X_valid))
y_pred_train_xgb = np.zeros(len(X_train))
y_pred_test_xgb = np.zeros(len(X_test))
    
model_xgb_1 = XGBRegressor(**XGB_params_1)
model_xgb_1.fit(X_train, y_train,
                eval_set = [(X_train, y_train), (X_valid, y_valid)],
                verbose = 500)
    
y_pred_val_xgb   = model_xgb_1.predict(X_valid)
y_pred_train_xgb = model_xgb_1.predict(X_train)
y_pred_test_xgb  = model_xgb_1.predict(X_test[feature_columns])
    
fold_rmse_train = mean_squared_error(y_train, y_pred_train_xgb) ** 0.5
fold_r2_train = r2_score(y_train, y_pred_train_xgb)
    
fold_rmse_valid = mean_squared_error(y_valid, y_pred_val_xgb) ** 0.5
fold_r2_valid = r2_score(y_valid, y_pred_val_xgb)

print(f"Final XGBoost RMSE on validation data = {round(fold_rmse_valid, 5)}")
print(f"Final XGBoost R2 on validation data = {round(fold_r2_valid, 5)}")

print(f"Final XGBoost RMSE on train data = {round(fold_rmse_train, 5)}")
print(f"Final XGBoost R2 on train data = {round(fold_r2_train, 5)}")


plt.figure(figsize = (12, 8))
feature_imp = pd.DataFrame(sorted(zip(model_xgb_1.feature_importances_, X_train.columns)), columns = ['Value','Feature'])
sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value", ascending = False).iloc[:25], palette = "husl");


results_xgb = model_xgb_1.evals_result()
plt.figure(figsize=(10,5))
plt.plot(results_xgb["validation_1"]["rmse"], label = "Validation loss")
plt.plot(results_xgb["validation_0"]["rmse"], label = "Training loss")
plt.axvline(1200, color="gray", label = "Optimal tree number")
plt.xlabel("Count of trees")
plt.ylabel("Loss")
plt.title('Loss function for XGBoostRegressor model')
plt.legend();


explainer = shap.TreeExplainer(model_xgb_1)
shap_values = explainer(X_train, check_additivity = False)

fig1 = plt.figure(figsize=(25,15))
ax1 = fig1.add_subplot(111)
shap.plots.beeswarm(shap_values, max_display = 23, show = False, color = plt.get_cmap("cool"))
plt.gca()
plt.title("Beeswarm Shap Plot for XGBoost model")
plt.tight_layout()
plt.show();


explainer = shap.TreeExplainer(model_xgb_1)
shap_values = explainer(X_train, check_additivity = False)
shap.plots.bar(shap_values, max_display = 23, show = False)
plt.title("Bar Shap Plot for XGBoost model")
plt.tight_layout()
plt.show()


from scipy.stats import f
y_true = y_valid
y_pred = y_pred_val_xgb

# Calculation of statistics
n = len(y_true)
p = X_train.shape[1]
sst = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares
ssr = np.sum((y_pred - np.mean(y_true))**2)  # Explained sum of squares
sse = sst - ssr  # Residual sum of squares

mst = sst / (n - 1)  # Total variance
mse = sse / (n - p - 1)  # Residual variance

# F-statistics
f_stat = mst / mse

# Table value for F-riterion (alpha=0.05)
f_critical = f.ppf(0.95, (n-1), ((n - p - 1)), loc=0, scale=1)
print("-----------------------------------------------------")
print(f"F-statistics: {f_stat:.3f}")
print(f"Critical value F: {f_critical:.3f}")

# Autocorrelation test
residuals = y_true - y_pred
dw_stat = durbin_watson(residuals)
print("-----------------------------------------------------")
print(f"Darbin-Watson statistics: {dw_stat:.3f}")
print("lack of autocorrelation: ~2 (1.5-2.5 is acceptable)")
print("-----------------------------------------------------")

# Test for the normality of the residuals distibution
from scipy.stats import shapiro
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk's statistics: {shapiro_stat:.3f}, P-value: {shapiro_p*10**(24):.2f}*10^(-24)")
print("Normality of the distribution if - (p > 0.05)")
print(f"Normality of the distribution: {shapiro_p > 0.05}")
print("-----------------------------------------------------")

# Checking the significance of the R2
r_squared = ssr / sst
print(f"RÂ²: {r_squared:.4f}")
print(f"RÂ² significant: {f_stat > f_critical}")
print("-----------------------------------------------------")

# Heteroscedasticity test
X_valid_fold1_const = add_constant(X_valid)
if X_valid_fold1_const.shape[1] < 2:
    raise ValueError("The test is not possible, and more then 2 variables are required in the predictors")
bp_test = het_breuschpagan(residuals, X_valid_fold1_const)
print(f"Breusch-Pagan Test: p-value = {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print("Heteroscedasticity is present (p < 0.05)")
else:
    print("Heteroscedasticity is not present")

# Residual histogram
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde = True)
plt.title("Distribution of residuals the model")
plt.xlabel("Residuals")
plt.show()

# Graphic of residuals and predictions
plt.figure(figsize = (8, 3))
sns.scatterplot(x = y_pred, y = residuals, size = 0.1, legend = False)
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()


def objective(trial):
    y_pred = np.zeros(len(X_valid))
    
    lgbm_params = {'n_estimators': trial.suggest_int("n_estimators", 500, 2500, step = 100),
                   'num_leaves': trial.suggest_int("num_leaves", 16, 128, step = 2),
                   'max_depth': trial.suggest_int("max_depth", 8, 14),
                   'learning_rate': trial.suggest_float("learning_rate", 1e-3, 1e-1, log = True),
                   'subsample': trial.suggest_float("subsample", 0.5, 0.95),
                   'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 0.95),
                   'device': 'gpu',
                   'random_state': 42,
                   'verbose': -1}

    
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train,
              eval_set = [(X_valid, y_valid)])
    y_pred = model.predict(X_valid)
    score = np.sqrt(mean_squared_error(y_valid,  y_pred))
    return score

#sampler = TPESampler(seed = 42)
#study_2 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_2.optimize(objective, n_trials = 100)


# Optuna 100 iterations
# 26.43647 on validation data
lgbm_params_2 = {'n_estimators': 2000, 
                 'num_leaves': 46, 
                 'max_depth': 10, 
                 'learning_rate': 0.0011469122730138714, 
                 'subsample': 0.565835846998457, 
                 'colsample_bytree': 0.6054124006627148,
                         'device': 'gpu',
                         'random_state': 42,
                         'verbose': -1}

# For metrics
y_pred_val_lgbm = np.zeros(len(X_valid))
y_pred_train_lgbm = np.zeros(len(X_train))
y_pred_test_lgbm = np.zeros(len(X_test))
    
model_lgbm_2 = lgb.LGBMRegressor(**lgbm_params_2)
model_lgbm_2.fit(X_train, y_train, 
                 eval_set = (X_valid, y_valid))
    
y_pred_val_lgbm   = model_lgbm_2.predict(X_valid)
y_pred_train_lgbm = model_lgbm_2.predict(X_train)
y_pred_test_lgbm += model_lgbm_2.predict(X_test[feature_columns])
    
fold_rmse_train = mean_squared_error(y_train, y_pred_train_lgbm) ** 0.5
fold_r2_train = r2_score(y_train, y_pred_train_lgbm)
    
fold_rmse_valid = mean_squared_error(y_valid, y_pred_val_lgbm) ** 0.5
fold_r2_valid = r2_score(y_valid, y_pred_val_lgbm)

print(f"Final LGBM RMSE on validation data = {round(fold_rmse_valid, 5)}")
print(f"Final LGBM R2 on validation data = {round(fold_r2_valid, 5)}")

print(f"Final LGBM RMSE on train data = {round(fold_rmse_train, 5)}")
print(f"Final LGBM R2 on train data = {round(fold_r2_train, 5)}")


plt.figure(figsize = (12, 8))
feature_imp = pd.DataFrame(sorted(zip(model_lgbm_2.feature_importances_, X_train.columns)), columns = ['Value','Feature'])
sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value", ascending = False).iloc[:25], palette = "husl");


results_lgbm = model_lgbm_2.evals_result_
plt.figure(figsize=(10,5))
plt.plot(results_lgbm["valid_0"]["l2"], label = "Validation loss")
plt.axvline(2000, color = "gray", label = "Optimal tree number")
plt.xlabel("Count of trees")
plt.ylabel("Loss")
plt.title('Loss function for LightGBMRegressor model')
plt.legend();


# Lack of video memory

#explainer = shap.TreeExplainer(model_lgbm_2)
#shap_values = explainer(X_train, check_additivity = False)

#fig1 = plt.figure(figsize=(25,15))
#ax1 = fig1.add_subplot(111)
#shap.plots.beeswarm(shap_values, max_display = 23, show = False, color = plt.get_cmap("cool"))
#plt.gca()
#plt.title("Beeswarm Shap Plot for LightGBM model")
#plt.tight_layout()
#plt.show();


y_true = y_valid
y_pred = y_pred_val_lgbm

n = len(y_true)
p = X_train.shape[1]
sst = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares
ssr = np.sum((y_pred - np.mean(y_true))**2)  # Explained sum of squares
sse = sst - ssr  # Residual sum of squares

mst = sst / (n - 1)  # Total variance
mse = sse / (n - p - 1)  # Residual variance

# F-statistics
f_stat = mst / mse

# Table value for F-riterion (alpha=0.05)
f_critical = f.ppf(0.95, (n-1), ((n - p - 1)), loc=0, scale=1)
print("-----------------------------------------------------")
print(f"F-statistics: {f_stat:.3f}")
print(f"Critical value F: {f_critical:.3f}")

# Autocorrelation test
residuals = y_true - y_pred
dw_stat = durbin_watson(residuals)
print("-----------------------------------------------------")
print(f"Darbin-Watson statistics: {dw_stat:.3f}")
print("lack of autocorrelation: ~2 (1.5-2.5 is acceptable)")
print("-----------------------------------------------------")

# Test for the normality of the residuals distibution
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk's statistics: {shapiro_stat:.3f}, P-value: {shapiro_p*10**(24):.2f}*10^(-24)")
print("Normality of the distribution if - (p > 0.05)")
print(f"Normality of the distribution: {shapiro_p > 0.05}")
print("-----------------------------------------------------")

# Checking the significance of the R2
r_squared = ssr / sst
print(f"RÂ²: {r_squared:.4f}")
print(f"RÂ² significant: {f_stat > f_critical}")
print("-----------------------------------------------------")

# Heteroscedasticity test
X_valid_fold1_const = add_constant(X_valid)
if X_valid_fold1_const.shape[1] < 2:
    raise ValueError("The test is not possible, and more then 2 variables are required in the predictors")
bp_test = het_breuschpagan(residuals, X_valid_fold1_const)
print(f"Breusch-Pagan Test: p-value = {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print("Heteroscedasticity is present (p < 0.05)")
else:
    print("Heteroscedasticity is not present")

# Residual histogram
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde = True)
plt.title("Distribution of residuals the model")
plt.xlabel("Residuals")
plt.show()

# Graphic of residuals and predictions
plt.figure(figsize = (8, 3))
sns.scatterplot(x = y_pred, y = residuals, size = 0.1, legend = False)
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()


def objective(trial):
    y_pred = np.zeros(len(X_valid))
    
    cat_params_3 = {
        'iterations': trial.suggest_int("iterations", 500, 2500, step=100),
        'depth': trial.suggest_int("depth", 6, 12),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        'bootstrap_type': 'Bayesian',
        'task_type': 'GPU',
        'random_seed': 42,
        'verbose': 0
    }

    
    model = CatBoostRegressor(**cat_params_3)
    model.fit(X_train, y_train,
              eval_set=(X_valid, y_valid),
              use_best_model = True)
    y_pred = model.predict(X_valid)
    score = np.sqrt(mean_squared_error(y_valid,  y_pred))
    return score

#sampler = TPESampler(seed = 42)
#study_3 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_3.optimize(objective, n_trials = 100)


# Optuna 100 iterations
# 26.43778 on validation data
catboost_params_3 = {'iterations': 1500, 
                     'depth': 6, 
                     'learning_rate': 0.00451741822710335, 
                     'l2_leaf_reg': 0.08901748261979717, 
                     'random_strength': 0.29804762855129413,
                             'bootstrap_type': 'Bayesian',
                             'task_type': 'GPU',
                             'random_seed': 42,
                             'verbose': 0}

y_pred_val_catboost   = np.zeros(len(X_valid))
y_pred_train_catboost = np.zeros(len(X_train))
y_pred_test_catboost  = np.zeros(len(X_test))
    
model_catboost_3 = CatBoostRegressor(**catboost_params_3)
model_catboost_3.fit(X_train, y_train,
                     eval_set=(X_valid, y_valid),
                     use_best_model = True)
    
y_pred_val_catboost   = model_catboost_3.predict(X_valid)
y_pred_train_catboost = model_catboost_3.predict(X_train)
y_pred_test_catboost += model_catboost_3.predict(X_test[feature_columns])
    
fold_rmse_train = mean_squared_error(y_train, y_pred_train_catboost) ** 0.5
fold_r2_train = r2_score(y_train, y_pred_train_catboost)
    
fold_rmse_valid = mean_squared_error(y_valid, y_pred_val_catboost) ** 0.5
fold_r2_valid = r2_score(y_valid, y_pred_val_catboost)

print(f"Final CatBoost RMSE on validation data = {round(fold_rmse_valid, 5)}")
print(f"Final CatBoost R2 on validation data = {round(fold_r2_valid, 5)}")

print(f"Final CatBoost RMSE on train data = {round(fold_rmse_train, 5)}")
print(f"Final CatBoost R2 on train data = {round(fold_r2_train, 5)}")


plt.figure(figsize = (12, 8))
feature_imp = pd.DataFrame(sorted(zip(model_catboost_3.feature_importances_, X_train.columns)), columns = ['Value','Feature'])
sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value", ascending = False).iloc[:25], palette = "husl");


results_catboost = model_catboost_3.evals_result_
plt.figure(figsize=(10,5))
plt.plot(results_catboost["learn"]["RMSE"], label = "Validation loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title('Loss function for CatBoostRegressor model')
plt.legend();


# Lack of video memory

#explainer = shap.TreeExplainer(model_catboost_3)
#shap_values = explainer(X_train, check_additivity = False)

#fig1 = plt.figure(figsize=(25,15))
#ax1 = fig1.add_subplot(111)
#shap.plots.beeswarm(shap_values, max_display = 23, show = False, color = plt.get_cmap("cool"))
#plt.gca()
#plt.title("Beeswarm Shap Plot for LightGBM model")
#plt.tight_layout()
#plt.show();


y_true = y_valid
y_pred = y_pred_val_catboost

# Calculation of statistics
n = len(y_true)
p = X_train.shape[1]
sst = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares
ssr = np.sum((y_pred - np.mean(y_true))**2)  # Explained sum of squares
sse = sst - ssr  # Residual sum of squares

mst = sst / (n - 1)  # Total variance
mse = sse / (n - p - 1)  # Residual variance

# F-statistics
f_stat = mst / mse

# Table value for F-riterion (alpha=0.05)
f_critical = f.ppf(0.95, (n-1), ((n - p - 1)), loc=0, scale=1)
print("-----------------------------------------------------")
print(f"F-statistics: {f_stat:.3f}")
print(f"Critical value F: {f_critical:.3f}")

# Autocorrelation test
residuals = y_true - y_pred
dw_stat = durbin_watson(residuals)
print("-----------------------------------------------------")
print(f"Darbin-Watson statistics: {dw_stat:.3f}")
print("lack of autocorrelation: ~2 (1.5-2.5 is acceptable)")
print("-----------------------------------------------------")

# Test for the normality of the residuals distibution
from scipy.stats import shapiro
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk's statistics: {shapiro_stat:.3f}, P-value: {shapiro_p*10**(24):.2f}*10^(-24)")
print("Normality of the distribution if - (p > 0.05)")
print(f"Normality of the distribution: {shapiro_p > 0.05}")
print("-----------------------------------------------------")

# Checking the significance of the R2
r_squared = ssr / sst
print(f"RÂ²: {r_squared:.4f}")
print(f"RÂ² significant: {f_stat > f_critical}")
print("-----------------------------------------------------")

# Heteroscedasticity test
X_valid_fold1_const = add_constant(X_valid)
if X_valid_fold1_const.shape[1] < 2:
    raise ValueError("The test is not possible, and more then 2 variables are required in the predictors")
bp_test = het_breuschpagan(residuals, X_valid_fold1_const)
print(f"Breusch-Pagan Test: p-value = {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print("Heteroscedasticity is present (p < 0.05)")
else:
    print("Heteroscedasticity is not present")

# Residual histogram
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde = True)
plt.title("Distribution of residuals the model")
plt.xlabel("Residuals")
plt.show()

# Graphic of residuals and predictions
plt.figure(figsize = (8, 3))
sns.scatterplot(x = y_pred, y = residuals, size = 0.1, legend = False)
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()


def build_model_optuna(hyperparams, input_shape):
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    n_layers = hyperparams.suggest_int("n_layers", 3, 8, 1)

    for i in range(n_layers):
        n_units = hyperparams.suggest_int(f"units_{i}", 16, 512, step = 2)
        activation = hyperparams.suggest_categorical(f"activation_{i}", ["relu", "tanh"])

        if i % 2 == 1:  # Ğ�ĞµÑ‡ĞµÑ‚Ğ½Ñ‹Ğµ Ñ�Ğ»Ğ¾Ğ¸ (1Ğ¹, 3Ğ¹, 5Ğ¹...)
            l1_reg = hyperparams.suggest_float(f"l1_reg_{i}", 1e-5, 1e-1, log=True)
            l2_reg = hyperparams.suggest_float(f"l2_reg_{i}", 1e-5, 1e-1, log=True)
            kernel_reg = regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        else:
            kernel_reg = None

        model.add(layers.Dense(units = n_units,
                               activation = activation,
                               kernel_regularizer = kernel_reg))
        model.add(layers.BatchNormalization())
    
        if i == 2 and n_layers == 5:
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        if i == 3 and n_layers == 6:
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        if (i == 2 or i == 4) and (n_layers == 7):
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))
        if (i == 3 or i == 5) and (n_layers == 8):
            dropout_rate = hyperparams.suggest_float(f"dropout_rate_{i}", 0.1, 0.5)
            model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(1, activation='linear'))
    
    optim = hyperparams.suggest_categorical("optimizer", ["adam", "rmsprop"])
    learning_rate = hyperparams.suggest_float("learning_rate", 1e-4, 0.5, log = True)
    
    if optim == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)

    model.compile(optimizer=optimizer, loss="mse", metrics = [keras.metrics.MeanSquaredError()])
    return model


def objective(trial):
    y_pred = np.zeros(len(X_valid))
    
    model = build_model_optuna(trial, X_train.shape[1])
    model.fit(X_train, y_train,
              epochs = 50, 
              validation_data = [X_valid, y_valid], 
              batch_size = 512,
              verbose = 1)
    y_pred = model.predict(X_valid).flatten()
    score = np.sqrt(mean_squared_error(y_valid,  y_pred))
    return score

#import time
#start_time = time.time()
#sampler = TPESampler(seed = 42)
#study_4 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_4.optimize(objective, n_trials = 100)
#print("--- %s seconds ---" % (time.time() - start_time))


def Evaluate_Optuna_Model(hyperparams, input_shape):
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    n_layers = hyperparams.get("n_layers", 4)
    for i in range(n_layers):
        n_units = hyperparams.get(f"units_{i}", 64)
        activation = hyperparams.get(f"activation_{i}", "relu")
        l1_reg = hyperparams.get(f"l1_reg_{i}", 0.0)
        l2_reg = hyperparams.get(f"l2_reg_{i}", 0.0)
        model.add(layers.Dense(units=n_units,
                               activation=activation,
                               kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg)))
        model.add(layers.BatchNormalization())
        if f"dropout_rate_{i}" in hyperparams:
            dropout_rate = hyperparams[f"dropout_rate_{i}"]
            model.add(layers.Dropout(rate=dropout_rate))

    model.add(layers.Dense(1, activation='linear'))
    optim = hyperparams.get("optimizer", "adam")
    learning_rate = hyperparams.get("learning_rate", 1e-3)
    optimizer = tf.keras.optimizers.get(optim)
    optimizer.learning_rate = learning_rate
    model.compile(optimizer=optimizer, loss="mse", metrics = [keras.metrics.MeanSquaredError()])
    return model


def fit_history(X_tr, y_tr, X_vl, y_vl):
    history = []  
    best_NN_params = {'n_layers': 5, 
                      'units_0': 422, 
                      'activation_0': 
                      'relu', 'units_1': 270, 'activation_1': 'relu', 'l1_reg_1': 3.0162092627967762e-05, 'l2_reg_1': 0.000224109716191095, 
                      'units_2': 484, 'activation_2': 'tanh', 'dropout_rate_2': 0.38120758355807116, 
                      'units_3': 196, 'activation_3': 'relu', 'l1_reg_3': 0.00010165510266418732, 'l2_reg_3': 0.0009749762207436118, 
                      'units_4': 164, 'activation_4': 'relu', 
                      'optimizer': 'adam', 
                      'learning_rate': 0.00015503093158719095}
        
    model_4 = Evaluate_Optuna_Model(best_NN_params, X_tr.shape[1])
    history.append(model_4.fit(X_tr, y_tr, epochs = 50, validation_data = [X_vl, y_vl], batch_size = 512, verbose = 0))
    print("Ready")
    return history


best_NN_params = {'n_layers': 5, 
                  'units_0': 422, 
                  'activation_0': 
                  'relu', 'units_1': 270, 'activation_1': 'relu', 'l1_reg_1': 3.0162092627967762e-05, 'l2_reg_1': 0.000224109716191095, 
                  'units_2': 484, 'activation_2': 'tanh', 'dropout_rate_2': 0.38120758355807116, 
                  'units_3': 196, 'activation_3': 'relu', 'l1_reg_3': 0.00010165510266418732, 'l2_reg_3': 0.0009749762207436118, 
                  'units_4': 164, 'activation_4': 'relu', 
                  'optimizer': 'adam', 
                  'learning_rate': 0.00015503093158719095}
model_keras_4 = Evaluate_Optuna_Model(best_NN_params, X_train.shape[1])
model_keras_4.summary()

history = fit_history(X_train, y_train, X_valid, y_valid)


history = history[0]
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)


plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs, ĞºĞ¾Ğ»-Ğ²Ğ¾')
plt.ylabel('MSE, Ğ¼ĞµÑ‚Ñ€Ñ‹')
plt.ylim([680, 1500])
plt.legend()
plt.grid(True)
plt.show()


# Optuna 100 iterations
# 26.44508 on validation data
best_NN_params = {'n_layers': 5, 
                  'units_0': 422, 
                  'activation_0': 
                  'relu', 'units_1': 270, 'activation_1': 'relu', 'l1_reg_1': 3.0162092627967762e-05, 'l2_reg_1': 0.000224109716191095, 
                  'units_2': 484, 'activation_2': 'tanh', 'dropout_rate_2': 0.38120758355807116, 
                  'units_3': 196, 'activation_3': 'relu', 'l1_reg_3': 0.00010165510266418732, 'l2_reg_3': 0.0009749762207436118, 
                  'units_4': 164, 'activation_4': 'relu', 
                  'optimizer': 'adam',
                  'learning_rate': 0.00015503093158719095}

y_pred_val_keras   = np.zeros(len(X_valid))
y_pred_train_keras = np.zeros(len(X_train))
y_pred_test_keras  = np.zeros(len(X_test))
    
model_keras_4 = Evaluate_Optuna_Model(best_NN_params, X_train.shape[1])
model_keras_4.fit(X_train, y_train,
                  epochs = 50, 
                  validation_data = [X_valid, y_valid], 
                  batch_size = 512,
                  verbose = 0)
    
y_pred_val_keras   = model_keras_4.predict(X_valid).flatten()
y_pred_train_keras = model_keras_4.predict(X_train).flatten()
y_pred_test_keras  = model_keras_4.predict(X_test[feature_columns]).flatten()
    
fold_rmse_train = mean_squared_error(y_train, y_pred_train_keras) ** 0.5
fold_r2_train = r2_score(y_train, y_pred_train_keras)
    
fold_rmse_valid = mean_squared_error(y_valid, y_pred_val_keras) ** 0.5
fold_r2_valid = r2_score(y_valid, y_pred_val_keras)

print(f"Final Keras RMSE on validation data = {round(fold_rmse_valid, 5)}")
print(f"Final Keras R2 on validation data = {round(fold_r2_valid, 5)}")

print(f"Final Keras RMSE on train data = {round(fold_rmse_train, 5)}")
print(f"Final Keras R2 on train data = {round(fold_r2_train, 5)}")


y_true = y_valid
y_pred = y_pred_val_keras

# Calculation of statistics
n = len(y_true)
p = X_train.shape[1]
sst = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares
ssr = np.sum((y_pred - np.mean(y_true))**2)  # Explained sum of squares
sse = sst - ssr  # Residual sum of squares

mst = sst / (n - 1)  # Total variance
mse = sse / (n - p - 1)  # Residual variance

# F-statistics
f_stat = mst / mse

# Table value for F-riterion (alpha=0.05)
f_critical = f.ppf(0.95, (n-1), ((n - p - 1)), loc=0, scale=1)
print("-----------------------------------------------------")
print(f"F-statistics: {f_stat:.3f}")
print(f"Critical value F: {f_critical:.3f}")

# Autocorrelation test
residuals = y_true - y_pred
dw_stat = durbin_watson(residuals)
print("-----------------------------------------------------")
print(f"Darbin-Watson statistics: {dw_stat:.3f}")
print("lack of autocorrelation: ~2 (1.5-2.5 is acceptable)")
print("-----------------------------------------------------")

# Test for the normality of the residuals distibution
from scipy.stats import shapiro
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk's statistics: {shapiro_stat:.3f}, P-value: {shapiro_p*10**24:.2f}*10^(-24)")
print("Normality of the distribution if - (P-value > 0.05)")
print(f"Normality of the distribution: {shapiro_p > 0.05}")
print("-----------------------------------------------------")

# Checking the significance of the R2
r_squared = ssr / sst
print(f"RÂ²: {r_squared:.4f}")
print(f"RÂ² significant: {f_stat > f_critical}")
print("-----------------------------------------------------")

# Heteroscedasticity test
X_valid_fold1_const = add_constant(X_valid)
if X_valid_fold1_const.shape[1] < 2:
    raise ValueError("The test is not possible, and more then 2 variables are required in the predictors")
bp_test = het_breuschpagan(residuals, X_valid_fold1_const)
print(f"Breusch-Pagan Test: p-value = {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print("Heteroscedasticity is present (p < 0.05)")
else:
    print("Heteroscedasticity is not present")

# Residual histogram
plt.figure(figsize=(8, 3))
sns.histplot(residuals, kde = True)
plt.title("Distribution of residuals the model")
plt.xlabel("Residuals")
plt.show()

# Graphic of residuals and predictions
plt.figure(figsize = (8, 3))
sns.scatterplot(x = y_pred, y = residuals, size = 0.1, legend = False)
plt.axhline(y = 0, color = 'r', linestyle = '--')
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()


def objective(trial):
    y_pred = np.zeros(len(X_valid_scaled))
    ridge_params = {'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log = True),
                    #'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    #'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-2, log = True),
                    'max_iter': trial.suggest_int('max_iter', 100, 2000)}

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_valid_scaled)
    score = np.sqrt(mean_squared_error(y_valid,  y_pred))
    return score

#sampler = TPESampler(seed = 42)
#study_5 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_5.optimize(objective, n_trials = 100)


# Optuna 100 iterations
# 26.44342 on validation data
ridge_params_5 = {'alpha': 0.1767016940294795, 'tol': 0.0071144760093434225, 'max_iter': 1491}

y_pred_val_ridge   = np.zeros(len(X_valid_scaled))
y_pred_train_ridge = np.zeros(len(X_train_scaled))
y_pred_test_ridge  = np.zeros(len(X_test_scaled))
    
model_ridge_5 = Ridge(**ridge_params_5)
model_ridge_5.fit(X_train_scaled, y_train)
    
y_pred_val_ridge   = model_ridge_5.predict(X_valid_scaled)
y_pred_train_ridge = model_ridge_5.predict(X_train_scaled)
y_pred_test_ridge += model_ridge_5.predict(X_test_scaled[feature_columns])
    
fold_rmse_train = mean_squared_error(y_train, y_pred_train_ridge) ** 0.5
fold_r2_train = r2_score(y_train, y_pred_train_ridge)
fold_rmse_valid = mean_squared_error(y_valid, y_pred_val_ridge) ** 0.5
fold_r2_valid = r2_score(y_valid, y_pred_val_ridge)

print(f"Final Ridge RMSE on validation data = {round(fold_rmse_valid, 5)}")
print(f"Final Ridge R2 on validation data = {round(fold_r2_valid, 5)}")

print(f"Final Ridge RMSE on train data = {round(fold_rmse_train, 5)}")
print(f"Final Ridge R2 on train data = {round(fold_r2_train, 5)}")


figure = plt.figure(figsize = (15, 5))
plt.plot(X_train_scaled.columns, model_ridge_5.coef_, linewidth = 2, color = 'red', label = 'Ğ’ĞµÑ�Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸')
plt.plot(X_train_scaled.columns, np.zeros(23,), linewidth = 2, color = 'blue')
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.ylabel("Weight in model")
plt.title("Distribution of final weights for the Ridge regression model after training on 20% of validation data");


df_stacking_train = pd.DataFrame({'XGB': y_pred_train_xgb,
                                  'LGBM': y_pred_train_lgbm,
                                  'CatBoost': y_pred_train_catboost,
                                  'Tensorflow': y_pred_train_keras,
                                  'Ridge': y_pred_train_ridge})
df_stacking_train.head()


plt.figure(figsize=(10, 6))

plt.hist(df_stacking_train['XGB'], bins = 30, alpha = 0.5, label = 'XGB')
plt.hist(df_stacking_train['LGBM'], bins = 30, alpha = 0.5, label = 'LGBM')
plt.hist(df_stacking_train['CatBoost'], bins = 30, alpha = 0.5, label = 'CatBoost')
plt.hist(df_stacking_train['Tensorflow'], bins = 30, alpha = 0.5, label = 'TensorFlow')
plt.hist(df_stacking_train['Ridge'], bins = 30, alpha = 0.5, label = 'Ridge')

plt.title('Distribution of model predictions on train data')
plt.xlabel('The meaning of prediction')
plt.ylabel('Frequency')
plt.legend(loc = 'best')
plt.grid(True, alpha = 0.3)
plt.show()


df_stacking_valid = pd.DataFrame({'XGB': y_pred_val_xgb,
                                  'LGBM': y_pred_val_lgbm,
                                  'CatBoost': y_pred_val_catboost,
                                  'Tensorflow': y_pred_val_keras,
                                  'Ridge': y_pred_val_ridge})
df_stacking_valid.head()


plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 6))

plt.hist(df_stacking_valid['XGB'], bins = 30, alpha = 0.5, label = 'XGB')
plt.hist(df_stacking_valid['LGBM'], bins = 30, alpha = 0.5, label = 'LGBM')
plt.hist(df_stacking_valid['CatBoost'], bins = 30, alpha = 0.5, label = 'CatBoost')
plt.hist(df_stacking_valid['Tensorflow'], bins = 30, alpha = 0.5, label = 'TensorFlow')
plt.hist(df_stacking_valid['Ridge'], bins = 30, alpha = 0.5, label = 'Ridge')

plt.title('Distribution of model predictions on validation data')
plt.xlabel('The meaning of prediction')
plt.ylabel('Frequency')
plt.legend(loc = 'best')
plt.grid(True, alpha = 0.3)
plt.show()


df_stacking_test = pd.DataFrame({'XGB': y_pred_test_xgb,
                                 'LGBM': y_pred_test_lgbm,
                                 'CatBoost': y_pred_test_catboost,
                                 'Tensorflow': y_pred_test_keras,
                                 'Ridge': y_pred_test_ridge})
df_stacking_test.head()


plt.style.use('seaborn-v0_8')
plt.figure(figsize=(10, 6))

plt.hist(df_stacking_test['XGB'], bins = 30, alpha = 0.5, label = 'XGB')
plt.hist(df_stacking_test['LGBM'], bins = 30, alpha = 0.5, label = 'LGBM')
plt.hist(df_stacking_test['CatBoost'], bins = 30, alpha = 0.5, label = 'CatBoost')
plt.hist(df_stacking_test['Tensorflow'], bins = 30, alpha = 0.5, label = 'TensorFlow')
plt.hist(df_stacking_test['Ridge'], bins = 30, alpha = 0.5, label = 'Ridge')

plt.title('Distribution of model predictions on test data')
plt.xlabel('The meaning of prediction')
plt.ylabel('Frequency')
plt.legend(loc = 'best')
plt.grid(True, alpha = 0.3)
plt.show()


def objective(trial):
    df_stacking_train_valid = pd.concat([df_stacking_train, df_stacking_valid], axis = 0)
    y_pred = np.zeros(len(df_stacking_valid))
    
    ridge_params = {'alpha': trial.suggest_float('alpha', 1e-3, 1e3, log = True),
                    #'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    #'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']),
                    'tol': trial.suggest_float('tol', 1e-5, 1e-2, log = True),
                    'max_iter': trial.suggest_int('max_iter', 100, 2000)}

    model = Ridge(**ridge_params)
    model.fit(df_stacking_train, y_train)
    y_pred = model.predict(df_stacking_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    return rmse

#sampler = TPESampler(seed = 42)
#study_6 = optuna.create_study(direction = "minimize", sampler = sampler)
#study_6.optimize(objective, n_trials = 100)


ridge_params_3 = {'alpha': 995.2232225244844, 'tol': 0.0011738464043739066, 'max_iter': 1008}
cv = KFold(n_splits = 5, random_state = 42, shuffle = True)
df_stacking_train_valid = pd.concat([df_stacking_train, df_stacking_valid], axis = 0)

rmse_scores_train = []
rmse_scores_valid = []
r2_scores_train = []
r2_scores_valid = []

y_pred_val_stacking = np.zeros(len(df_stacking_train_valid))
y_pred_train_stacking = np.zeros(len(df_stacking_train_valid))
y_pred_test_stacking = np.zeros(len(df_stacking_test))

X_valid_fold1 = None
y_valid_fold1 = None
y_pred_fold1 = None

for i, (idx_train, idx_valid) in enumerate(cv.split(df_stacking_train_valid, y_final)):
    
    X_train = df_stacking_train_valid.iloc[idx_train].copy()
    X_valid = df_stacking_train_valid.iloc[idx_valid].copy()
    y_train = y_final.iloc[idx_train].copy()
    y_valid = y_final.iloc[idx_valid].copy()
    
    model_stacking_3 = Ridge(**ridge_params_3)
    model_stacking_3.fit(X_train, y_train)
    
    y_pred_val_stacking[idx_valid] = model_stacking_3.predict(X_valid)

    if i == 0:
        X_valid_fold1 = X_valid.copy()
        y_valid_fold1 = y_valid.copy()
        y_pred_fold1 = y_pred_val_stacking[idx_valid].copy()
    
    y_pred_train_stacking[idx_train] = model_stacking_3.predict(X_train)
    y_pred_test_stacking            += model_stacking_3.predict(df_stacking_test)
    
    fold_rmse_train = mean_squared_error(y_train, y_pred_train_stacking[idx_train]) ** 0.5
    fold_r2_train = r2_score(y_train, y_pred_train_stacking[idx_train])
    rmse_scores_train.append(fold_rmse_train)
    r2_scores_train.append(fold_r2_train)
    print(f"Fold (Stacking) {i + 1} RMSE Train: {fold_rmse_train:.5f}")
    
    fold_rmse_valid = mean_squared_error(y_valid, y_pred_val_stacking[idx_valid]) ** 0.5
    fold_r2_valid = r2_score(y_valid, y_pred_val_stacking[idx_valid])
    rmse_scores_valid.append(fold_rmse_valid)
    r2_scores_valid.append(fold_r2_valid)
    print(f"Fold (Stacking) {i + 1} RMSE Valid: {fold_rmse_valid:.5f}")

print(f"RMSE on valid data = {round(np.mean(rmse_scores_valid), 5)}")
y_pred_test_stacking /= 5


figure = plt.figure(figsize = (15, 5))
plt.plot(df_stacking_train_valid.columns, model_stacking_3.coef_, linewidth = 2, color = 'red', label = 'Ğ’ĞµÑ�Ğ° Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸')
plt.plot(df_stacking_train_valid.columns, np.zeros(5,), linewidth = 2, color = 'blue')
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.ylabel("Weight in model")
plt.title("Distribution of final weights for the Ridge ensemble model after training on 20% of validation data");


sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
lgbm_v3 = pd.read_csv("/kaggle/input/lgbm-v3/LGBM_V3.csv") # The results of valdiation (KFold, 5 folds) by the LightGBM model based on the same features. LB = 26.38739


sub.head()


lgbm_v3.head()


weight = [0.70, 0.30]


sub['BeatsPerMinute'] = (weight[0]*y_pred_test_lgbm + weight[1]*lgbm_v3["BeatsPerMinute"])
sub.head()


sub.to_csv("submission.csv", index = False)

