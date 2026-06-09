import warnings
import numpy as np 
import pandas as pd 
import seaborn as sns
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col=0)


train_df.head()


train_df.shape


train_df.info()


train_df.isna().sum()


test_df.isna().sum()


train_df.describe()


train_df.duplicated().sum() # No dupicates


categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'object']
data_lengths = {}

for column in categorical_columns:
    original_length = len(set(train_df[column]))
    # Remove all spaces (even between words) and convert to lowercase
    cleaned_column = train_df[column].str.replace(' ', '').str.lower()
    
    unique_values_length = len(set(cleaned_column))
    
    # Store the lengths before and after cleaning
    data_lengths[column] = {'original': original_length, 'unique_after_cleaning': unique_values_length}

for column, lengths in data_lengths.items():
    print(f"Column: {column}")
    print(f"Original length: {lengths['original']}")
    print(f"Unique values after cleaning: {lengths['unique_after_cleaning']}")
    print("-" * 50)


missing_data = train_df.isna().sum() / len(train_df) * 100
missing_data = missing_data[missing_data > 0]

plt.figure(figsize=(10, 6))
missing_data.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Percentage of missing data for each feature', fontsize=16)
plt.ylabel('Missing data percentage (%)')
plt.xlabel('Features')
plt.xticks(rotation=0)
plt.show()


train_df.dropna(subset=['Number_of_Ads'], inplace = True)
test_df.dropna(subset=['Number_of_Ads'], inplace = True)
train_df['Number_of_Ads'].isna().sum()


# 1. Distribution of Episode Length
plt.figure(figsize=(12, 6))
sns.histplot(train_df['Episode_Length_minutes'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Episode Length (in minutes)', fontsize=16)
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Frequency')
plt.show()

# 2. Boxplot of Episode Length by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Episode_Length_minutes', data=train_df)
plt.title('Boxplot of Episode Length by Genre', fontsize=16)
plt.xlabel('Genre')
plt.ylabel('Episode Length (minutes)')
plt.xticks(rotation=45)
plt.show()

# 3. Hexbin plot of Host Popularity vs Episode Length
plt.figure(figsize=(12, 6))
plt.hexbin(train_df['Host_Popularity_percentage'], train_df['Episode_Length_minutes'], gridsize=50, cmap='Blues')
plt.colorbar(label='Frequency')
plt.title('Hexbin Plot: Host Popularity vs Episode Length', fontsize=16)
plt.xlabel('Host Popularity Percentage')
plt.ylabel('Episode Length (minutes)')
plt.show()

# 4. Hexbin plot of Guest Popularity vs Episode Length
plt.figure(figsize=(12, 6))
plt.hexbin(train_df['Guest_Popularity_percentage'], train_df['Episode_Length_minutes'], gridsize=50, cmap='Blues')
plt.colorbar(label='Frequency')
plt.title('Hexbin Plot: Guest Popularity vs Episode Length', fontsize=16)
plt.xlabel('Guest Popularity Percentage')
plt.ylabel('Episode Length (minutes)')
plt.show()

# 5. Hexbin plot of Host Popularity vs Guest Popularity
plt.figure(figsize=(12, 6))
plt.hexbin(train_df['Host_Popularity_percentage'], train_df['Guest_Popularity_percentage'], gridsize=50, cmap='Blues')
plt.colorbar(label='Frequency')
plt.title('Hexbin Plot: Host Popularity vs Guest Popularity', fontsize=16)
plt.xlabel('Host Popularity Percentage')
plt.ylabel('Guest Popularity Percentage')
plt.show()



numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()

def remove_iqr_outliers(df, columns, multiplier=1.5):
    df_clean = df.copy()
    outlier_indices = set()

    for col in columns:
        if col == 'Listening_Time_minutes':
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)].index
        outlier_indices.update(outliers)

        print(f"{col}: Removed {len(outliers)} outliers")

    print(f"Total rows removed: {len(outlier_indices)}")
    df_clean = df_clean.drop(index=outlier_indices)

    return df_clean

print(f"Original rows: {len(train_df)}")
train_df = remove_iqr_outliers(train_df, numerical_columns)
print(f"Cleaned rows: {len(train_df)}")



plt.figure(figsize=(12, 6))
sns.boxplot(x='Genre', y='Episode_Length_minutes', data=train_df)
plt.title('Boxplot of Episode Length by Genre', fontsize=16)
plt.xlabel('Genre')
plt.ylabel('Episode Length (minutes)')
plt.xticks(rotation=45)
plt.show()


label_encoders = {}

for col in categorical_columns:
    label_encoder = LabelEncoder()
    train_df[col] = label_encoder.fit_transform(train_df[col])
    test_df[col] = label_encoder.fit_transform(test_df[col])
    label_encoders[col] = label_encoder

train_df.head()


correlation_matrix = train_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()


# --- Episode_Length_minutes Imputation with LightGBM ---

train_no_missing_ep = train_df[train_df['Episode_Length_minutes'].notnull()]

features_ep = [
    'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads'
]

X_ep = train_no_missing_ep[features_ep]
y_ep = train_no_missing_ep['Episode_Length_minutes']


lgbm_model_ep = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)
lgbm_model_ep.fit(X_ep, y_ep)

missing_ep_idx = train_df['Episode_Length_minutes'].isna()
X_missing_ep = train_df.loc[missing_ep_idx, features_ep]

train_df.loc[missing_ep_idx, 'Episode_Length_minutes'] = lgbm_model_ep.predict(X_missing_ep)


# --- Guest_Popularity_percentage Imputatuin with CatBoost ---

train_no_missing_gp = train_df[train_df['Guest_Popularity_percentage'].notnull()]

features_gp = [
    'Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
    'Episode_Sentiment', 'Host_Popularity_percentage', 'Number_of_Ads',
    'Episode_Length_minutes'
]

X_gp = train_no_missing_gp[features_gp]
y_gp = train_no_missing_gp['Guest_Popularity_percentage']

cat_features_indices = [features_gp.index(f) for f in ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Episode_Sentiment']]

cb_model_gp = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    cat_features=cat_features_indices,
    verbose=0,
    random_state=42
)

cb_model_gp.fit(X_gp, y_gp)

missing_gp_idx = train_df['Guest_Popularity_percentage'].isna()
X_missing_gp = train_df.loc[missing_gp_idx, features_gp]

train_df.loc[missing_gp_idx, 'Guest_Popularity_percentage'] = cb_model_gp.predict(X_missing_gp)

print(train_df.isna().sum())


train_copy = train_df.copy()
test_copy = test_df.copy()

target_column_name = 'Listening_Time_minutes'
train_copy.drop(columns=[target_column_name], inplace=True)

# Function to generate feature combinations
def generate_feature_combinations(X, max_per_pair=2, selected_features=None):
    X = X.copy()
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    generated_feature_names = [] if selected_features is None else selected_features

    for i in range(len(numeric_features)):
        for j in range(i + 1, len(numeric_features)):
            f1, f2 = numeric_features[i], numeric_features[j]

            # All potential transformations
            all_candidates = {
                f'{f1}_plus_{f2}': X[f1] + X[f2],
                f'{f1}_minus_{f2}': X[f1] - X[f2],
                f'{f1}_times_{f2}': X[f1] * X[f2],
                f'{f1}_log_{f2}': np.log1p(np.abs(X[f1])) - np.log1p(np.abs(X[f2])),
                f'{f1}_sqrt_{f2}': np.sqrt(np.abs(X[f1])) - np.sqrt(np.abs(X[f2])),
                f'{f1}_abs_diff_{f2}': np.abs(X[f1] - X[f2])
            }


            # Filter if we're in test df
            if selected_features is not None:
                all_candidates = {k: v for k, v in all_candidates.items() if k in selected_features}
                X = X.assign(**all_candidates)
                X.replace([np.inf, -np.inf], np.nan, inplace=True)
                X.fillna(0, inplace=True)
                continue

            # Otherwise (training mode), compute and select top features by variance
            temp_df = pd.DataFrame(all_candidates)
            temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            temp_df.fillna(0, inplace=True)

            variances = temp_df.var()
            top_features = variances.sort_values(ascending=False).head(max_per_pair).index.tolist()

            for fname in top_features:
                X[fname] = temp_df[fname]
                generated_feature_names.append(fname)
                
    return X, generated_feature_names


train_copy, selected_feature_names = generate_feature_combinations(train_copy, max_per_pair=2)
test_copy, _ = generate_feature_combinations(test_copy, selected_features=selected_feature_names)

train_df_target_column = train_df[target_column_name]
len(train_copy.columns), len(test_copy.columns)


mapping = {i: value for i, value in enumerate(label_encoders['Publication_Day'].classes_)}

print("Encoded to Original Mapping for Publication_Day:")
for encoded_value, original_value in mapping.items():
    print(f"{encoded_value} -> {original_value}")



train_copy['Host_Guest_Popularity_Ratio'] = (train_copy['Host_Popularity_percentage'] / (train_copy['Guest_Popularity_percentage'] + 1e-6))
test_copy['Host_Guest_Popularity_Ratio'] = (test_copy['Host_Popularity_percentage'] / (train_df['Guest_Popularity_percentage'] + 1e-6))

train_copy['Is_Weekend'] = train_copy['Publication_Day'].apply(lambda x: 1 if x in [2, 3] else 0)
test_copy['Is_Weekend'] = test_copy['Publication_Day'].apply(lambda x: 1 if x in [2, 3] else 0)
len(train_copy.columns), len(test_copy.columns)




clustering_features = ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Episode_Length_minutes']

scaler = StandardScaler()
X_clustering = scaler.fit_transform(train_copy[clustering_features])

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
train_copy['Podcast_Cluster'] = kmeans.fit_predict(X_clustering)

X_clustering_test = scaler.transform(test_copy[clustering_features])
test_copy['Podcast_Cluster'] = kmeans.predict(X_clustering_test)
len(train_copy.columns), len(test_copy.columns)


corr_matrix = train_copy.corr()
thresh = 0.95  # correlation threshold
high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > thresh:
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

print("Removed highly correlated features:", len(high_corr_features))
train_copy.drop(columns=high_corr_features, inplace=True)
test_copy.drop(columns=high_corr_features, inplace=True)


# Compute correlation of features with the target variable
target_corr = train_copy.corrwith(train_df_target_column).abs().sort_values(ascending=False)
corr_cutoff = 0.03
selected_features_corr = target_corr[target_corr > corr_cutoff].index.tolist()
print("Selected features after correlation filtering:", selected_features_corr)

# Keep only selected features
train_copy = train_copy[selected_features_corr]
test_copy = test_copy[selected_features_corr]


len(train_copy.columns), len(test_copy.columns)



mi_scores = mutual_info_regression(train_copy, train_df_target_column)
mi_series = pd.Series(mi_scores, index=train_copy.columns).sort_values(ascending=False)

top_k = int(len(mi_series) * 0.75)
selected_features = mi_series.head(top_k).index.tolist()

print("Selected features after MI filtering:", selected_features)
train_copy = train_copy[selected_features]
train_copy[target_column_name] = train_df_target_column
test_copy = test_copy[selected_features]
len(train_copy.columns), len(test_copy.columns)


# Correlation Heatmap
plt.figure(figsize=(19, 19))
sns.heatmap(train_copy.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Correlation Matrix of Selected Features")
plt.show()

# Mutual Information Bar Plot
plt.figure(figsize=(19, 12))
sns.barplot(x=mi_series[selected_features].values, y=mi_series[selected_features].index, palette='viridis')
plt.xlabel("Mutual Information Score")
plt.ylabel("Feature")
plt.title("Mutual Information Scores of Selected Features to Target ")
plt.show()


X = train_copy.drop(columns=[target_column_name]).values.astype(np.float32)
y = train_copy[target_column_name].values.astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)


# lgb_train = lgb.Dataset(X_train, label=y_train)
# lgb_val = lgb.Dataset(X_val, label=y_val)

# params = {
#     'objective': 'regression',
#     'metric': 'rmse',
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.05,
#     'max_depth': 10,
#     'num_leaves': 128,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'reg_lambda': 2.0,
#     'reg_alpha': 1.5,
#     'verbosity': -1,
#     'seed': 42
# }

# model = lgb.train(
#     params,
#     lgb_train,
#     valid_sets=[lgb_train, lgb_val],
#     num_boost_round=300000,
#     callbacks=[
#         early_stopping(stopping_rounds=100),
#         log_evaluation(period=1000)
#     ]
# )



model_xgb = xgb.XGBRegressor(
        n_estimators=30000,
        learning_rate=0.01,
        max_depth=15,
        objective='reg:squarederror',
        random_state=42,
        verbosity=1,
        eval_metric='rmse'
    )

model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=300,
    verbose=1000
)


# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# rmse_scores = []
# models = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     lgb_train = lgb.Dataset(X_train, label=y_train)
#     lgb_val = lgb.Dataset(X_val, label=y_val)

#     params = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'boosting_type': 'gbdt',
#         'learning_rate': 0.05,
#         'max_depth': 10,
#         'num_leaves': 128,
#         'feature_fraction': 0.8,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 5,
#         'reg_lambda': 2.0,
#         'reg_alpha': 1.5,
#         'verbosity': -1,
#         'seed': 42
#     }

#     model = lgb.train(
#         params,
#         lgb_train,
#         valid_sets=[lgb_train, lgb_val],
#         num_boost_round=300000,
#         callbacks=[
#             early_stopping(stopping_rounds=100),
#             log_evaluation(period=1000)
#         ]
#     )

#     preds = model.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, preds))
#     rmse_scores.append(rmse)
#     models.append(model)

#     print(f"Fold {fold} RMSE: {rmse:.4f}")

# print(f"\nAverage RMSE across folds: {np.mean(rmse_scores):.4f}")




# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# rmse_scores_cb = []
# models_cb = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     model_cb = cb.CatBoostRegressor(
#         iterations=10000,
#         learning_rate=0.05,
#         depth=10,
#         l2_leaf_reg=5.0,
#         random_strength=1.5,
#         loss_function='RMSE',
#         eval_metric='RMSE',
#         random_state=42,
#         od_type="Iter",
#         od_wait=100,
#         verbose=1000
#     )

#     model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))

#     preds = model_cb.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, preds))
#     rmse_scores_cb.append(rmse)
#     models_cb.append(model_cb)

#     print(f"Fold {fold} RMSE: {rmse:.4f}")

# print(f"\nAverage RMSE across folds (CatBoost): {np.mean(rmse_scores_cb):.4f}")



# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# rmse_scores_xgb = []
# models_xgb = []

# for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     model_xgb = xgb.XGBRegressor(
#         n_estimators=20000,
#         learning_rate=0.05,
#         max_depth=10,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_lambda=2.0,
#         reg_alpha=1.5,
#         objective='reg:squarederror',
#         random_state=42,
#         verbosity=1,
#         eval_metric='rmse'
#     )

#     model_xgb.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=300,
#         verbose=1000
#     )

#     preds = model_xgb.predict(X_val)
#     rmse = np.sqrt(mean_squared_error(y_val, preds))
#     rmse_scores_xgb.append(rmse)
#     models_xgb.append(model_xgb)

#     print(f"Fold {fold} RMSE: {rmse:.4f}")

# print(f"\nAverage RMSE across folds (XGBoost): {np.mean(rmse_scores_xgb):.4f}")



test_copy_predictions = model_xgb.predict(test_copy)

submission = pd.DataFrame({
    'id': test_copy.index,
    'Listening_Time_minutes': test_copy_predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

