import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm 
import random
sns.set_theme(style="whitegrid")

from itertools import combinations
from sklearn.dummy import DummyRegressor 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor 

from sklearn.utils import shuffle 
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna
from optuna.samplers import TPESampler
import shap


import warnings 
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df.head()


print("shape of the dataset ", df.shape)
num_samples = df.shape[0]


df.describe()


df.info()


df.isnull().sum()


print('Duplicate samples available in the data --> ', df.shape[0] != df.drop_duplicates().shape[0])


fig, axs = plt.subplots(nrows=3, ncols=2)
fig.suptitle("Individual histograms")
fig.set_size_inches(28, 12)

num_features = list(df.describe().columns) 

for i in range(len(num_features)):
    fig.add_subplot(3,2,i+1)
    fig.tight_layout()
    plt.hist(df[num_features[i]], bins = 50)
    plt.title(num_features[i])
    plt.xlabel('Range')
    plt.ylabel('Frequency')


df.drop('id', axis=1).plot.box(vert=False);


X = df.iloc[:, :-1]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
shuffle_split_cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


def dummy_regressor_baseline(strategy: str = "median", constant_val: float = None, quantile_val: float = None) -> pd.Series :
    baseline_model_median = DummyRegressor(
        strategy=strategy, constant=constant_val, quantile=quantile_val
    )

    baseline_median_cv_results = cross_validate(
        baseline_model_median, X_train, y_train, cv=shuffle_split_cv, 
        scoring="neg_root_mean_squared_error", n_jobs=2
    )
    
    return pd.Series(-baseline_median_cv_results["test_score"], name="Dummy regressor error")


baseline_median_cv_results_error = dummy_regressor_baseline(strategy = 'median')
baseline_mean_cv_results_error = dummy_regressor_baseline(strategy = 'mean')
baseline_constant_cv_results_error = dummy_regressor_baseline(strategy = 'constant', constant_val=2)
baseline_quantile_cv_results_error = dummy_regressor_baseline(strategy = 'quantile', quantile_val=0.55)

dummy_error_df = pd.concat([
    baseline_median_cv_results_error, baseline_mean_cv_results_error,
    baseline_constant_cv_results_error, baseline_quantile_cv_results_error
    ], axis=1
)
            
dummy_error_df.columns = ['Median cv', 'Mean cv', 'Constant cv', 'Quantile cv']
dummy_error_df


dummy_error_df.plot.hist(bins=50, density=True, edgecolor="black")
plt.legend(bbox_to_anchor=(1.05, 0.8), loc="upper left")
plt.xlabel("Mean absolute error ($k$)")
_ = plt.title("Distribution of the testing errors")


selected_features = [
    'Episode_Length_minutes', 'Episode_Title', 'Host_Popularity_percentage', 
    'Guest_Popularity_percentage', 'Number_of_Ads', 'Podcast_Name'
]

interactions, n_interactions = [], 4
for r in range(2, n_interactions+1): 
    interactions.extend(
        list(combinations(selected_features, r))
    )



encoded_columns, interaction_columns = [], []
for comb in tqdm(interactions):
    name = '_'.join(comb)
    interaction_columns.append(name)
        
    if len(comb) == 2:
        df[name] = df[comb[0]].astype(str) + '_' + df[comb[1]].astype(str)
        df_test[name] = df_test[comb[0]].astype(str) + '_' + df_test[comb[1]].astype(str)
        
    elif len(comb) == 3:
        df[name] = (df[comb[0]].astype(str) + '_' +
                       df[comb[1]].astype(str) + '_' +
                       df[comb[2]].astype(str))
        df_test[name] = (df_test[comb[0]].astype(str) + '_' +
                      df_test[comb[1]].astype(str) + '_' +
                      df_test[comb[2]].astype(str))
        
    elif len(comb) == 4:
        df[name] = (df[comb[0]].astype(str) + '_' +
                       df[comb[1]].astype(str) + '_' +
                       df[comb[2]].astype(str) + '_' +
                       df[comb[3]].astype(str))
        df_test[name] = (df_test[comb[0]].astype(str) + '_' +
                      df_test[comb[1]].astype(str) + '_' +
                      df_test[comb[2]].astype(str) + '_' +
                      df_test[comb[3]].astype(str))
    
    encoded_columns.append(name)

df[encoded_columns] = df[encoded_columns].astype('category')
df_test[encoded_columns] = df_test[encoded_columns].astype('category')


df_with_null = df.copy()
df_test_with_null = df_test.copy()


def remove_outlier_using_IQR(feature_list: list[str], df: pd.DataFrame) -> pd.DataFrame: 
    for feature in feature_list : 
        Q1, Q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[feature] >= Q1 - 1.5 * IQR) & (df[feature] <= Q3 + 1.5 * IQR) | df.Episode_Length_minutes.isnull()]
    return df 

df = remove_outlier_using_IQR(['Number_of_Ads', 'Episode_Length_minutes'], df)
df.drop('id', axis=1).plot.box(vert = False)
print(f'{round(((num_samples - df.shape[0])/num_samples)*100, 3)} % of data removed, number of samples removed {num_samples - df.shape[0]}')


plt.figure(figsize=(18, 12), facecolor='w')
df.Episode_Title.value_counts().plot(kind= 'bar');


df['Episode_Title'] = df['Episode_Title'].str.split().str[-1].astype(int)
df_test['Episode_Title'] = df_test['Episode_Title'].str.split().str[-1].astype(int)
df['Episode_Title']


df_with_null['Episode_Title'] = df_with_null['Episode_Title'].str.split().str[-1].astype(int)
df_test_with_null['Episode_Title'] = df_test_with_null['Episode_Title'].str.split().str[-1].astype(int)


df.Genre.value_counts().plot(kind= 'bar');


sns.violinplot(x='Genre', y='Listening_Time_minutes', data=df)
plt.xticks(rotation=90)
plt.show()


plt.scatter(x = df['Host_Popularity_percentage'], y = df['Listening_Time_minutes'])
plt.xlabel('Host_Popularity_percentage')
plt.ylabel('Listening_Time_minutes')
plt.show();


df['Host_Popularity_percentage'].value_counts().reset_index().sort_values(by = 'Host_Popularity_percentage').head(10)


df['Host_Popularity_percentage'] = df['Host_Popularity_percentage'].clip(lower= 20, upper=100)
df_test['Host_Popularity_percentage'] = df_test['Host_Popularity_percentage'].clip(lower=20, upper=100)
print(f'{((num_samples - df.shape[0])/num_samples)*100} % of data removed')


df_with_null['Host_Popularity_percentage'] = df_with_null['Host_Popularity_percentage'].clip(lower= 20, upper=100)
df_test_with_null['Host_Popularity_percentage'] = df_test_with_null['Host_Popularity_percentage'].clip(lower=20, upper=100)


sns.violinplot(x='Publication_Day', y='Listening_Time_minutes', data=df)
plt.xticks(rotation=90)
plt.show()


df['is_weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
df_test['is_weekend'] = df_test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
df.Publication_Day.value_counts().plot(kind = 'bar');


df_with_null['is_weekend'] = df_with_null['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
df_test_with_null['is_weekend'] = df_test_with_null['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)


sns.violinplot(x='Publication_Time', y='Listening_Time_minutes', data=df)
plt.xticks(rotation=90)
plt.show()


df['is_day'] = df['Publication_Time'].isin(['Afternoon', 'Morning']).astype(int)
df_test['is_day'] = df_test['Publication_Time'].isin(['Afternoon', 'Morning']).astype(int)
df.Publication_Time.value_counts().plot(kind = 'bar');


df_with_null['is_day'] = df_with_null['Publication_Time'].isin(['Afternoon', 'Morning']).astype(int)
df_test_with_null['is_day'] = df_test_with_null['Publication_Time'].isin(['Afternoon', 'Morning']).astype(int)


df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].clip(upper=100)
df_test['Guest_Popularity_percentage'] = df_test['Guest_Popularity_percentage'].clip(upper=100)


df_with_null['Guest_Popularity_percentage'] = df_with_null['Guest_Popularity_percentage'].clip(upper=100)
df_test_with_null['Guest_Popularity_percentage'] = df_test_with_null['Guest_Popularity_percentage'].clip(upper=100)


plt.scatter(x = df['Guest_Popularity_percentage'], y = df['Listening_Time_minutes'])
plt.xlabel('Guest_Popularity_percentage')
plt.ylabel('Listening_Time_minutes')
plt.show();


mask = df['Guest_Popularity_percentage'].isna()
num_missing = mask.sum()
non_null_values = df['Guest_Popularity_percentage'].dropna().values
random_samples = np.random.choice(non_null_values, size=num_missing, replace=True)
df.loc[mask, 'Guest_Popularity_percentage'] = random_samples

plt.hist(df['Guest_Popularity_percentage'], bins = 50);


mask = df_test['Guest_Popularity_percentage'].isna()
num_missing = mask.sum()
non_null_values = df_test['Guest_Popularity_percentage'].dropna().values
random_samples = np.random.choice(non_null_values, size=num_missing, replace=True)
df_test.loc[mask, 'Guest_Popularity_percentage'] = random_samples


plt.scatter(x = df['Episode_Length_minutes'], y = df['Listening_Time_minutes'])
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.show();


df[[
    'Episode_Title', 'Episode_Length_minutes', 'Host_Popularity_percentage', 
    'Number_of_Ads', 'Guest_Popularity_percentage'
]].corr()


sns.pairplot(df[['Episode_Title', 'Episode_Length_minutes', 'Host_Popularity_percentage', 'Number_of_Ads']]);


df_copy = df.copy()
df_copy.dropna(inplace = True)
x = df_copy['Episode_Length_minutes'].values.reshape(-1, 1)
y = df_copy['Listening_Time_minutes'].values

reg = LinearRegression()
reg.fit(x, y)
y_pred = reg.predict(x)
 
residuals = y - y_pred
std_dev = np.std(residuals)
x_normalized = (x - x.min()) / (x.max() - x.min())
 
alpha = 2.5   
upper_threshold = y_pred + std_dev * (1 + alpha * x_normalized.flatten())
lower_threshold = y_pred - std_dev * (1 + alpha*1.5 * x_normalized.flatten())
 
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, label="Data")
plt.plot(x, y_pred, color='black', label='Fitted Line', linewidth=2)
plt.plot(x, upper_threshold, 'r--', label='+2 Std Dev')
plt.plot(x, lower_threshold, 'r--', label='-2 Std Dev')
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.title('Listening Time vs Guest Popularity with Outlier Thresholds')
plt.legend()
plt.grid(True)
plt.show()


print("Coefficient:", reg.coef_[0])
print("Intercept:", reg.intercept_)


inlier_mask = (y >= lower_threshold) & (y <= upper_threshold)
dropped_indexes = df_copy[~inlier_mask].index 
df = df.drop(index=dropped_indexes)
print(f'{round(((num_samples - df.shape[0])/num_samples)*100, 3)} % of data removed, number of samples removed {num_samples - df.shape[0]}')


plt.scatter(x = df['Episode_Length_minutes'], y = df['Listening_Time_minutes'])
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.show();


def fill_with_regline(df) :
    mask_null = df['Episode_Length_minutes'].isnull()
    
    df.loc[
    mask_null, 'Episode_Length_minutes'] = (
        (df.loc[mask_null, 'Listening_Time_minutes'] - reg.intercept_)/reg.coef_[0]
    ).clip(lower=0, upper=120)
    return df 


episode_time_mapper = dict()
for i in tqdm(range(len(df))) :   
    if df.Podcast_Name.iloc[i] not in episode_time_mapper : 
        episode_time_mapper[df.Podcast_Name.iloc[i]] = dict()
    if df.Episode_Title.iloc[i] not in episode_time_mapper[df.Podcast_Name.iloc[i]] : 
        episode_time_mapper[df.Podcast_Name.iloc[i]][df.Episode_Title.iloc[i]] = set()
    if pd.notnull(df.Episode_Length_minutes.iloc[i]) :
        episode_time_mapper[df.Podcast_Name.iloc[i]][df.Episode_Title.iloc[i]].add(df.Episode_Length_minutes.iloc[i])


podcast_sample = random.sample(list(episode_time_mapper.keys()), k = 5)
for podcast in podcast_sample :  
    title_sample = random.sample(list(episode_time_mapper[podcast].keys()), k= 3)  
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ['salmon', 'skyblue', 'lightgreen']
    for i in range(3) : 
        axes[i].hist(
            list(episode_time_mapper[podcast][title_sample[i]]), bins=5, 
            color=colors[i], edgecolor='black'
        )
        axes[i].set_title(f'Histogram of Episode {title_sample[i]} : {podcast}')
        axes[i].set_xlabel('Episode Length minutes')
        axes[i].set_ylabel('Count')
        
    plt.tight_layout()
    plt.show()


df_copy = df.copy()
def fill_with_values_dict() : 
    for i in tqdm(range(len(df_copy))) : 
        if not pd.notnull(df_copy['Episode_Length_minutes'].iloc[i]) : 
            podcast, title = df_copy['Podcast_Name'].iloc[i], df_copy['Episode_Title'].iloc[i]
            df_copy['Episode_Length_minutes'].iloc[i] = np.median(
                list(episode_time_mapper[podcast][title])
            ) 
    return df_copy

df_copy = fill_with_values_dict()


plt.scatter(x = df_copy['Episode_Length_minutes'], y = df_copy['Listening_Time_minutes'])
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.show();


num_features = ['Host_Popularity_percentage', 'Episode_Title']
cat_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
features = num_features + cat_features 
target = 'Episode_Length_minutes'
df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

preprocessor = ColumnTransformer([
    ('title_enc', OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1
    ), cat_features),
], remainder='passthrough')  

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
print(f'Validation MSE: {mse:.2f}')


df_copy = df.copy()
missing_mask = df_copy['Episode_Length_minutes'].isnull() 
df_copy.loc[missing_mask, 'Episode_Length_minutes'] = pipeline.predict(df_copy.loc[missing_mask, features])


plt.scatter(x = df_copy['Episode_Length_minutes'], y = df_copy['Listening_Time_minutes'])
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.show();


num_features = ['Host_Popularity_percentage', 'Episode_Title'] 
cat_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features)
    ]
)

knn_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("imputer", KNNImputer(n_neighbors=3))
])

X = df[num_features + cat_features]
X_imputed = knn_pipeline.fit_transform(X)


df_imputed = df.copy()
df_imputed.loc[df['Episode_Length_minutes'].isnull(), 'Episode_Length_minutes'] = \
    X_imputed[df['Episode_Length_minutes'].isnull(), 0]



plt.scatter(x = df_imputed['Episode_Length_minutes'], y = df_imputed['Listening_Time_minutes'])
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.show();


df_copy = df.copy()


x_col = 'Episode_Length_minutes'
y_col = 'Listening_Time_minutes'
 
other_features = [
    'Host_Popularity_percentage', 'Guest_Popularity_percentage', 
    'Number_of_Ads', 'Episode_Title'
]
 
impute_df = df[[x_col, y_col] + other_features].copy()
impute_df = impute_df.astype(float)
 
imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42,
    sample_posterior=True,  
    imputation_order='ascending'
)
 
imputed_array = imputer.fit_transform(impute_df)
 
df_copy[x_col] = df_copy[x_col].fillna(pd.Series(imputed_array[:, 0], index=df.index))


df_copy['Episode_Length_minutes'] = df_copy['Episode_Length_minutes'].clip(lower = 0, upper=120)
imputed_mask = df[x_col].isna()
 
plt.scatter(
    df_copy.loc[~imputed_mask, 'Episode_Length_minutes'],
    df_copy.loc[~imputed_mask, 'Listening_Time_minutes'],
    color='blue', label='Actual', alpha=0.9
)
 
plt.scatter(
    df_copy.loc[imputed_mask, 'Episode_Length_minutes'],
    df_copy.loc[imputed_mask, 'Listening_Time_minutes'],
    color='green', label='Imputed', alpha=0.01
)
 
plt.xlabel('Episode_Length_minutes')
plt.ylabel('Listening_Time_minutes')
plt.title('Actual vs Imputed Episode Length')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


df[x_col] = df[x_col].fillna(pd.Series(imputed_array[:, 0], index=df.index))

# Prepare test data with same order of columns but insert placeholder for y_col
impute_df_test = df_test[[x_col] + other_features].astype(float)
# Insert a dummy y_col with NaNs just to match column structure
impute_df_test.insert(1, y_col, np.nan)

imputed_array_test = imputer.transform(impute_df_test)
df_test[x_col] = df_test[x_col].fillna(pd.Series(imputed_array_test[:, 0], index=df_test.index))


period = 60
df['Episode_Length_sin'] = np.sin(2 * np.pi * df['Episode_Length_minutes'] / period)
df['Episode_Length_cos'] = np.cos(2 * np.pi * df['Episode_Length_minutes'] / period)

df_with_null['Episode_Length_sin'] = np.sin(2 * np.pi * df_with_null['Episode_Length_minutes'] / period)
df_with_null['Episode_Length_cos'] = np.cos(2 * np.pi * df_with_null['Episode_Length_minutes'] / period)

df_test_with_null['Episode_Length_sin'] = np.sin(2 * np.pi * df_test_with_null['Episode_Length_minutes'] / period)
df_test_with_null['Episode_Length_cos'] = np.cos(2 * np.pi * df_test_with_null['Episode_Length_minutes'] / period)

df_test['Episode_Length_sin'] = np.sin(2 * np.pi * df_test['Episode_Length_minutes'] / period)
df_test['Episode_Length_cos'] = np.cos(2 * np.pi * df_test['Episode_Length_minutes'] / period)


df['_sin_Host_Popularity_percentage'] = np.sin(2*np.pi * df['Host_Popularity_percentage'] / 20).astype('float32')
df['_cos_Host_Popularity_percentage'] = np.cos(2*np.pi * df['Host_Popularity_percentage'] / 20).astype('float32')

df_test_with_null['_sin_Host_Popularity_percentage'] = np.sin(2*np.pi * df_test_with_null['Host_Popularity_percentage'] / 20).astype('float32')
df_test_with_null['_cos_Host_Popularity_percentage'] = np.cos(2*np.pi * df_test_with_null['Host_Popularity_percentage'] / 20).astype('float32')

df_with_null['_sin_Host_Popularity_percentage'] = np.sin(2*np.pi * df_with_null['Host_Popularity_percentage'] / 20).astype('float32')
df_with_null['_cos_Host_Popularity_percentage'] = np.cos(2*np.pi * df_with_null['Host_Popularity_percentage'] / 20).astype('float32')

df_test['_sin_Host_Popularity_percentage'] = np.sin(2*np.pi * df_test['Host_Popularity_percentage'] / 20).astype('float32')
df_test['_cos_Host_Popularity_percentage'] = np.cos(2*np.pi * df_test['Host_Popularity_percentage'] / 20).astype('float32')


sns.violinplot(x='Number_of_Ads', y='Listening_Time_minutes', data=df)
plt.xticks(rotation=90)
plt.show()


df.Episode_Sentiment.value_counts().plot(kind='bar');


sns.violinplot(x='Episode_Sentiment', y='Listening_Time_minutes', data=df)
plt.xticks(rotation=90)
plt.show()


num_features = [
    'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 
    'Episode_Title', 'Episode_Length_sin', 'Episode_Length_cos', 'is_weekend', 'is_day', 
    '_sin_Host_Popularity_percentage', '_cos_Host_Popularity_percentage'
]
cat_features = ['Publication_Day', 'Publication_Time', 'Episode_Sentiment']
interaction_features = interaction_columns + ['Podcast_Name', 'Genre']
custom_order = [
    ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
    ['Morning', 'Afternoon', 'Evening', 'Night'],
    ['Negative', 'Neutral', 'Positive']
]

X, y = df.drop(['id', 'Listening_Time_minutes'], axis = 1), df['Listening_Time_minutes']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ("ordinal_encoder", OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, 
        categories = custom_order
    ))
])
interaction_feature_transformer = Pipeline(steps=[
    ("labelencoder", LabelEncoder())
])
 
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])


X_wn, y_wn = df_with_null.drop(['id', 'Listening_Time_minutes'], axis = 1), df_with_null['Listening_Time_minutes']
X_wn_train_full, X_wn_test, y_wn_train_full, y_wn_test = train_test_split(X_wn, y_wn, test_size=0.2, random_state=42)
X_wn_train, X_wn_val, y_wn_train, y_wn_val = train_test_split(X_wn_train_full, y_wn_train_full, test_size=0.2, random_state=42)


lgbm_wn_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),("lgbm_wn_reg", LGBMRegressor(n_jobs=-1))
]) 

lgbm_wn_reg_pipeline.fit(X_wn_train_full, y_wn_train_full)
y_wn_pred = lgbm_wn_reg_pipeline.predict(X_wn_test)
lgbm_wn_reg_test_score = round(pow(mean_squared_error(y_wn_pred, y_wn_test), 0.5), 3)
print(lgbm_wn_reg_test_score)


val_scores = []
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "random_state": 42,
        "n_jobs": -1, 
        "verbosity": -1
    }

    X_wn_train_p, X_wn_val_p = preprocessor.transform(X_wn_train), preprocessor.transform(X_wn_val)
    model = LGBMRegressor(**params)
    
    model.fit(
        X_wn_train_p, y_wn_train,
        eval_set=[(X_wn_val_p, y_wn_val)],
        eval_metric="rmse",
        callbacks=[
            log_evaluation(0), early_stopping(50)
        ]
    )
    
    preds = model.predict(X_wn_val_p)
    rmse = mean_squared_error(y_wn_val, preds, squared=False)
    val_scores.append(rmse)
    return rmse

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print("Best Params:", study.best_params)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_scores)+1), val_scores, marker="o")
plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE per Optuna Trial")
plt.grid(True)
plt.tight_layout()
plt.show()


final_lgbm_wn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("lgbm_wn", LGBMRegressor(**study.best_params, random_state=42))
])

final_lgbm_wn_pipeline.fit(X_wn_train_full, y_wn_train_full)
y_pred = final_lgbm_wn_pipeline.predict(X_wn_test)
lgbm_wn_final_test_score = round(pow(mean_squared_error(y_pred, y_wn_test), 0.5), 3)
print(lgbm_wn_final_test_score)


y_pred = final_lgbm_wn_pipeline.predict(df_test_with_null.drop('id', axis=1))
submission = pd.DataFrame({
    "id": df_test_with_null['id'], "Listening_Time_minutes": y_pred
})
submission.to_csv("submission_wn_lgbm.csv", index=False)
print("submission.csv created âœ…")


def explain_with_shap(pipeline, model_name: str = 'lgbm_wn') :
    trained_lgbm_model = pipeline.named_steps[model_name]
    fitted_preprocessor = pipeline.named_steps['preprocessor']
    
    X_wn_test_transformed = fitted_preprocessor.transform(X_wn_test)
    feature_names = fitted_preprocessor.get_feature_names_out(
        input_features=X_wn_test.columns
    )
    
    sample_indices = np.random.choice(X_wn_test_transformed.shape[0], size=10000, replace=False)
    sample_data = X_wn_test_transformed[sample_indices]
    
    explainer = shap.Explainer(trained_lgbm_model)
    shap_values = explainer(sample_data)
    return shap_values, sample_data, feature_names 

shap_values, sample_data, feature_names = explain_with_shap(
    pipeline = final_lgbm_wn_pipeline, model_name = 'lgbm_wn'
)

shap.summary_plot(shap_values, features=sample_data, feature_names=feature_names)


shap.plots.waterfall(shap_values[0])


shap.plots.force(shap_values[0], matplotlib=True)


catboost_wn_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("catboost_wn_reg", CatBoostRegressor(
        verbose=0, random_state=42, allow_writing_files=False  
    ))
])
 
catboost_wn_reg_pipeline.fit(X_wn_train_full, y_wn_train_full)

y_wn_pred_catboost = catboost_wn_reg_pipeline.predict(X_wn_test)
catboost_wn_reg_test_score = round(
    pow(mean_squared_error(y_wn_test, y_wn_pred_catboost), 0.5), 3
)
print(catboost_wn_reg_test_score)


val_scores = []

def objective(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
 
    boosting_type = (
        trial.suggest_categorical("boosting_type", ["Plain", "Ordered"])
        if grow_policy == "SymmetricTree"
        else "Plain"
    )

    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "grow_policy": grow_policy,
        "boosting_type": boosting_type,
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "verbose": 0
    }

    X_train_p = preprocessor.transform(X_wn_train)
    X_val_p = preprocessor.transform(X_wn_val)

    model = CatBoostRegressor(**params)
    model.fit(
        X_train_p, y_wn_train,
        eval_set=(X_val_p, y_wn_val)
    )

    preds = model.predict(X_val_p)
    rmse = mean_squared_error(y_wn_val, preds, squared=False)
    val_scores.append(rmse)
    return rmse

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print("Best Params:", study.best_params)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_scores) + 1), val_scores, marker="o")
plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE per Optuna Trial (CatBoost)")
plt.grid(True)
plt.tight_layout()
plt.show()


final_catb_wn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("catb_wn", CatBoostRegressor(
        **study.best_params, verbose=0, random_state=42, 
        allow_writing_files=False
    ))
])

final_catb_wn_pipeline.fit(X_wn_train_full, y_wn_train_full)
y_pred = final_catb_wn_pipeline.predict(X_wn_test)
catb_wn_final_test_score = round(pow(mean_squared_error(y_pred, y_wn_test), 0.5), 3)
print(catb_wn_final_test_score)


y_pred = final_catb_wn_pipeline.predict(df_test_with_null.drop('id', axis=1))
submission = pd.DataFrame({
    "id": df_test_with_null['id'], "Listening_Time_minutes": y_pred
})
submission.to_csv("submission_wn_catboost.csv", index=False)
print("submission_wn_catboost.csv created âœ…")


xgb_wn_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("xgb_wn_reg", XGBRegressor(objective='reg:squarederror', random_state=42))
])
 
xgb_wn_reg_pipeline.fit(X_wn_train_full, y_wn_train_full)

y_wn_pred_xgb = xgb_wn_reg_pipeline.predict(X_wn_test)
xgb_wn_reg_test_score = round(
    pow(mean_squared_error(y_wn_test, y_wn_pred_xgb), 0.5), 3
)
print(xgb_wn_reg_test_score)


val_scores = []

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        "verbosity": 0,           
        "random_state": 42,
        "n_jobs": -1
    }

    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    model = XGBRegressor(**params)
    
    model.fit(
        X_train_p, y_train,
        eval_set=[(X_val_p, y_val)],
        eval_metric="rmse",
        verbose=False  
    )
    
    preds = model.predict(X_val_p)
    rmse = mean_squared_error(y_val, preds, squared=False)
    val_scores.append(rmse)
    return rmse

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print("Best Params:", study.best_params)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_scores)+1), val_scores, marker="o")
plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE per Optuna Trial")
plt.grid(True)
plt.tight_layout()
plt.show()


final_xgb_wn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb_wn", XGBRegressor(**study.best_params, random_state=42))
])

final_xgb_wn_pipeline.fit(X_wn_train_full, y_wn_train_full)
y_pred = final_xgb_wn_pipeline.predict(X_wn_test)
xgb_wn_final_test_score = round(pow(mean_squared_error(y_pred, y_wn_test), 0.5), 3)
print(xgb_wn_final_test_score)


estimators = [ 
    ('xgb_wn', final_xgb_wn_pipeline), 
    ('catb_wn', final_catb_wn_pipeline),
    ('lgbm_wn', final_lgbm_wn_pipeline)
]

stacked_reg_wn_model = StackingRegressor(
    estimators=estimators, final_estimator=LGBMRegressor(),
    cv=5, n_jobs=-1, passthrough=False
)

stacked_reg_wn_model.fit(X_wn_train_full, y_wn_train_full)
y_pred = stacked_reg_wn_model.predict(X_wn_test)
stacked_reg_wn_final_test_score = round(pow(mean_squared_error(y_pred, y_wn_test), 0.5), 3)
print(stacked_reg_wn_final_test_score)


estimators = [ 
    ('xgb_wn', final_xgb_wn_pipeline), 
    ('catb_wn', final_catb_wn_pipeline),
    ('lgbm_wn', final_lgbm_wn_pipeline)
]

stacked_reg_wn_model = StackingRegressor(
    estimators=estimators, final_estimator=LGBMRegressor(),
    cv=5, n_jobs=-1, passthrough=False
)

stacked_reg_wn_model.fit(X_wn, y_wn) 


y_pred = stacked_reg_wn_model.predict(df_test_with_null.drop('id', axis=1))
submission = pd.DataFrame({
    "id": df_test_with_null['id'], "Listening_Time_minutes": y_pred
})
submission.to_csv("submission->(LGBM+CATB+XGB)optuna->LGBM.csv", index=False)
print("submission->(LGBM+CATB+XGB)optuna->LGBM.csv created âœ…")





reg_model_creators = {
    "linear_reg":            lambda: LinearRegression(),
    "ridge_reg":             lambda: Ridge(),
    "lasso_reg":             lambda: Lasso(),
    "elastic_net_reg":       lambda: ElasticNet(),
    "decision_tree_reg":     lambda: DecisionTreeRegressor(),
    "random_forest_reg":     lambda: RandomForestRegressor(n_jobs=-1),
    "gradient_boosting_reg": lambda: GradientBoostingRegressor(), 
    "knn_reg":               lambda: KNeighborsRegressor(n_jobs=-1),
    "adaboost_reg":          lambda: AdaBoostRegressor(),
    "bayesian_ridge":        lambda: BayesianRidge(),
    "xgboost_reg":           lambda: XGBRegressor(verbosity=0, n_jobs=-1),
    "lightgbm_reg":          lambda: LGBMRegressor(n_jobs=-1), 
    "catboost_reg":          lambda: CatBoostRegressor(verbose=0, random_state=42, allow_writing_files=False)
}

X_sampled, _, y_sampled, _ = train_test_split(
    X_train, y_train, train_size=0.1, random_state=42
)

result = []
for name in reg_model_creators :
    model = reg_model_creators[name]()
    reg = Pipeline(steps = [("preprocessor", preprocessor),("reg", model)])

    start_time = time.time()
    reg.fit(X_sampled, y_sampled)
    end_time = time.time()
    train_time = round(end_time - start_time, 2)

    start_time = time.time()
    y_val_pred = reg.predict(X_val)
    test_score = round(pow(mean_squared_error(y_val_pred, y_val), 0.5), 3)
    end_time = time.time()
    test_time = round(end_time - start_time, 2)

    print("="*35)
    print(f"{name} passed in {train_time} secs | RMSE --> {test_score}")
    result.append([name, test_score, train_time, test_time])


result_df = pd.DataFrame(result, columns = ['model', 'score', 'train_time', 'test_time'])
result_df.sort_values(by = 'score', inplace = True)
result_df


fig, ax1 = plt.subplots()
 
ax1.plot(result_df['model'], result_df['score'], color='tab:blue', label='Score')
ax1.set_ylabel('Score', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
 
ax1.set_xticks(range(len(result_df['model'])))
ax1.set_xticklabels(result_df['model'], rotation=90)
 
ax2 = ax1.twinx()
ax2.plot(result_df['model'], result_df['train_time'], color='tab:red', label='Train Time')
ax2.set_ylabel('Train Time (s)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Model Score vs Train Time')
plt.tight_layout()
plt.show()


lin_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),("lin_reg", LinearRegression())
]) 

lin_reg_pipeline.fit(X_train_full, y_train_full)
y_pred = lin_reg_pipeline.predict(X_test)
lin_reg_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(lin_reg_test_score)


ridge_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),("ridge_reg", Ridge())
]) 

ridge_reg_pipeline.fit(X_train_full, y_train_full)
y_pred = ridge_reg_pipeline.predict(X_test)
ridge_reg_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(ridge_reg_test_score)


from sklearn.model_selection import cross_val_score
def rmse_cv(model, X, y):
    return -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5).mean()

def objective(trial):
    alpha = trial.suggest_loguniform('ridge_reg__alpha', 1e-4, 1e1)

    ridge_reg_pipeline = Pipeline(steps = [
        ("preprocessor", preprocessor), ("ridge_reg", Ridge(alpha = alpha))
    ])

    score = rmse_cv(ridge_reg_pipeline, X_train, y_train)
    return score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

best_alpha = study.best_params['ridge_reg__alpha']
print(f"Best alpha: {best_alpha}")
print(f"Best RMSE: {round(study.best_value, 3)}")


optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title("Optuna Ridge Alpha Tuning - RMSE") 
plt.show()


ridge_reg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("ridge_reg", Ridge())
])

best_params = study.best_params
ridge_reg_pipeline.set_params(**best_params)

ridge_reg_pipeline.fit(X_train_full, y_train_full)
y_pred = ridge_reg_pipeline.predict(X_test)

ridge_reg_test_score = round(mean_squared_error(y_test, y_pred, squared=False), 3)
print(f"Test RMSE: {ridge_reg_test_score}")


lgbm_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),("lgbm_reg", LGBMRegressor(n_jobs=-1))
]) 

lgbm_reg_pipeline.fit(X_train_full, y_train_full)
y_pred = lgbm_reg_pipeline.predict(X_test)
lgbm_reg_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(lgbm_reg_test_score)


val_scores = []
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "n_jobs": -1, 
        "verbosity": -1
    }

    X_train_p, X_val_p = preprocessor.transform(X_train), preprocessor.transform(X_val)
    model = LGBMRegressor(**params)
    
    model.fit(
        X_train_p, y_train,
        eval_set=[(X_val_p, y_val)],
        eval_metric="rmse"
    )
    
    preds = model.predict(X_val_p)
    rmse = mean_squared_error(y_val, preds, squared=False)
    val_scores.append(rmse)
    return rmse

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print("Best Params:", study.best_params)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_scores)+1), val_scores, marker="o")
plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE per Optuna Trial")
plt.grid(True)
plt.tight_layout()
plt.show()


final_lgbm_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("lgbm", LGBMRegressor(**study.best_params, random_state=42))
])

final_lgbm_pipeline.fit(X_train_full, y_train_full)
y_pred = final_lgbm_pipeline.predict(X_test)
lgbm_final_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(lgbm_final_test_score)


catb_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),("catb_reg", CatBoostRegressor())
]) 

catb_reg_pipeline.fit(X_train_full, y_train_full)
y_pred = catb_reg_pipeline.predict(X_test)
catb_reg_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(catb_reg_test_score)


val_scores = []

def objective(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])
 
    boosting_type = (
        trial.suggest_categorical("boosting_type", ["Plain", "Ordered"])
        if grow_policy == "SymmetricTree"
        else "Plain"
    )

    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "grow_policy": grow_policy,
        "boosting_type": boosting_type,
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "verbose": 0
    }

    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    model = CatBoostRegressor(**params)
    model.fit(
        X_train_p, y_wn_train,
        eval_set=(X_val_p, y_wn_val)
    )

    preds = model.predict(X_val_p)
    rmse = mean_squared_error(y_wn_val, preds, squared=False)
    val_scores.append(rmse)
    return rmse

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print("Best Params:", study.best_params)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_scores) + 1), val_scores, marker="o")
plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE per Optuna Trial (CatBoost)")
plt.grid(True)
plt.tight_layout()
plt.show()


final_catb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("catb", CatBoostRegressor(**study.best_params, random_state=42))
])

final_catb_pipeline.fit(X_train_full, y_train_full)
y_pred = final_catb_pipeline.predict(X_test)
catb_final_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(catb_final_test_score)


xgb_reg_pipeline = Pipeline(steps = [
    ("preprocessor", preprocessor),
    ("xgb_reg", XGBRegressor(objective='reg:squarederror', random_state=42))
])

xgb_reg_pipeline.fit(X_train_full, y_train_full)
y_pred = xgb_reg_pipeline.predict(X_test)
xgb_reg_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(xgb_reg_test_score)


val_scores = []
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0, 10.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        "verbosity": 0,           
        "random_state": 42,
        "n_jobs": -1
    }

    X_train_p, X_val_p = preprocessor.transform(X_train), preprocessor.transform(X_val)
    model = XGBRegressor(**params)
    
    model.fit(
        X_train_p, y_train,
        eval_set=[(X_val_p, y_val)],
        eval_metric="rmse", 
        verbose=False  
    )
    
    preds = model.predict(X_val_p)
    rmse = mean_squared_error(y_val, preds, squared=False)
    val_scores.append(rmse)
    return rmse

study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print("Best Params:", study.best_params)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(val_scores)+1), val_scores, marker="o")
plt.xlabel("Trial")
plt.ylabel("Validation RMSE")
plt.title("Validation RMSE per Optuna Trial")
plt.grid(True)
plt.tight_layout()
plt.show()


final_xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(**study.best_params, random_state=42))
])

final_xgb_pipeline.fit(X_train_full, y_train_full)
y_pred = final_xgb_pipeline.predict(X_test)
xgb_final_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(xgb_final_test_score)


estimators = [ 
    ('xgb', final_xgb_pipeline), 
    ('lin_reg', lin_reg_pipeline),
    ('lgbm', final_lgbm_pipeline),
    ('catb', final_catb_pipeline)
]

stacked_reg_model = StackingRegressor(
    estimators=estimators, final_estimator=LGBMRegressor(),
    cv=5, n_jobs=-1, passthrough=False
)

stacked_reg_model.fit(X_train_full, y_train_full)
y_pred = stacked_reg_model.predict(X_test)
stacked_reg_final_test_score = round(pow(mean_squared_error(y_pred, y_test), 0.5), 3)
print(stacked_reg_final_test_score)


estimators = [
    ('xgb', final_xgb_pipeline), 
    ('lin_reg', lin_reg_pipeline),
    ('lgbm', final_lgbm_pipeline),    
    ('catb', final_catb_pipeline),
]

stacked_reg_model = StackingRegressor(
    estimators=estimators, final_estimator=LGBMRegressor(),
    cv=5, n_jobs=-1, passthrough=False
)

stacked_reg_model.fit(X, y)


y_pred = stacked_reg_model.predict(df_test.drop('id', axis=1))
submission = pd.DataFrame({
    "id": df_test['id'], "Listening_Time_minutes": y_pred
})
submission.to_csv("submission_ln_xgb_lgbm-lgbm.csv", index=False)
print("submission_ln_xgb_lgbm-lgbm.csv created âœ…")


base_preds = np.column_stack([
    model.predict(X_test) for name, model in stacked_reg_model.named_estimators_.items()
])

final_estimator = stacked_reg_model.final_estimator_
explainer = shap.Explainer(stacked_reg_model.final_estimator_)
shap_values = explainer(base_preds)


shap.summary_plot(shap_values, base_preds, feature_names=[name for name, _ in estimators])


def explain_with_shap(model_name: str = 'model') : 
    model_pipeline = stacked_reg_model.named_estimators_[model_name]
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(X_test)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    sample_test_data = X_test_transformed_df.sample(100, random_state=42)
    explainer = shap.Explainer(model_pipeline.named_steps[model_name])
    shap_values = explainer(sample_test_data)

    shap.plots.beeswarm(shap_values)
    shap.plots.waterfall(shap_values[0])
    plt.show() 


explain_with_shap('xgb')


explain_with_shap('lgbm')


submission_wn = pd.read_csv('/kaggle/working/submission->(LGBM+CATB+XGB)optuna->LGBM.csv')
submission = pd.read_csv('/kaggle/working/submission_ln_xgb_lgbm-lgbm.csv')


submission['new'] = (0.997*submission_wn['Listening_Time_minutes']) + (0.003*submission['Listening_Time_minutes'])
submission.drop('Listening_Time_minutes', axis = 1, inplace = True)
submission.rename(columns = {'new' : 'Listening_Time_minutes'}, inplace = True)
submission.to_csv("linear_comb_wn&n.csv", index=False)
print("linear_comb_wn&n.csv created âœ…") 













