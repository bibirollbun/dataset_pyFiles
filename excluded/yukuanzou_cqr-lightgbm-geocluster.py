import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

base_path = Path('/kaggle/input/prediction-interval-competition-ii-house-price')
alpha = 0.1

test_data = pd.read_csv(base_path / 'test.csv')
train_data = pd.read_csv(base_path / 'dataset.csv')


def features_pre_process(dataset):
    dataset = dataset.copy()
    columns_to_drop = ["id", "join_year", "subdivision", "submarket",'city', 'sale_nbr']
    columns_to_category = [
        'month', 'sale_warning', 'join_status', 'zoning', 'present_use', 
        'year_built', 'year_reno', 'grade', 'fbsmt_grade', 'condition', 
        'stories', 'beds', 'bath_full', 'bath_3qtr', 'bath_half', 'garb_sqft', 
        'gara_sqft', 'wfnt','golf', 'greenbelt', 'noise_traffic', 'view_rainier',
        'view_olympics', 'view_cascades', 'view_territorial', 'view_skyline', 'view_sound',
        'view_lakewash', 'view_lakesamm', 'view_otherwater', 'view_other'
    ]
    columns_to_int = [
        'sale_price', 'sale_nbr', 'area', 'land_val', 'imp_val',
        'sqft_lot', 'sqft', 'sqft_1', 'sqft_fbsmt'
    ]
    columns_to_float = ['latitude', 'longitude']
    
    if 'sale_date' in dataset.columns:
        dataset['sale_date'] = pd.to_datetime(dataset['sale_date'], errors='coerce')
        dataset['month'] = dataset['sale_date'].dt.month.astype('category')
        dataset['year'] = dataset['sale_date'].dt.year.astype('category')
    
    if 'zoning' in dataset.columns:
        dataset['zoning'] = dataset['zoning'].apply(
            lambda x: 'NR' if pd.notna(x) and 'NR' in str(x) else 'Non_NR'
        )
    
    if 'year_reno' in dataset.columns:
        dataset['year_reno'] = dataset['year_reno'].apply(
            lambda x: 'N' if pd.notna(x) and x == 0 else 'Y'
        )
    
    cols_to_drop = [col for col in columns_to_drop if col in dataset.columns]
    dataset.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    for col in columns_to_int:
        if col in dataset.columns:
            dataset[col] = dataset[col].astype('int32')
    
    for col in columns_to_category:
        if col in dataset.columns and col != 'month':
            dataset[col] = dataset[col].astype('category')
    
    for col in columns_to_float:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce').astype('float32')

    dataset.drop(columns='sale_date', inplace=True, errors='ignore')
    
    return dataset


train_data_p = features_pre_process(train_data)
test_data_p = features_pre_process(test_data)
# print(train_data_p)
train_data_p.info()


import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def geographic_clustering(data):
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    geo_df = gpd.GeoDataFrame(data, geometry=geometry)
    
    coords = data[['latitude', 'longitude']].values
    
    coords_scaled = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    
    n_clusters = max(5, len(coords) // 5000)
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=1000,
        max_iter=50,
        random_state=42
    )
    
    data['geo_cluster'] = kmeans.fit_predict(coords_scaled)
    
    centroids = kmeans.cluster_centers_ * coords.std(axis=0) + coords.mean(axis=0)
    data['dist_to_centroid_km'] = [
        np.linalg.norm(np.array([lat, lon])* 111)  # 1°≈111km
        for lat, lon in coords - centroids[data['geo_cluster']]
    ]
    print(f"Automatically select the number of clusters: {n_clusters}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.scatter(
        data['longitude'],
        data['latitude'],
        c=data['geo_cluster'],
        cmap='tab20',
        s=1,
        alpha=0.7
    )
    
    plt.scatter(
        centroids[:, 1],
        centroids[:, 0],
        c='red',
        marker='X',
        s=100,
        label='Center'
    )
    
    plt.colorbar(label='label')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('K-means Visualization')
    plt.legend()
    plt.show()
    
    return data


train_with_clu = geographic_clustering(train_data_p)
test_with_clu = geographic_clustering(test_data_p)
# train_with_clu.info()
# test_with_clu.info()


from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(train_with_clu, groups=train_data_p['geo_cluster']))

train_set = train_with_clu.iloc[train_idx]
val_set = train_with_clu.iloc[val_idx]


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from typing import Tuple, Dict

class ConformalizedQuantileRegression:
    def __init__(self, alpha: float = 0.1, seed: int = 42,
                 lower_n_estimators: int = 100, lower_learning_rate: float = 0.1, lower_max_depth: int = 5,
                 upper_n_estimators: int = 100, upper_learning_rate: float = 0.1, upper_max_depth: int = 5,
                 lower_num_leaves: int = 31, lower_subsample: float = 1.0, lower_colsample_bytree: float = 1.0,
                 lower_reg_alpha: float = 0.0, lower_reg_lambda: float = 0.0,
                 upper_num_leaves: int = 31, upper_subsample: float = 1.0, upper_colsample_bytree: float = 1.0,
                 upper_reg_alpha: float = 0.0, upper_reg_lambda: float = 0.0):
        self.alpha = alpha
        self.seed = seed
        self.lower_n_estimators = lower_n_estimators
        self.lower_learning_rate = lower_learning_rate
        self.lower_max_depth = lower_max_depth
        self.upper_n_estimators = upper_n_estimators
        self.upper_learning_rate = upper_learning_rate
        self.upper_max_depth = upper_max_depth
        self.lower_num_leaves = lower_num_leaves
        self.lower_subsample = lower_subsample
        self.lower_colsample_bytree = lower_colsample_bytree
        self.lower_reg_alpha = lower_reg_alpha
        self.lower_reg_lambda = lower_reg_lambda
        self.upper_num_leaves = upper_num_leaves
        self.upper_subsample = upper_subsample
        self.upper_colsample_bytree = upper_colsample_bytree
        self.upper_reg_alpha = upper_reg_alpha
        self.upper_reg_lambda = upper_reg_lambda
        self.lower_model = None  
        self.upper_model = None  
        self.q_hat = None        

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_calib: pd.DataFrame, y_calib: pd.Series) -> None:
        self.lower_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=self.alpha/2,
            random_state=self.seed,
            verbose=-1,
            n_estimators=self.lower_n_estimators,
            learning_rate=self.lower_learning_rate,
            max_depth=self.lower_max_depth,
            num_leaves=self.lower_num_leaves,
            subsample=self.lower_subsample,
            colsample_bytree=self.lower_colsample_bytree,
            reg_alpha=self.lower_reg_alpha,
            reg_lambda=self.lower_reg_lambda
        ).fit(X_train, y_train)
        
        self.upper_model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=1-self.alpha/2,
            random_state=self.seed,
            verbose=-1,
            n_estimators=self.upper_n_estimators,
            learning_rate=self.upper_learning_rate,
            max_depth=self.upper_max_depth,
            num_leaves=self.upper_num_leaves,
            subsample=self.upper_subsample,
            colsample_bytree=self.upper_colsample_bytree,
            reg_alpha=self.upper_reg_alpha,
            reg_lambda=self.upper_reg_lambda
        ).fit(X_train, y_train)

        calib_lower = self.lower_model.predict(X_calib)
        calib_upper = self.upper_model.predict(X_calib)
        E_scores = np.maximum(y_calib - calib_upper, calib_lower - y_calib)
        
        n = len(y_calib)
        self.q_hat = np.quantile(
            E_scores,
            np.ceil((n + 1) * (1 - self.alpha)) / n,
            interpolation='higher'
        )

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        lower = self.lower_model.predict(X) - self.q_hat  
        upper = self.upper_model.predict(X) + self.q_hat  
        return lower, upper

def evaluate_cqr(coverage_target: float,
                lower_n_estimators: int,
                lower_learning_rate: float,
                lower_max_depth: int,
                lower_num_leaves: int,
                lower_subsample: float,
                lower_colsample_bytree: float,
                lower_reg_alpha: float,
                lower_reg_lambda: float,
                upper_n_estimators: int,
                upper_learning_rate: float,
                upper_max_depth: int,
                upper_num_leaves: int,
                upper_subsample: float,
                upper_colsample_bytree: float,
                upper_reg_alpha: float,
                upper_reg_lambda: float,
                train_set: pd.DataFrame,
                val_set: pd.DataFrame,
                target_col: str = 'sale_price') -> Dict[str, float]:
    X_train, X_calib, y_train, y_calib = train_test_split(
        train_set.drop(target_col, axis=1),
        train_set[target_col],
        test_size=0.3,
        random_state=42
    )
    
    cqr = ConformalizedQuantileRegression(
        alpha=coverage_target,
        lower_n_estimators=int(lower_n_estimators),
        lower_learning_rate=lower_learning_rate,
        lower_max_depth=int(lower_max_depth),
        lower_num_leaves=int(lower_num_leaves),
        lower_subsample=lower_subsample,
        lower_colsample_bytree=lower_colsample_bytree,
        lower_reg_alpha=lower_reg_alpha,
        lower_reg_lambda=lower_reg_lambda,
        upper_n_estimators=int(upper_n_estimators),
        upper_learning_rate=upper_learning_rate,
        upper_max_depth=int(upper_max_depth),
        upper_num_leaves=int(upper_num_leaves),
        upper_subsample=upper_subsample,
        upper_colsample_bytree=upper_colsample_bytree,
        upper_reg_alpha=upper_reg_alpha,
        upper_reg_lambda=upper_reg_lambda
    )
    cqr.fit(X_train, y_train, X_calib, y_calib)
    
    X_val = val_set.drop(target_col, axis=1)
    y_val = val_set[target_col].values
    lower, upper = cqr.predict(X_val)
    
    coverage = np.mean((y_val >= lower) & (y_val <= upper))
    avg_width = np.mean(upper - lower)
    
    winkler_scores = []
    for y, l, u in zip(y_val, lower, upper):
        if y < l:
            penalty = (2/coverage_target) * (l - y)
        elif y > u:
            penalty = (2/coverage_target) * (y - u)
        else:
            penalty = 0
        winkler_scores.append((u - l) + penalty)
    
    return {
        'coverage': coverage,
        'avg_width': avg_width,
        'winkler': np.mean(winkler_scores),
        'undercover': np.mean(y_val < lower),
        'overcover': np.mean(y_val > upper),
        'model': cqr
    }

# Define the objective function to minimize
def objective(params):
    alpha, lower_n_estimators, lower_learning_rate, lower_max_depth, lower_num_leaves, lower_subsample, lower_colsample_bytree, lower_reg_alpha, lower_reg_lambda, \
    upper_n_estimators, upper_learning_rate, upper_max_depth, upper_num_leaves, upper_subsample, upper_colsample_bytree, upper_reg_alpha, upper_reg_lambda = params
    results = evaluate_cqr(
        coverage_target=alpha,
        lower_n_estimators=int(lower_n_estimators),
        lower_learning_rate=lower_learning_rate,
        lower_max_depth=int(lower_max_depth),
        lower_num_leaves=int(lower_num_leaves),
        lower_subsample=lower_subsample,
        lower_colsample_bytree=lower_colsample_bytree,
        lower_reg_alpha=lower_reg_alpha,
        lower_reg_lambda=lower_reg_lambda,
        upper_n_estimators=int(upper_n_estimators),
        upper_learning_rate=upper_learning_rate,
        upper_max_depth=int(upper_max_depth),
        upper_num_leaves=int(upper_num_leaves),
        upper_subsample=upper_subsample,
        upper_colsample_bytree=upper_colsample_bytree,
        upper_reg_alpha=upper_reg_alpha,
        upper_reg_lambda=upper_reg_lambda,
        train_set=train_set,
        val_set=val_set,
        target_col='sale_price'
    )
    print(f"Alpha: {alpha}, "
          f"Lower NE: {lower_n_estimators}, LR: {lower_learning_rate}, MD: {lower_max_depth}, NL: {lower_num_leaves}, SS: {lower_subsample}, CBT: {lower_colsample_bytree}, RA: {lower_reg_alpha}, RL: {lower_reg_lambda}, "
          f"Upper NE: {upper_n_estimators}, LR: {upper_learning_rate}, MD: {upper_max_depth}, NL: {upper_num_leaves}, SS: {upper_subsample}, CBT: {upper_colsample_bytree}, RA: {upper_reg_alpha}, RL: {upper_reg_lambda}, "
          f"Winkler Score: {results['winkler']}")
    return results['winkler']

# Define the search space
search_space = [
    Real(0.1, 0.4, name='alpha'),
    Integer(50, 200, name='lower_n_estimators'),
    Real(0.01, 0.1, name='lower_learning_rate'),
    Integer(3, 10, name='lower_max_depth'),
    Integer(10, 100, name='lower_num_leaves'),
    Real(0.6, 1.0, name='lower_subsample'),
    Real(0.6, 1.0, name='lower_colsample_bytree'),
    Real(0, 1.0, name='lower_reg_alpha'),
    Real(0, 1.0, name='lower_reg_lambda'),
    Integer(50, 200, name='upper_n_estimators'),
    Real(0.01, 0.1, name='upper_learning_rate'),
    Integer(3, 10, name='upper_max_depth'),
    Integer(10, 100, name='upper_num_leaves'),
    Real(0.6, 1.0, name='upper_subsample'),
    Real(0.6, 1.0, name='upper_colsample_bytree'),
    Real(0, 1.0, name='upper_reg_alpha'),
    Real(0, 1.0, name='upper_reg_lambda')
]

# Perform Bayesian optimization
result = gp_minimize(objective, search_space, n_calls=20, random_state=42, verbose=True)

print(f"Best Winkler Score: {result.fun}")
print(f"Best Parameters: {result.x}")


# Use the best parameters to evaluate the model
best_params = result.x
best_alpha = best_params[0]
best_lower_n_estimators = int(best_params[1])
best_lower_learning_rate = best_params[2]
best_lower_max_depth = int(best_params[3])
best_lower_num_leaves = int(best_params[4])
best_lower_subsample = best_params[5]
best_lower_colsample_bytree = best_params[6]
best_lower_reg_alpha = best_params[7]
best_lower_reg_lambda = best_params[8]
best_upper_n_estimators = int(best_params[9])
best_upper_learning_rate = best_params[10]
best_upper_max_depth = int(best_params[11])
best_upper_num_leaves = int(best_params[12])
best_upper_subsample = best_params[13]
best_upper_colsample_bytree = best_params[14]
best_upper_reg_alpha = best_params[15]
best_upper_reg_lambda = best_params[16]

results = evaluate_cqr(
    coverage_target=best_alpha,
    lower_n_estimators=best_lower_n_estimators,
    lower_learning_rate=best_lower_learning_rate,
    lower_max_depth=best_lower_max_depth,
    lower_num_leaves=best_lower_num_leaves,
    lower_subsample=best_lower_subsample,
    lower_colsample_bytree=best_lower_colsample_bytree,
    lower_reg_alpha=best_lower_reg_alpha,
    lower_reg_lambda=best_lower_reg_lambda,
    upper_n_estimators=best_upper_n_estimators,
    upper_learning_rate=best_upper_learning_rate,
    upper_max_depth=best_upper_max_depth,
    upper_num_leaves=best_upper_num_leaves,
    upper_subsample=best_upper_subsample,
    upper_colsample_bytree=best_upper_colsample_bytree,
    upper_reg_alpha=best_upper_reg_alpha,
    upper_reg_lambda=best_upper_reg_lambda,
    train_set=train_set,
    val_set=val_set,
    target_col='sale_price'
)

print(f"Coverage: {results['coverage']:.3f} (Target: 0.9)")
print(f"Interval Width: {results['avg_width']:.2f}")
print(f"Winkler Score: {results['winkler']:.2f}")

cqr_model = results['model']


def plot_dual_feature_importance(lower_model, upper_model, feature_names, top_n=20):
    lower_imp = pd.Series(lower_model.feature_importances_, index=feature_names)
    upper_imp = pd.Series(upper_model.feature_importances_, index=feature_names)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
    lower_imp.nlargest(top_n).plot.barh(ax=ax1, title='Lower Quantile Features')
    upper_imp.nlargest(top_n).plot.barh(ax=ax2, title='Upper Quantile Features')
    plt.tight_layout()


X_train = train_set.drop('sale_price', axis=1)
plot_dual_feature_importance(cqr_model.lower_model, 
                           cqr_model.upper_model,
                           feature_names=X_train.columns)


lower_pred, upper_pred = cqr_model.predict(test_with_clu)
print(lower_pred, upper_pred)


cross_ratio = np.mean(lower_pred > upper_pred)  # 理想值应为0
print(f"分位数交叉比例: {cross_ratio:.2%}")


submission = pd.DataFrame({
    'id': range(200000, 200000 + 200000),
    'pi_lower': lower_pred,
    'pi_upper': upper_pred
})

submission.to_csv('submission.csv', index=False)
print("Saved successful, let's see first five lines:")
print(submission.head())

