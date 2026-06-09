from scipy.optimize import minimize

import polars as pl

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_pinball_loss, mean_squared_error

from xgboost import XGBRegressor

import warnings
warnings.simplefilter('ignore')


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """Compute the Winkler Interval Score for prediction intervals.

    Args:
        y_true (array-like): True observed values.
        lower (array-like): Lower bounds of prediction intervals.
        upper (array-like): Upper bounds of prediction intervals.
        alpha (float): Significance level (e.g., 0.1 for 90% intervals).
        return_coverage (bool): If True, also return empirical coverage.

    Returns:
        score (float): Mean Winkler Score.
        coverage (float, optional): Proportion of true values within intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


PATH = '../input/prediction-interval-competition-ii-house-price/'

# Đọc và xử lý file train
train = pd.read_csv(PATH + 'dataset.csv', parse_dates=['sale_date'])
train['sale_year'] = train['sale_date'].dt.year

# Đọc và xử lý file test
test = pd.read_csv(PATH + 'test.csv', parse_dates=['sale_date'])
test['sale_year'] = test['sale_date'].dt.year

# Lấy giá trị lớn nhất và nhỏ nhất của cột sale_price
MAX_PRICE = train['sale_price'].max()
MIN_PRICE = train['sale_price'].min()



import pandas as pd
import numpy as np

# Count zoning & keep top 32
zoning_vc = train['zoning'].value_counts().reset_index()
zoning_vc.columns = ['zoning', 'CE_zoning']
top_zoning = zoning_vc['zoning'].tolist()[:32]

# Count subdivision
subdivision_vc = train['subdivision'].value_counts().reset_index()
subdivision_vc.columns = ['subdivision', 'CE_subdivision']

# Cột cần xử lý
columns_to_use = [c for c in test.columns if c != 'id']
positive_cols = ['view_rainier', 'view_olympics', 'view_cascades',
                 'view_territorial', 'view_skyline', 'view_sound',
                 'view_lakewash', 'view_lakesamm', 'view_otherwater',
                 'view_other']

cat_str_cols = ['join_status', 'city', 'zoning', 'submarket']
drop_cols = ['sale_warning', 'sale_date', 'latitude', 'longitude', 'year_reno']



def base_encoder(input_df):
    df = input_df[columns_to_use].copy()

    # Join CE_zoning
    df = df.merge(zoning_vc, how='left', on='zoning')

    # Join CE_subdivision
    df = df.merge(subdivision_vc, how='left', on='subdivision')

    # Thay 0 sqft thành NaN
    df['sqft'] = df['sqft'].replace(0, np.nan)

    # Tính các biến mới
    df['total_living_sqft'] = df['sqft'] + df['sqft_fbsmt']
    df['lot_to_living_ratio'] = df['sqft_lot'] / df['sqft']
    df['fbsmt_to_sqft_ratio'] = df['sqft_fbsmt'] / df['sqft']

    # sale_year và sale_month
    df['sale_year'] = input_df['sale_date'].dt.year
    df['sale_month'] = input_df['sale_date'].dt.month

    # Có cảnh báo không
    df['has_warning'] = (input_df['sale_warning'] != "   ").astype(np.int8)

    # Thay zoning không nằm trong top_zoning bằng 'other'
    df['zoning'] = df['zoning'].apply(lambda x: x if x in top_zoning else 'other')

    # Xác định year_current_state và has_renovated
    df['year_current_state'] = np.where(input_df['year_reno'] == 0,
                                        input_df['year_built'],
                                        input_df['year_reno'])
    df['has_renovated'] = (input_df['year_reno'] != 0).astype(np.int8)

    # Điền giá trị null cho submarket
    df['submarket'] = df['submarket'].fillna('NA')

    # Tổng điểm view tích cực
    df['positive_view'] = input_df[positive_cols].sum(axis=1)

    # Tổng giá trị đất + xây dựng
    df['total_val'] = df['land_val'] + df['imp_val']

    # Tuổi nhà và tuổi trạng thái hiện tại
    df['age'] = df['sale_year'] - input_df['year_built']
    df['age_current_state'] = df['sale_year'] - df['year_current_state']

    # Categorical encoding (dùng dtype 'category')
    for c in cat_str_cols:
        df[c] = df[c].astype('category')

    # Log transform với log1p
    for c in ['land_val', 'imp_val', 'total_val', 'sqft', 'sqft_fbsmt',
              'sqft_lot', 'sqft_1', 'total_living_sqft']:
        df[c] = np.log1p(df[c])

    # Drop các cột không cần
    df = df.drop(columns=drop_cols, errors='ignore')

    return df



x0 = base_encoder(train)
test_x0 = base_encoder(test)



# 1. Nối dữ liệu lại
agg_df = pd.concat([x0, test_x0], axis=0, ignore_index=True)[[
    'subdivision', 'sale_year', 'submarket', 'area',
    'imp_val', 'land_val', 'total_val',
    'grade', 'sqft', 'year_built', 'total_living_sqft'
]]



def aggregation(key):
    grouped = agg_df.groupby(key).agg({
        'total_living_sqft': 'mean',
        'imp_val': 'mean',
        'land_val': 'mean',
        'grade': 'median',
        'sqft': 'mean',
        'year_built': 'mean',
        'total_val': 'mean'
    }).reset_index()

    # Đổi tên cột cho rõ ràng
    grouped.columns = [key,
                       f'agg_{key}_living_sqft',
                       f'agg_{key}_imp_val',
                       f'agg_{key}_land_val',
                       f'agg_{key}_grade',
                       f'agg_{key}_sqft',
                       f'agg_{key}_year_built',
                       f'agg_{key}_total_val']
    
    return grouped



# đặc trưng tổng hợp
subdiv_df = aggregation('subdivision')
submarket_df = aggregation('submarket')
sale_year_df = aggregation('sale_year')
area_df = aggregation('area')



# Merge với train
x0 = x0.merge(subdiv_df, how='left', on='subdivision')
x0 = x0.merge(sale_year_df, how='left', on='sale_year')
x0 = x0.merge(submarket_df, how='left', on='submarket')
x0 = x0.merge(area_df, how='left', on='area')
x0 = x0.drop(columns='subdivision')

# Merge với test
test_x0 = test_x0.merge(subdiv_df, how='left', on='subdivision')
test_x0 = test_x0.merge(sale_year_df, how='left', on='sale_year')
test_x0 = test_x0.merge(submarket_df, how='left', on='submarket')
test_x0 = test_x0.drop(columns='subdivision')



# Lưu tên cột
feature_names = x0.columns.tolist()

# Tìm các cột categorical (dtype = category)
cat_columns = [col for col in x0.columns if pd.api.types.is_categorical_dtype(x0[col])]

# Chuyển category sang int32
for c in cat_columns:
    x0[c] = x0[c].cat.codes.astype('int32')
    test_x0[c] = test_x0[c].cat.codes.astype('int32')



# Gộp lại
allx = pd.concat([x0, test_x0], axis=0, ignore_index=True)

# Tách lại thành train và test
x = allx.iloc[:len(x0)].to_numpy()
test_x = allx.iloc[len(x0):].to_numpy()

# Target y
y = train['sale_price'].to_numpy()



def run_log_xgb(alpha, params, folds, stratify, seed):
    N_FOLDS = folds
    lny = np.log1p(y)
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for i, (train_idx, valid_idx) in enumerate(skf.split(train, stratify)):
        x_train, y_train = x[train_idx], lny[train_idx]
        x_valid, y_valid = x[valid_idx], lny[valid_idx]

        model = XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=alpha,
            n_estimators=10000,
            random_state=seed,
            enable_categorical=True,
            tree_method='gpu_hist',
            early_stopping_rounds=100,
            verbose=False,
            **params
        )

        model.fit(x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    verbose=False)

        oof[valid_idx] = np.expm1(model.predict(x_valid))
        pred += model.predict(test_x) / N_FOLDS

    tot_pinball = mean_pinball_loss(y, oof, alpha=alpha)
    print(f'Pinball Loss={tot_pinball.astype(int):,}')

    return oof, np.expm1(pred)


def winkler_breakdown(y_true, lower, upper, alpha=0.1):
    '''
    Utility function to break down the Winkler Score into its components:
    interval width, lower penalty, and upper penalty.
    '''
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    penalty_lower = np.where(y_true < lower, penalty_lower, 0)
    penalty_upper = np.where(y_true > upper, penalty_upper, 0)
    score += penalty_lower
    score += penalty_upper

    return np.mean(score), np.mean(width), np.mean(penalty_lower), np.mean(penalty_upper)


alphas = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

oofs = []
preds = []
#XGB_params = {'max_depth':6, 'colsample_bytree': 0.7, 'learning_rate': 0.06}

XGB_params = {
    'max_depth': 7,
    'learning_rate': 0.03,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'min_child_weight': 3,
    'gamma': 0.2,
    'lambda': 1.5,
    'alpha': 0.5,
    'n_jobs': -1,
    'verbosity': 0,
    'seed': 42
}

for seed, alpha in enumerate(alphas):
    print(f"{alpha=}")
    oof, pred = run_log_xgb(alpha, XGB_params, 5, train['grade'], seed)
    oofs.append(oof)
    preds.append(pred)
    print()


q_keys = ['q05', 'q10', 'q25', 'q50', 'q75', 'q90', 'q95']
n_q = len(q_keys)

# Tập oof và prediction từ các mô hình tương ứng từng quantile
q_oof = dict(zip(q_keys, oofs))     # oofs: list of arrays
q_pred = dict(zip(q_keys, preds))   # preds: list of arrays

# Gộp lại thành DataFrame
oof_df = pd.DataFrame(q_oof)


oof_lower = np.zeros(len(train))
oof_upper = np.zeros(len(train))
pred_lower = np.zeros(len(test))
pred_upper = np.zeros(len(test))
print(oof_df.shape)
print(oof_df.head())



n_folds = 7
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

for i, (train_idx, valid_idx) in enumerate(skf.split(oof_df, train['sale_year'])):
    train_oof = oof_df.iloc[train_idx]
    valid_oof = oof_df.iloc[valid_idx]
    y_train = y[train_idx]
    y_valid = y[valid_idx]

    # Hàm tính lower & upper
    def compute_bounds(weights, df):
        weights = np.clip(weights, 0, 1)
        w_lower = dict(zip(q_keys, weights[:n_q]))
        w_upper = dict(zip(q_keys, weights[n_q:]))
        lower = sum(w_lower[k] * df[k] for k in q_keys)
        upper = sum(w_upper[k] * df[k] for k in q_keys)
        return lower, upper

    # Mục tiêu tối ưu: Winkler loss
    def objective(weights):
        lower, upper = compute_bounds(weights, train_oof)
        winkler, _, _, _ = winkler_breakdown(y_train, lower, upper, alpha=0.1)
        return winkler

    # Bắt đầu tối ưu
    init_weights = np.ones(n_q * 2) / n_q
    bounds = [(0.0, 1.0)] * (n_q * 2)
    res = minimize(objective, init_weights, method='L-BFGS-B', bounds=bounds)

    # Áp dụng trọng số tối ưu
    opt_weights = np.clip(res.x, 0, 1)
    w_lower = dict(zip(q_keys, opt_weights[:n_q]))
    w_upper = dict(zip(q_keys, opt_weights[n_q:]))

    # Dự đoán valid fold
    oof_lower[valid_idx] = sum(w_lower[k] * valid_oof[k] for k in q_keys)
    oof_upper[valid_idx] = sum(w_upper[k] * valid_oof[k] for k in q_keys)

    # Tính dự đoán trên test set
    pred_lower += sum(w_lower[k] * q_pred[k] for k in q_keys) / n_folds
    pred_upper += sum(w_upper[k] * q_pred[k] for k in q_keys) / n_folds

    # Đánh giá fold
    winkler, _, _, _ = winkler_breakdown(y_valid, oof_lower[valid_idx], oof_upper[valid_idx], alpha=0.1)
    print(f"Fold_{i}: Winkler = {int(winkler):,}")



winkler, width, penalty_lower, penalty_upper = winkler_breakdown(
    y, oof_lower, oof_upper, alpha=0.1
)

print(f"Winkler = {int(winkler):,}")
print(f"\twidth = {int(width):,}")
print(f"\tpenalty_lower = {int(penalty_lower):,}, penalty_upper = {int(penalty_upper):,}")



# clip estimates
pred_lower = np.clip(pred_lower, MIN_PRICE, MAX_PRICE)
pred_upper = np.clip(pred_upper, MIN_PRICE, MAX_PRICE)

# Create DataFrame and save to CSV
submission = pd.DataFrame({
    'id': test['id'],
    'pi_lower': pred_lower,
    'pi_upper': pred_upper
})
submission.to_csv('submission.csv', index=False)



from IPython.display import FileLink
FileLink('submission.csv')



from sklearn.linear_model import Ridge
import pandas as pd

# ===== Train =====
X_train = pd.DataFrame(q_oof)[q_keys]
y_true = y  # hoặc: train['sale_price'].to_numpy()

# Học lower bound (nhỏ hơn q50)
target_lower = (y_true < X_train['q50']).astype(float)
reg_lower = Ridge(alpha=1.0)
reg_lower.fit(X_train, target_lower)

# Học upper bound (lớn hơn q50)
target_upper = (y_true > X_train['q50']).astype(float)
reg_upper = Ridge(alpha=1.0)
reg_upper.fit(X_train, target_upper)

# ===== Áp dụng trên tập train (OOF) =====
w_lower_lr = dict(zip(q_keys, reg_lower.coef_))
w_upper_lr = dict(zip(q_keys, reg_upper.coef_))

oof_lower_lr = sum(w_lower_lr[k] * X_train[k] for k in q_keys)
oof_upper_lr = sum(w_upper_lr[k] * X_train[k] for k in q_keys)

# Tính winkler trên tập train
winkler_lr, width_lr, p_low_lr, p_up_lr = winkler_breakdown(y_true, oof_lower_lr, oof_upper_lr, alpha=0.1)
print(f"[Cách 2] Winkler: {winkler_lr:.2f} | Width: {width_lr:.2f} | Penalty lower: {p_low_lr:.2f}, upper: {p_up_lr:.2f}")

# ===== Áp dụng trên tập test =====
X_test = pd.DataFrame(q_pred)[q_keys]
pred_lower_lr = sum(w_lower_lr[k] * X_test[k] for k in q_keys)
pred_upper_lr = sum(w_upper_lr[k] * X_test[k] for k in q_keys)


from sklearn.ensemble import GradientBoostingRegressor

meta_X = pd.DataFrame(q_oof)[q_keys]
meta_y_lower = (y < meta_X['q50']).astype(int)
meta_y_upper = (y > meta_X['q50']).astype(int)

stack_model_lower = GradientBoostingRegressor()
stack_model_upper = GradientBoostingRegressor()
stack_model_lower.fit(meta_X, meta_y_lower)
stack_model_upper.fit(meta_X, meta_y_upper)

# Predict on train
oof_lower_stack = stack_model_lower.predict(meta_X)
oof_upper_stack = stack_model_upper.predict(meta_X)
winkler, _, _, _ = winkler_breakdown(y, oof_lower_stack, oof_upper_stack, alpha=0.1)
print(f"[Stack] Winkler = {winkler:.2f}")


