# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')  # Replace with actual path
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train.drop(columns=['id'], inplace=True)
test_id = test['id']
test.drop(columns=['id'], inplace=True)


train.head()


test.head()


train.describe()


train.duplicated()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, columns):
    """
    Plots distributions:
    - Numeric columns: histogram + boxplot
    - Categorical columns: countplot
    """
    for col in columns:
        plt.figure(figsize=(12, 4))

        if df[col].dtype.name in ['category', 'object']:
            # Categorical: countplot
            sns.countplot(x=df[col], order=df[col].value_counts().index)
            plt.title(f'Countplot of {col}')
            plt.xlabel(col)
        else:
            # Numeric: histogram + boxplot
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Histogram of {col}')
            plt.xlabel(col)

            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)

        plt.tight_layout()
        plt.show()

# Ensure 'Sex' is categorical
train['Sex'] = train['Sex'].astype('category')

columns = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
plot_distributions(train, columns)

# Ensure 'Sex' is categorical
test['Sex'] = test['Sex'].astype('category')

columns = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
plot_distributions(test, columns)



import pandas as pd

def clip_dataset(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """
    Clips specified columns of df to given (low, high) bounds.

    Args:
      df (pd.DataFrame): input DataFrame
      bounds (dict): mapping column â†’ (low, high) tuple
    
    Returns:
      pd.DataFrame: a new DataFrame with values clipped
    """
    df = df.copy()
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df

# Define your physiological bounds
bounds = {
    'Height':     (140, 210),
    'Weight':     (45,  120),
    'Heart_Rate': (60,  130),
    'Body_Temp':  (37,  41),
    'Calories':   (0,   300)
}

# Apply clipping to both
train_clipped = clip_dataset(train, bounds)
bounds = {
    'Height':     (140, 210),
    'Weight':     (45,  120),
    'Heart_Rate': (60,  130),
    'Body_Temp':  (37,  41)
}

# Apply clipping to both

test_clipped  = clip_dataset(test,  bounds)

# Verify
print("Train min/max after clip:\n", train_clipped.describe().loc[['min','max'], bounds.keys()])
print("Test  min/max after clip:\n", test_clipped.describe().loc[['min','max'], bounds.keys()])



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# 2) Your prepare_features function (corrected name)
import numpy as np
import pandas as pd

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Clips & bins Body_Temp
    - Centers & squares Age
    - Adds Temp_Dev & Temp_Cat
    - Adds core, non-linear & interaction features
    - Bins Age & BMI
    """
    df = df.copy()
    
    # --- 1) Categorical & dtypes ---
    if 'Sex' in df:
        df['Sex'] = df['Sex'].astype('category')
    df['Age'] = df['Age'].astype('int8')
    for c in ['Height','Weight','Duration','Heart_Rate','Body_Temp','Calories']:
        if c in df:
            df[c] = df[c].astype('float32')
    
    # --- 2) Clip noisy Body_Temp & derive new temp features ---
    df['Body_Temp'] = df['Body_Temp'].clip(lower=37.5, upper=41.5)
    df['Temp_Dev']  = df['Body_Temp'] - 37.0
    # Temperature category: mild (37.5â€“38.5), mod (38.5â€“39.5), high (39.5â€“41.5)
    df['Temp_Cat'] = pd.cut(
        df['Body_Temp'],
        bins=[37.5, 38.5, 39.5, 41.5],
        labels=['mild','moderate','high']
    ).astype('category')
    
    # --- 3) Core engineered features ---
    df['BMI']             = df['Weight'] / ((df['Height']/100)**2)
    df['Temp_Heart']      = df['Body_Temp'] * df['Heart_Rate']
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['HR_per_min']      = df['Heart_Rate'] / df['Duration']
    df['High_Temp_Flag']  = (df['Body_Temp'] > 40.5).astype('int8')
    
    # --- 4) Non-linear transforms ---
    df['BMI_sq']         = df['BMI'] ** 2
    df['Duration_log']   = np.log1p(df['Duration'])
    df['HR_sq']          = df['Heart_Rate'] ** 2
    df['Age_cen']        = df['Age'] - df['Age'].mean()
    df['Age_sq']         = df['Age_cen'] ** 2
    
    # --- 5) Interaction terms ---
    df['Age_Duration']   = df['Age'] * df['Duration']
    df['Age_HR']         = df['Age'] * df['Heart_Rate']
    df['Weight_Temp']    = df['Weight'] * df['Body_Temp']
    df['Age_TempDev']    = df['Age'] * df['Temp_Dev']
    
    # --- 6) Binned features ---
    df['Age_bin'] = pd.cut(
        df['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=False
    )
    df['BMI_grp'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=False
    )
    
    # --- 7) Oneâ€�hot encode new categoricals ---
    df = pd.get_dummies(df, columns=['Temp_Cat'], drop_first=True)
    
    return df



train_fe = prepare_features(train)
test_fe  = prepare_features(test)

# 3) Safe one-hot for Sex
def safe_one_hot(df, col='Sex'):
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)
    else:
        df[f'{col}_male'] = 0
    return df

train_fe = safe_one_hot(train_fe, 'Sex')
test_fe  = safe_one_hot(test_fe,  'Sex')

# 4) Now define features & proceed
features = [c for c in train_fe.columns if c not in ['id','Calories']]

poly   = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

X          = poly.fit_transform(train_fe[features])
X_scaled   = scaler.fit_transform(X)
y          = np.log1p(train_fe['Calories'])
X_test     = poly.transform(test_fe[features])
X_test_scaled = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Pipeline ready. Features:", features)



# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
# import matplotlib.pyplot as plt


# # 4) Sample to speed up MI estimation
# sample_df = train_fe.sample(n=50000, random_state=42)
# X = sample_df[features]
# y = sample_df['Calories']

# # 5) Compute MI
# mi = mutual_info_regression(X, y, random_state=42)
# mi_df = pd.DataFrame({'Feature': features, 'MI': mi}).set_index('Feature')

# mi_df_sorted = mi_df.sort_values('MI', ascending=False)

# # 6) Plot heatmap
# plt.figure(figsize=(6,8))
# sns.heatmap(mi_df_sorted, annot=True, fmt=".4f", cmap="YlGnBu")
# plt.title("Mutual Information between Features and Calories")
# plt.tight_layout()
# plt.show()



import pandas as pd
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

# 1) Fit on full training set
est = XGBRegressor(**best_params)
est.fit(X_train, y_train)

# 2) Pull importances
imp = pd.Series(est.feature_importances_, index=poly.get_feature_names_out(features))
imp = imp.sort_values(ascending=False)
print("Top 20 importances:\n", imp.head(20))

# 3) Automatically select features above a threshold
sfm = SelectFromModel(est, threshold="mean", prefit=True)
X_selected = sfm.transform(X_train)

# 4) Trainâ€�val split & reâ€�fit
X_tr_sel, X_va_sel, y_tr, y_va = train_test_split(
    X_selected, y_train, test_size=0.2, random_state=42
)
est2 = XGBRegressor(**best_params)
est2.fit(X_tr_sel, y_tr)
print("Reducedâ€�feature CV MSLE:", mean_squared_log_error(
    np.expm1(y_va), np.expm1(est2.predict(X_va_sel))
))



from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold

kf = KFold(3, shuffle=True, random_state=42)
rfe = RFECV(
    estimator=XGBRegressor(**best_params),
    step=0.1,               # remove 10% of features at each iteration
    cv=kf,
    scoring='neg_mean_squared_log_error',
    n_jobs=-1
)
rfe.fit(X_train, y_train)

print("Optimal number of features:", rfe.n_features_)
selected = np.array(poly.get_feature_names_out(features))[rfe.support_]
print("Selected features:", selected)



import pandas as pd
import numpy as np
import warnings, logging
from itertools import product
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer
from xgboost import XGBRegressor

# 1) Updated base features (pre-polynomial)
base_features = [
    'Age','Height','Weight','Duration','Heart_Rate','Body_Temp',
    'Temp_Dev','BMI','Temp_Heart','Duration_per_kg','HR_per_min',
    'High_Temp_Flag','BMI_sq','Duration_log','HR_sq','Age_cen',
    'Age_sq','Age_Duration','Age_HR','Weight_Temp','Age_TempDev',
    'Age_bin','BMI_grp','Temp_Cat_moderate','Temp_Cat_high','Sex_male'
]

# 2) Build poly & scaler
poly   = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

# 3) Transform train & test
X_poly_train = poly.fit_transform(train_fe[base_features])
X_poly_test  = poly.transform   (test_fe [base_features])

# 4) Get poly feature names
poly_names = poly.get_feature_names_out(base_features)

# 5) RFECV-selected 49 feature names from that poly matrix
selected = [
 'Heart_Rate','Temp_Heart','Duration_per_kg','HR_per_min','Sex_male',
 'Age Duration_per_kg','Age HR_sq','Age Weight_Temp','Age Sex_male',
 'Height HR_sq','Height Age_Duration','Height Sex_male','Weight Heart_Rate',
 'Weight HR_sq','Weight Age_Duration','Weight Weight_Temp','Weight Sex_male',
 'Duration Heart_Rate','Duration Temp_Heart','Duration HR_per_min',
 'Duration HR_sq','Duration Age_Duration','Duration Weight_Temp',
 'Duration Age_bin','Duration Sex_male','Heart_Rate Duration_per_kg',
 'Heart_Rate Duration_log','Heart_Rate Age_bin','Heart_Rate Sex_male',
 'Body_Temp Temp_Heart','Body_Temp Duration_per_kg',
 'Temp_Heart Duration_per_kg','Temp_Heart Duration_log',
 'Temp_Heart Age_Duration','Temp_Heart Weight_Temp','Temp_Heart Age_bin',
 'Duration_per_kg HR_sq','Duration_per_kg Age_Duration',
 'Duration_per_kg Age_bin','Duration_per_kg Sex_male',
 'HR_per_min Weight_Temp','Duration_log HR_sq','HR_sq Age_Duration',
 'HR_sq Weight_Temp','Age_Duration Weight_Temp','Age_Duration Age_bin',
 'Age_Duration Sex_male','Weight_Temp Age_bin','Weight_Temp Sex_male'
]

# 6) Find their indices
keep_idx = [i for i,n in enumerate(poly_names) if n in selected]

# 7) Slice & scale
X_train_sel   = X_poly_train[:, keep_idx]
X_test_sel    = X_poly_test [:, keep_idx]
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled  = scaler.transform(X_test_sel)

# 8) Train/val split
y = np.log1p(train_fe['Calories']).values
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=42
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 9) CV sweep on reduced features
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
msle_scorer = make_scorer(
    lambda yt, yp: mean_squared_log_error(np.expm1(yt), np.expm1(yp)),
    greater_is_better=False
)

param_grid = {
    'n_estimators':    [400],
    'max_depth':       [8],
    'learning_rate':   [0.07],
    'subsample':       [1.0],
    'colsample_bytree':[0.6],
    'min_child_weight':[1],
    'gamma':           [0],
    'reg_alpha':       [1],
    'reg_lambda':      [1],
    'tree_method':     ['gpu_hist'],
    'predictor':       ['gpu_predictor']
}

best_msle   = np.inf
best_params = None
best_model  = None

for combo in product(*(param_grid[p] for p in param_grid)):
    params = dict(zip(param_grid.keys(), combo),
                  random_state=42, verbosity=0)
    model = XGBRegressor(**params)
    msle = -cross_val_score(
        model, X_tr, y_tr,
        cv=kf, scoring=msle_scorer, n_jobs=-1
    ).mean()
    print(f"Params {params} â†’ MSLE {msle:.6f}")
    if msle < best_msle:
        best_msle   = msle
        best_params = params
        # refit on all reduced training data
        best_model = XGBRegressor(**params).fit(X_train_scaled, y)

print(f"\nğŸ�† Best reduced-feature model: {best_params} â†’ MSLE {best_msle:.6f}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 10) Final predictions & submission
pred_test = np.clip(
    np.expm1(best_model.predict(X_test_scaled)),
    0, None
)

submission = pd.DataFrame({
    'id': test_id,
    'Calories': pred_test
})
submission.to_csv('xgb_reduced_updated_submission.csv', index=False)
print("Saved xgb_reduced_updated_submission.csv")



# ### import numpy as np
# import pandas as pd
# from itertools import product
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.metrics import mean_squared_log_error, make_scorer
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# import warnings, logging

# # 1) Globally ignore Python warnings (e.g. FutureWarning, UserWarning)
# warnings.filterwarnings('ignore')

# # 2) Silence LightGBMâ€™s internal logger
# logging.getLogger('lightgbm').setLevel(logging.ERROR)

# param_grids = {
#     'xgboost': {
#         'n_estimators': [350], 'max_depth': [8], 'tree_method':['gpu_hist'], 'predictor':['gpu_predictor'], 'learning_rate': [0.07], 'subsample': [1.0], 'colsample_bytree': [0.6], 'min_child_weight': [1], 'gamma': [0], 'reg_alpha': [1], 'reg_lambda': [1]
#     }
# }

# # --- 3) Set up crossâ€�validation and scorer ---
# kf = KFold(n_splits=3, shuffle=True, random_state=42)
# msle_scorer = make_scorer(
#     lambda yt, yp: mean_squared_log_error(np.expm1(yt), np.expm1(yp)),
#     greater_is_better=False
# )

# results = {}            # best MSLE per model
# best_params = {}        # best params per model
# best_estimators = {}    # best fitted model per model
# threshold = 0.1 #0.003697
# # --- 4) Loop over each model and each hyperparam combo ---
# for name, grid in param_grids.items():
#     print(f"\n=== Manual CV for {name.upper()} ===")
#     best_msle   = np.inf
#     best_combo  = None
#     best_model  = None
    
#     for combo in product(*(grid[k] for k in grid)):
#         params = dict(zip(grid.keys(), combo))
#         if name == 'xgboost':
#             model = XGBRegressor(random_state=42, verbosity=0, **params)
#         else:
#             model = LGBMRegressor(random_state=42, **params)

#         # Run cross_val_score
            

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model.fit(X_train, y_train)    
#             scores = cross_val_score(
#                 model, X_train, y_train,
#                 cv=kf,
#                 scoring=msle_scorer,
#                 # early_stopping_rounds=50,
#                 n_jobs=-1
#             )
#         msle = -scores.mean()
#         if msle < threshold:
#             print(f"Params: {params} â†’ CV MSLE: {msle:.6f}")
#             threshold = msle
#         else:
#             print(f"Params: {params}")
#         # check if this is the best so far for this model
#         if msle < best_msle:
#             best_msle = msle
#             best_combo = params
#             # fit on full training set so we have a predictor ready
#             model.fit(X_train, y_train)
#             best_model = model
    
#     # after trying all combos, record the winner
#     results[name] = best_msle
#     best_params[name] = best_combo
#     best_estimators[name] = best_model
    
#     print(f"\n>>> Best for {name}: {best_combo} â†’ MSLE {best_msle:.6f}")



import numpy as np
import pandas as pd
import warnings, logging
from itertools import product
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer
from xgboost import XGBRegressor


# 1) Define your updated base features (preâ€�poly)
base_features = [
    'Age','Height','Weight','Duration','Heart_Rate','Body_Temp',
    'Temp_Dev','BMI','Temp_Heart','Duration_per_kg','HR_per_min',
    'High_Temp_Flag','BMI_sq','Duration_log','HR_sq','Age_cen',
    'Age_sq','Age_Duration','Age_HR','Weight_Temp','Age_TempDev',
    'Age_bin','BMI_grp','Temp_Cat_moderate','Temp_Cat_high','Sex_male'
]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2) Build polynomial features & capture names
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(train_fe[base_features])
X_poly_test  = poly.transform(   test_fe[ base_features])
poly_names   = poly.get_feature_names_out(base_features)

# 3) RFECVâ€�selected 49 names (from your previous step)
selected = [
 'Heart_Rate','Temp_Heart','Duration_per_kg','HR_per_min','Sex_male',
 'Age Duration_per_kg','Age HR_sq','Age Weight_Temp','Age Sex_male',
 'Height HR_sq','Height Age_Duration','Height Sex_male','Weight Heart_Rate',
 'Weight HR_sq','Weight Age_Duration','Weight Weight_Temp','Weight Sex_male',
 'Duration Heart_Rate','Duration Temp_Heart','Duration HR_per_min',
 'Duration HR_sq','Duration Age_Duration','Duration Weight_Temp',
 'Duration Age_bin','Duration Sex_male','Heart_Rate Duration_per_kg',
 'Heart_Rate Duration_log','Heart_Rate Age_bin','Heart_Rate Sex_male',
 'Body_Temp Temp_Heart','Body_Temp Duration_per_kg',
 'Temp_Heart Duration_per_kg','Temp_Heart Duration_log',
 'Temp_Heart Age_Duration','Temp_Heart Weight_Temp','Temp_Heart Age_bin',
 'Duration_per_kg HR_sq','Duration_per_kg Age_Duration',
 'Duration_per_kg Age_bin','Duration_per_kg Sex_male',
 'HR_per_min Weight_Temp','Duration_log HR_sq','HR_sq Age_Duration',
 'HR_sq Weight_Temp','Age_Duration Weight_Temp','Age_Duration Age_bin',
 'Age_Duration Sex_male','Weight_Temp Age_bin','Weight_Temp Sex_male'
]

# 4) Map those names back to indices
keep_idx = [i for i, name in enumerate(poly_names) if name in selected]

# 5) Subset & scale
scaler = StandardScaler()
X_train_sel   = X_poly_train[:, keep_idx]
X_test_sel    = X_poly_test[ :, keep_idx]
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled  = scaler.transform(   X_test_sel)

# 6) Train/val split
y = np.log1p(train_fe['Calories']).values
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=42
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7) CV sweep on reduced features
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)

kf = KFold(n_splits=3, shuffle=True, random_state=42)
msle_scorer = make_scorer(
    lambda yt, yp: mean_squared_log_error(np.expm1(yt), np.expm1(yp)),
    greater_is_better=False
)

param_grid = {
     'n_estimators': [350], 'max_depth': [8], 'tree_method':['gpu_hist'], 'predictor':['gpu_predictor'], 'learning_rate': [0.07], 'subsample': [1.0], 'colsample_bytree': [0.6], 'min_child_weight': [1], 'gamma': [0], 'reg_alpha': [1], 'reg_lambda': [1]
}

best_msle   = np.inf
best_params = None
best_model  = None

print("Running manual CV on reduced 49-feature matrixâ€¦")
for combo in product(*(param_grid[p] for p in param_grid)):
    params = dict(zip(param_grid.keys(), combo), random_state=42, verbosity=0)
    model = XGBRegressor(**params)
    msle = -cross_val_score(
        model, X_tr, y_tr,
        cv=kf, scoring=msle_scorer, n_jobs=-1
    ).mean()
    print(f" â†’ Params: {params}  MSLE: {msle:.6f}")
    if msle < best_msle:
        best_msle   = msle
        best_params = params
        # Refit on ALL reduced training data
        best_model = XGBRegressor(**params).fit(X_train_scaled, y)

print(f"\nğŸ�† Best model: {best_params} â†’ CV MSLE {best_msle:.6f}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8) Final predictions & submission

pred_test = np.clip(np.expm1(best_model.predict(X_test_scaled)), 0, None)

submission = pd.DataFrame({'id': test_id, 'Calories': pred_test})
submission.to_csv('xgb_final_reduced_submission.csv', index=False)
print("Saved â†’ xgb_final_reduced_submission.csv")


