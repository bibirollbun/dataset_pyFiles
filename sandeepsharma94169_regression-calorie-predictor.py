import warnings 
warnings.filterwarnings('ignore')


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import KFold 
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error
from IPython.display import display


train = pd.read_csv(r"/kaggle/input/playground-series-s5e5/train.csv").drop('id',axis=1)
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv').drop('id',axis=1)
original = pd.read_csv(r'/kaggle/input/orginal-calories-data/calories.csv').drop('User_ID',axis=1)



train.head()


from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df, name="Dataset", show_plots=False, cat_limit=10):
    """
    A reusable EDA function for tabular datasets.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - name (str): Name to display
    - show_plots (bool): Whether to plot histograms and bar charts
    - cat_limit (int): Max unique values in categorical columns to show value counts
    
    Returns:
    - None (just displays info)
    """
    data = df.copy()
    
    print(f"\n\n{'='*15} EDA Report for {name} {'='*15}\n")

    # Shape and columns
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}\n")

    # Dtypes
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = data.select_dtypes(include='object').columns.tolist()

    print(f"Numerical Columns: {num_cols}")
    print(f"Categorical Columns: {cat_cols}\n")

    # Missing values
    missing = data.isnull().sum()
    if missing.any():
        print("Missing Values:")
        display(missing[missing > 0])
    else:
        print("No missing values.\n")

    # Unique values
    print("Unique Value Count:")
    display(pd.DataFrame(data.nunique()).T)

    # Summary stats
    print("\nSummary Statistics:")
    display(data.describe().T.style.background_gradient(cmap='Blues').format(precision=2))

    # Correlation matrix
    if len(num_cols) > 1:
        print("\nCorrelation Matrix:")
        corr = data[num_cols].corr()
        display(corr.style.background_gradient(cmap='Blues').format(precision=2))
    
    # Categorical column value counts
    print("\nCategorical Value Counts (if < cat_limit):")
    for col in cat_cols:
        if data[col].nunique() <= cat_limit:
            print(f"\n▶ {col}:")
            display(data[col].value_counts(dropna=False).to_frame().rename(columns={col: "Count"}))

    # Optional: Plots
    if show_plots:
        print("\nPlotting distributions for numerical columns...")
        data[num_cols].hist(bins=30, figsize=(15, len(num_cols)*2), layout=(len(num_cols)//3+1, 3))
        plt.tight_layout()
        plt.show()

        print("\nBar plots for small categorical columns...")
        for col in cat_cols:
            if data[col].nunique() <= cat_limit:
                plt.figure(figsize=(6, 3))
                sns.countplot(y=col, data=data, order=data[col].value_counts().index)
                plt.title(f"Value counts: {col}")
                plt.show()
eda(train, name="Train Data", show_plots=True)
eda(test, name="Test Data")
eda(original, name="Original Data", show_plots=False)



train['Sex'].unique()


def feature_engineering(data):
    df = data.copy()
    df['Sex'] = df['Sex'].map({'male':0,'female':'1'})
    df['BMI'] = df['Weight'] / ( (df['Height'] / 100) ** 2 )
    # df.drop('Height',axis=1,inplace = True)
    df['Duration_Heart_Rate'] = df['Duration']*df['Heart_Rate']
    df['Duration_Body_Temp'] = df['Duration']*df['Body_Temp']
    df['Sex'] = df['Sex'].astype('int')
    df['Duration_Age'] = df['Duration']*df['Age']
    df['Heart_body_duration'] = df['Heart_Rate']* df['Body_Temp'] * df['Duration'] 
    df['Age_Duration_Temp'] = df['Age'] * df['Duration'] * df['Body_Temp']
    df['Intensity_Index'] = df['Heart_Rate'] / df['Duration']
    df['Weight_Intensity_Index'] = df['Weight'] * df['Intensity_Index']
    df['Height_Intensity_Index'] = df['Height'] * df['Intensity_Index']
    
    return df 
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)


from itertools import combinations
import pandas as pd

from itertools import combinations
import pandas as pd

def interaction_feat(data, test, cancat_cols, agg_cols, stats=['mean']):
    df = data.copy()
    df2 = test.copy()

    for i in [2, 3]:  # combinations of 2 and 3 columns
        comb = combinations(cancat_cols, r=i)

        for j in comb:
            # Create interaction key
            group_key = '_'.join(j)
            df[group_key] = df[list(j)].astype(str).agg('_'.join, axis=1)
            df2[group_key] = df2[list(j)].astype(str).agg('_'.join, axis=1)

            # Fix: align categories between train and test
            df[group_key] = df[group_key].astype('category')
            df2[group_key] = pd.Categorical(df2[group_key], categories=df[group_key].cat.categories)

            # Group by interaction and aggregate
            agg_df = df.groupby(group_key)[agg_cols].agg(stats)

            # Flatten column names
            agg_df.columns = [f"{group_key}_{col}_{stat}" for col, stat in agg_df.columns]
            agg_df = agg_df.reset_index()

            # Merge aggregated features back to both train and test
            df = pd.merge(df, agg_df, on=group_key, how='left')
            df2 = pd.merge(df2, agg_df, on=group_key, how='left')

    return df, df2

train_fe,test_fe = interaction_feat(
    train_fe,test_fe,
    cancat_cols=['Age', 'Height', 'Weight'],
    agg_cols=['Duration', 'Heart_Rate', 'Body_Temp']
)



len(train_fe.columns),len(test_fe.columns)


len(train_fe.select_dtypes('category').columns)


cols_to_drop = ['Age_Height','Age_Weight','Height_Weight','Age_Height_Weight','Height']
train_fe.drop(cols_to_drop,axis=1,inplace=True)
test_fe.drop(cols_to_drop,axis=1,inplace=True)


target = 'Calories' 
X = train_fe.drop(columns=target,axis=1)
y = train_fe[target]


# kf = KFold(n_splits=5,shuffle=True,random_state=40)
# preds = np.zeros(len(test_fe))
# oof_preds = np.zeros(len(X))
# rmsle_scores = []

# for fold,(train_idx,test_idx) in enumerate(kf.split(X,y),1):
#     x_train,y_train = X.iloc[train_idx],y[train_idx]
#     x_test,y_test = X.iloc[test_idx],y[test_idx]

#     model_lgb = LGBMRegressor(n_estimators = 1200,
#                               max_depth = 14,
#                               enable_categorical=True,
#                               random_state=42, 
#                               loss_function = 'RMSE', 
#                               eval_metric='RMSE',
#                               verbose=-1)
#     model_lgb.fit(x_train,y_train)
    
#     y_pred = model_lgb.predict(x_test) 
#     y_pred = np.maximum(0, y_pred)
#     y_test = np.maximum(0, y_test)
#     fold_rmsle = np.sqrt(mean_squared_log_error(y_test,y_pred))

#     rmsle_scores.append(fold_rmsle)
#     print("=" * 5,f"Fold {fold} training completes","=" * 5)
#     print(f"Rmsle{fold_rmsle:>10.5f}")
#     preds+=model_lgb.predict(test_fe)
#     oof_preds[test_idx] += y_pred
# preds /= kf.n_splits
# print(f"\nAverage RMSLE: {np.mean(rmsle_scores):.4f}")


from catboost import CatBoostRegressor,Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=40)
preds_cat = np.zeros(len(test_fe))
oof_preds_cat = np.zeros(len(X))
rmsle_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    x_train, y_train = X.iloc[train_idx], y[train_idx]
    x_test, y_test = X.iloc[test_idx], y[test_idx]

    train_pool = Pool(data=x_train, label=y_train)
    val_pool = Pool(data=x_test, label=y_test)
    test_pool = Pool(data=test_fe)

    model_cat = CatBoostRegressor(
        iterations=1200,
        depth=14,
        learning_rate=0.03,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0
    )
    model_cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    y_pred = model_cat.predict(x_test)
    y_pred = np.maximum(0, y_pred)
    y_test = np.maximum(0, y_test)

    fold_rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    rmsle_scores.append(fold_rmsle)

    print("=" * 5, f"Fold {fold} CatBoost training completes", "=" * 5)
    print(f"RMSLE: {fold_rmsle:>10.5f}")

    preds_cat += model_cat.predict(test_pool)
    oof_preds_cat[test_idx] += y_pred

preds_cat /= kf.n_splits
print(f"\nAverage RMSLE (CatBoost): {np.mean(rmsle_scores):.4f}")



# from xgboost import XGBRegressor
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_log_error
# import numpy as np

# kf = KFold(n_splits=5, shuffle=True, random_state=40)
# preds_xgb = np.zeros(len(test_fe))
# oof_preds_xgb = np.zeros(len(X))
# rmsle_scores = []

# for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
#     x_train, y_train = X.iloc[train_idx], y[train_idx]
#     x_test, y_test = X.iloc[test_idx], y[test_idx]

#     model_xgb = XGBRegressor(
#         n_estimators=1200,
#         max_depth=14,
#         objective='reg:squarederror',
#         eval_metric='rmse',
#         tree_method='hist',  # or 'auto'
#         random_state=42,
#         verbosity=0
#     )
#     model_xgb.fit(x_train, y_train)

#     y_pred = model_xgb.predict(x_test)
#     y_pred = np.maximum(0, y_pred)
#     y_test = np.maximum(0, y_test)

#     fold_rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
#     rmsle_scores.append(fold_rmsle)

#     print("=" * 5, f"Fold {fold} XGBoost training completes", "=" * 5)
#     print(f"RMSLE: {fold_rmsle:>10.5f}")

#     preds_xgb += model_xgb.predict(test_fe)
#     oof_preds_xgb[test_idx] += y_pred

# preds_xgb /= kf.n_splits
# print(f"\nAverage RMSLE (XGBoost): {np.mean(rmsle_scores):.4f}")



sub_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub_df['Calories'] = preds_cat


sub_df.to_csv('/kaggle/working/calories_result.csv',index=False)




