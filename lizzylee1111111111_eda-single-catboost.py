import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde 
import matplotlib.cm as cm
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, roc_curve,auc,precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.append("/kaggle/input/useful-eda-package")
from KAGGLE20250401UsefulMethods import plot

train_path = "/kaggle/input/playground-series-s5e4/train.csv"
test_path  = "/kaggle/input/playground-series-s5e4/test.csv"
train_extra=pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")


df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)


df_train = pd.concat([df_train, train_extra], axis=0, ignore_index=True)
df_train.head(2)


df_train = df_train.drop_duplicates()
df_train.shape


def report_missing_values(df, name="DataFrame"):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print(f"{name} has no missing values.\n")
    else:
        print(f"Missing Values Report for {name}:\n")
        for col, count in missing.items():
            print(f"• {col}: {count} missing value(s)")
        print()

# Usage
report_missing_values(df_train, name="Train Set")
report_missing_values(df_test, name="Test Set")



categorical_cols_1 = ['Podcast_Name',"Number_of_Ads","Episode_Title"]
categorical_cols_2 = ['Publication_Day','Publication_Time', 'Episode_Sentiment',"Genre"]
continuous_cols = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Listening_Time_minutes']
target_col = ['Listening_Time_minutes']


def plot_pdf_multiple_features(df, features_per_row=3, linespace=500, bw_method='scott'):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    num_features = df.shape[1]
    num_rows = (num_features + features_per_row - 1) // features_per_row
    fig, axes = plt.subplots(num_rows, features_per_row, figsize=(features_per_row * 4, num_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        feature_data = df[col].dropna().values
        if len(feature_data) == 0:
            ax.set_visible(False)
            continue
        kde = gaussian_kde(feature_data, bw_method=bw_method)
        x = np.linspace(feature_data.min(), feature_data.max(), linespace)
        ax.plot(x, kde(x), color='black', linewidth=0.8, label=col)
        ax.set_title(col)
        ax.legend()

    # Hide any unused subplots
    for j in range(num_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

print("train")
plot_pdf_multiple_features(df_train[continuous_cols], features_per_row=4)



plot.plot_correlation_heatmap(df_train[continuous_cols + ["Number_of_Ads"]])


long_ads = df_train[df_train['Number_of_Ads'] >= 4]
long_ads[["id","Number_of_Ads","Listening_Time_minutes"]].head(20)


columns_to_fill = ['Guest_Popularity_percentage', 'Episode_Length_minutes']

def fill_and_mark_missing(df, columns):
    for col in columns:
        df[col + '_missing'] = df[col].isna().astype(int)
        df[col].fillna(0, inplace=True)
    return df

df_train_copy = df_train.copy()
df_test_copy = df_test.copy()

df_train_copy = fill_and_mark_missing(df_train_copy, columns_to_fill)
df_test_copy = fill_and_mark_missing(df_test_copy, columns_to_fill)
df_train_copy.dropna(inplace=True)

report_missing_values(df_train_copy, name="Train Set")
report_missing_values(df_test_copy, name="Test Set")

df_test_copy.head(2)


for df in [df_train_copy,df_test_copy]:
    df['Episode_Number'] = df['Episode_Title'].str.extract(r'(\d+)').astype(float)
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1e-5)
    df['Genre_Sentiment'] = df['Genre'] + "_" + df['Episode_Sentiment']


from sklearn.preprocessing import LabelEncoder

df_train_copy.loc[df_train_copy['Number_of_Ads'] > 4, 'Number_of_Ads'] = 0
df_test_copy.loc[df_test_copy['Number_of_Ads'] > 4, 'Number_of_Ads'] = 0

cat_cols = df_train_copy.select_dtypes(include='object').columns.tolist()

for col in cat_cols:
    
    le = LabelEncoder()
    all_values = pd.concat([df_train_copy[col], df_test_copy[col]], axis=0)
    le.fit(all_values)
    
    df_train_copy[col] = le.transform(df_train_copy[col])
    df_test_copy[col] = le.transform(df_test_copy[col])


from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

TARGET = ['Listening_Time_minutes']

cat_features = [
    col for col in df_train_copy.select_dtypes(include='int').columns
    if col not in ['id'] and col not in target_col
]

for col in cat_features:
    df_train_copy[col] =  df_train_copy[col].astype(str)
    df_test_copy[col] =  df_test_copy[col].astype(str)

FEATURES = [
    col for col in df_train_copy.columns
    if col not in ['id'] + target_col
]

report_missing_values(df_train_copy, name="Train Set")
report_missing_values(df_test_copy, name="Test Set")


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(len(df_train_copy))
pred_cat = np.zeros(len(df_test_copy))

for i, (train_index, val_index) in enumerate(kf.split(df_train_copy)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)
    
    X_tr = df_train_copy.iloc[train_index][FEATURES]
    y_tr = df_train_copy.iloc[train_index][TARGET[0]]
    X_val = df_train_copy.iloc[val_index][FEATURES]
    y_val = df_train_copy.iloc[val_index][TARGET[0]]
    X_test = df_test_copy[FEATURES]  # 这不需要每轮 copy，但保留也没错

    model_cat = CatBoostRegressor(
        loss_function='MAE',
        iterations=2000,
        learning_rate=0.08777,
        depth=8,
        l2_leaf_reg=0.12596,
        bootstrap_type='Bayesian',
        random_strength=4.27e-08,
        bagging_temperature=0.35995,
        od_type='Iter',
        od_wait=39,
        verbose=200,
        allow_writing_files=False,
        task_type='GPU',  # 可选：若没有 GPU 可删去
        cat_features=cat_features,
        random_seed=42
    )
    
    model_cat.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=500,
        verbose=200
    )
    
    oof_cat[val_index] = model_cat.predict(X_val)
    pred_cat += model_cat.predict(X_test)

pred_cat /= FOLDS

final_rmse = mean_squared_error(df_train_copy[TARGET[0]], oof_cat, squared=False)
print(f"\n✅ Final OOF RMSE: {final_rmse:.4f}")



#investigate bad samples
oof_cat_series = pd.Series(oof_cat, index=df_train_copy.index)
errors = oof_cat_series - df_train_copy['Listening_Time_minutes']

bad_indices = errors[np.abs(errors) > 30].index

bad_samples = df_train_copy.loc[bad_indices].copy()
bad_samples['Prediction'] = oof_cat_series.loc[bad_indices]
bad_samples['Error'] = errors.loc[bad_indices]

bad_samples.head(3)




len_bad_all = len(bad_samples[bad_samples["Episode_Length_minutes"] == 0.00])
print (len_bad_all)
print(len_bad_all / 87093)

col for col in bad_samples.select_dtypes(include='int').columns


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

sub.Listening_Time_minutes =pred_cat 
#sub.loc[sub['id'] == 787939, 'Listening_Time_minutes'] = 89.12

sub.to_csv("submission.csv", index=False)

print("Sub shape:", sub.shape)
sub.head()

