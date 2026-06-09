import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')



def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# target summary with cat cols
def target_summary_cat_cols(dataframe,target,categorical_col):
    summary_df = dataframe.groupby(categorical_col)[target].mean().reset_index()
    summary_df.columns = [categorical_col,"TARGET_MEAN"]

    print(summary_df, end="\n\n\n")

    #Create histogram
    plt.figure(figsize=(15,6))
    plt.bar(summary_df[categorical_col].astype(str),summary_df["TARGET_MEAN"], color='skyblue')
    plt.xlabel(categorical_col)
    plt.ylabel(target)
    plt.title(f"{target} mean for {categorical_col}")
    plt.xticks(rotation=45)
    plt.show()
    


train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv', parse_dates=['date'])
test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')

df = pd.concat([train, test], sort=False)





df['date'].min(), df['date'].max()

check_df(df)





df["date"].min(), df["date"].max()



for col in ["store", "item"] : 
    target_summary_cat_cols(df, "sales", col)




df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})



def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_week'] = df.date.dt.dayofweek
    df['day_of_year'] = df.date.dt.dayofyear
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.dayofweek >= 5 
    df["is_month_start"] = df.date.dt.is_month_start.astype(int)
    df["is_month_end"] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)



df.groupby(["store", "item", "month"])["sales"].mean().head(36)


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))





df.sort_values(by=['store','item', 'date'], axis=0, inplace=True)



def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])





def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])


    
                                                      


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)




df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])



df['sales'] = np.log1p(df["sales"].values)




# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False




test.date.max(), test.date.min()


# Train set: Data up to the beginning of 2017 (end of 2016).
train = df.loc[(df["date"] < "2017-01-01"), :]

# Validation set: The first 3 months of 2017.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# Selecting features and target variables
cols = [col for col in train.columns if col not in ['date', 'id', 'sales', 'year']]

Y_train = train['sales']  # Target variable for the training set
X_train = train[cols]     # Feature variables for the training set

Y_val = val['sales']      # Target variable for the validation set
X_val = val[cols]         # Feature variables for the validation set

# Display the shapes of the target and feature variables for both training and validation sets
Y_train.shape, X_train.shape, Y_val.shape, X_val.shape












# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

# Train the model
model = lgb.train(
    lgb_params, 
    lgbtrain,
    valid_sets=[lgbtrain, lgbval],
    callbacks=[lgb.early_stopping(lgb_params['early_stopping_rounds'])]
)

lgb_param = {'num_leaves': 10,
             'learning_rate': 0.02,
             'feature_fraction':0.8,
             'max_depth' : 5,
             'verbose': 0,
             'num_boost_round': 1000,
             'early_stopping_rounds': 200,
             'nthread': -1}
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val,reference=lgbtrain,feature_name=cols)
#Train the model
model= lgb.train(lgb_params,
                 lgbtrain,
                 valid_sets=[lgbtrain,lgbval],
                 callbacks=[lgb.early_stopping(lgb_params['early_stopping_rounds'])])




y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))




def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp


# Plot the top 30 feature importances and display the top 200 features
plot_lgb_importances(model, num=200)
plot_lgb_importances(model, num=30, plot=True)


# Get the feature importances
feat_imp = plot_lgb_importances(model, num=200)


# Identify features with zero importance
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

# Filter out features with zero importance
imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)



train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)
test_preds


submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)
submission_df.to_csv("submission_demand.csv", index=False)




