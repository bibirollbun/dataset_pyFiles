import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")
test=pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/test.csv")


train.shape, test.shape


train.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


df = pd.concat([train, test], sort=False)


def create_date_features(df, date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = (df[date_column].dt.weekday // 4).astype(int)
    df['is_month_start'] =df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    #df[date_column].dt.day_name()
    return df


df.date = pd.to_datetime(df.date)
df = create_date_features(df, "date")


def random_noise(dataframe):# bağımlı değişkenin standart sapması 1.6 olacak şekilde gürültü ekler
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])["sales"].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91,92,170,171,172,173,174,175,176,177,178,179,
                       180,181,182,183,184,185,186,187,188,189,190,
                       350,351,352,352,354,355,356,357,358,359,360,
                       361,362,363,364,365,366,367,368,369,370,538,
                       539,540,541,542,718,719,720,721,722])


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe \
        .groupby(["store", "item"])["sales"] \
        .transform(lambda x: x.shift(1) \
                   .rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(dataframe)
    return dataframe
    


df = roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,
                             450,451,539,540,541,629,630,631,720])


def ewm_features(dataframe, alphas, lags):
    new_columns = {}
    for alpha in alphas:
        for lag in lags:
            new_column_name = f'sales_ewm_alpha_{str(alpha).replace(".", "")}_lag_{lag}'
            new_columns[new_column_name] = dataframe.groupby(["store", "item"])["sales"].transform(
                lambda x: x.shift(lag).ewm(alpha=alpha).mean()
            )
    return pd.concat([dataframe, pd.DataFrame(new_columns)], axis=1)


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720]
df = ewm_features(df, alphas, lags)


df.shape


df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])
df['sales'] = np.log1p(df["sales"].values)


# Training set will include all data before 2017
train = df.loc[(df["date"] < "2017-01-01"), :]

# Validation set will include the first 3 months of 2017
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]


cols = [col for col in train.columns if col not in ["date", "id", "sales", "year"]]

Y_train = train["sales"]
X_train = train[cols]

Y_val = val["sales"]
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Define the model's hyperparameters in a dictionary
lgb_params = {
    'num_leaves': 10,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'max_depth': 5,
    'verbose': 0,
    'num_boost_round': 1000,
    'nthread': -1
}


def smape(preds, target):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    n = len(preds)
    # Mask to handle the case where both prediction and target are zero
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    """
    A wrapper function to use SMAPE as an evaluation metric in LightGBM.
    """
    labels = train_data.get_label()
    # We must convert predictions and labels back from log-scale to original scale
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    # LightGBM feval function requires this return format: (metric_name, metric_value, is_higher_better)
    return 'SMAPE', smape_val, False


# Create LightGBM's special Dataset objects for training and validation
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

# Train the model
model = lgb.train(lgb_params,
                  lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  callbacks=[
                      lgb.early_stopping(stopping_rounds=200),
                      lgb.log_evaluation(period=100)
                  ],
                  feval=lgbm_smape)

# Make predictions on the validation set using the best model
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# Calculate the final SMAPE score on the validation data
smape(np.expm1(y_pred_val), np.expm1(Y_val))


def plot_lgb_importances(model, plot=False, num=10):
    """
    Calculates and optionally plots feature importances from a trained LightGBM model.
    """
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({
        'feature': model.feature_name(),
        'split': model.feature_importance('split'),
        'gain': 100 * gain / gain.sum()
    }).sort_values('gain', ascending=False)

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


# Get the full DataFrame of feature importances
feat_imp = plot_lgb_importances(model, num=264)


# Identify features with zero importance
importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values
len(importance_zero)


# Create a new list of features that excludes the zero-importance ones
imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[imp_feats]


test = df.loc[df.sales.isna()]
X_test = test[imp_feats]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}


# Create the final LightGBM Dataset using all training data
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=imp_feats)

# Train the final model
final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

# Make predictions on the test data
test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)


# Create a new DataFrame for the submission file
submission_df = test.loc[:, ["id", "sales"]]

# Replace the placeholder 'sales' column with our predictions
# We must use np.expm1() to reverse the log-transformation
submission_df['sales'] = np.expm1(test_preds)

# Ensure sales are integers, as we can't sell fractions of items
submission_df['sales'] = np.round(submission_df['sales']).astype(int)

# Display the first few rows of the submission file as a final check
submission_df.head()


# Ensure the 'id' column is of integer type
submission_df['id'] = submission_df.id.astype(int)

# Save the DataFrame to a CSV file
submission_df.to_csv("submission.csv", index=False)




