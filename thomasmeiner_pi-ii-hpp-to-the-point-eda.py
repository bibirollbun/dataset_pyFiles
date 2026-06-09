!pip install bluecast -q

from bluecast.eda.analyse import plot_distribution_by_time
import itertools
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.imputation import mice
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_tree

import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")
test = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/test.csv")

submission = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/sample_submission.csv")


target = 'sale_price'
print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)


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


train.info()


train.nunique()


train = pd.read_csv("/kaggle/input/prediction-interval-competition-ii-house-price/dataset.csv")

train['sale_date'] = pd.to_datetime(train['sale_date'])

plot_distribution_by_time(
    df=train,
    col_to_plot="sale_price",
    date_col="sale_date",
    xlabel="Week",
    ylabel="Feature distribution",
    title="Weekly distribution of the the sale_price (target)",
    freq="W", # use any of: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
)


# numeric columns *excluding* the target itself
num_cols = train.select_dtypes(exclude=['object', 'category']).columns.difference([target])

# vector of correlations feature ↔ target
corr_to_target = train[num_cols].corrwith(train[target])

# turn it into a tidy one-column DataFrame (easier to sort / style)
corr_df = (
    corr_to_target
    .to_frame(name=f'r({target})')      # name the column something meaningful
    .sort_values(by=f'r({target})', ascending=False)
)

# pretty display
display(
    corr_df.style
            .background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
            .format('{:.2f}')
)


num_cols = train.select_dtypes(exclude=['object', 'category']).columns
num_mat = train[num_cols]

corr = num_mat.corr(method='pearson') 
display(corr.style.background_gradient(cmap='viridis').format('{:.2f}'))


num_cols = ["sqft", "grade", "imp_val", "land_val"]
num_mat = train[num_cols]

corr = num_mat.corr(method='pearson') 
display(corr.style.background_gradient(cmap='viridis').format('{:.2f}'))


for grade in train.sort_values("grade")["grade"].unique():
    print(f"Checking corr matrix within grade {grade}")
    temp_df = train.loc[(train["grade"] == grade)]
    num_mat = temp_df[num_cols]

    corr = num_mat.corr(method='pearson') 
    display(corr.style.background_gradient(cmap='viridis').format('{:.2f}'))


#Quantifying the amount of missing data for a variable   
column = "sale_nbr"

min_data_df = train.copy()
min_data_df[column] = np.where(min_data_df[column].isna(), min_data_df[column].min(), 
                             min_data_df[column])

max_data_df = train.copy()
max_data_df[column] = np.where(max_data_df[column].isna(), max_data_df[column].max(), 
                             max_data_df[column])


print(ols(f"{target}~{column}", data=min_data_df).fit().summary())
print(ols(f"{target}~{column}", data=max_data_df).fit().summary())


cat_cols = train.select_dtypes(include='object').columns
cat_cols_with_na = [c for c in cat_cols if train[c].isna().any()]
miss_mat = train[cat_cols_with_na].isna().astype(int) # transform to 0/1 values
miss_mat[target] = train[target].values

corr = miss_mat.corr(method='pearson') 
display(corr.style.background_gradient(cmap='viridis').format('{:.2f}'))

