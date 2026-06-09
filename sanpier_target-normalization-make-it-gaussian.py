!pip install pingouin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.special import inv_boxcox
from scipy.stats import boxcox, norm as stats_norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer


# set path
project_path = "/kaggle/input/prediction-interval-competition-ii-house-price"


# load dataset
df_train = pd.read_csv(project_path + "/dataset.csv")
print(df_train.shape)
df_train.head(1)


def distribution_vs_gaussian(target):
    """ show the distribution of given target column vs gaussian
    """     
    _, axes = plt.subplots(1, 2, figsize=(20, 7))
    sns.histplot(df_train[target], kde=True, stat="density", ax=axes[0])
    mu, std = stats_norm.fit(df_train[target])
    xmin, xmax = axes[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats_norm.pdf(x, mu, std)
    axes[0].plot(x, p, 'k', linewidth=2)
    axes[0].set(title='Distribution of target vs Gaussian')
    pg.qqplot(df_train[target], dist='norm', ax=axes[1])
    axes[1].set(title='QQ-Plot of Target')
        
def check_best_target_transformation(data, target):
    """ check best possible target transformation method for closest-to-gaussian: 
        how close the transformed target to normal (gaussian) distribution by checking R2-score
    """    
    def calculate_vs_gaussian_r2(series):
        """ calculate the r-squared between the given series and a normal distribution.
        """
        # dropna first
        series = series.dropna().sort_values()
        # generate theoretical normal quantiles
        mean, std = series.mean(), series.std()
        theoretical_quantiles = stats_norm.ppf(np.linspace(0.01, 0.99, len(series)), loc=mean, scale=std)
        return round(r2_score(series, theoretical_quantiles), 4)
    # different approaches
    norm = data[target].copy()
    log = np.log1p(data[target])
    sqrt = np.sqrt(data[target])
    cbrt = np.cbrt(data[target])
    box, lam = boxcox(data[target] + 1)
    transformer = PowerTransformer(method='yeo-johnson')
    yeo = transformer.fit_transform(data[target].values.reshape(-1, 1)).flatten()
    # dict of different approaches
    series_dict = {'norm': norm, 'log': log, 'sqrt': sqrt, 'cbrt': cbrt, 'box': box, 'yeo': yeo}
    # calculate r-squared for each
    r2_scores = {method: calculate_vs_gaussian_r2(pd.Series(data)) for method, data in series_dict.items()}
    print(r2_scores)
    # find best methodology
    best_method = max(r2_scores, key=r2_scores.get)
    best_r2 = r2_scores[best_method]
    print(f"The best transformation is '{best_method}' with an R-squared of {best_r2:.4f}")


# distribution of original target
distribution_vs_gaussian("sale_price")


check_best_target_transformation(df_train, "sale_price")


def target_transformation(df, target, method="log"):
    """ do transformation on the target variable. methods can be as follows:
         + log:     logarithmic transformation
         + sqrt:    square root
         + cbrt:    cubic root
         + box:     box-cox transformation
         + yeo:     yeo-johnson transformation
    """   
    new_target = f"{target}_{method}"
    if method == "log":
        df[new_target] = np.log1p(df[target])
    elif method == "sqrt":
        df[new_target] = np.sqrt(df[target])
    elif method == "cbrt":
        df[new_target] = np.cbrt(df[target])
    elif method == "box":
        df[new_target], lam = boxcox(df[target] + 1)
        return df, lam
    elif method == "yeo":
        transformer = PowerTransformer(method='yeo-johnson')
        df[new_target] = transformer.fit_transform(df[target].values.reshape(-1, 1)).flatten()
        return df, transformer
    return df

def inverse_transformation(series, method, inverser=None):
    """ do inverse transformation on the given series. methods can be as follows:
         + log:     logarithmic transformation
         + sqrt:    square root
         + cbrt:    cubic root
         + box:     box-cox transformation
         + yeo:     yeo-johnson transformation
    """   
    if method == "norm":
        return series
    elif method == "log":
        return np.expm1(series)
    elif method == "sqrt":
        return np.square(series)
    elif method == "cbrt":
        return np.power(series, 3)
    elif method == "box":
        return inv_boxcox(series, inverser) - 1
    elif method == "yeo":
        return inverser.inverse_transform(series.reshape(-1, 1)).flatten()


# apply target transformation
df_train, yeo_transformer = target_transformation(df_train, "sale_price", method="yeo")
distribution_vs_gaussian("sale_price_yeo")

