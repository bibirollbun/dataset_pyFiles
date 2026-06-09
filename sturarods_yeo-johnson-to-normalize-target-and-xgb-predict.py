## ğŸ“š Libraries
!pip install pingouin
!pip install statstests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as ww
from ydata_profiling import ProfileReport
from scipy.stats import probplot, yeojohnson, f_oneway, chi2_contingency
from pingouin import rcorr
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from statsmodels.api import OLS
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statstests.tests import shapiro_francia
from statstests.process import stepwise

%matplotlib inline
ww.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## ğŸ“‚ Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


## Missing Values Heatmap
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("TRAIN - MISSING VALUES", weight="bold")
sns.heatmap(data=train.isna())
plt.subplot(1,2,2)
plt.title("TEST - MISSING VALUES", weight="bold")
sns.heatmap(data=test.isna())
plt.tight_layout()


### Variable Distributions
plt.figure(figsize=(12,6))
plt.title("TRAIN SET DISTRIBUTIONS\n", weight="bold")
plt.subplot(1,2,1)
sns.histplot(data=train['Guest_Popularity_percentage'], kde=True, bins=20)
plt.subplot(1,2,2)
sns.histplot(data=train['Episode_Length_minutes'], kde=True, bins=20)
plt.tight_layout()


## ğŸ§¹ Handling Missing Data
train[['Guest_Popularity_percentage', 'Episode_Length_minutes']].fillna(train[['Guest_Popularity_percentage', 'Episode_Length_minutes']].median(), inplace=True)
test[['Guest_Popularity_percentage', 'Episode_Length_minutes']].fillna(train[['Guest_Popularity_percentage', 'Episode_Length_minutes']].median(), inplace=True)
train.dropna(inplace=True)


## ğŸ“� Outlier Removal using IQR Method
# We remove outliers from the target variable `Listening_Time_minutes` using the interquartile range (IQR) technique:
Q1 = train['Listening_Time_minutes'].quantile(0.25)
Q3 = train['Listening_Time_minutes'].quantile(0.75)
IQR = Q3 - Q1
LCI = Q1 - 1.5 * IQR
LCS = Q3 + 1.5 * IQR
train = train[(train['Listening_Time_minutes'] >= LCI) & (train['Listening_Time_minutes'] <= LCS)]


from scipy.stats import yeojohnson
from sklearn.preprocessing import PowerTransformer
transformador_yeojohnson = PowerTransformer(method='yeo-johnson', standardize=False)
train['Listening_Time_minutes_yeojohnson'] = transformador_yeojohnson.fit_transform(train['Listening_Time_minutes'].values.reshape(-1,1))


from statstests.tests import shapiro_francia
shapiro_francia(train['Listening_Time_minutes_yeojohnson'])


### Cleaning `Episode_Title`
train['Episode_Title'].replace('Episode ', '', regex=True, inplace=True)
train['Episode_Title'] = pd.to_numeric(train['Episode_Title'])
test['Episode_Title'].replace('Episode ', '', regex=True, inplace=True)
test['Episode_Title'] = pd.to_numeric(train['Episode_Title'])


from scipy.stats import chi2_contingency
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    k, r = confusion_matrix.shape
    return np.sqrt((chi2 / n) / (min(k - 1, r - 1)))

print("CramÃ©râ€™s V:", cramers_v(train['Genre'], train['Podcast_Name']))


from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Separa os dados de Listening_Time_minutes por sentimento
grupo_negativo = train[train['Episode_Sentiment'] == 'Negative']['Listening_Time_minutes']
grupo_neutro = train[train['Episode_Sentiment'] == 'Neutral']['Listening_Time_minutes']
grupo_positivo = train[train['Episode_Sentiment'] == 'Positive']['Listening_Time_minutes']

# ANOVA test
f_stat, p_valor = f_oneway(grupo_negativo, grupo_neutro, grupo_positivo)

# Tukey's HSD test
tukey_result = pairwise_tukeyhsd(
    endog=train['Listening_Time_minutes'],
    groups=train['Episode_Sentiment'],
    alpha=0.05)

print(tukey_result)


# Plot Tukey Result w/ Confiance Interval
fig = tukey_result.plot_simultaneous(comparison_name=None, ylabel='Episode_Sentiment')
plt.title('Teste de Tukey - Intervalos de ConfianÃ§a das DiferenÃ§as de MÃ©dias')
plt.grid(True)
plt.show()


from statsmodels.api import OLS
from statstests.process import stepwise

train = train[['Episode_Title', 'Episode_Length_minutes', 'Podcast_Name',
       'Genre', 'Host_Popularity_percentage', 'Publication_Day',
       'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads',
       'Episode_Sentiment', 'Listening_Time_minutes_yeojohnson']]
test = test[['Episode_Title', 'Episode_Length_minutes', 'Podcast_Name',
       'Genre', 'Host_Popularity_percentage', 'Publication_Day',
       'Publication_Time', 'Guest_Popularity_percentage', 'Number_of_Ads',
       'Episode_Sentiment']]

train = pd.get_dummies(train, drop_first=True, dtype='int')
test = pd.get_dummies(test, drop_first=True, dtype='int')

train.columns = train.columns.str.strip().str.replace(' ', '_')
train.columns = train.columns.str.strip().str.replace('_&_', '_')
test.columns = test.columns.str.strip().str.replace(' ', '_')
test.columns = test.columns.str.strip().str.replace('_&_', '_')

X = train.drop('Listening_Time_minutes_yeojohnson', axis=1)
y = train['Listening_Time_minutes_yeojohnson']

variaveis = " + ".join(X.columns)
formula = "Listening_Time_minutes_yeojohnson ~ " + variaveis
print(formula)
base_model = OLS.from_formula(formula, data=train).fit()
stepwise_model = stepwise(base_model)


def inverse_yeojohnson(x_trans, lmbda):
    """
    Inverte a transformaÃ§Ã£o Yeo-Johnson.

    ParÃ¢metros:
        x_trans: array-like, valores transformados
        lmbda: float, parÃ¢metro lambda usado na transformaÃ§Ã£o Yeo-Johnson

    Retorna:
        x_original: array-like, valores antes da transformaÃ§Ã£o
    """
    x_trans = np.asarray(x_trans)
    x_original = np.zeros_like(x_trans)

    # Para x >= 0
    pos_mask = x_trans >= 0
    if lmbda != 0:
        x_original[pos_mask] = np.power(x_trans[pos_mask] * lmbda + 1, 1 / lmbda) - 1
    else:
        x_original[pos_mask] = np.exp(x_trans[pos_mask]) - 1

    # Para x < 0
    neg_mask = x_trans < 0
    if lmbda != 2:
        x_original[neg_mask] = 1 - np.power(-(2 - lmbda) * x_trans[neg_mask] + 1, 1 / (2 - lmbda))
    else:
        x_original[neg_mask] = 1 - np.exp(-x_trans[neg_mask])

    return x_original

ypred_submission = stepwise_model.predict(test)
ypred_submission = inverse_yeojohnson(ypred_submission, transformador_yeojohnson.lambdas_)


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold

xgb = XGBRegressor(objective='reg:squarederror', random_state=42)


# RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Scorer to use w/ GridSearchCV
rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Start the Model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# Params Grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# Cruzader Validations
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search w/ RMSE
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=cv,
    verbose=1,
    n_jobs=-1
)

# Training the Model
X = X[['Episode_Title', 'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Podcast_Name_Brain_Boost', 'Podcast_Name_Business_Briefs', 'Podcast_Name_Business_Insights', 'Podcast_Name_Comedy_Corner', 'Podcast_Name_Criminal_Minds', 'Podcast_Name_Digital_Digest', 'Podcast_Name_Educational_Nuggets', 'Podcast_Name_Fashion_Forward', 'Podcast_Name_Global_News', 'Podcast_Name_Humor_Hub', 'Podcast_Name_Innovators', 'Podcast_Name_Joke_Junction', 'Podcast_Name_Laugh_Line', 'Podcast_Name_Learning_Lab', 'Podcast_Name_Lifestyle_Lounge', 'Podcast_Name_Market_Masters', 'Podcast_Name_Melody_Mix', 'Podcast_Name_Money_Matters', 'Podcast_Name_Music_Matters', 'Podcast_Name_Mystery_Matters', 'Podcast_Name_Sound_Waves', 'Podcast_Name_Sport_Spot', 'Podcast_Name_Sports_Central', 'Podcast_Name_Tech_Trends', 'Podcast_Name_World_Watch', 'Genre_Comedy', 'Genre_Education', 'Genre_Lifestyle', 'Genre_Music', 'Genre_News', 'Genre_Sports', 'Genre_Technology', 'Genre_True_Crime', 'Publication_Day_Saturday', 'Publication_Day_Sunday', 'Publication_Day_Tuesday', 'Publication_Day_Wednesday', 'Publication_Time_Evening', 'Publication_Time_Night', 'Episode_Sentiment_Neutral', 'Episode_Sentiment_Positive']]
grid_search.fit(X.iloc[:500,:], y.iloc[:500])

# Results
print(f"Best params: {grid_search.best_params_}")
print(f"Best RMSE (negative): {grid_search.best_score_}")


#%% PREDICT W/ TEST
features_stepwise = X.columns
ypred_xgb = grid_search.predict(test[features_stepwise])
ypred_xgb = inverse_yeojohnson(ypred_xgb, transformador_yeojohnson.lambdas_)

submission = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission['Listening_Time_minutes'] = ypred_xgb
submission[['id', 'Listening_Time_minutes']]
submission.to_csv('/kaggle/working/submisson.csv', index=False)

