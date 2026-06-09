!pip install scikit-learn==1.6.1  # Cambia il numero di versione a quello desiderato



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# all libraries:
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale, 
    OneHotEncoder, 
    OrdinalEncoder, 
)

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


class FindScaling():
    def __init__(self):
        self.distribution = None
        self.feature_all_transformations = None


    def _normal_test(self, variable):
        
        stat, pval = stats.shapiro(variable)
        return stat, pval

    def scaling_pipe(self, feature):

        if type(feature) == pd.Series:
            feature = feature.values.reshape(-1, 1)
        
        distributions = [
            ("Unscaled data", feature),
            ("Data after standard scaling", StandardScaler().fit_transform(feature)),
            ("Data after min-max scaling", MinMaxScaler().fit_transform(feature)),
            ("Data after max-abs scaling", MaxAbsScaler().fit_transform(feature)),
            ("Data after robust scaling", RobustScaler(quantile_range=(25, 75)).fit_transform(feature)),
            ("Data after power transformation (Yeo-Johnson)", PowerTransformer(method="yeo-johnson").fit_transform(feature)),
            ("Data after power transformation (Box-Cox)", PowerTransformer(method="box-cox").fit_transform(feature)),
            ("Data after quantile transformation (uniform pdf)", QuantileTransformer(output_distribution="uniform", random_state=42).fit_transform(feature)),
            ("Data after quantile transformation (gaussian pdf)", QuantileTransformer(output_distribution="normal", random_state=42).fit_transform(feature)),
        ]
        
        self.distribution = distributions

        dict_dist = {}
        for i, e in distributions:
            dict_dist[i] = e.reshape(1,-1)[0]
        self.feature_all_transformations = pd.DataFrame(dict_dist)

        return distributions


    def ranking_scale(self, feature):

        all_scales = self.scaling_pipe(feature)
        
        # normal test:
        norm_results = {
            'scale_method':[],
            'pvals':[],
            'stats':[],
        }

        for name, val in all_scales:
            t, p = self._normal_test(val)
            norm_results['scale_method'].append(name)
            norm_results['pvals'].append(p)
            norm_results['stats'].append(t)
        
        df_norm = pd.DataFrame(norm_results)

        # kutosis test:
        kurtosis_results = {
            'scale_method':[],
            'kurtosis':[],
        }
        for i in self.feature_all_transformations.columns:
            kurt = stats.kurtosis(self.feature_all_transformations[i], fisher=False)
            kurtosis_results['scale_method'].append(i)
            kurtosis_results['kurtosis'].append(kurt)
        
        df_kurt = pd.DataFrame(kurtosis_results)
        df_kurt['kurt_target'] = 3
        df_kurt['kurt_dist'] = np.abs(df_kurt['kurt_target'] - df_kurt['kurtosis'])

        # merge two statistics:
        result = pd.merge(df_norm, df_kurt, how="inner", on=["scale_method"])

        return result.sort_values(['pvals', 'kurt_dist'], ascending=False)




def target_vs_indip_plot(data, y, indip_feature=None):
    """
    This function creates a pair of plots: a histogram and a boxplot to visualize the distribution of the target variable
    and its relationship with an independent feature.

    Parameters:
        data (DataFrame): The dataframe containing the data.
        y (str): The target variable column name.
        indip_feature (str, optional): The independent feature column name. If not provided, the function will just 
                                       plot the target variable's distribution and boxplot.
    
    Returns:
        None: Displays the plots.
    """

    # Set the figure size and create subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))

    # Plot based on whether an independent feature is provided
    if indip_feature:
        sns.histplot(data=data, x=y, hue=indip_feature, element="poly", palette='Set2', kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {y} by {indip_feature}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(y, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)

        sns.boxplot(data=data, x=indip_feature, y=y, palette='Set2', ax=ax2)
        ax2.set_title(f'{y} vs {indip_feature}', fontsize=14, fontweight='bold')
        ax2.set_xlabel(indip_feature, fontsize=12)
        ax2.set_ylabel(y, fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.6)
    else:
        sns.histplot(data=data, x=y, element="poly", palette='Set2', kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {y}', fontsize=14, fontweight='bold')
        ax1.set_xlabel(y, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)

        sns.boxplot(data=data, y=y, palette='Set2', ax=ax2)
        ax2.set_title(f'Boxplot of {y}', fontsize=14, fontweight='bold')
        ax2.set_xlabel(' ', fontsize=12)
        ax2.set_ylabel(y, fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.6)

    # Adjust spacing between subplots
    plt.tight_layout(pad=3.0)
    
    # Display the plots
    plt.show()


def plot_3d_scatter(x, y, z, color='blue', size=30, alpha=0.8, title='3D Scatter Plot'):
    """
    Creates a 3D scatter plot with customizable styling.

    Parameters:
    - x, y, z: array-like, coordinates of the points.
    - color: color of the points (default: 'blue').
    - size: size of the points (default: 30).
    - alpha: transparency of the points (default: 0.8).
    - title: title of the plot (default: '3D Scatter Plot').
    """
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')  # Defining a 3D axis

    # Scatter plot
    scatter = ax.scatter(x, y, z, c=color, s=size, alpha=alpha, cmap='coolwarm', edgecolors='k')

    # Set labels for axes
    ax.set_xlabel('X Axis', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Axis', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z Axis', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Add color bar for better visual representation
    plt.colorbar(scatter, ax=ax, label='Color scale')

    # Improve layout and show plot
    plt.tight_layout()
    plt.show()


def residual_diagnostic(model, X, y):
    """
    This function returns a 2x2 subplot grid showing residual analysis and distribution of y-hat vs y.

    Parameters:
        model: A trained model.
        X: Matrix of independent features.
        y: 1D array of the target variable.

    Returns:
        None
    """
    y_hat = model.predict(X)
    residuals = y - y_hat

    # Calculate IQR for outliers
    residuals_q1 = residuals.quantile(q=0.25)
    residuals_q3 = residuals.quantile(q=0.75)
    residuals_iqr = residuals_q3 - residuals_q1
    residuals_limit_low = residuals_q1 - (residuals_iqr * 1.5)
    residuals_limit_high = residuals_q3 + (residuals_iqr * 1.5)

    residuals_outlier = np.where(
        residuals <= residuals_limit_low, -1,
        np.where(residuals >= residuals_limit_high, -1, 0)
    )

    residuals_abs = np.abs(residuals)

    # Calculate leverage
    X_ = np.column_stack([np.ones(X.shape[0]), X])  # Adding intercept
    H = X_ @ np.linalg.inv(X_.T @ X_) @ X_.T  # Hat matrix (leverage)
    leverages = np.diag(H)

    s = np.sqrt(np.sum(residuals**2) / (len(y) - X.shape[1] - 1))

    studentized_residuals = residuals / (s * np.sqrt(1 - leverages))

    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 10))

    # ECDF of absolute residuals
    sns.ecdfplot(x=residuals_abs, palette="coolwarm", linewidth=2, ax=ax[0, 0])
    ax[0, 0].set_xlabel("Absolute Error", fontsize=12, fontweight='bold')
    ax[0, 0].set_ylabel("Cumulative Probability", fontsize=12, fontweight='bold')
    ax[0, 0].set_title("Empirical CDF of Absolute Errors", fontsize=14, fontweight='bold', pad=15)
    ax[0, 0].grid(True, linestyle="--", alpha=0.6)

    # Studentized residuals vs predicted values
    sns.scatterplot(x=y_hat, y=studentized_residuals, alpha=0.7, color='blue', ax=ax[0, 1])
    ax[0, 1].axhspan(-4, -3, facecolor='red', alpha=0.3, label='Influential Outliers (residual â‰¤ -3)')
    ax[0, 1].axhspan(3, 4, facecolor='red', alpha=0.3)
    ax[0, 1].axhspan(-3, -2, facecolor='yellow', alpha=0.3, label='Possible Outliers (-3 â‰¤ residual â‰¤ -2)')
    ax[0, 1].axhspan(2, 3, facecolor='yellow', alpha=0.3)
    ax[0, 1].axhline(y=0, color='black', linestyle='--')
    ax[0, 1].set_title("Studentized Residuals vs Predicted Values", fontsize=14, fontweight="bold")
    ax[0, 1].set_xlabel("Predicted Values", fontsize=12)
    ax[0, 1].set_ylabel("Studentized Residuals", fontsize=12)
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    # Residuals vs observed values
    sns.scatterplot(x=y, y=residuals, hue=residuals_outlier, palette='coolwarm', size=abs(residuals), sizes=(20, 200), alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax[1, 0])
    ax[1, 0].set_title("Scatter Plot of Errors", fontsize=14, fontweight='bold', pad=15)
    ax[1, 0].set_xlabel("Observed Values (y)", fontsize=12, fontweight='bold')
    ax[1, 0].set_ylabel("Prediction Error", fontsize=12, fontweight='bold')
    ax[1, 0].legend(title="Anomalous Error", title_fontsize=12, fontsize=10, loc="upper right")
    ax[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    ax[1, 0].grid(True, linestyle="--", alpha=0.6)

    # KDE for actual vs predicted values
    sns.kdeplot(x=y, color='red', label='Actual Value', fill=True, alpha=0.4, linewidth=2, ax=ax[1, 1])
    sns.kdeplot(x=y_hat, color='blue', label='Predicted Value', fill=True, alpha=0.4, linewidth=2, ax=ax[1, 1])
    ax[1, 1].set_title("Distribution of Actual vs Predicted Values", fontsize=14, fontweight='bold', pad=15)
    ax[1, 1].set_xlabel("Value", fontsize=12, fontweight='bold')
    ax[1, 1].set_ylabel("Density", fontsize=12, fontweight='bold')
    ax[1, 1].legend(title="Legend", title_fontsize=12, fontsize=10, loc="upper right")
    ax[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.subplots_adjust(hspace=0.4)
    plt.show()



class NullManager:
    def __init__(self):
        pass

    def check_nulls(self, dataframe):
        """Return True if dataframe have nulls value"""
        cases = dataframe.isnull().value_counts().reset_index()
        
        columns = cases.drop('count', axis=1).columns
        null = False
        null_columns = []

        for row in cases.values:
            if row.__contains__(True):
                null = True
                null_columns = [columns[i] for i in range(len(row) -1) if row[i] == 1]

        return null, null_columns

    def fill_if_null(self, dataset, method='ffill'):

        conferm, null_columns = self.check_nulls(dataset)

        if conferm:
            print('Fixed null value with {} method'.format(method))
            return dataset.fillna(method=method)
        else:
            print('No null valued in your dataset')
            return dataset
        


###############################################################################################################
def day_sequentially_fix(dataset, annual_limit=365, starting_from=1):
    """Fix days sequencial:"""
    n = starting_from-1

    day_fixed = []
    for i in range(0, dataset.shape[0]):
        n += 1
        if n % annual_limit == 0:
            day_fixed.append(n)
            n = 0
        else:
            day_fixed.append(n)
    
    return day_fixed


###############################################################################################################
def get_seasons_from_day(day:np.array):
    """Given a array of integer days the function returns another array with specific season"""

    # Valori corrispondenti alle condizioni
    seasons = np.array(["winter", "spring", "summer", "autumn", "winter"])

    cond = [
        (day >= 1) & (day <= 79),   # winter
        (day >= 80) & (day <= 151), # spring
        (day >= 152) & (day <= 243),# summer
        (day >= 244) & (day <= 334),# autumn
        (day >= 335) & (day <= 365) # winter
    ]

    return np.select(cond, seasons, default='unknown')



###############################################################################################################
def sin_loop(array:np.array, period):
    """Transform a np.array to a sin distribution with given period"""

    return np.sin((2*np.pi * array) / period)



reading_sample_submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"
reading_train_path = "/kaggle/input/playground-series-s5e3/train.csv"
reading_test_path = "/kaggle/input/playground-series-s5e3/test.csv"

model_path = "/kaggle/input/playgroud_s5e3/scikitlearn/default/1/playground-series-s5e3.pkl"


# read dataframes:

# training-set:
df_train = pd.read_csv(reading_train_path)

# test-set:
df_test = pd.read_csv(reading_test_path)

# sample of submission data:
sample_submission = pd.read_csv(reading_sample_submission_path)



# numerical columns list:
numerical_columns = [
    'pressure', 
    'maxtemp', 
    'temparature', 
    'mintemp',
    'dewpoint',
    'humidity', 
    'cloud', 'sunshine',
    'winddirection', 
    'windspeed'
]

# integer columns:
int_columns = ['day']

# taget columns:
target_columns = 'rainfall'

# id columns
id_column = 'id'


### Check and fix null value with ffill method:
METHOD = 'ffill'

# instantiate de object that fix null values:
null_manager = NullManager()

# fix train dataset if it needs:
df_train = null_manager.fill_if_null(df_train, method=METHOD)

# fix test dataset if it needs:
df_test = null_manager.fill_if_null(df_test, method=METHOD)


# check if days are sequentially or not:

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 4))
sns.scatterplot(data=df_train, x='id', y='day', ax=ax1)
ax1.set_title('Sequential days om Training Set:')
ax1.grid()

sns.scatterplot(data=df_test, x='id', y='day', ax=ax2)
ax2.set_title('Sequential days on Test Set:')
ax2.grid()

plt.show()


# fix sequential days in training set:
df_train['day'] = day_sequentially_fix(df_train, annual_limit=365, starting_from=int(df_train['day'][0]))

# fix sequential days in test set:
df_test['day'] = day_sequentially_fix(df_test, annual_limit=365, starting_from=int(df_test['day'][0]))



# check if days are sequentially or not:

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 4))
sns.scatterplot(data=df_train, x='id', y='day', ax=ax1)
ax1.set_title('Sequential days om Training Set:')
ax1.grid()

sns.scatterplot(data=df_test, x='id', y='day', ax=ax2)
ax2.set_title('Sequential days on Test Set:')
ax2.grid()

plt.show()



for columns in numerical_columns:
    target_vs_indip_plot(data=df_train, y=columns, indip_feature='rainfall')


# Creazione del layout con sottoplot
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
fig.suptitle("ðŸ“Š Seasonal Trends of Weather Variables", fontsize=18, fontweight="bold")

# Palette di colori per diversificare le linee
palette = sns.color_palette("husl", len(numerical_columns))

# Loop sulle variabili numeriche per creare i grafici
for i, column in enumerate(numerical_columns):
    row, col = divmod(i, 2)  # Organizza i subplot in una griglia
    ax = axes[row, col]
    
    sns.lineplot(data=df_train, x='id', y=column, color=palette[i], ax=ax, label=column)
    sns.lineplot(data=df_test, x='id', y=column, color=palette[i], ax=ax)
    ax.vlines(x=df_test['id'].min(), ymin=df_test[column].min(), ymax=df_test[column].max(), linestyles='dashed', label='Test-set Starting point')
    
    ax.set_title(f"ðŸ“ˆ Trend of {column.capitalize()}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Day of the Year", fontsize=12)
    ax.set_ylabel(column.capitalize(), fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

# Migliora la spaziatura tra i grafici
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


PERIOD = 365



# Train Dataset:

# Get seasons from day:
df_train['seasons'] = get_seasons_from_day(df_train['day'])
# get sin-loop (with base 365) from day
df_train['day_sin'] = sin_loop(df_train['day'], period=PERIOD)



# Test Dataset:

# Get seasons from day:
df_test['seasons'] = get_seasons_from_day(df_test['day'])
# get sin-loop (with base 365) from day
df_test['day_sin'] = sin_loop(df_test['day'], period=PERIOD)
df_test.head()


plt.figure(figsize=(20, 5))
for season in df_train.seasons.unique():
    sns.scatterplot(data=df_train[df_train['seasons'] == season], x='id', y='temparature', label=season, palette='Set2') 
    sns.scatterplot(data=df_test[df_test['seasons'] == season], x='id', y='temparature', label=season, palette='Set2') 
    plt.vlines(x=df_test['id'].min(), ymin=df_test['temparature'].min(), ymax=df_test['temparature'].max(), linestyles='dashed')
plt.grid()


# numerical columns list:
numerical_columns = [
    'pressure', 
    'maxtemp', 
    'temparature', 
    'mintemp',
    'dewpoint',
    'humidity', 
    'cloud', 'sunshine',
    'winddirection', 
    'windspeed',
    'day_sin'
]

# integer columns:
categorical_columns = ['seasons']

# taget columns:
target_columns = 'rainfall'

# id columns
id_column = 'id'




# Composing matrix X and vector y:
X = df_train[[*numerical_columns, *categorical_columns]]
y = df_train[target_columns]
X_submission = df_test[[*numerical_columns, *categorical_columns]]



# Split into train-set and test-set:
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)



# Create preprocessor-column-transformer:
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    ('standard', StandardScaler(), numerical_columns)
    ],
    verbose_feature_names_out=False
)


# transforms all the datasets:
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
X_scaled = preprocessor.transform(X)
X_submission_scaled = preprocessor.transform(X_submission)


# Takes the features name trasformed:
feature_names = preprocessor.get_feature_names_out()

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
X_submission_scaled = pd.DataFrame(X_submission_scaled, columns=feature_names, index=X_submission.index)



plt.figure(figsize=(16, 10))

sns.heatmap(data=X_scaled.corr(method='spearman'), vmin=-1, vmax=1, annot=True)
plt.title('Spearman Correlation Matrix:')


# # USE THIS CODE IF YOU WANT ESTIMATE THE BEST RANDOM-FOREST-CLASSIFIER:

# # define a dictionary with:
# # - keys: name of Randome-Forest parameters
# # - values: a list of optional values that we wont to test 
# params_grid = {
#     'n_estimators': [100, 200], 
#     'max_depth': [3, 4, 6, 8, 10],
#     'min_samples_split': [2, 4, 6, 8, 12, 16],
#     'min_samples_leaf': [2, 4, 5, 6, 8, 12, 16]
# }

# # Define our Grid-Search-CV method and the feed with our chosen parameters and the number of folds we want to use:
# search = GridSearchCV(RandomForestClassifier(), params_grid, cv=6)
# search.fit(X_train_scaled, y_train)


# # Best Model:
# print("Best Parameters:", search.best_params_)
# print("Best Score:", search.best_score_)

# best_forest = search.best_estimator_
# best_forest


model_local_estimated ='/kaggle/input/playgroud_s5e3/scikitlearn/default/1/playground-series-s5e3.pkl'

best_forest = pickle.load(open(model_local_estimated, mode="rb"))
best_forest



y_train_best = best_forest.predict(X_train_scaled)
y_train_best_proba = best_forest.predict_proba(X_train_scaled)[:, 1]

y_test_best = best_forest.predict(X_test_scaled)
y_test_best_proba = best_forest.predict_proba(X_test_scaled)[:, 1]

print('\t\t Random Forest:')
print('accuracy_score train:', round(accuracy_score(y_train, y_train_best), 2), 'accuracy_score test:', round(accuracy_score(y_test, y_test_best), 2))
print('precision_score train:', round(precision_score(y_train, y_train_best), 2), 'precision_score test:', round(precision_score(y_test, y_test_best), 2))
print('recall_score train:', round(recall_score(y_train, y_train_best), 2), 'recall_score test:', round(recall_score(y_test, y_test_best), 2))
print('f1_score train:', round(f1_score(y_train, y_train_best), 2), 'f1_score test:', round(f1_score(y_test, y_test_best), 2))
print('roc-auc train:', round(roc_auc_score(y_train, y_train_best_proba), 2), 'roc-auc test:', round(roc_auc_score(y_test, y_test_best_proba), 2))


feature_importances = pd.DataFrame(
    {
        'features':X_train_scaled.columns,
        'importances':best_forest.feature_importances_
    }
)
feature_importances.sort_values(by='importances', ascending=False, inplace=True)

sns.barplot(data=feature_importances, x='importances', y='features', hue='features')
plt.grid()
plt.title('Feature Importances:')
plt.show()



# compute the fainfall probability for submission dataset:
rainfall_proba = best_forest.predict_proba(X_submission_scaled)[:, 1]

# create submission dataframe:
df_for_submission = df_test.copy()
df_for_submission['rainfall'] = rainfall_proba

submission_file = df_for_submission[['id', 'rainfall']]
submission_file.head()


# Save
submission_file.to_csv('submission.csv', index=False)




