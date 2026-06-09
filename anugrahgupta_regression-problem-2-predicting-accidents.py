# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt

def plot_categorical_bar(df, column, top_n=None, figsize=(8,5), color='#4287f5', show_percent=False, sort='value'):
    """
    Plots a bar chart for a categorical column in a DataFrame,
    optionally annotating bars with percentages and custom sorting.

    Parameters:
        df (pd.DataFrame): Data frame.
        column (str): Column to plot.
        top_n (int, optional): Show top N categories only.
        figsize (tuple, optional): Size of figure.
        color (str, optional): Bar color.
        show_percent (bool, optional): Annotate bars with percentages.
        sort (str, optional): 'frequency' (default) or 'value'

    Returns:
        matplotlib.axes.Axes: Plotted axes.
    """
    counts = df[column].value_counts()
    if sort == 'value':
        counts = counts.sort_index()
    elif sort == 'frequency':
        counts = counts.sort_values(ascending=False)
    if top_n is not None:
        counts = counts.head(top_n)
    percent = counts / counts.sum() * 100

    ax = counts.plot(kind='bar', figsize=figsize, color=color)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {column}')
    plt.xticks(rotation=45)

    if show_percent:
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{percent.iloc[i]:.1f}%', 
                        (p.get_x() + p.get_width()/2, p.get_height()), 
                        ha='center', va='bottom', fontsize=11, color='black')

    plt.tight_layout()
    plt.show()
    return ax



def plot_numerical_hist(df, column, bins=10, figsize=(8,5), color='#4287f5', show_percent=False):
    """
    Plots a histogram showing the distribution of values for a numerical column in a DataFrame,
    optionally annotating bars with percentages.

    Parameters:
        df (pd.DataFrame): The data frame containing the data.
        column (str): Column to plot.
        bins (int or list, optional): Number of bins or bin edges.
        figsize (tuple, optional): Size of figure.
        color (str, optional): Bar color.
        show_percent (bool, optional): Annotate bars with percentages if True.

    Returns:
        matplotlib.axes.Axes: The plotted axes.
    """
    data = df[column].dropna()

    fig, ax = plt.subplots(figsize=figsize)
    counts, bin_edges, patches = ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.8)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {column}')
    plt.xticks(rotation=45)

    if show_percent:
        total = len(data)
        for count, patch in zip(counts, patches):
            if count > 0:
                # Place annotation at center top of each bar
                x = patch.get_x() + patch.get_width() / 2
                percent = 100 * count / total
                ax.annotate(f'{percent:.1f}%', (x, count), ha='center', va='bottom', fontsize=11, color='black')

    plt.tight_layout()
    plt.show()
    return ax



def plot_bivariate_bar(df, cat_col, target_col, top_n=None, figsize=(8,5), color='#4287f5', sort='value', show_count=False):
    """
    Plots a bar chart of the mean target value for each category in a categorical column.
    
    Parameters:
        df (pd.DataFrame): Data frame.
        cat_col (str): Categorical column.
        target_col (str): Target column (numerical).
        top_n (int, optional): Show top N categories only.
        figsize (tuple, optional): Size of figure.
        color (str, optional): Bar color.
        sort (str, optional): 'value' (alphabetical/category index) or 'mean' (target mean) or 'count' (category frequency).
        show_count (bool, optional): Show count of each category above bars.
        
    Returns:
        matplotlib.axes.Axes: Plotted axes.
    """
    gp = df.groupby(cat_col)[target_col].mean()
    cnt = df[cat_col].value_counts()
    
    # Optionally sort
    if sort == 'value':
        gp = gp.sort_index()
    elif sort == 'mean':
        gp = gp.sort_values(ascending=False)
    elif sort == 'count':
        gp = gp[cnt.loc[gp.index].sort_values(ascending=False).index]
    
    if top_n is not None:
        gp = gp.head(top_n)

    ax = gp.plot(kind='bar', figsize=figsize, color=color)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(f'Mean {target_col}')
    ax.set_title(f'{target_col} by {cat_col}')
    plt.xticks(rotation=45)
    
    if show_count:
        counts_subset = cnt.loc[gp.index]
        for i, p in enumerate(ax.patches):
            ax.annotate(f'n={counts_subset.iloc[i]}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=11, color='black')
    
    plt.tight_layout()
    plt.show()
    return ax

import matplotlib.pyplot as plt

def plot_scatter(df, x_col, y_col, figsize=(8,5), color='#4287f5', alpha=0.6, add_regline=False, title=None):
    """
    Plots a scatter plot for two numerical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Data frame.
        x_col (str): Column name for x-axis (numerical).
        y_col (str): Column name for y-axis (numerical).
        figsize (tuple, optional): Size of figure.
        color (str, optional): Point color.
        alpha (float, optional): Point transparency.
        add_regline (bool, optional): Overlay a regression line.
        title (str, optional): Custom plot title.

    Returns:
        matplotlib.axes.Axes: Plotted axes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], color=color, alpha=alpha)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f'{y_col} vs {x_col}')
    
    # Optionally add regression line
    if add_regline:
        import numpy as np
        from sklearn.linear_model import LinearRegression
        # Drop missing values for both columns
        X = df[[x_col, y_col]].dropna()
        x = X[x_col].values.reshape(-1, 1)
        y = X[y_col].values
        lr = LinearRegression()
        lr.fit(x, y)
        y_pred = lr.predict(x)
        ax.plot(x, y_pred, color='red', linewidth=2, label='Regression line')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    return ax



def assign_bins(df, column, bins=5, labels=None, method='quantile', new_col=None, include_lowest=True):
    """
    Assigns bins to a numerical column and adds a new binned column to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame to modify.
        column (str): Name of the numerical column to bin.
        bins (int or list): Number of bins or explicit bin edges.
        labels (list, optional): Labels for the bins. If None, auto-label.
        method (str): 'quantile' (equal-frequency, pd.qcut) or 'uniform' (equal-width, pd.cut)
        new_col (str, optional): Name for the new binned column. Default: column+'_bin'
        include_lowest (bool): Include the lowest value in first bin.

    Returns:
        pd.Series: The newly created binned column.
    """
    if new_col is None:
        new_col = f"{column}_bin"

    col_data = df[column]

    if method == 'quantile':
        # Equal-frequency bins using pd.qcut
        binned = pd.qcut(col_data, q=bins, labels=labels, duplicates='drop', include_lowest=include_lowest)
    else:
        # Equal-width bins using pd.cut
        binned = pd.cut(col_data, bins=bins, labels=labels, include_lowest=include_lowest)
        
    df[new_col] = binned
    return binned

# Example usage:
# assign_bins(df, 'accident_risk', bins=4, method='quantile')
# assign_bins(df, 'num_lanes', bins=[0,2,4,6,8], labels=['1-2','3-4','5-6','7-8'], method='uniform')


from scipy.stats import chi2_contingency

def get_chi2(df, row_col, col_col):
    """
    Computes and prints the chi-square statistic and p-value for two categorical columns.
    """
    table = pd.crosstab(df[row_col], df[col_col])
    chi2, p, dof, expected = chi2_contingency(table)
    print(f"Chi2: {chi2:.6f}, p-value: {p:.6f}")
    return chi2, p
    

import seaborn as sns
import matplotlib.pyplot as plt

def plot_crosstab_heatmap(df, row_col, col_col, normalize='row', cmap='coolwarm', fmt='.2f', figsize=(7,5)):
    """
    Plots a heatmap of normalized proportions for two categorical columns.
    """
    table = pd.crosstab(df[row_col], df[col_col])

    if normalize == 'row':
        proportions = table.div(table.sum(axis=1), axis=0)
    elif normalize == 'column':
        proportions = table.div(table.sum(axis=0), axis=1)
    else:
        proportions = table

    plt.figure(figsize=figsize)
    sns.heatmap(proportions, annot=True, fmt=fmt, cmap=cmap, cbar=True)
    plt.title(f'Proportion of {col_col} within {row_col}')
    plt.ylabel(row_col)
    plt.xlabel(col_col)
    plt.tight_layout()
    plt.show()



# loading the dataset

df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df.head()


df.head()


df.shape


df.dtypes


df.describe()


# Check for duplicate rows in a DataFrame
duplicate_rows = df.duplicated()

# Count total duplicates
num_duplicates = duplicate_rows.sum()
print(f'Total duplicate rows: {num_duplicates}')

# View all duplicate rows (excluding their first occurrence)
duplicates_df = df[duplicate_rows]
print(duplicates_df)


# Count of null (missing) values for every column in a DataFrame
null_counts = df.isnull().sum()

print(null_counts)



# For unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"Unique values in '{col}': {unique_vals}")


# For mean and std in numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns
for col in numerical_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"Mean and STD of '{col}': Mean = {mean_val}, Std = {std_val}")


# Visualize target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['accident_risk'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Distribution of Accident Risk', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Accident Risk')
axes[0].set_ylabel('Frequency')

axes[1].boxplot(df['accident_risk'])
axes[1].set_title('Boxplot of Accident Risk', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Accident Risk')

plt.tight_layout()
plt.show()

print(f'Target Variable Statistics:')
print(df['accident_risk'].describe())


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create bins for accident_risk with width 0.2
bins = np.arange(0, 1.2, 0.1)  # [0.0, 0.2, 0.4, ..., 1.0]
df['risk_bin'] = pd.cut(df['accident_risk'], bins=bins, include_lowest=True)

# Plot count of records in each bin
sns.countplot(x='risk_bin', data=df)
plt.xlabel('Accident Risk (bin width 0.2)')
plt.ylabel('Count')
plt.title('Distribution of Accident Risk (Binned)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import seaborn as sns
sns.countplot(x='num_reported_accidents', data=df)


import seaborn as sns
import matplotlib.pyplot as plt

df['high_risk'] = (df['accident_risk'] > 0.7).astype(int)

plot_categorical_bar(df, 'high_risk', color='#4287f5', show_percent=True)


print(df['accident_risk'].describe())
print('Skewness:', df['accident_risk'].skew())


df.drop(['risk_bin', 'high_risk'], axis=1, inplace=True)


df.describe()


df['accident_risk'].quantile([0.25, 0.5, 0.75, 0.9])


df.hist()


# Find categorical columns by dtype
cat_by_dtype = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Find columns (any dtype) with unique values < 10 (excluding target/numeric IDs as needed)
cat_by_nunique = [col for col in df.columns if df[col].nunique(dropna=False) < 10]

# Combine and deduplicate
categorical_cols = list(set(cat_by_dtype + cat_by_nunique))

# For remaining columns, treat as numerical (excluding categorical and non-numeric columns)
numerical_cols = [col for col in df.columns 
                  if col not in categorical_cols and np.issubdtype(df[col].dtype, np.number)]

# Plot categorical bar plots
for col in categorical_cols:
    plot_categorical_bar(df, col, color='#4287f5', show_percent=True)

# Plot histograms for numerical columns
for col in numerical_cols:
    if col!='id':
        plot_numerical_hist(df, col, bins=10, show_percent=True)



features = df.drop('accident_risk', axis=1)


corr_matrix = df.corr(numeric_only=True)  # or df.corr() if all num
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='seismic', center=0)
plt.title("Correlation Matrix")
plt.show()


for col in categorical_cols:
    plot_bivariate_bar(df, col, 'accident_risk', top_n=None, figsize=(8,5), color='#4287f5', sort='value', show_count=False)


df[df['num_reported_accidents']==7]


df.info()


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df,
    x='curvature',          # x-axis column
    y='accident_risk',      # y-axis column
    hue='road_signs_present',        # (optional) adds color for categories
    palette='Set2',         # (optional) choose color palette
    alpha=0.7               # transparency
)
plt.title('Accident Risk vs curvature by road_signs_present')
plt.show()


fig_df = df.copy()

fig_df['high_curvature'] = fig_df['curvature']>0.5


fig_df.groupby(['high_curvature','lighting'])['accident_risk'].mean()


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(
    data=fig_df,
    x='high_curvature',
    y='accident_risk',
    hue='lighting',
    estimator='mean',
    ci=None
)
plt.title('Mean Accident Risk by High Curvature and lighting')
plt.ylabel('Mean Accident Risk')
plt.xlabel('high_curvature')
plt.tight_layout()
plt.show()


fig_df.groupby(['high_curvature','num_lanes'])['accident_risk'].mean()


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(
    data=fig_df,
    x='high_curvature',
    y='accident_risk',
    hue='num_lanes',
    estimator='mean',
    ci=None
)
plt.title('Mean Accident Risk by High Curvature and Number of Lanes')
plt.ylabel('Mean Accident Risk')
plt.xlabel('Number of Lanes')
plt.tight_layout()
plt.show()


fig_df = df.copy()

fig_df.groupby(['lighting','time_of_day'])['accident_risk'].mean()


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(
    data=fig_df,
    x='lighting',
    y='accident_risk',
    hue='time_of_day',
    estimator='mean',
    ci=None
)
plt.title('Mean Accident Risk by High Curvature and Number of Lanes')
plt.ylabel('Mean Accident Risk')
plt.xlabel('Number of Lanes')
plt.tight_layout()
plt.show()


not_imp_features = {}

fig_df['high_risk']= fig_df['accident_risk']>0.7
for column in categorical_cols: 
    print(f'investigation for {column}')
    chi_square, p=  get_chi2(fig_df, 'high_risk', column)
    if p>0.05:
        not_imp_features[column]={'chi_square':chi_square, 
                               'p':  p}


not_imp_features


for column in categorical_cols: 
    plot_crosstab_heatmap(fig_df, 'high_risk', column, normalize='row', cmap='coolwarm', fmt='.2f', figsize=(7,5))


import seaborn as sns
import matplotlib.pyplot as plt


sns.scatterplot(
    data=df,
    x='curvature',
    y='accident_risk',
    hue='road_signs_present',
    style='road_type',
    palette='Set2',
    alpha=0.7
)
plt.title('Accident Risk vs Curvature by Road Signs and Lighting')
plt.legend()
plt.show()


# Assuming road_signs_present is numeric (e.g. count or binary)
pivot = fig_df.pivot_table(
    index='lighting',
    columns='high_risk',     # Make sure df["high_risk"] is defined: df['accident_risk'] > 0.5
    values='road_signs_present',
    aggfunc='mean'
)

print(pivot)



import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df,
    x='curvature',          # x-axis column
    y='accident_risk',      # y-axis column
    hue='time_of_day',        # (optional) adds color for categories
    palette='Set2',         # (optional) choose color palette
    alpha=0.7               # transparency
)
plt.title('Accident Risk vs Number of Lanes by Road Type')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=df,
    x='curvature',          # x-axis column
    y='accident_risk',      # y-axis column
    hue='road_signs_present',        # (optional) adds color for categories
    palette='Set2',         # (optional) choose color palette
    alpha=0.7               # transparency
)
plt.title('Accident Risk vs Number of Lanes by Road Type')
plt.show()


sns.boxplot(x='lighting', y='accident_risk', data=df)


sns.boxplot(x='time_of_day', y='accident_risk', data=df)


sns.violinplot(x='lighting', y='accident_risk', data=df)


sns.violinplot(x='time_of_day', y='accident_risk', data=df)


sns.boxplot(x='road_type', y='accident_risk', data=df)


sns.violinplot(x='num_lanes', y='accident_risk', data=df)


sns.boxplot(x='weather', y='accident_risk', data=df)


sns.boxplot(x='road_signs_present', y='accident_risk', data=df)


sns.boxplot(x='speed_limit', y='accident_risk', data=df)


fig_df['high_risk'] = fig_df['accident_risk']>0.7
fig_df['high_speed'] = fig_df['speed_limit']>=50


from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

table = pd.crosstab(fig_df['high_risk'], fig_df['speed_limit'])


# Normalize by row (lighting)
proportions = table.div(table.sum(axis=1), axis=0)
sns.heatmap(proportions, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Proportion of speed limit within high risk')
plt.show()



pd.crosstab(fig_df['high_speed'], fig_df['high_risk'])


from scipy.stats import chi2_contingency

table = pd.crosstab(fig_df['high_risk'], fig_df['high_speed'])
proportions = table.div(table.sum(axis=1), axis=0)
sns.heatmap(proportions, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Proportion of high_speed vs high risk')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(
    data=fig_df,
    x='curvature',          # x-axis column
    y='accident_risk',      # y-axis column
    hue='high_speed',        # (optional) adds color for categories
    palette='Set2',         # (optional) choose color palette
    alpha=0.7               # transparency
)
plt.title('Accident Risk vs Number of Lanes by Road Type')
plt.show()


sns.boxplot(x='high_speed', y='accident_risk', data=fig_df)


fig_df['is_clear']=fig_df['weather']=='clear'
fig_df['is_dark']= fig_df['lighting']=='night'
fig_df['is_evening']= fig_df['time_of_day'] =='evening'
fig_df['is_foggy']=fig_df['weather']=='foggy'
fig_df['is_urban']=fig_df['road_type']=='urban'


fig_df.corr(numeric_only=True)  # or df.corr() if all num


df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


df = df[df['num_reported_accidents']!=7]
df = df[(df['accident_risk']!=0) &(df['accident_risk']!=0)  ]


from sklearn.model_selection import train_test_split

y = df["accident_risk"]
X = df.drop('accident_risk', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------
# 1ï¸�âƒ£ Pre-cleaner (run before feature engineering)
# -------------------------------------------------
class PreCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans NaNs, drops high-missing columns, fixes categorical issues.
    """
    def __init__(self,
                 num_strategy='median',
                 cat_strategy='mode',
                 drop_threshold=0.9):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.drop_threshold = drop_threshold
        self.numeric_fill_values_ = {}
        self.categorical_fill_values_ = {}
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        X = X.copy()

        # Drop columns with excessive missing
        missing_ratio = X.isnull().mean()
        self.columns_to_drop_ = missing_ratio[missing_ratio > self.drop_threshold].index.tolist()

        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        # Numeric fill values
        for c in num_cols:
            if c in self.columns_to_drop_: continue
            if self.num_strategy == 'mean':
                self.numeric_fill_values_[c] = X[c].mean()
            else:
                self.numeric_fill_values_[c] = X[c].median()

        # Categorical fill values
        for c in cat_cols:
            if c in self.columns_to_drop_: continue
            if self.cat_strategy == 'mode':
                self.categorical_fill_values_[c] = X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown"
            else:
                self.categorical_fill_values_[c] = "Unknown"

        return self

    def transform(self, X):
        X = X.copy()

        # Drop high-missing columns
        X.drop(columns=self.columns_to_drop_, inplace=True, errors='ignore')

        # Fill numeric
        for c, v in self.numeric_fill_values_.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)

        # Fill categorical + clean strings
        for c, v in self.categorical_fill_values_.items():
            if c in X.columns:
                X[c] = X[c].replace(['?', 'NA', 'N/A', '', 'None'], np.nan).fillna(v)
                X[c] = X[c].astype(str).str.strip().str.title()

        return X


# -------------------------------------------------
# 2ï¸�âƒ£ Post-cleaner (run after feature engineering)
# -------------------------------------------------
class PostCleaner(BaseEstimator, TransformerMixin):
    """
    Removes categorical or temporary columns after feature engineering.
    """
    def __init__(self,
                 drop_categoricals=True,
                 drop_columns=None):
        self.drop_categoricals = drop_categoricals
        self.drop_columns = drop_columns if drop_columns is not None else []
        self.categorical_cols_ = []

    def fit(self, X, y=None):
        if self.drop_categoricals:
            self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        if self.drop_categoricals:
            X.drop(columns=self.categorical_cols_, inplace=True, errors='ignore')
        if self.drop_columns:
            X.drop(columns=self.drop_columns, inplace=True, errors='ignore')
        return X


# 1. Create an instance of your transformer
pre_cleaner = PreCleaner()          # <-- instance, not class

# 2. Call fit_transform on the instance
X_train = pre_cleaner.fit_transform(X=X_train)
X_train


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class FlexibleCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Mix of OneHotEncoder and LabelEncoder for flexible encoding. 
    Returns DataFrame with all columns (encoded and untouched).
    """
    def __init__(self, onehot_columns=None, label_columns=None, drop_first=False, dtype=int):
        self.onehot_columns = onehot_columns if onehot_columns is not None else []
        self.label_columns = label_columns if label_columns is not None else []
        self.drop_first = drop_first
        self.dtype = dtype
        self.onehot_encoders_ = {}
        self.label_encoders_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        # Fit OneHotEncoders
        for col in self.onehot_columns:
            encoder = OneHotEncoder(drop='first' if self.drop_first else None, sparse=False, dtype=self.dtype, handle_unknown='ignore')
            encoder.fit(X[[col]])
            self.onehot_encoders_[col] = encoder

        # Fit LabelEncoders
        for col in self.label_columns:
            encoder = LabelEncoder()
            encoder.fit(X[col].astype(str))
            self.label_encoders_[col] = encoder
        return self

    def transform(self, X):
        X = X.copy()
        new_cols = []

        # One-hot encoding
        for col in self.onehot_columns:
            encoder = self.onehot_encoders_[col]
            ohe = encoder.transform(X[[col]])
            ohe_cols = encoder.get_feature_names_out([col])
            ohe_df = pd.DataFrame(ohe, columns=ohe_cols, index=X.index)
            new_cols.append(ohe_df)
            X.drop(columns=col, inplace=True)

        # Label encoding
        for col in self.label_columns:
            encoder = self.label_encoders_[col]
            X[col] = encoder.transform(X[col].astype(str))

        # Concatenate new one-hot columns to result
        if new_cols:
            X = pd.concat([X] + new_cols, axis=1)
        return X

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from scipy.stats import chi2_contingency

class FeatureEngineeringbeforeEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, corr_threshold=0.1, pvalue_threshold=0.05):
        self.corr_threshold = corr_threshold
        self.pvalue_threshold = pvalue_threshold
        self.num_cols_to_keep = None
        self.cat_cols_to_keep = None

    def fit(self, X, y):
        X = X.copy()
        
        # Get numerical columns (excluding the target if present)
        num_cols = X.select_dtypes(include='number').columns
        # Pearson correlation (absolute value)
        corr = X[num_cols].corrwith(y).abs()
        self.num_cols_to_keep = list(corr[corr >= self.corr_threshold].index)
        
        # Get categorical columns
        cat_cols = X.select_dtypes(['object', 'category']).columns
        pvalues = []
        for col in cat_cols:
            table = pd.crosstab(X[col], y > 0.5)
            if table.shape[0] > 1 and table.shape[1] > 1:
                _, p, _, _ = chi2_contingency(table)
            else:
                p = 1  # ignore if only one value
            pvalues.append((col, p))
        self.cat_cols_to_keep = [col for col, p in pvalues if p <= self.pvalue_threshold]
        self.columns_to_keep = self.cat_cols_to_keep+ self.num_cols_to_keep
        return self

    def transform(self, X):
        X = X.copy()
    
        # Derived features
        # X['is_dark'] = X['lighting'] == 'night'
        X['high_speed'] = X['speed_limit'] > 50
        X['high_speed_high_curv'] = (X['high_speed']) & (X['curvature']>0.5)
        X['high_speed_fog']= (X['high_speed']) & (X['weather']=='foggy')
        X['high_speed_no_sign']= (X['high_speed']) & (X['road_signs_present']==False)
    
        
        cols = self.columns_to_keep  + ['is_dark', 'high_speed', 'high_speed_high_curv', 'high_speed_fog', 'high_speed_no_sign']

        # convert bools to int

        for col in X.columns:
            # Is column all bools, or 'True'/'False' strings (excluding NaNs)?
            unique_vals = set(X[col].dropna().unique())
            if unique_vals.issubset({True, False, 'True', 'False'}):
                # Convert to bool (if needed), then to int in one line:
                X[col] = X[col].astype(str).map({'True': 1, 'False': 0, '1': 1, '0': 0, 'True': 1, 'False': 0}).astype(int)


        # Select only columns that are in X (to avoid KeyError)
        final_cols = [col for col in cols if col in X.columns]
        
        out_df = X[final_cols]
        
    
        return out_df



X_train.info()


X_train


from sklearn.pipeline import Pipeline

pre_process_pipe = Pipeline([
    ("pre_clean", PreCleaner(num_strategy='median', cat_strategy='mode', drop_threshold=0.8)),
    ("feature_eng", FeatureEngineeringbeforeEncoding( corr_threshold=0.01, pvalue_threshold=0.05)),
    ("one_hot_enc", FlexibleCategoricalEncoder(onehot_columns=['road_type', 'weather', 'lighting'])),
    ("post_clean", PostCleaner(drop_categoricals=True))
])

# Fit the pipeline and transform your data
X_train_transformed = pre_process_pipe.fit_transform(X_train, y_train)  # y_train passed to relevant steps
X_train_transformed


from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np

# If installed:
from xgboost import XGBRegressor

models = {
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_features=0.5, min_samples_split=5, min_samples_leaf = 4),
    "hist_gbm": HistGradientBoostingRegressor(random_state=42),
    # "svm": SVR(kernel='rbf', C=1.0, epsilon=0.1),         # You can tune kernel/C/epsilon as needed
    # "xgboost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Assuming you already have:
# - X_train, y_train, X_test, y_test
# - preprocessing pipeline: train_pipe (without model step)


# linear_regressor = LinearRegression()
# ridge_regressor =  Ridge(alpha=1.0)
# random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# svm = SVR(kernel='poly')
# # Define the models you want to train
# models = {
#     "linear_regression": linear_regressor,
#     "ridge_regression": ridge_regressor,
#     "random_forest": random_forest_regressor,
#     ""
    
#  }

# Create full pipelines with your preprocessing + model
model_pipelines = {}
for name, model in models.items():
    pipe = Pipeline([
        ("preprocessing", pre_process_pipe),
        ("model", model)
    ])
    model_pipelines[name] = pipe


from sklearn.model_selection import cross_val_score
import numpy as np

# Use negative mean squared error for regression accuracy (lower is better)
scoring = 'neg_root_mean_squared_error'
cv = 5  # 5-fold CV

for name, pipeline in model_pipelines.items():
    print(f"Evaluating {name} ...")
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    mae_scores = -scores  # Convert negative MAE to positive
    print(f"{name}:")
    print(f"  Mean RMSE: {mae_scores.mean():.4f}")

    print(f"  Std RMSE: {mae_scores.std():.4f}")
    print(f"% Error RMSE:  {mae_scores.mean()*100/y_train.mean()}")
    print(f"  All folds RMSE: {mae_scores}\n")


# Train all models
for name, pipeline in model_pipelines.items():
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    print(f"{name} training completed.")

# Dictionaries to store predictions and evaluation metrics
preds, evaluation = {}, {}

for name, pipeline in model_pipelines.items():
    preds[name] = pipeline.predict(X_test)
    y_pred = preds[name]
    
    # Calculate common metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    evaluation[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2
    }


evaluation


evaluation  # after adding more high risk feature


evaluation # without removing but adding more high_risk feature


# loading the dataset

test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

for name, pipeline in model_pipelines.items():
    y_pred = pipeline.predict(test_df)
    out_df = pd.DataFrame({
        "id": test_df['id'],  # or X_test.index if 'id' is the index
        "accident_risk": y_pred
    })
    out_df.to_csv(f"{name}_submission_updated_without_zero_risk.csv", index=False)


preds['random_forest']


evaluation # substituting features speed limit and light with high speed and night a





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# df_compare_sorted already defined as above
# preds already contains your model predictions

# Combine all predictions into a DataFrame for easy plotting
df_compare = pd.DataFrame({'Actual': y_test})
for name in preds:
    df_compare[name] = preds[name]

# Reset index to make charts cleaner
df_compare_sorted = df_compare.sort_values('Actual').reset_index(drop=True)

# 1. Calculate the MAE for each model
mae_scores = {}
for name in preds:
    mae_scores[name] = np.mean(np.abs(df_compare_sorted['Actual'] - df_compare_sorted[name]))

# 2. Sort models by MAE, lowest error LAST so it is plotted on top!
ordered_models = sorted(mae_scores.keys(), key=lambda k: mae_scores[k], reverse=True)

plt.figure(figsize=(16, 7))

# Plot actual values first (choose linewidth and style for clarity)

# Plot model predictions in order (lowest error will be plotted last, hence on top)
for name in ordered_models:
    plt.plot(
        df_compare_sorted[name],
        label=f"{name} (MAE={mae_scores[name]:.1f})",
        linewidth=2 if mae_scores[name] == min(mae_scores.values()) else 1,
        linestyle='-' if mae_scores[name] == min(mae_scores.values()) else '--',
        alpha=0.75
    )
    
plt.plot(df_compare_sorted['Actual'], label='Actual', linewidth=2, color='black')

plt.legend()
plt.title('Actual vs. Predicted Values (Best Model on Top)')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.tight_layout()
plt.show()


df_compare.hist()


plt.figure(figsize=(16, 7))
plt.scatter(y_test, preds['random_forest'], alpha=0.2, label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()


plt.figure(figsize=(16, 7))
plt.scatter(y_test, preds['hist_gbm'], alpha=0.2, label='Hist GBM')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram for HistGradientBoosting predictions
axes[0].hist(preds['hist_gbm'], bins=50, edgecolor='black', alpha=0.7, color='orangered')
axes[0].set_title('HistGradientBoosting Predictions', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Accident Risk')
axes[0].set_ylabel('Frequency')

# Histogram for RandomForest predictions
axes[1].hist(preds['random_forest'], bins=50, edgecolor='black', alpha=0.7, color='mediumseagreen')
axes[1].set_title('RandomForest Predictions', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Accident Risk')
axes[1].set_ylabel('Frequency')

# Histogram for actual values
axes[2].hist(y_test, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[2].set_title('Actual Accident Risk', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Actual Accident Risk')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print('Target Variable and Prediction Statistics:')
print(df['accident_risk'].describe())





