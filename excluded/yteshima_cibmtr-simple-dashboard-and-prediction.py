!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


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


import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from datetime import datetime, timedelta

import ipywidgets as widgets
from ipywidgets import interact, Layout
from IPython.display import HTML, display, clear_output
from IPython.display import IFrame

import matplotlib.pyplot as plt

from scipy import optimize

from xgboost import XGBRegressor,XGBClassifier, DMatrix
from lightgbm import LGBMRegressor, LGBMClassifier, log_evaluation, early_stopping
import lightgbm as lgb

from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error ,mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_curve, auc


from matplotlib.colors import LinearSegmentedColormap

from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.utils import concordance_index


custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', ['blue', 'white', 'red']
)

import random

from tqdm import tqdm

from colorama import Fore, Style, init;

import optuna
import shap

from optuna.samplers import TPESampler

from scipy import optimize

# ignore wornings
import warnings
warnings.filterwarnings("ignore")

# Get the execution mode of the Kaggle environment
run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')


# Load the training and test data from CSV files

df_sample_submission = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col = 0)
df_train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col=0)
df_dictionary = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv', index_col=0)

print(f'N_train = {len(df_train)}, N_test = {len(df_test)}, N_sample_submission = {len(df_sample_submission)}, N_dictionary = {len(df_dictionary)}')

# Assign a new column 'train_test' with the value 'train' to the training dataset, and 'test' to the test dataset respectively
df_train['train_test'] = 'train'
df_test['train_test'] = 'test'

# Nelson-Aalen Label Transformation
naf = NelsonAalenFitter()
naf.fit(df_train['efs_time'], df_train['efs'])
df_train['naf_label'] = -naf.cumulative_hazard_at_times(df_train['efs_time']).values

df_train['cox'] = df_train['efs_time']
df_train.loc[df_train['efs']==0, 'cox'] *= -1

# df_train.loc[df_train['efs'] == 0, 'naf_label'] -= 0.1

# Specify the target column for the analysis or model
target_col = 'naf_label'

# Combine the training and test datasets into a single DataFrame for unified processing
df_all = pd.concat([
    df_train,
    df_test,
])




df_dictionary


# Define a custom format function to format float values for display
def custom_format(x):
    if isinstance(x, float):
        # Format float values to 3 decimal places and remove trailing zeros and decimal points
        return ('{0:.3f}'.format(x)).rstrip('0').rstrip('.')
    return x

# Display detailed information about the DataFrame, including statistics and metadata
def display_dfinfo(df):
    display(HTML('<h4>Display head of the dataframe</h4>'))
    display(df_all.sample(3))
    display(HTML('<br><h4>Display numerical data infomations</h4>'))
    df_disp = []
    for tt in df['train_test'].unique():
        # Generate descriptive statistics for numeric columns
        tmp = df.select_dtypes(include=[int, float]).loc[df['train_test'] == tt].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )

        # Add skewness and kurtosis for numeric columns
        tmp.loc['skew'] = df.loc[df['train_test'] == tt].select_dtypes(include=[int, float]).skew()
        tmp.loc['kurtosis'] = df.loc[df['train_test'] == tt].select_dtypes(include=[int, float]).kurtosis()

        # Add data type and NaN count for all columns
        tmp.loc['dtype'] = df.loc[df['train_test'] == tt].dtypes
        tmp.loc['NaN count'] = df.loc[df['train_test'] == tt].isna().sum(axis=0)

        tmp.loc['N unique'] = df.loc[df['train_test'] == tt].nunique()
        tmp.columns = pd.MultiIndex.from_product([tmp.columns, [tt]])  # Add multi-level columns
        df_disp.append(tmp)

    df_disp = pd.concat(df_disp, axis=1)  # Combine statistics for all 'train_test' groups
    df_disp = df_disp[df_disp.columns.get_level_values(0).unique()]  # Remove duplicate columns
    # Reorganize the DataFrame and filter relevant statistics
    df_disp = df_disp.T
    df_disp = df_disp.loc[
        df.select_dtypes(include=[int, float]).columns, [
            'count', 'NaN count', 'N unique', 'dtype',
            'mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max',
            'std', 'skew', 'kurtosis'
        ]
    ]
    formatter = {}
    # Display the DataFrame with custom formatting and background gradients for numeric stats
    display(
        df_disp.style.format(formatter=custom_format).background_gradient(
            subset=['mean', 'min', '5%', '25%', '50%', '75%', '95%', 'max'],
            cmap='Reds', axis=1
        )
    )
    
    display(HTML('<br><h4>Display categorical data infomations</h4>'))
    col = df.columns[0]
    for col in [col for col in df.select_dtypes(include='object') if col != 'train_test']:
        df_disp = []
        for tt in df['train_test'].unique():
            tmp = df.loc[df['train_test']==tt,col]
            tmp.fillna('nan', inplace = True)
            tmp = pd.DataFrame(tmp.value_counts()).T
            df_disp.append(tmp)

        df_tmp = df[[col, target_col]].copy()
        df_tmp[col].fillna('nan', inplace = True)

        df_disp.append(df_tmp.groupby(col, dropna = False)[target_col].describe()[['mean', 'std']].T)
        df_disp = pd.concat(df_disp)

        df_disp.index = pd.MultiIndex.from_product([
            [col],
            [f'{trates} count' for trates in list(df['train_test'].unique())] + [f'{target_col} mean', f'{target_col} std']
        ])

        df_disp.columns.names = ['']
        display(df_disp.style.set_table_styles([
            {'selector': 'th.index_name', 'props': [('width', '60px')]},  # width of 1st row
            {'selector': 'th.row_heading', 'props': [('width', '90px')]},  # width of 2nd row
        ]
       ).background_gradient(cmap='Reds', axis=1))


# Function to plot a correlation matrix for numeric columns
def plot_correlation_matrix(df, num_cols, plottype='sns'):
    if plottype == 'plotly':
        fig = px.imshow(
            df[num_cols].corr(), zmax=1, zmin=-1, color_continuous_scale='rdbu_r',  # Red-blue color scale
            text_auto=".2f"
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=max(min(len(num_cols) * 100, 600), 600),
            height=max(min(len(num_cols) * 80, 600), 400),
            title='Correlation matrix'
        )
        fig.show()
    elif plottype == 'sns':
        plt.figure(figsize=(min(len(df.columns) * 0.6, 15), min(len(df.columns) * 0.15, 12)))
        sns.heatmap(
            df[num_cols].corr(), annot=True, vmax=1, vmin=-1,
            cmap=custom_cmap, fmt='.2f'  # Use a custom color map and format values
        )
        plt.title('Correlation Matrix')
        plt.show()



num_cols = df_all.select_dtypes(include=[float, int]).columns
display_dfinfo(df_all[df_all['train_test']=='train'])
plot_correlation_matrix(df_all, num_cols,  plottype='sns')


# Function to plot a histogram for a given numeric column, optionally grouped by a categorical column
def hist_df(df, num_col1, cat_col, n_data, displaytype='density', plottype='sns'):

    # Sample the data if a specific number of rows is specified, otherwise use the full DataFrame
    if np.isnan(n_data):
        df_plot = df.copy()
    else:
        df_plot = df.sample(n_data)
    
    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        if num_col1 != cat_col:  # If a categorical column is provided, color the histogram by category
            fig = px.histogram(
                df_plot,
                x=num_col1, marginal='violin', color=cat_col,
                # nbins=100,
                histnorm=displaytype,  # Normalize the histogram (e.g., density)
                barmode='relative', opacity=0.5
            )
        else:  # If no categorical column, create a plain histogram
            fig = px.histogram(
                df_plot,
                x=num_col1,
                histnorm=displaytype,
                barmode='relative',
                opacity=0.5
            )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=900, height=350,
            margin=dict(l=0, r=0, b=0, t=20),  # Adjust margins
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=15)),
        )
        fig.show()  # Display the Plotly figure

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        if num_col1 != cat_col:  # If a categorical column is provided, create a colored histogram
            fig = sns.histplot(
                df_plot,
                x=num_col1, hue=cat_col,  # Group by the categorical column
                edgecolor=None, alpha=0.5
            )
            plt.show()
        else:  # If no categorical column, create a plain histogram
            fig = sns.histplot(
                df_plot,
                x=num_col1,
                edgecolor=None, alpha=0.5
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
            plt.show()


# Function to create a scatter plot for two numeric columns, optionally grouped by a categorical column
def scatter_df(df, num_col1, num_col2, cat_col, n_data=10000, plottype='sns'):
    # Sample the data if a specific number of rows is specified, otherwise use the full DataFrame
    if np.isnan(n_data):
        df_plot = df.copy()
    else:
        df_plot = df.sample(n_data)
    
    # Create a temporary DataFrame with the selected columns
    df_tmp = df_plot[[num_col1, num_col2, cat_col]]
    # Add spaces to column names to avoid conflicts with Plotly or Seaborn
    df_tmp.columns = [c + ' ' * i for i, c in enumerate(df_tmp.columns)]

    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        fig = px.scatter(
            df_tmp,
            x=num_col1 + '', y=num_col2 + ' ', color=cat_col + '  ',  # Scatter plot with color grouping
            marginal_x='histogram', marginal_y='violin', opacity=0.5,  # Add marginal plots
            color_continuous_scale=px.colors.sequential.Rainbow,  # Use a rainbow color scale
            # trendline='ols',  # Add a trendline    
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=750, height=550,
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins
            xaxis1=dict(domain=[0.1, 0.75]),  # Specify the area of the x-axis for the main plot
            yaxis1=dict(domain=[0.1, 0.8]),  # Specify the area of the y-axis for the main plot
            xaxis2=dict(domain=[0.76, 1.0]),  # Specify the area of the x-axis for marginal plots
            yaxis2=dict(domain=[0.1, 0.8]),  # Specify the area of the y-axis for marginal plots
            xaxis3=dict(domain=[0.1, 0.75]),  # Specify the area of the x-axis for marginal plots
            yaxis3=dict(domain=[0.81, 1.0]),  # Specify the area of the y-axis for marginal plots
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=15)),
        )
        fig.update_traces(marker=dict(size=3), selector=dict(mode='markers'))
        fig.show()  # Display the Plotly figure

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        # If the categorical column is an object or has fewer than 10 unique values, use a JointGrid
        if (df_tmp[cat_col + '  '].dtype == object) or (df_tmp[cat_col + '  '].nunique() <= 10):
            g = sns.JointGrid(
                data=df_tmp,
                x=num_col1 + '', y=num_col2 + ' ', hue=cat_col + '  ',  # Scatter plot with color grouping
                palette='rainbow'
            )
            # Plot the scatterplot in the center
            g.plot_joint(sns.scatterplot, color="blue")
            # Plot KDE (Kernel Density Estimation) on the margins
            try:
                g.plot_marginals(sns.kdeplot, color="skyblue", alpha=0.4, fill=True)
            except: pass
            g.ax_joint.legend(bbox_to_anchor=(1.25, 1), loc='upper left', borderaxespad=0 )
            plt.tight_layout()
            plt.show()
        else:
            # Create a standard scatter plot for non-categorical grouping
            try:
                fig = sns.scatterplot(
                    data=df_tmp,
                    x=num_col1 + '', y=num_col2 + ' ', hue=cat_col + '  ',
                    palette='rainbow'
                )
            except:
                fig = sns.scatterplot(
                    data=df_tmp,
                    x=num_col1 + '', y=num_col2 + ' ',
                    palette='rainbow'
                )
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
            plt.show()

# Function to create a violin plot for two numeric columns, optionally grouped by a categorical column
def violin_df(df, num_col1, num_col2, cat_col, n_data=10000, plottype='sns'):
    # Sample the data if a specific number of rows is specified, otherwise use the full DataFrame
    if np.isnan(n_data):
        df_plot = df.copy()
    else:
        df_plot = df.sample(n_data)
    
    # Check if the categorical column is one of the numeric columns
    col_in_num = cat_col in [num_col1, num_col2]
    if col_in_num:
        df_tmp = df_plot[[num_col1, num_col2]]  # Exclude the categorical column if it's numeric
    else:
        df_tmp = df_plot[[num_col1, num_col2, cat_col]]
        df_tmp[cat_col] = df_tmp[cat_col].astype(object)  # Convert the categorical column to object type

    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        if df_tmp[num_col1].dtype in [int, float]:
            bw = (df_tmp[num_col1].max() - df_tmp[num_col1].min())/50
            plot_height = min(1200, 120 * df_tmp[num_col1].nunique())
            plot_width = 450
        else:
            bw = (df_tmp[num_col2].max() - df_tmp[num_col2].min())/50
            plot_height = 450
            plot_width = min(1200, 120 * df_tmp[num_col1].nunique())
        
        if col_in_num or df_tmp[cat_col].nunique() > 10:  # Skip coloring if too many unique categories
            fig = px.violin(
                df_tmp, x=num_col1, y=num_col2,
            )
            
        else:  # Include coloring by the categorical column
            fig = px.violin(
                df_tmp, x=num_col1, y=num_col2, color=cat_col,violinmode = 'group'
            )
            plot_width+=120

        # Customize the layout of the Plotly figure
        fig.update_layout(
            width = plot_width,
            height = plot_height,
            margin = dict(l=0, r=0, b=0, t=20),  # Adjust margins
            xaxis = dict(title_font=dict(size=20)),
            yaxis = dict(title_font=dict(size=20)),
            legend = dict(font=dict(size=12)),
        )
        # unify the violin width
        fig.update_traces(
            scalemode='width',
            points=False,
            bandwidth=bw,    
            width=0.95
        )

        fig.show()

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        fig = plt.figure(figsize=(max(min(df_tmp[num_col1].nunique() * df_tmp[cat_col].nunique() / 2,10),4),4))
        if col_in_num or df_tmp[cat_col].nunique() > 10:  # Skip coloring if too many unique categories
            sns.violinplot(
                df_tmp, x=num_col1, y=num_col2,
                linewidth=0.1
            )
        else:  # Include coloring by the categorical column
            sns.violinplot(
                df_tmp, x=num_col1, y=num_col2, hue=cat_col,
                linewidth=0.5
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
        plt.show()


# Function to create a cross-tabulation or pivot table and visualize it
def cross_df(df, num_col2, num_col1, cat_col, n_data, plottype='sns'):
    if df[cat_col].dtype == object:  # If the categorical column is an object
        df_plot = pd.crosstab(df_all[num_col1], df_all[num_col2])  # Create a cross-tabulation
        label = cat_col
    else:  # If the categorical column is numeric
        df_plot = pd.pivot(
            df_all.groupby([num_col1, num_col2])[cat_col].mean().reset_index(),
            columns=num_col1, index=num_col2, values=cat_col
        )
        label = 'count'
    
    # Use Plotly for visualization if specified
    if plottype == 'plotly':
        fig = px.imshow(
            df_plot,
            text_auto=True,
            color_continuous_scale='blues',  # Use a blue color scale
            labels={"color": cat_col}
        )
        # Customize the layout of the Plotly figure
        fig.update_layout(
            width=950, height=550,
            margin=dict(l=0, r=0, b=0, t=0),  # Adjust margins
            xaxis=dict(title_font=dict(size=20)),
            yaxis=dict(title_font=dict(size=20)),
            legend=dict(font=dict(size=12)),
        )
        fig.show()

    # Use Seaborn for visualization if specified
    elif plottype == 'sns':
        fig = sns.heatmap(
            df_plot,
            cmap='coolwarm',  # Use a cool-warm color scale
            annot=True,  # Annotate cells with data values
            fmt = 'd'
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0 )
        plt.show()

# Function to create an interactive plot with dropdowns for column selection
def plot_interact(df_in, disp_cols, n_data=np.nan, n_int_thresh = 3, plottype='sns'):
    # Columns where nunique is less than n_int_thresh are displayed as str.
    df = df_in.copy()
    df.loc[:,df.nunique()<n_int_thresh] = df.loc[:,df.nunique()<n_int_thresh].astype(str)
    button_width=110
    col_dict = {}
    for a, b in zip(df.columns, df.dtypes):
        col_dict[f'({b}) {a}'] = a
    
    # Create dropdowns for selecting numeric and categorical columns
    num_col_button1 = widgets.Dropdown(
        options = col_dict,
        button_style='info',
        description='X'
    )
    num_col_button1.style.description_width = '10px'
    num_col_button1.style.font_size = '1px'

    num_col_button2 = widgets.Dropdown(
        options=col_dict,
        value = target_col,
        button_style='primary',
        description='Y',
    )
    num_col_button2.style.button_width = f'{button_width}px'
    num_col_button2.style.description_width = '10px'

    cat_col_button = widgets.Dropdown(
        options = col_dict, button_style='warning', description='color'
    )
    cat_col_button.style.button_width = '80px'
    cat_col_button.style.description_width = '50px'
    
    # Create an interactive function to update the plot based on user selections
    @interact(
        num_col1=num_col_button1, num_col2=num_col_button2,
        cat_col=cat_col_button,
    )
    def plot_df(num_col1, num_col2, cat_col):
        clear_output()  # Clear the output to avoid clutter
        print(f'type({num_col1}):{df[num_col1].dtype}, type({num_col2}):{df[num_col2].dtype}')
        
        if num_col1 == num_col2:  # If the same column is selected for X and Y, plot a histogram
            hist_df(df, num_col1, cat_col, n_data, plottype=plottype)
        elif (df[num_col1].dtype != object) and (df[num_col2].dtype != object):  # Scatter plot for numeric columns
            scatter_df(df, num_col1, num_col2, cat_col, n_data, plottype=plottype)
        elif df[num_col1].dtype != object:  # Violin plot for numeric vs categorical
            df[num_col2].fillna('nan', inplace = True)
            violin_df(df, num_col1, num_col2, cat_col, n_data, plottype=plottype)
        elif df[num_col2].dtype != object:  # Violin plot for categorical vs numeric
            df[num_col1].fillna('nan', inplace = True)
            violin_df(df, num_col1, num_col2, cat_col, n_data, plottype=plottype)
        else:  # Cross-tabulation for categorical columns
            df[num_col1].fillna('nan', inplace = True)
            df[num_col2].fillna('nan', inplace = True)
            cross_df(df, num_col2, num_col1, cat_col, n_data, plottype=plottype)


for col in df_all.columns:
    df_all[col] = df_all[col].astype(float, errors='ignore')
# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    disp_cols = df_all.columns
    plot_interact(df_all, disp_cols, n_int_thresh = 15, plottype='plotly')
else:
    print('Run in interactive mode to display plots.')


def safe_transform(encoder, labels):
    known_labels = set(encoder.classes_)
    return [
        encoder.transform([label])[0] if label in known_labels else -1 for label in labels
    ]

def target_encoder(df, input_col, target_col):
    tmp = df[[input_col, target_col]]
    means = df.groupby(input_col)[target_col].mean()
    for ind in means.index:
        tmp.loc[tmp[f'{input_col}']==ind, f'{input_col}_te'] = means[ind]

    return tmp[f'{input_col}_te'].values

def preprocessing(df, num_cols, cat_cols, target_col,train_test = 'train_test'):
    df_pp = df[num_cols].astype(float).copy()
    
    for i, cat_col in enumerate(cat_cols):
        print(cat_col, end = ' / ')
        # target encoding
        df_pp[f'{cat_col}_te'] = target_encoder(df, cat_col, target_col)

    imputer = IterativeImputer()
    df_pp = pd.DataFrame(imputer.fit_transform(df_pp), index = df_pp.index, columns = df_pp.columns)
    df_pp[target_col] = df[target_col]
    df_pp[train_test] = df[train_test]
    return df_pp

def adversarial_validation(df_adv):
#     Return a list of train data indistinguishable from test data
    xgb = XGBClassifier()
    X_adv = df_adv.drop('train_test',axis = 1)
    y_adv = df_adv['train_test'].map({'train':0,'train_extra':0, 'test':1})
    
    xgb.fit(X_adv, y_adv)
    predict_adv = pd.DataFrame(
        xgb.predict_proba(X_adv.loc[y_adv==0])[:,0], columns=['train'],
        index = X_adv.index[y_adv==0]
    )
    predict_adv.sort_values(by='train',inplace = True)
    return predict_adv.index

# Inputation
def catboostimputer(df, target_col):
    df_out = df.copy()
    for col in [column for column in df.columns if column != target_col]:
        df_nan = df.loc[df[col].isna()]
        df_exist = df.loc[~df[col].isna()]

        if len(df_nan) > 0:
            X_train = df_exist.drop(col, axis=1)
            X_test = df_nan.drop(col, axis=1)
            y_train = df_exist[col]
    
            for col_x in X_train.select_dtypes(include=object).columns:
                X_train[col_x] = X_train[col_x].astype(str)
                X_test[col_x] = X_test[col_x].astype(str)     
                
            if y_train.dtype == object:
                print(col, 'object', end=' / ')
                cat_model = CatBoostClassifier(iterations=30, depth=5, learning_rate=0.1, verbose=0)
            else:
                print(col, 'num', end=' / ')
                cat_model = CatBoostRegressor(iterations=30, depth=5, learning_rate=0.1, verbose=0)
            cat_model.fit(
                X_train, y_train,
                cat_features = list(X_train.select_dtypes(include=object).columns)
            )
            df_out.loc[df_nan.index, col] = cat_model.predict(X_test)
    return df_out


# display numerical columns and categorical columns
print(df_all.select_dtypes(exclude=[int,float]).columns)
print(df_all.select_dtypes(include=[int,float]).columns)


df_all[target_col] = df_all[target_col].astype(float)
num_cols = [
    'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6', 'hla_high_res_10',
    'hla_nmdp_6', 'year_hct', 'donor_age', 'age_at_hct','comorbidity_score',
    'karnofsky_score', 'hla_low_res_8', 'hla_low_res_10', 'efs_time','naf_label'
]
cat_cols = [
    'dri_score', 'psych_disturb', 'cyto_score', 'diabetes',
   'hla_match_c_high', 'tbi_status', 'arrhythmia', 'graft_type',
   'vent_hist', 'renal_issue', 'pulm_severe', 'prim_disease_hct',
   'cmv_status', 'hla_match_dqb1_high', 'tce_imm_match', 'hla_match_c_low',
   'rituximab', 'hla_match_drb1_low', 'hla_match_dqb1_low', 'prod_type',
   'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity',
   'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hla_match_a_high',
   'hepatic_severe', 'prior_tumor', 'hla_match_b_low', 'peptic_ulcer',
   'hla_match_a_low', 'gvhd_proph', 'rheum_issue', 'sex_match',
   'hla_match_b_high', 'race_group', 'hepatic_mild', 'tce_div_match',
   'donor_related', 'melphalan_dose', 'cardiac', 'hla_match_drb1_high',
   'pulm_moderate', 'efs']
print('Imputing start', end=' → ')
df_all_pp = catboostimputer(df_all, target_col)
print('Encoding start', end=' → ')
df_all_pp = preprocessing(
    df_all,
    num_cols, cat_cols,
    target_col
)


input_coaggregatels = list(df_all_pp.columns)
input_cols = list(df_all_pp.columns)

input_cols.remove('efs_te')
input_cols.remove('efs_time')
# input_cols.remove('naf_label')

input_cols.remove(target_col)

df_target = df_all_pp.loc[:, input_cols + [target_col]].copy()
df_target.drop(
    df_target.index[
        (df_target['train_test'].isin(['train'])) &
        (df_target[target_col].isna())
    ], axis=0, inplace = True
)

train_data = df_target.loc[
    df_target['train_test'].isin(['train'])
].drop('train_test', axis=1)
test_data =  df_target.loc[
    df_target['train_test']=='test'
].drop('train_test', axis=1)

X_train, X_val, y_train, y_val = train_test_split(
    train_data.drop(target_col, axis = 1),
    train_data[target_col], 
    test_size=0.2, random_state=42)

X_test = test_data.drop(target_col, axis=1)


def bayese_objective(X, y, Regressor, metric):
    def bayese_trial(trial):
        if Regressor == XGBRegressor:
            params = {
                'grow_policy': trial.suggest_categorical('grow_policy', ["depthwise", "lossguide"]),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                'gamma' : trial.suggest_float('gamma', 1e-9, 0.5),
                'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'max_depth': trial.suggest_int('max_depth', 0, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 100.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 100.0, log=True),

                'random_state': 42,
                'booster':'gbtree',
                'device':"cuda",
                'verbosity': 0,
                'tree_method':"hist",
                "timeout_request_budget": 180,
                'eval_metric': metrics['XGB'],

                
            }
        elif Regressor == LGBMRegressor:
            params = {
                "n_estimators": trial.suggest_int('n_estimators', 50, 1000, step=10),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "max_depth": trial.suggest_int('max_depth', 3, 15),
                "min_child_samples": trial.suggest_int('lgbm_min_child_samples', 1, 20),
                "subsample": trial.suggest_float('subsample', 0.5, 1.0),
                "colsample_bytree": trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'verbose':-1,
                
                'random_state': 42,
                "time_budget": 180,
                'verbose':-1,
                'metric': metrics['LGBM']
            }
        elif Regressor == CatBoostRegressor:
            params = {
                "iterations": trial.suggest_int('iterations', 50, 1000, step=10),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
                "depth": trial.suggest_int('depth', 3, 15),
                "l2_leaf_reg": trial.suggest_float('l2_leaf_reg', 1e-3, 1),
                
                'random_state': 42,
                "verbose": False,
                # 'time_limit':300,
                'eval_metric': metrics['CatBoost'],
            }
        
        cv = KFold(n_splits=10, shuffle=True, random_state=0)

        # X_reduced = X.sample(n_sample)
        # y_reduced = y.loc[X_reduced.index]

        cv_splits = cv.split(X, y = y)
        cv_scores = list()
        
        for train_idx, val_idx in cv_splits:
            model = Regressor()
            model.set_params(**params)
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train_fold, y_train_fold)
            
            y_val_prob = model.predict(X_val_fold)
            score = mean_squared_error(y_val_fold, y_val_prob)

            cv_scores.append(score)
        return np.mean(cv_scores)
    return bayese_trial


%%time
best_params1 = []
best_scores1 = []
_optim = False

metrics = {
    'XGB': 'rmse',
    'LGBM': 'rmse',
    'CatBoost': 'RMSE'
}
models = {
    'XGB':XGBRegressor,
    'LGBM':LGBMRegressor,
    'Cat':CatBoostRegressor
}
if _optim:# run Bayese Optimization
    for key, model in models.items():
        print(f'{key}:')
        try:
            study = optuna.create_study(
                direction = 'minimize',
                sampler=optuna.samplers.TPESampler(seed=0),
                study_name=f"{key}_study", storage=f"sqlite:///{key}_study.db", load_if_exists=True
            )
                        
            study.optimize(
                bayese_objective(X_train, y_train, model, metrics),
                n_trials=200, timeout=3600 * 1, n_jobs = -1
            )

            best_params1.append(study.best_trial.params)
            best_scores1.append(study.best_trial.value)
            print(f'{key}:')
            print('best params:')
            print(best_params1[-1])
            print('best scores:')
            print(best_scores1[-1])
        except:
            print(f'{model} failed')

else:
    best_params1 = {
        'XGB':{
            'grow_policy': 'lossguide', 'n_estimators': 889, 'learning_rate': 0.09310536004117909, 'gamma': 0.0010177904814213112, 'subsample': 0.8611230552485879, 'colsample_bytree': 0.4203181188964791, 'max_depth': 6, 'min_child_weight': 1, 'reg_lambda': 98.74006376979398, 'reg_alpha': 0.00919985915342561,
            
            'random_state': 42,
            'booster':'gbtree',
            'device':"cuda",
            'verbosity': 0,
            'tree_method':"hist",
            'eval_metric': metrics['XGB'],
        },'LGBM':{
            'n_estimators': 640, 'learning_rate': 0.041387005681257576, 'max_depth': 8, 'lgbm_min_child_samples': 6, 'subsample': 0.8282564969064214, 'colsample_bytree': 0.5120222771641108, 'num_leaves': 108,

            'random_state': 42,
            'verbose':-1,
            'metric': metrics['LGBM']
        },'CatBoost':{
            'iterations': 580, 'learning_rate': 0.02831219596642158, 'depth': 4, 'l2_leaf_reg': 0.580023654181232,
            
            'random_state': 42,
            "verbose": False,
            'eval_metric': metrics['CatBoost'],
        }}



%%time
models_best = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
predict_cols = ['XGB','LGBM', 'CatBoost']

predict_trains = []
predict_vals = []
predict_tests = []
for i,model in enumerate(models_best):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(X_train, y_train)
    predict_trains.append(model.predict(X_train).flatten())
    predict_vals.append(model.predict(X_val).flatten())
    predict_tests.append(model.predict(X_test).flatten())
print('finished')


predict_trains = pd.DataFrame(
    np.array(predict_trains).T, columns = predict_cols, index=X_train.index
)
predict_vals = pd.DataFrame(
    np.array(predict_vals).T, columns = predict_cols, index = X_val.index
)
predict_tests = pd.DataFrame(
    np.array(predict_tests).T, columns = predict_cols, index = X_test.index
)
predict_cols = predict_cols + ['blend']

predict_trains['blend'] = predict_trains.mean(axis=1)
predict_vals['blend'] = predict_vals.mean(axis=1)
predict_tests['blend'] = predict_tests.mean(axis=1)

predict_trains['True'] = y_train
predict_trains['train_val'] = 'train'
predict_vals['True'] = y_val
predict_vals['train_val'] = 'val'


target_max = max(predict_trains.max()[:-1].max(), predict_vals.max()[:-1].max())
target_max = max(target_max, df_all[target_col].max())
target_min = min(np.abs(predict_trains.min()[:-1]).min(), np.abs(predict_vals.min()[:-1]).min())
target_min = min(target_min, df_all[target_col].min())
fig, ax = plt.subplots(nrows = 3, ncols = len(models_best)+1, figsize = (12,9))
for i in range(len(models_best)+1):
    cc = np.sqrt(mean_squared_error(predict_trains['True'], np.abs(predict_trains.iloc[:,i])))
    ax[0][i].scatter(predict_trains['True'],predict_trains.iloc[:,i], s = 1, alpha=0.1)    
    ax[0][i].plot([target_min, target_max],[target_min, target_max], c='k', lw=1, ls=':')
    ax[0][i].set_title(f'{predict＿cols[i]} Train\nRMSE={cc:.5}', fontdict={'size':12})
    ax[0][i].set_xlabel('True'); ax[0][i].set_ylabel('Predict');
    ax[0][i].set_xlim(target_min, target_max); ax[0][i].set_ylim(target_min, target_max)
    ax[0][i].set_aspect(1)
    ax[0][i].grid()
    # ax[0][i].set_xscale('log'); ax[0][i].set_yscale('log');
    
    cc = np.sqrt(mean_squared_error(predict_vals['True'], np.abs(predict_vals.iloc[:,i])))
    ax[1][i].scatter(predict_vals['True'],predict_vals.iloc[:,i], s = 1, alpha=0.3)
    ax[1][i].plot([target_min, target_max],[target_min, target_max], c='k', lw=1, ls=':')
    ax[1][i].set_title(f'{predict_cols[i]} Valid \nRMSE={cc:.5}', fontdict={'size':12})
    ax[1][i].set_xlim(target_min, target_max); ax[1][i].set_ylim(target_min, target_max)
    ax[1][i].set_xlabel('True'); ax[1][i].set_ylabel('Predict');
    ax[1][i].set_aspect(1)
    ax[1][i].grid()
    # ax[1][i].set_xscale('log'); ax[1][i].set_yscale('log');
    
    fpr_train, tpr_train, _ = roc_curve(df_all.loc[predict_trains.index, 'efs'], predict_trains.iloc[:,i])
    fpr_val, tpr_val, _ = roc_curve(df_all.loc[predict_vals.index, 'efs'], predict_vals.iloc[:,i])
    c_index_train = concordance_index(df_all.loc[predict_trains.index, 'efs'], predict_trains.iloc[:,i])
    c_index_val = concordance_index(df_all.loc[predict_vals.index, 'efs'], predict_vals.iloc[:,i])

    auc_train = auc(fpr_train, tpr_train)
    auc_val = auc(fpr_val, tpr_val)
    ax[2][i].plot(fpr_train, tpr_train, c='r', label=f'train auc={c_index_train:.4f}')
    ax[2][i].plot(fpr_val, tpr_val, c='b', label = f'val auc={c_index_val:.4f}')
    ax[2][i].legend()
    ax[2][i].grid()
    ax[2][i].set_xlabel('fpr'); ax[1][i].set_ylabel('tpr');
    ax[2][i].set_aspect(1)



    
plt.tight_layout()  


# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    shap.initjs()
    model_button = widgets.ToggleButtons(
        options = predict_cols[:-1],
        button_style='info',description = 'model:'
    )
    model_button.style.button_width = f'100px'
    model_button.style.description_width = '90px'
    
    type_button = widgets.ToggleButtons(
        options = ['dot','bar'],
        button_style='warning',description = 'type:'
    )
    type_button.style.button_width = f'100px'
    type_button.style.description_width = '90px'
    
    max_disp_slider = widgets.IntSlider(
        value=min(6,len(df_train.columns)), min=0, max=len(X_train.columns), step=1, 
        description='max_display:', orientation='horizontal'
    )
    max_disp_slider.style.button_width = f'100px'
    max_disp_slider.style.description_width = '90px'
    
    
    @interact(model_name = model_button, plot_type = type_button, max_display = max_disp_slider)
    def plot_re(model_name, plot_type, max_display):
        df_train = X_train.sample(500)
        i = list(predict_cols).index(model_name)
    
        model = models_best[i]
            
        explainer = shap.TreeExplainer(model=model, model_output='raw')
        shap_values = explainer.shap_values(X=df_train)
        shap.summary_plot(shap_values, df_train, plot_type=plot_type, max_display=max_display)
else:
    print('Run in interactive mode to display plots.')


models_final = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
predict_tests = []
for i,model in enumerate(models_final):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(train_data.drop(target_col,axis=1), train_data[target_col])
    predict_tests.append(model.predict(X_test).flatten())

predict_tests = pd.DataFrame(
    np.array(predict_tests).T, columns = predict_cols[:-1], index = X_test.index
)

predict_tests['blend'] = predict_tests.mean(axis=1)


y_test_predict = predict_tests['blend']
# y_test_predict = df_all['prediction']
df_submit = df_sample_submission.copy()
df_submit['prediction'] = y_test_predict
df_submit.to_csv('submission.csv', index = True)

display(df_submit)




