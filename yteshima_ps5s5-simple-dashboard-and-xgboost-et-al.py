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

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import HTML, display, clear_output

import matplotlib.pyplot as plt


from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.decomposition import PCA

from matplotlib.colors import LinearSegmentedColormap
get_ipython().run_line_magic('matplotlib', 'inline')

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap', ['blue', 'white', 'red']
)

import random

import shap


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping


# ignore wornings
import warnings
warnings.filterwarnings("ignore")

# Get the execution mode of the Kaggle environment
run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'Interactive')


# Load the training and test data from CSV files

df_sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv',index_col = 0)
df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col=0)

print(f'N_train = {len(df_train)}, N_test = {len(df_test)}')

# Assign a new column 'train_test' with the value 'train' to the training dataset, and 'test' to the test dataset respectively
df_train['train_test'] = 'train'
df_test['train_test'] = 'test'

# Create reduced versions of the training and test datasets by randomly sampling 1/20th of the rows
df_train_reduced = df_train.sample(len(df_train) // 5)
df_test_reduced = df_test.sample(len(df_test) // 5)
print(f'N_train_reduced = {len(df_train_reduced)}, N_test_reduced = {len(df_test_reduced)}')

# Specify the target column for the analysis or model
target_col = df_train.columns[-2]

# Combine the training and test datasets into a single DataFrame for unified processing
df_all = pd.concat([
    df_train,
    df_test,
    # Uncomment the lines below to include the reduced datasets if needed
    # df_train_reduced,
    # df_test_reduced
])



df_all_pp = df_all.copy()
target_col = target_col.replace('_log','')
df_all_pp['Body_Temp^2'] = (df_all_pp['Body_Temp'] - 37 ) ** 2
df_all_pp['Body_Temp^3'] = (df_all_pp['Body_Temp'] - 37 ) ** 3
df_all_pp['Body_Temp^4'] = (df_all_pp['Body_Temp'] - 37 ) ** 4

pca_cols = ['Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp^4']

pca = PCA()

df_pca = pd.DataFrame(pca.fit_transform(df_all_pp[pca_cols]), columns = [f'pca_{i}' for i in range(len(pca_cols))], index = df_all.index)

df_all_pp = pd.concat([df_all_pp, df_pca], axis = 1)

df_all_pp[f'{target_col}_log'] = np.log(df_all_pp[target_col] + 1)
target_col = target_col + '_log'


# Define a custom format function to format float values for display
def custom_format(x):
    if isinstance(x, float):
        # Format float values to 3 decimal places and remove trailing zeros and decimal points
        return ('{0:.3f}'.format(x)).rstrip('0').rstrip('.')
    return x

# Display detailed information about the DataFrame, including statistics and metadata
def display_dfinfo(df):
    display(HTML('<br><h2>Display head of the dataframe</h2>'))
    display(df.sample(3))
    display(HTML('<br><h2>Display numerical data infomations</h2>'))
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
    
    if len([col for col in df.select_dtypes(include='object') if col != 'train_test']) > 0:
        display(HTML('<br><h2>Display categorical data infomations</h2>'))
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
            width=max(min(len(num_cols) * 100, 600), 400),
            height=max(min(len(num_cols) * 80, 600), 400),
            title='Correlation matrix'
        )
        fig.show()
    elif plottype == 'sns':
        plt.figure(figsize=(min(len(df.columns) * 0.6, 12), min(len(df.columns) * 0.25, 12)))
        sns.heatmap(
            df[num_cols].corr(), annot=True, vmax=1, vmin=-1,
            cmap=custom_cmap, fmt='.2f'  # Use a custom color map and format values
        )
        plt.title('Correlation Matrix')
        plt.show()



num_cols = df_all_pp.select_dtypes(include=[float, int]).columns
disp_cols = df_all_pp.columns
cat_cols = [c for c in df_all_pp.columns if c!= 'date']
display_dfinfo(df_all_pp)
plot_correlation_matrix(df_all_pp, num_cols,  plottype='sns')


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
            hover_data = {'index':df_tmp.index}
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
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
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
            plot_height = min(1200, 120 * df_tmp[num_col2].nunique())
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
def plot_interact(df_in, disp_cols, n_int_thresh = 10, n_data=np.nan, plottype='sns'):
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
        options = col_dict,
        value = target_col,
        button_style='warning',
        description='color'
    )
    cat_col_button.style.button_width = '80px'
    cat_col_button.style.description_width = '50px'
    
    # Create an interactive function to update the plot based on user selections
    @interact(
        num_col1=num_col_button1, num_col2=num_col_button2,
        cat_col=cat_col_button
    )
    def plot_df(num_col1, num_col2, cat_col):
        clear_output()  # Clear the output to avoid clutter
        
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


# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    disp_cols = df_all.columns
    plot_interact(df_all_pp, disp_cols, n_data = 5000, plottype='plotly')
else:
    print('Run in interactive mode to display plots.')
    df_plot = df_all_pp.copy()
    scatter_df(df_plot, 'Age', target_col, 'Sex', n_data=5000, plottype='plotly')


def target_encoder(df, input_col, target_col):
    tmp = df[[input_col, target_col]]
    means = df.groupby(input_col)[target_col].mean()
    for ind in means.index:
        tmp.loc[tmp[f'{input_col}']==ind, f'{input_col}_te'] = means[ind]

    return tmp[f'{input_col}_te'].values

def impute_df(df, impute_cols):
    df_imputed = df.copy()

    for col in impute_cols:
        print(col, end = ', ')
        impute_model = LGBMRegressor(verbose=-1)
        x = df.dropna(subset=col).drop([col], axis = 1)
        y = df.dropna(subset=col)[[col]]
        impute_model.fit(x,y)
        df_imputed.loc[df_imputed[col].isna(), col] = df_imputed.loc[df_imputed[col].isna(),:].drop(col, axis = 1)
    return df_imputed

def preprocessing(df, num_cols, cat_cols, target_col,train_test = 'train_test'):    
    df_pp = df[num_cols].copy()
    for i, cat_col in enumerate(cat_cols):
        print(cat_col, end = ' / ')
        # target encoding
        df_pp[f'{cat_col}_te'] = target_encoder(df, cat_col, target_col)

    impute_cols = df_pp.columns[df_pp.isna().sum() > 0]
    df_pp = impute_df(df_pp, impute_cols)


    df_pp[target_col] = df[target_col]
    df_pp[train_test] = df[train_test]
    return df_pp

def adversarial_validation(df_adv):
#     Return a list of train data indistinguishable from test data
    xgb = XGBClassifier()
    X_adv = df_adv.drop('train_test',axis = 1)
    y_adv = df_adv['train_test'].map({'train':0,'train_extra':0, 'original':0, 'test':1})
    xgb.fit(X_adv, y_adv)
    predict_adv = pd.DataFrame(
        xgb.predict_proba(X_adv.loc[y_adv==0])[:,0], columns=['train'],
        index = X_adv.index[y_adv==0]
    )
    predict_adv.sort_values(by='train',inplace = True)
    return predict_adv.index


print(df_all_pp.select_dtypes(exclude=[int,float]).columns)
print(df_all_pp.select_dtypes(include=[int,float]).columns)


df_all_pp[target_col] = df_all_pp[target_col].astype(float)

cat_cols = ['Sex']
num_cols = [
    'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
    'pca_0','pca_1', 'pca_2', 'pca_3', 'pca_4']
print('Preprocessing start', end=' → ')
# df_all_pp = catboostimputer(df_all, target_col)
df_all_pp = preprocessing(
    df_all_pp,
    num_cols, cat_cols,
    target_col
)


df_all_pp[target_col]=df_all_pp[target_col].astype(float)

input_coaggregatels = list(df_all_pp.columns)
input_cols = list(df_all_pp.columns)
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

valid_indices = adversarial_validation(df_target.drop(target_col, axis = 1))
valid_indices = valid_indices[:round(len(valid_indices)*0.2)]

X_train = train_data.drop(target_col,axis=1).drop(valid_indices)
X_val = train_data.drop(target_col,axis=1).loc[valid_indices]
X_test = test_data.drop(target_col, axis = 1)

y_train = train_data[target_col].drop(valid_indices)
y_val = train_data[target_col].loc[valid_indices]

print(f'X_train.shape:{X_train.shape}, X_val.shape:{X_val.shape}, X_test.shape:{X_test.shape}')
print(f'y_train.shape:{y_train.shape}, y_val.shape:{y_val.shape}')


def bayese_objective(X, y, Classifier, metric, n_sample = np.nan):
    def bayese_trial(trial):
        if Classifier in [XGBClassifier, XGBRegressor]:
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
        elif Classifier in [LGBMClassifier, LGBMRegressor]:
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
        elif Classifier in [CatBoostClassifier, CatBoostRegressor]:
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
        
        cv = KFold(n_splits=20, shuffle=True, random_state=0)

        if np.isnan(n_sample):
            X_reduced = X.copy()
            y_reduced = y.copy()
        else:
            X_reduced = X.sample(n_sample)
            y_reduced = y.loc[X_reduced.index]

        cv_splits = cv.split(X_reduced, y = y_reduced)
        cv_scores = list()
        
        for train_idx, val_idx in cv_splits:
            model = Classifier()
            model.set_params(**params)
            X_train_fold, X_val_fold = X_reduced.iloc[train_idx], X_reduced.iloc[val_idx]
            y_train_fold, y_val_fold = y_reduced.iloc[train_idx], y_reduced.iloc[val_idx]
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
        print(f'{key} {model}:')
        try:
            study = optuna.create_study(
                direction = 'minimize',
                sampler=optuna.samplers.TPESampler(seed=0),
                study_name=f"{key}_study", storage=f"sqlite:///{key}_study.db", load_if_exists=True
            )
                        
            study.optimize(
                bayese_objective(X_train, y_train, model, metrics, n_sample = np.nan),
                n_trials=1000, timeout=3600 * 5, n_jobs = -1
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
            'grow_policy': 'depthwise', 'n_estimators': 500, 'learning_rate': 0.01715593354307259, 'gamma': 0.0022055593919613897, 'subsample': 0.7033884210642536, 'colsample_bytree': 0.8001731059025385, 'max_depth': 11, 'min_child_weight': 1, 'reg_lambda': 22.110975795697072, 'reg_alpha': 0.009824212444424779,
            
            'random_state': 42,
            'booster':'gbtree',
            'device':"cuda",
            'verbosity': 0,
            'tree_method':"hist",
            'eval_metric': metrics['XGB'],
        },'LGBM':{
            'n_estimators': 920, 'learning_rate': 0.013456127113316898, 'max_depth': 9, 'lgbm_min_child_samples': 13, 'subsample': 0.7421470990629568, 'colsample_bytree': 0.7349380726883565, 'num_leaves': 177,
            
            'random_state': 42,
            'verbose':-1,
            'metric': metrics['LGBM']
        },'CatBoost':{
            'iterations': 900, 'learning_rate': 0.015915191518969852, 'depth': 11, 'l2_leaf_reg': 0.8696274225723688,
            
            'random_state': 42,
            "verbose": False,
            'eval_metric': metrics['CatBoost'],
        }}



%%time
models_best = [
    XGBRegressor(),
    LGBMRegressor(),
    CatBoostRegressor()
]
predict_trains = pd.DataFrame(index = X_train.index)
predict_vals = pd.DataFrame(index = X_val.index)
predict_tests = pd.DataFrame(X_test.index)

predict_cols = ['XGB', 'LGBM', 'CatBoost']

for i,model in enumerate(models_best):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(X_train, y_train)
    predict_trains[predict_cols[i]] = np.exp(model.predict(X_train))
    predict_vals[predict_cols[i]] = np.exp(model.predict(X_val))
    predict_tests[predict_cols[i]] = np.exp(model.predict(X_test))
print('finished')


# ss_input = StandardScaler()
# train_data_input_ss = ss_input.fit_transform(train_data.drop(target_col,axis=1))
# train_data_input_ss = pd.DataFrame(
#     train_data_input_ss,
#     index = train_data.index,
#     columns = train_data.drop(target_col, axis = 1).columns
# )
# test_data_input_ss = ss_input.transform(test_data.drop(target_col,axis=1))
# test_data_input_ss = pd.DataFrame(
#     test_data_input_ss,
#     index = test_data.index,
#     columns = test_data.drop(target_col,axis=1).columns
# )


# X_train_ss = train_data_input_ss.drop(valid_indices)
# X_val_ss = train_data_input_ss.loc[valid_indices]
# X_test_ss = test_data_input_ss


# early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# # Initialize Neural Network
# model = Sequential([
#     Dense(
#         128, activation='relu', kernel_initializer='he_normal',
#         input_shape=(X_train_ss.shape[1],)
#     ),
#     Dropout(0.3),
#     Dense(
#         64, activation='relu', kernel_initializer='he_normal'
#     ),
#     Dropout(0.2),
#     Dense(
#         32,activation='relu', kernel_initializer='he_normal'
#     ),
#     Dropout(0.2),
#     Dense(
#         16,activation='relu', kernel_initializer='he_normal'
#     ),
#     Dense(1, activation='linear')  # Binary classification
# ])

# # Compile Model
# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

# # Train Model
# history = model.fit(
#     X_train_ss, y_train,
#     epochs=500, batch_size=32, validation_split=0.2,            
#     callbacks=[early_stopping], verbose=0)

# # Make Predictions
# predict_trains['keras'] = np.exp(model.predict(X_train_ss).flatten())
# predict_vals['keras'] = np.exp(model.predict(X_val_ss).flatten())

# y_test_pred_keras = np.exp(model.predict(X_test_ss).flatten())


predict_trains['blend'] = predict_trains.mean(axis=1)
predict_vals['blend'] = predict_vals.mean(axis=1)
predict_tests['blend'] = predict_tests.mean(axis=1)

predict_trains['True'] = np.exp(y_train)
predict_vals['True'] = np.exp(y_val)
predict_vals['train_val'] = 'val'


predict_cols = predict_trains.columns[:-2]
target_max = max(predict_trains.max()[:-1].max(), predict_vals.max()[:-1].max())
target_max = np.exp(max(target_max, df_all_pp[target_col].max()))
target_min = min(np.abs(predict_trains.min()[:-1]).min(), np.abs(predict_vals.min()[:-1]).min())
target_min = np.exp(min(target_min, df_all_pp[target_col].min()))
fig, ax = plt.subplots(nrows = 1, ncols = len(predict_cols)+1, figsize = ((len(predict_cols)+1)*3,4))
for i, model in enumerate(predict_trains.columns[:-1]):
    index_sample = random.sample(list(predict_trains.index), min(len(predict_trains), 2000))
    score_train = np.sqrt(mean_squared_log_error(predict_trains['True'].loc[index_sample], predict_trains[model].loc[index_sample]))

    index_sample = random.sample(list(predict_vals.index), min(len(predict_vals), 2000))
    score_val = np.sqrt(mean_squared_log_error(predict_vals['True'].loc[index_sample], predict_vals[model].loc[index_sample]))
    ax[i].scatter(predict_trains['True'],predict_trains[model], c='r', label=f'train rmsle={score_train:.4f}', s=2, alpha=0.2)
    ax[i].scatter(predict_vals['True'],predict_vals[model], c='b', label = f'val rmsle={score_val:.4f}',s = 2, alpha= 0.2)
    ax[i].legend()
    ax[i].grid()
    ax[i].set_title(model)
    ax[i].set_xlabel('true')
    ax[i].set_ylabel('predict');
    ax[i].set_aspect(1)
    
plt.tight_layout()  


# ipywidgets　will run only in interactive mode
if run_type == 'Interactive':
    shap.initjs()
    model_button = widgets.ToggleButtons(
        options = ['XGB', 'LGBM', 'CatBoost'],
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
        df_train = X_train.sample(1000)
        i = list(predict_cols).index(model_name)
    
        model = models_best[i]
            
        explainer = shap.TreeExplainer(model=model, model_output='raw')
        shap_values = explainer.shap_values(X=df_train)
        shap.summary_plot(shap_values, df_train, plot_type=plot_type, max_display=max_display)
else:
    print('Run in interactive mode to display plots.')
    df_train = X_train.sample(1000)
    model = models_best[0]
    explainer = shap.TreeExplainer(model=model, model_output='raw')
    shap_values = explainer.shap_values(X=df_train)
    shap.summary_plot(shap_values, df_train, plot_type='dot', max_display=7)



models_final = [XGBRegressor(), LGBMRegressor(), CatBoostRegressor()]
predict_tests = []
for i,model in enumerate(models_final):
    print(predict_cols[i], end = ' / ')
    model.set_params(**best_params1[predict_cols[i]])
    model.fit(train_data.drop(target_col,axis=1), train_data[target_col])
    predict_tests.append(model.predict(X_test))

predict_tests = pd.DataFrame(
    np.array(predict_tests).T, columns = ['XGB', 'LGBM', 'CatBoost'], index = X_test.index
)

# predict_tests['keras'] = y_test_pred_keras

predict_tests['blend'] = predict_tests['XGB'] * 0.5 + predict_tests['LGBM'] * 0.25 + predict_tests['CatBoost'] * 0.25 # + predict_tests['keras'] * 0.0


target_col = target_col.replace('_log', '')
y_test_predict = np.exp(predict_tests['blend'])-1
# y_test_predict = df_all['prediction']
df_submit = df_sample_submission.copy()
df_submit[target_col] = y_test_predict
df_submit.to_csv('submission.csv', index = True)
print(df_submit.isna().sum())
display(df_submit)




