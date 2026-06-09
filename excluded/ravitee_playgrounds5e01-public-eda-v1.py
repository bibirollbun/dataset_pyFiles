%%time
#EDA Reference - https://www.kaggle.com/code/khsamaha/simple-lgbm-shap-kaggle-sticker-sales-py

# Suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

#use IPython's display package to display HTML content instead of plain text in this notebook
from IPython.display import display_html, clear_output;


import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', 50);
pd.set_option('display.max_rows', 50);

#Provides the ability to add colored output
from colorama import Fore, Style

import ctypes, gc
from os import getpid
from psutil import Process

def CleanMemory():
    gc.collect()
    # libc = ctypes.CDLL("libc.so.6")
    
    # libc.malloc_trim(0)
    pid        = getpid()
    py         = Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    print(f"\nRAM usage = {memory_use :.4} GB") 
    
clear_output()
CleanMemory()


%%time 

# Setting rc parameters in seaborn for plots and graphs- 
# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-
# To alter this, refer to matplotlib.rcParams.keys()
import matplotlib.pyplot as plt

sns.set({"axes.facecolor"       : "#ffffff",
         "figure.facecolor"     : "#ffffff",
         "axes.edgecolor"       : "#000000",
         "grid.color"           : "#ffffff",
         "font.family"          : ['Cambria'],
         "axes.labelcolor"      : "#000000",
         "xtick.color"          : "#000000",
         "ytick.color"          : "#000000",
         "grid.linewidth"       : 0.75,  
         "grid.linestyle"       : "--",
         "axes.titlecolor"      : '#0099e6',
         'axes.titlesize'       : 8.5,
         'axes.labelweight'     : "bold",
         'legend.fontsize'      : 7.0,
         'legend.title_fontsize': 7.0,
         'font.size'            : 7.5,
         'xtick.labelsize'      : 7.5,
         'ytick.labelsize'      : 7.5,
        });


# Color printing    
def PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    "Prints color outputs using colorama using a text F-string";
    print(style + color + text + Style.RESET_ALL); 


CleanMemory();


%%time

import plotly.express as px

from IPython.display import display, HTML
def styled_heading(text):
    return f"""
    <p style="
        font-family: 'Arial'; 
        font-size: 3rem; 
        color: black; 
        text-align: center; 
        margin: 0; 
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); 
        background-color: #73B1F4; 
        padding: 2px; 
        border-radius: 20px; 
        border: 7px dark slate gray; 
        width:95%">
        {text}
    </p>
    """
def styled_heading2(text):
    return f"""
    <p style="
        font-family: 'Arial'; 
        font-size: 2rem; 
        color: black; 
        text-align: center; 
        margin: 0; 
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); 
        background-color: #8e82e5; 
        padding: 2px; 
        border-radius: 15px; 
        border: 7px dark slate gray; 
        width:85%">
        {text}
    </p>
    """

# Helper function to generate colored horizontal line
def colored_line(color='#323c6a'):
    return ""

def print_error(err_msg):
    display(HTML(styled_heading("Error!!!!!!!")))
    print(f"An error occurred: {err_msg}")

def print_rows(df, title, n_top):
    #print top n values from train and test set
    heading = styled_heading(f"{title}")
    display(HTML(colored_line()))
    display(HTML(heading))
    display(HTML(df.head(n_top).to_html()))
    display(HTML(colored_line()))
    

def print_top_n_rows(df, train_or_test_ds, n_top):
    #print top n values from train and test set
    heading = styled_heading(f"Top {n_top} rows of {train_or_test_ds} dataset")
    display(HTML(colored_line()))
    display(HTML(heading))
    display(HTML(df.head(n_top).to_html()))
    display(HTML(colored_line()))
    # display(HTML(f"Shape of {train_or_test_ds} is {df.shape}"))
    PrintColor(f"Shape of {train_or_test_ds} is {df.shape}");

    
def print_gen_info(df):
    heading = styled_heading(f"General Information of train dataset")
    display(HTML(colored_line()))
    display(HTML(heading))
    display(HTML(df.info()))
    display(HTML(colored_line()))


def print_summary(df):
    heading = styled_heading(f"Statistical description of train dataset")
    display(HTML(colored_line()))
    display(HTML(heading))
    #create a bar chart for the mean column
    #set the background for 'std' column
    #set the background for 50% column
    # display(HTML(df.describe().T\
    #                 .round(2).style.format(precision=2)\
    #                 .style.bar(subset=['mean'], color=px.colors.qualitative.G10[2])\
    #                 .background_gradient(subset=['std'], cmap='Blues')\
    #                 .background_gradient(subset=['50%'], cmap='BuGn')\
    #                 .to_html()))
    display(HTML(df.describe().round(2).style.format(precision=2).background_gradient(cmap="Blues").to_html()))
    display(HTML(colored_line()))

def print_null_values(df, train_or_test_ds):
    heading = styled_heading(f"Null values in {train_or_test_ds} dataset")
    display(HTML(colored_line()))
    display(HTML(heading))

    null_count = df.isnull().sum()

    if null_count.sum() == 0:
        display(HTML(f"<h3><p>No null values in the {train_or_test_ds} dataset.</p></h3>"))
    else:
        display(HTML(null_count[null_count > 0].to_frame().to_html()))
    display(HTML(colored_line()))    

def print_duplicate_values(df, train_or_test_ds):
    heading = styled_heading(f"Duplicate values in {train_or_test_ds} dataset")
    display(HTML(colored_line()))
    display(HTML(heading))

    duplicate_count = df.duplicated().sum()

    if duplicate_count.sum() == 0:
        display(HTML(f"<h3><p>No duplicate values in the {train_or_test_ds} dataset.</p></h3>"))
    else:
        display(HTML(duplicate_count[duplicate_count > 0].to_frame().to_html()))
    display(HTML(colored_line()))
    

def print_unique_values(df):
    try:
        heading = styled_heading("Unique Values in Training Dataset")
        
        display(HTML(colored_line()))
        display(HTML(heading))
        display(HTML(colored_line()))
        
        unique_values_table = "<table border='1'><tr><th>Column Name</th><th>Data Type</th><th># of Unique Values</th><th>Top 7 Unique Values</th></tr>"
        
        for column in df.columns:
            tot_num_of_unique_values = df[column].nunique()
            unique_values = df[column].unique()[:7]  # Taking at least 7 unique values
            unique_values_str = ', '.join(map(str, unique_values))
            data_type = df[column].dtype
            unique_values_table += f"<tr><td>{column}</td><td>{data_type}</td><td>{tot_num_of_unique_values}</td><td>{unique_values_str}</td></tr>"
        
        unique_values_table += "</table>"
        display(HTML(unique_values_table))
    
    except Exception as e:
        print_error(str(e))
        
def print_training_and_target_columns(train_df, test_df,):
    target = [f for f in train_df.columns if f not in test_df.columns][0]
    original_features = list(test_df.columns)
    
    try:
        heading = styled_heading("Features available for training")
        
        display(HTML(colored_line()))
        display(HTML(heading))
        display(HTML(colored_line()))
        
        unique_values_table = "<table border='1'><tr><th>Column Name</th><th>Data Type</th><th>Unique Values</th></tr>"
        
        for column in df.columns:
            unique_values = df[column].unique()[:7]  # Taking at least 7 unique values
            unique_values_str = ', '.join(map(str, unique_values))
            data_type = df[column].dtype
            unique_values_table += f"<tr><td>{column}</td><td>{data_type}</td><td>{unique_values_str}</td></tr>"
        
        unique_values_table += "</table>"
        display(HTML(unique_values_table))
    
    except Exception as e:
        print_error(str(e))
        
#print the counts of categorical variables to get an idea about the 
def print_value_counts_less_than(df, threshold=50):
    try:
        heading = styled_heading("Value counts in training dataset")
        
        display(HTML(colored_line()))
        display(HTML(heading))
        display(HTML(colored_line()))
        
        value_counts_table = "<table border='1'><tr><th>Column Name</th><th>Data Type</th><th>Count</th></tr>"
        
        for column in df.columns:
            value_counts = df[column].value_counts()
            data_type = df[column].dtype
            if len(value_counts) < threshold: 
                value_counts_table += f"<tr><td>{column}</td><td>{data_type}</td><td>"
                for val, count in value_counts.items():
                    value_counts_table += (f"    {val}: {count}<br>")
                    
                value_counts_table += "</td></tr>"
        
        value_counts_table += "</table>"
        display(HTML(value_counts_table))
    
    except Exception as e:
        print_error(str(e))


def print_initial_analysis(train_df, test_df, n_top=5):
    try:
        #print top n values from train and test set
        print_top_n_rows(train_df, 'train', n_top)
        print_top_n_rows(test_df, 'test', n_top)
        
        #print general information of test dataset
        print_gen_info(train_df)
    
        #print test dataset summary
        print_summary(train_df)
        
        #print null values in train and test set
        print_null_values(train_df, 'train')
        print_null_values(test_df, 'test')
        
        #print duplicate values in train and test set
        print_duplicate_values(train_df, 'train')
        print_duplicate_values(test_df, 'test')
        
        #print unique values in train set
        print_unique_values(train_df)
        
        # print distribution of categorical features
        print_value_counts_less_than(train_df, threshold=30)

        
    except Exception as e:
        print_error(str(e))


%%time 

# Configuration class:-
class CFG:
    """
    Configuration class for parameters and CV strategy for tuning and training
    Some parameters may be unused here as this is a general configuration class
    """;
    
    # Data preparation:-   
    n_version_nb         = 1; 
    n_season             = 5;
    n_episode            = 1;
    path               = f"/kaggle/input/playground-series-s{n_season}e{n_episode}";
    target             = 'num_sold';
    n_state              = 42;
    n_splits             = 5; #number of splits to create during cross validation
    init_analysis_reqd   = True;  #whether initial analysis required (True/False). If True execute function 'print_initial_analysis(df_train, df_test)'
    fe_holidays          = False; #should holidays be feature engineered
    
    dtl_preproc_req    = "Y";
    

print();
PrintColor(f"--> Configuration done!\n");
CleanMemory();


%%time
import os

train_df = pd.read_csv(os.path.join(CFG.path, 'train.csv'), index_col='id', parse_dates=['date'])

if CFG.init_analysis_reqd == True:
    test_df = pd.read_csv(os.path.join(CFG.path, 'test.csv'), index_col='id', parse_dates=['date'])
    
    print_initial_analysis(train_df, test_df)

CleanMemory()


%%time

plt.figure(figsize=(12,4))
ax =sns.lineplot(data=train_df, x='date', y=CFG.target, linewidth=0.4)

ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel('Numbers sold', fontsize=10)

plt.title('Numbers sold every year', size=11)
plt.show()
CleanMemory()


%%time

plt.figure(figsize=(12,4))
ax =sns.lineplot(data=train_df, x='date', y=CFG.target, linewidth=0.4, hue='country', palette ='Dark2')

ax.set_xlabel('Year')
ax.set_ylabel('Numbers sold')

plt.title('Numbers sold every year for each country')
plt.show()
CleanMemory()


%%time

plt.figure(figsize=(12,4))
ax =sns.lineplot(data=train_df, x='date', y=CFG.target, linewidth=0.4, hue='store')

ax.set_xlabel('Year')
ax.set_ylabel('Numbers sold')

plt.title('Numbers sold every year from each store')
plt.show()
CleanMemory()


%%time

plt.figure(figsize=(12,4))
ax =sns.lineplot(data=train_df, x='date', y=CFG.target, linewidth=0.4, hue='product')

ax.set_xlabel('Year')
ax.set_ylabel('Numbers sold')

plt.title('Products sold every year')
plt.show()
CleanMemory()


df = train_df.copy()
df['year'] = df['date'].dt.year.astype("int")
#select a subset of the dataframe containing year and num_sold
#group that by year and take the mean of num_sold
avg_sales_per_year = df[['year', 'country', 'num_sold']]\
                    .groupby(['year', 'country'])['num_sold'].mean()\
                    .reset_index()

display(avg_sales_per_year.round(2).style.format(precision=2).background_gradient(cmap="Blues"))

del df
CleanMemory()


plt.figure(figsize=(9,4))

ax = sns.barplot(data=avg_sales_per_year, y='num_sold', x='year', hue='country')
ax.set_xlabel('Year')
ax.set_ylabel('Average Sales')

plt.title('Average Sales Per Year Per Country')
plt.show()


mean_num_sold = train_df['num_sold'].mean().round(2).squeeze()
plt.figure(figsize=(10,4))

ax = sns.histplot(train_df["num_sold"], kde=True)
ax.axvline(x=mean_num_sold, color="darkred", ls="--", lw=1.5)
ax.text(mean_num_sold+50, ax.get_ylim()[1] * 0.7, f'Mean: {mean_num_sold:.2f}', \
        color="darkred", fontsize=10)
ax.set_xlabel('Sales')
ax.set_ylabel('Count')

plt.title("Distribution of Sales")
plt.show()


def plot_num_transformation(nan_handled_data, global_title):
    df=nan_handled_data.copy()

    #log transform to reduce the impact of large values
    # log1p is used instead of np.log to handle zeros by computing log(1+num_sold)
    df['log_num_sold'] = np.log1p(df['num_sold'])

    # Square root transformation to reduce positive skewness. Less aggressive than log transform
    df['sqrt_num_sold'] = np.sqrt(df['num_sold'])

    # Box Cox transformation to handle wider range of data distributions. Requires all values to be positive.
    # +1 ensures all values are positive
    # df['boxcox_num_sold'] = stats.boxcox(df['num_sold']+1)

    # Yeo-Johnson transformation to handle wider range of data distributions along with zero and negative values. 
    pt = PowerTransformer(method="yeo-johnson")
    df['yeojohnson_num_sold'] = pt.fit_transform(df[['num_sold']])

    # Original variable 
    plt.figure(figsize=(14, 6)) 
    plt.subplot(2, 2, 1) 

    sns.histplot(df['num_sold'], kde=True) 
    plt.title('Original') 

    # Log transformation 
    plt.subplot(2, 2, 2) 
    sns.histplot(df['log_num_sold'], kde=True) 
    plt.title('Log Transformation') 

    # Square root transformation 
    plt.subplot(2, 2, 3) 
    sns.histplot(df['sqrt_num_sold'], kde=True) 
    plt.title('Square Root Transformation') 

    # Yeo-Johnson transformation 
    plt.subplot(2, 2, 4) 
    sns.histplot(df['yeojohnson_num_sold'], kde=True) 
    plt.title('Yeo-Johnson Transformation') 

    # Box-Cox transformation 
    # plt.subplot(2, 2, 4) 
    # sns.histplot(df['boxcox_num_sold'], kde=True) 
    # plt.title('Box-Cox Transformation') 

    #add a global title
    plt.suptitle(f"{global_title}", fontsize=16, color = "#006bb3")
    plt.tight_layout() 
    plt.show()

    del df


%%time
import numpy as np
from sklearn.preprocessing import PowerTransformer

def handle_nan_and_plot(nan_strategy, df, plot_function):
    data = df.copy()

    if nan_strategy == 'drop':
        data = data.dropna(subset='num_sold').reset_index(drop=True)
        title = "Comparison of different transformations on target after dropping NaN"
    elif nan_strategy == 'fill_mean':
        mean_num_sold = data['num_sold'].mean().round(2) 
        data['num_sold'] = data['num_sold'].fillna(mean_num_sold) 
        title = "Comparison of different transformations on target - NaN Mean Handling"
    elif nan_strategy == 'interpolate':
        data['num_sold'] = data['num_sold'].interpolate()
        title = "Comparison of different transformations on target - NaN Interpolate"
    elif nan_strategy == 'ffill':
        data['num_sold'] = data['num_sold'].ffill()
        title = "Comparison of different transformations on target - NaN ffill"
    elif nan_strategy == 'bfill':
        data['num_sold'] = data['num_sold'].bfill()
        title = "Comparison of different transformations on target - NaN bfill"

    plot_function(nan_handled_data=data, global_title=title)

#list of strategies to apply

strategies = ['drop', 'fill_mean', 'interpolate', 'ffill', 'bfill']

# Function to apply all strategies recursively 
def apply_strategies_recursively(df, strategies, plot_function):
    for strategy in strategies:
        handle_nan_and_plot(nan_strategy=strategy, df=df, plot_function=plot_function)

# Call the function to apply all strategies
apply_strategies_recursively(df=train_df, strategies=strategies, plot_function=plot_num_transformation)

CleanMemory()


%%time
from scipy import stats

def evaluate_num_transformation(data):
    # Ensure positive values for Box-Cox
    data_bc, _ = stats.boxcox(data + 1)
    data_yeo = stats.yeojohnson(data)[0]
    data_log = np.log1p(data)
    data_sqrt = np.sqrt(data)
    
    # Create a dictionary to store results
    results = {
        "Original": data,
        "Log": data_log,
        "Square Root": data_sqrt,
        "Box-Cox": data_bc,
        "Yeo-Johnson": data_yeo
    }
    
    # Evaluate skewness and kurtosis
    for key, value in results.items():
        print(f"{key} - Skewness: {stats.skew(value)}, Kurtosis: {stats.kurtosis(value, fisher=False)}")
        
        # Kolmogorov-Smirnov Test
        stat, p_value = stats.kstest(value, 'norm')
        print(f"{key} - Kolmogorov-Smirnov Test: Statistics={stat}, p-value={p_value}")
        print()

# Apply the evaluation
evaluate_num_transformation(train_df['num_sold'].dropna())

CleanMemory()


import holidays

def transform_date(df):
    ## Add Holidays
    extract_country = dict(
        zip(np.sort(df.country.unique()), ["CA", "FI", "IT", "KE", "NO", "SG"]))
    holidays_dict = {
        c: holidays.country_holidays(a, years=range(2010, 2020))
        for c, a in extract_country.items()
    }
    df["is_holiday"] = 0
    for c in holidays_dict:
        df.loc[df.country == c, "is_holiday"] = df.date.isin(holidays_dict[c]).astype(int)

    #week day name such as Mon, Tue
    df["weekday"] = df["date"].dt.strftime("%a").astype("category")
    #week day number such as 0 for Sunday
    df["weekday_num"] = df["date"].dt.strftime("%w").astype("category")
    #day of the month 
    df["day_of_month"] = df["date"].dt.strftime("%d").astype("category")
    #name of the month such as Jan, Feb
    df["month_name"] = df["date"].dt.strftime("%b").astype("category")
    #number of the month such as 01 for Jan
    df["month_num"] = df["date"].dt.strftime("%m").astype("int")
    #year
    df["year"] = df["date"].dt.strftime("%Y").astype("int")
    #day of the year
    df["day_number_year"] = df["date"].dt.strftime("%j").astype("int")
    #Week number of the year. Monday as the first day of the week
    df["week_number_year"] = df["date"].dt.strftime("%W").astype("category")
    df["country"] = df["country"].astype("category")
    df["store"] = df["store"].astype("category")
    df["product"] = df["product"].astype("category")
    df["is_holiday"] = df["is_holiday"].astype("category")

    #sine and cosine transformations help encode cyclical features. This helps capture the start and end points of the cycle
    #For instance, both December and January are close in time.
    df["year_sin"] = np.sin(2 * np.pi * df["year"])
    df["year_cos"] = np.cos(2 * np.pi * df["year"])

    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df["day_number_year_sin"] = np.sin(2 * np.pi * df["day_number_year"] / 12)
    df["day_number_year_cos"] = np.cos(2 * np.pi * df["day_number_year"] / 12)

    # Define the order of months
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] 
    df.loc[:,'month_name'] = pd.Categorical(df['month_name'], categories=month_order, ordered=True)

    # Define the order of weekday 
    weekday_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] 
    df.loc[:,'weekday'] = pd.Categorical(df['weekday'], categories=weekday_order, ordered=True)


    return df



%%time 
train_fe_df = transform_date(train_df).dropna()

display(train_fe_df.head())
display(train_fe_df.shape)
CleanMemory()


%%time
sns.set(style="whitegrid", context="talk")

fig, (ax1) = plt.subplots(figsize=(12,6))

#boxplot
sns.boxplot(data=train_fe_df, x="weekday", y="num_sold", hue="store", ax=ax1)
ax1.set_title("Sales across different stores during a week")
ax1.set_xlabel("Day of week")
ax1.set_ylabel("Sales")
ax1.tick_params(axis='both')

ax1.legend(bbox_to_anchor=(1,1), ncols=1)

plt.tight_layout() 
plt.show()

CleanMemory()


%%time
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=train_fe_df, x="weekday", y="num_sold", hue="country", ax=ax)
ax.set_title("Sales across different countries during a week")
ax.set_xlabel("Day of week")
ax.set_ylabel("Sales")
ax.tick_params(axis='both')

ax.legend(bbox_to_anchor=(1,1), ncols=1)

plt.tight_layout() 
plt.show()

CleanMemory()


%%time
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=train_fe_df, x="month_name", y="num_sold", hue="country", ax=ax)
ax.set_title("Monthly Sales across different countries")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.tick_params(axis='both')

ax.legend(bbox_to_anchor=(1,1), ncols=1)

plt.tight_layout() 
plt.show()

CleanMemory()


%%time

condition = train_fe_df['country'] == 'Kenya'
filtered_df = train_fe_df[condition]
max_sales, min_sales = filtered_df['num_sold'].max(), filtered_df['num_sold'].min()

display(f"Range of sales numbers in Kenya are {min_sales}-{max_sales}")
display(filtered_df.sample(n=10, random_state=42))

del filtered_df
CleanMemory()


%%time

# Plot the data
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=train_fe_df, x="month_name", y="num_sold", hue="product", ax=ax)
ax.set_title("Monthly Sales by product")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.tick_params(axis='both')

ax.legend(bbox_to_anchor=(1,1), ncols=1)

plt.tight_layout() 
plt.show()

CleanMemory()


grouped_data = train_fe_df.groupby(['country', 'product']).size().reset_index(name='count')
display(grouped_data.head())


%%time

# Plot the data
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(data=grouped_data, x='country', y='count', hue='product', ax=ax)
ax.set_title("Country wise sales of products")
ax.set_xlabel("Country")
ax.set_ylabel("Count")
ax.tick_params(axis='both')

ax.legend(bbox_to_anchor=(1,1), ncols=1)

plt.tight_layout() 
plt.show()

CleanMemory()


%%time

g = sns.FacetGrid(
    train_fe_df,
    row="country", #create row for each country
    hue="year", #color the line based on the year
    col="product", #create a separate column for each product
    palette = "Dark2",
    height=7, #set the height of each facet(subplot) to 7 inches
    aspect=0.9,
    sharey=False, #do not share y-axis with other facets
    sharex=False, #do not share x-axis with other facets
)
g.map(sns.lineplot, "month_name", "num_sold", errorbar=None)
g.set_titles(
        col_template="\n---------------------\n{col_var} = {col_name}\n---------------------\n",
        size=14,
    )

g.add_legend(loc='upper right', title= "Year")
g.tick_params()
g.set_axis_labels(x_var="Month", y_var="Number Sold")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()

CleanMemory()




