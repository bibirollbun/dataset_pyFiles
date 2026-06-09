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


import sys
sys.path.append('/kaggle/working/')


!rm -rf /kaggle/working/scipy
!rm -rf /kaggle/working/statsmodels


!rm -rf scipy
!rm -rf statsmodels


!pip install numpy==1.26.4 scipy==1.11.4 statsmodels==0.14.0 --no-cache-dir --force-reinstall


#-------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt                  # To perform data visualisation
import seaborn as sns                                 # To perform data visualisation
import plotly.express as px                           # To perform data visualisation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
%matplotlib inline
                                             
from sklearn.linear_model import LinearRegression     # To perform prediction
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, ElasticNetCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import math

from scipy.stats import chi2_contingency              # To perform chi sqr test
from scipy import stats
from scipy.stats import skewtest, norm
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import StandardScaler      # To perform feature scaling
from sklearn.model_selection import train_test_split  # To perform train test split

from sklearn.model_selection import GridSearchCV      # To perform hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import pearsonr                       # To perform pearson correlation
from scipy.stats import loguniform
from scipy.stats import uniform
from random import randint
import copy

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import gc
#import mlflow

sns.set_theme(style="white", palette="muted")
%matplotlib inline

#-------------------------------------------------------------------------------------------
import warnings                                       # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")


#Class for importing datasets from the kaggle competition

filepath={'train_path':'/kaggle/input/drw-crypto-market-prediction/train.parquet',
         'test_path':'/kaggle/input/drw-crypto-market-prediction/test.parquet'}
class data_import():
    #Constructor
    def __init__(self, filepath):
        self.filepath = filepath
        self.y_train_DRW = None
        self.x_train_DRW = None
        self.y_test_DRW = None
        self.x_test_DRW = None
    
    #Function for importing train dataset
    def file_import_train(self,num_records=250000):
        train_path = self.filepath['train_path']
        df_train = pd.read_parquet(train_path)

        # Get the tail using df_train.tail() and then .compute()
        df_train = df_train.tail(num_records)
        
        self.y_train_DRW=df_train[['label']]
        self.x_train_DRW=df_train.drop(['label'],axis=1)
       
        return self.x_train_DRW,self.y_train_DRW

    #Function for importing test dataset
    def file_import_test(self,num_records=250000):
        test_path = self.filepath['test_path']
        df_test = pd.read_parquet(test_path)
       
        df_test = df_test.tail(num_records)
        
        self.y_test_DRW=df_test[['label']]
        self.x_test_DRW=df_test.drop(['label'],axis=1)
        
        return self.x_test_DRW,self.y_test_DRW

    def memory_optimize(self):
        self.file_import_train()
        self.file_import_test()

        for df in self.x_train_DRW, self.y_train_DRW, self.x_test_DRW, self.y_test_DRW:
            column_dtypes = df.dtypes
            for col in df.columns:
                current_dtype = column_dtypes[col] # Access the dtype from the Series of dtypes
                if str(current_dtype) == 'float64':
                    df[col]=df[col].astype('float32') # Converting to float 32 to optimize RAM
                elif str(current_dtype) == 'int64':
                    df[col]=df[col].astype('float32') # Converting to float 32 to optimize RAM
        print('All dtypes converted to float32')
        return self.x_train_DRW, self.y_train_DRW, self.x_test_DRW, self.y_test_DRW
        
#Calling the data_import class to populate train and test datasets from kaggle competition
data_import=data_import(filepath)
#x_train_DRW,y_train_DRW=data_import.file_import_train()  #Train split on train.parquet dataset
#x_test_DRW,y_test_DRW=data_import.file_import_test()     #Test split on test.parquet dataset
x_train_DRW, y_train_DRW, x_test_DRW, y_test_DRW=data_import.memory_optimize()


dataframes = {'x_train_DRW': x_train_DRW, 'x_test_DRW': x_test_DRW}

#Class to examine basic data characteristics - rows, columns, descriptive statistics, duplicates and missing values
class data_properties():
    
    # Constructor
    def __init__(self, dataframes_dict):
        # The __init__ should use the argument passed to it
        self.dataframes = dataframes_dict

    # Function to reflect shape, info, and descriptive statistics of each dataframe
    def dataset_description(self):
        # Loop through the key (name) and value (the DataFrame object)
        # in the dictionary.
        for name, df_object in self.dataframes.items():
            print(f"\n==========================================")
            print(f"Properties for DataFrame: '{name}'")
            print(f"==========================================")
            
            # Now, 'df_object' is an actual DataFrame, so you can use .shape, etc.
            print(f"Shape: {df_object.shape}\n")
            
            print("Info:")
            df_object.info() # .info() prints directly, so no need for a print() call
            
            print(f"\nDescriptive Statistics:")
            # .describe() returns a DataFrame, so we print it
            print(df_object.describe())

            print(f"Total missing values in DataFrame: {df_object.isna().sum().sum()}")
            
            #Treating duplicates if any
            if df_object.duplicated().any():
                print(f"Duplicates found in '{name}'. Removing duplicates.")
                df_object.drop_duplicates(inplace=True)
                print("Duplicates removed.")
            else:
                print(f"No duplicates found in {name}.")
                        
            print("\n")

            # Check for infinite values in dataset
            has_infinite = df_object.isin([np.inf, -np.inf]).any().any()
            
            if has_infinite:
                print(f"The DataFrame '{name}' contains infinite values.")
            
                # Finding which columns contain infinite values
                cols_with_inf = df_object.columns[df_object.isin([np.inf, -np.inf]).any()]
                print("Columns with infinite values:", list(cols_with_inf))

                # Dropping infinite values rows and columns
                for col in list(cols_with_inf):
                    if df_object[col].value_counts().values[0]==250000:   #if all values in a column are infinite
                        df_object.drop(col, axis=1, inplace=True)
            else:
                print(f"The DataFrame '{name}' does not contain infinite values.")

    #Function to test skewness - left or right 
    def skewness_test(self):
        left_skewed_col=[]
        right_skewed_col=[]
        for df_object,df_content in self.dataframes.items():
            for col in df_content.columns:
                if df_content[col].mean()<df_content[col].median():
                    left_skewed_col.append(col)
                elif df_content[col].mean()>df_content[col].median():
                    right_skewed_col.append(col)
            
            #Examining skewness for one feature through histogram and violin plot
            print(f"The feature 'X247' in {df_object} - Skewness depiction through Histogram, Violon Plot and Q-Q plot")
            # Create subplots: 1 row, 2 columns
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=('Histogram Distribution of X247', 'Violin plot of X247'))
            
            # Add Histogram
            fig.add_trace(go.Histogram(x=df_content['X247'], name='Histogram', nbinsx=500), 
                          row=1, col=1)
            
            # Add Violin plot
            fig.add_trace(go.Violin(y=df_content['X247'], name='Violin plot', box_visible=True, meanline_visible=True, points='outliers'), row=1, col=2)
            
            # Update layout for better appearance
            fig.update_layout(
                title_text="Distribution of X247", 
                showlegend=False)
            
            # Update x-axis label for histogram
            fig.update_xaxes(title_text="X247", row=1, col=1)
            
            # Update y-axis label for boxplot (optional, as it's the same as the x-axis for histogram)
            fig.update_yaxes(title_text="X247", row=1, col=2)
            fig.show()

            #Examining skewness through Q-Q plot
            sm.qqplot(df_content['X247'], line='s')
            plt.title('Q-Q plot for X247')
            plt.show()

        print(f"The columns with left skewness: {left_skewed_col}")
        print(f"The columns with right skewness: {right_skewed_col}")

        
# Create an instance of the class
data_prop_instance = data_properties(dataframes)

# Run the dataset_description method
dataset_prop=data_prop_instance.dataset_description()
dataset_prop

# Run the left skewness test method
dataset_skewness=data_prop_instance.skewness_test()
dataset_skewness

#Inference:
#1.Found no missing values in any column of train and test datasets.
#2.There is a presence of right skewness in some columns eg:- bid_qty and ask_qty (as listed in output) as mean is greater than the median.
#3.There is a presence of left skewness in some columns (as listed in output) eg:- X9, X17 etc.) as median is greater than the mean.
#4.Two kind of outliers found - one is the natural variation in values and other is unusual extremely high value (eg:- bid qty of 1114.93 when 75th percentile value is 13.08)
#5.Hence, these extreme values need to be clipped with IQR clipping technique.


from sklearn.model_selection import train_test_split

#Class for train test splitting

class DataSplitter():
    #constructor
    def __init__(self, x_dataset, y_dataset, test_size=None, random_state=None, shuffle=None):
        # The __init__ should use the argument passed to it
        self.x_dataset=x_dataset
        self.y_dataset=y_dataset
        self.test_size=test_size
        self.random_state=random_state
        self.shuffle=shuffle
        
    # Function to create train test set
    def train_test_set(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x_dataset, self.y_dataset, test_size=self.test_size, random_state=self.random_state, shuffle=self.shuffle)
        print('Train Test Split Created')
        return x_train, x_test, y_train, y_test

DataSplitter_instance=DataSplitter(x_train_DRW,y_train_DRW,test_size=0.3, random_state=42, shuffle=False)

#Train Test split on train.parquet dataset
x_train_original, x_test_original, y_train_original, y_test_original=DataSplitter_instance.train_test_set()


#Class for 1)Treating extreme outliers with IQR - upper and lower clipping technique and applying power transformation, 2)Feature Scaling and 
## 3)Memory Optimization

class outlier_treatement():
    #constructor
    def __init__(self, dataframe, scaling_fit_dataset, threshold_for_std_dev):
        # The __init__ should use the argument passed to it
        self.dataframe=dataframe
        self.scaling_fit_dataset=scaling_fit_dataset
        self.scaler = StandardScaler()
        self.pt = PowerTransformer(method='yeo-johnson', standardize=False)  # Initialize PowerTransformer with Yeo-Johnson method for positive/negative skewness post extreme values removal
        self.data_dict={'self.dataframe':self.dataframe,'self.scaling_fit_dataset':self.scaling_fit_dataset}
        self.skewness_threshold = 1.0
        self.kurtosis_threshold = 1.0
        self.skewed_cols=[]
        self.col_unique_one = []
        self.skewed_col_final = []
        self.dataframe_transformed = None
        self.scaling_fit_dataset_transformed = None
        self.zero_std_dev_col=[]
        self.threshold_for_std_dev = threshold_for_std_dev
        self.final_cols_scaling=[]
        self.final_cols_scaling_df=pd.DataFrame()
        
    # Function to handle extreme outliers (erroneous values, inputs etc) through IQR - Upper and Lower bound technique
    def outlier_handling(self):
        for df_name, df_object in self.data_dict.items():
            for col in df_object.columns:
                upper_bound=self.scaling_fit_dataset[col].quantile(0.95)
                lower_bound=self.scaling_fit_dataset[col].quantile(0.05)
                df_object[col] = df_object[col].clip(lower=lower_bound, upper=upper_bound)  
        
        #Check if the skewness still persists post outlier treatment and check for distribution normality
        for col in self.scaling_fit_dataset.columns:
            col_skewness= self.scaling_fit_dataset[col].skew()
            col_kurtosis=kurtosis(self.scaling_fit_dataset[col], fisher=True)
            
            if (abs(col_skewness)>self.skewness_threshold) or (abs(col_kurtosis)>self.kurtosis_threshold): #if the column still skewed (+ve) and kurtosis (+ve) post outlier treatment
                self.skewed_cols.append(col)

        # Creating copies of passed datasets
        self.dataframe_transformed = self.dataframe.copy()
        self.scaling_fit_dataset_transformed = self.scaling_fit_dataset.copy()

        # Transformation step for Train and Test dataset:
        # Only apply power transformation if skewed columns are found
        if self.skewed_cols:
            # Step 1: Identify skewed columns with ≤ 1 unique value (zero or near-zero variance)
            for col in self.skewed_cols:
                if self.scaling_fit_dataset[col].nunique() <= 1 and self.scaling_fit_dataset[col].std() < 1e-6:
                    self.col_unique_one.append(col)
            print(f'Columns with ≤1 unique value (excluded from power transform): {self.col_unique_one}')

            # Step 2: Filter final skewed columns with enough variance
            self.skewed_col_final = [col for col in self.skewed_cols if col not in self.col_unique_one]
            print(f'Columns with >1 unique value (included in power transform): {self.skewed_col_final}')
            
            if self.skewed_col_final:
                # Fit transformer only on valid columns
                self.pt.fit(self.scaling_fit_dataset[self.skewed_col_final])

                # Apply transform
                self.dataframe_transformed[self.skewed_col_final] = self.pt.transform(self.dataframe_transformed[self.skewed_col_final])
                self.scaling_fit_dataset_transformed[self.skewed_col_final] = self.pt.transform(self.scaling_fit_dataset_transformed[self.skewed_col_final])

                # Cap inf/very large values after power transformation 
                for df_t in [self.dataframe_transformed, self.scaling_fit_dataset_transformed]:
                    for col in self.skewed_col_final: # Only apply to columns that were power-transformed
                        max_finite_val = np.finfo(np.float32).max / 2
                        min_finite_val = np.finfo(np.float32).min / 2 
                        # Replace actual infinities
                        df_t[col] = df_t[col].replace([np.inf, -np.inf], [max_finite_val, min_finite_val])
                        # Clip any remaining values that are still too large/small,
                        # even if not strictly 'inf' (e.g., 1.7e308)
                        df_t[col] = df_t[col].clip(lower=min_finite_val, upper=max_finite_val)

                        # Clipping within 85th/15th percentile for max and min finite values as x_test_original has extreme value at 90th/10th percentile too
                        upper_bound= self.scaling_fit_dataset_transformed[col].quantile(0.90)
                        lower_bound= self.scaling_fit_dataset_transformed[col].quantile(0.10)
                        df_t[col] = df_t[col].clip(lower=lower_bound, upper=upper_bound)
            
            else:
                print("No valid skewed columns to transform after excluding low-variance ones.")

        else:
            print("No skewed columns found for power transformation.")

        return self.dataframe_transformed, self.scaling_fit_dataset_transformed

    # Function to scale features as per standard scaler
    def feature_standard_scaling(self):
        self.outlier_handling()
        
        # Creating copies of passed datasets
        self.dataframe_scaled = self.dataframe_transformed.copy() 
        self.scaling_fit_dataset_scaled = self.scaling_fit_dataset_transformed.copy() 

        # Check for columns standard deviation in train dataset
        for col in self.scaling_fit_dataset_scaled.columns:
            col_std_dev = self.scaling_fit_dataset_scaled[col].std()
            if col_std_dev < self.threshold_for_std_dev:  #if std deviation falls below threshold, they would possibly be close to zero and 
                                                          #thus create close to infinite values when applied standard scaling
                self.zero_std_dev_col.append(col)
            else:
                self.final_cols_scaling.append(col)       #Final cols with non-zero std deviation
        
        print(f'The features with close to zero std. devaiation: {self.zero_std_dev_col}')
        print(f'The final features to be scaled: {self.final_cols_scaling}')
            

        if self.final_cols_scaling:
            #Fitting standard scaler on train data
            self.scaler.fit(self.scaling_fit_dataset_transformed[self.final_cols_scaling])

            #Transforming Train and Test dataset through scaler
            self.scaling_fit_dataset_scaled[self.final_cols_scaling] = self.scaler.transform(self.scaling_fit_dataset_scaled[self.final_cols_scaling])
            self.dataframe_scaled[self.final_cols_scaling] = self.scaler.transform(self.dataframe_scaled[self.final_cols_scaling])

            #Retaining only above zero std dev features
            self.scaling_fit_dataset_scaled = self.scaling_fit_dataset_scaled[self.final_cols_scaling]
            self.dataframe_scaled = self.dataframe_scaled[self.final_cols_scaling]
            
        return self.dataframe_scaled

    # Function for memory optimization - converting 64 bits to 32 bits
    def memory_optimize(self):
        self.feature_standard_scaling()
        column_dtypes = self.dataframe_scaled.dtypes
        for col in self.dataframe_scaled.columns:
            current_dtype = column_dtypes[col] # Access the dtype from the Series of dtypes
    
            if str(current_dtype) != 'float32':
                self.dataframe_scaled[col]=self.dataframe_scaled[col].astype('float32') #Converting to float32 for RAM optimization

            gc.collect()

        #Treating duplicates columns if any
        self.dataframe_scaled = self.dataframe_scaled.loc[:, ~self.dataframe_scaled.columns.duplicated()]
        
        # Clipping again if ~infinite values exist
        self.dataframe_scaled.replace([np.inf, -np.inf], [np.finfo(np.float32).max, np.finfo(np.float32).min], inplace=True)

        return self.dataframe_scaled

processed_dataframes_final = {}

# outlier_treatement class methods call on x_train_original
x_processor = outlier_treatement(dataframe=x_train_original, scaling_fit_dataset=x_train_original, threshold_for_std_dev=1e-9)
x_train_original_processed = x_processor.memory_optimize()
processed_dataframes_final['x_train_original_processed'] = x_train_original_processed

# outlier_treatement class methods call on x_test_original
x_processor.dataframe = x_test_original.copy()
x_test_original_processed = x_processor.memory_optimize()
processed_dataframes_final['x_test_original_processed'] = x_test_original_processed

# outlier_treatement class methods call on y_train_original
y_processor = outlier_treatement(dataframe=y_train_original, scaling_fit_dataset=y_train_original, threshold_for_std_dev=1e-9)
y_train_original_processed = y_processor.memory_optimize()
processed_dataframes_final['y_train_original_processed'] = y_train_original_processed

# outlier_treatement class methods call on y_test_original
y_processor.dataframe = y_test_original.copy()
y_test_original_processed = y_processor.memory_optimize()
processed_dataframes_final['y_test_original_processed'] = y_test_original_processed

# Storing results in new variables
x_train_original_processed=processed_dataframes_final['x_train_original_processed']
x_test_original_processed=processed_dataframes_final['x_test_original_processed']
y_train_original_processed=processed_dataframes_final['y_train_original_processed']
y_test_original_processed=processed_dataframes_final['y_test_original_processed']

print('All dtypes converted to float32')


#ANALYSING THE ACTUAL DRIFT BETWEEN X_TRAIN AND X_TEST POST TRAIN TEST SPLIT DONE WITH SHUFFLE=FALSE
def analyze_drift(x_train, x_test):
    train_stats = x_train.describe().T[['mean', 'std']]
    test_stats = x_test.describe().T[['mean', 'std']]
    drift_df = train_stats.join(test_stats, lsuffix='_train', rsuffix='_test')
    drift_df['mean_diff'] = (drift_df['mean_test'] - drift_df['mean_train']).abs()
    drift_df['std_ratio'] = drift_df['std_test'] / drift_df['std_train']
    return drift_df.sort_values(by='mean_diff', ascending=False)

drift_report = analyze_drift(x_train_original_processed, x_test_original_processed)
drift_report


#Class for initial data exploration through pair plots, pearson/spearman's correlation through heatmaps

class initial_data_exploration():
    #constructor
    def __init__(self,x_train_df,y_train_df,label_col=None,num_features_for_plot=None, sample_rows_for_plot=None):
        # The __init__ should use the argument passed to it
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.label_col = label_col
        self.num_features_for_plot = num_features_for_plot
        self.sample_rows_for_plot = sample_rows_for_plot
        features_to_plot = self.x_train_df.iloc[:, :self.num_features_for_plot] # Assuming 0-indexed features
        self.merged_df = pd.concat([features_to_plot, self.y_train_df[[self.label_col]]], axis=1)
        print(f"Merged DataFrame for plots created with shape: {self.merged_df.shape}")
            
    # Function for quick visual analysis of linear/non-linear relationships through pairplots
    def plot_initial_relationship(self):
        
        # Generate the pair plot for a sample of 50 records for quick observation
        sns.pairplot(self.merged_df.iloc[0:self.sample_rows_for_plot,:])
        
        # Display the plot
        plt.tight_layout()
        plt.show()
        
        
    # Function for pearson correlation test
    def pearson_corr(self):
        #Pearson Correlation Examination
        corr_matrix_pearson=self.merged_df.corr(method='pearson')
        # Create the heatmap
        plt.figure(figsize=(12, 10)) # Adjust figure size as needed
        sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f")
        
        # Set the title of the heatmap
        plt.title('Pearson Correlation Heatmap')
        
        # Display the plot
        plt.tight_layout()
        plt.show()
        

    # Function for spearman's correlation test
    def spearman_corr(self):
        #Spearman's Correlation Examination
        corr_matrix_spearman=self.merged_df.corr(method='spearman')
        # Create the heatmap
        plt.figure(figsize=(12, 10)) # Adjust figure size as needed
        sns.heatmap(corr_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f")
        
        # Set the title of the heatmap
        plt.title("Spearman's Correlation Heatmap")
        
        # Display the plot
        plt.tight_layout()
        plt.show()
        
    
initial_data_explor_instance=initial_data_exploration(x_train_original_processed,y_train_original_processed,label_col='label',num_features_for_plot=10,sample_rows_for_plot=50)
initial_data_explor_instance.plot_initial_relationship()
initial_data_explor_instance.pearson_corr()
initial_data_explor_instance.spearman_corr()

# Inference:
#1.Some predictors like bid_qty,as_qty,buy_qty etc. have non-linear and complex relationship with label variable.
#2.Some predictors like X1,X2,X3 etc. have possible linearity in their relationship with label (can be tested by pearson's correlation)
#3.Also, there's high linearity amonst some predictors like between x1 and x2,x3,x4 each indicating multicollinear features (can be tested by pearson's correlation)
#4.Even though features like X1,X2 etc. percieved to have a linear relationships with label, their correlation percentage is way less - 3 to 4 percent.
#5.This indicates complex relationships of these features with label. Hence, there's need to run non-linear test to examine feature importance with label.
#6.But there's presence of high collinearity between predictors like X2 and X3, X3 and X4 (corr. percent atleast 90) indicating one feature of these
##multicollinear feature combinations should be retained and others should be discarded on the basis of pearson's correlation threshold.


#Class for removing multicollinearity
class remove_multicollinearity():
    #constructor
    def __init__(self,x_train_df,y_train_df,threshold=None,rows_to_include=None,label=None,method=None):
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.threshold = threshold
        self.rows_to_include = rows_to_include
        self.label = label
        self.method = method
        self.correlation_matrix_predictors = self.x_train_df.tail(rows_to_include).corr(method=self.method)

        self.mi_scores = mutual_info_regression(self.x_train_df.tail(self.rows_to_include), self.y_train_df[self.label].tail(self.rows_to_include))
        
        # Create a pandas dataframe to store feature names and their MI scores
        self.mi_df = pd.DataFrame({'MI Score':self.mi_scores, 'Feature':self.x_train_df.columns}).sort_values(by='MI Score',ascending=False)

        # Create a dictionary for faster MI score lookups
        self.mi_scores_dict = self.mi_df.set_index('Feature')['MI Score'].to_dict()


    #Function for removing multicollinearity
    def multicollinear_treatment(self):
        highly_correlated_pairs = []

        # Iterate through the matrix
        for i in range(len(self.correlation_matrix_predictors.columns)):
            for j in range(i + 1, len(self.correlation_matrix_predictors.columns)):
                col1 = self.correlation_matrix_predictors.columns[i]
                col2 = self.correlation_matrix_predictors.columns[j]
                correlation_value = self.correlation_matrix_predictors.iloc[i, j]
        
                # Check if the absolute correlation is above the threshold
                if abs(correlation_value) > self.threshold:
                    highly_correlated_pairs.append((col1, col2, correlation_value))
        
        # Print the highly correlated pairs
        print(f"Highly correlated predictor pairs (absolute correlation > {self.threshold}):")
        if highly_correlated_pairs:
            for col1, col2, corr_value in highly_correlated_pairs:
                print(f"  - {col1} and {col2}: {corr_value:.4f}")
        else:
            print("No highly correlated predictor pairs found above the specified threshold.")
        
        # Identify features to drop (On the basis of MI scores)
        features_to_drop = set()
        
        # mi_scores_dict for faster lookups
        for col1, col2, _ in highly_correlated_pairs:
            # Use .get() with a default of -np.inf for safety in case a feature is somehow missing
            mi_col1 = self.mi_scores_dict.get(col1, -np.inf)
            mi_col2 = self.mi_scores_dict.get(col2, -np.inf)

            if mi_col1 > mi_col2:
                features_to_drop.add(col2)
            else:
                features_to_drop.add(col1)
        
        # Drop the identified features from x_train_DRW
        x_train_reduced = self.x_train_df.drop(columns=list(features_to_drop), errors='ignore')
        
        # Print the count of unique dropped features
        print(f"\nNumber of unique features dropped due to high collinearity: {len(set(features_to_drop))}")
        print(f"Shape of x_train reduced post Multicollinearity Treatment: {x_train_reduced.shape}")   #Dataframe with removed multicollinearity
        
        return x_train_reduced

remove_multicollinearity_instance=remove_multicollinearity(x_train_original_processed,y_train_original_processed,threshold=0.80,rows_to_include=50000,label='label',method='pearson')
x_train_reduced_post_pearson=remove_multicollinearity_instance.multicollinear_treatment()
x_test_reduced_post_pearson=x_test_original_processed[x_train_reduced_post_pearson.columns]


# Class for analyzing and visualizing Mutual Information scores on x_train_reduced_post_pearson
class MIFeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, label_col=None, rows_to_include=None, threshold=None):
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.rows_to_include = rows_to_include
        self.label = label_col
        self.threshold=threshold
        self.mi_scores = mutual_info_regression(self.x_train_df.tail(self.rows_to_include), self.y_train_df[self.label].tail(self.rows_to_include))

        # Create a pandas dataframe to store feature names and their MI scores
        self.mi_df = pd.DataFrame({'MI Score':self.mi_scores, 'Feature':self.x_train_df.columns}).sort_values(by='MI Score',ascending=False)
        
        # Below Threshold features list
        self.below_threshold_features=list(self.mi_df[self.mi_df['MI Score']<self.threshold]['Feature'].values)

        # Above Threshold features df
        self.above_threshold_features_mi=self.mi_df[self.mi_df['MI Score']>=self.threshold]

    def MI_plotting(self):
        # --- Visualization ---
        # Top 50 Features basis MI score
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 5))
        sns.barplot(y='MI Score', x='Feature', data=self.mi_df.head(50),ax=my_ax[0])
        my_ax[0].set_title('Top 50 Features - Mutual Information Score')
        my_ax[0].set_xlabel('Importance - Mutual Information Score')
        my_ax[0].set_ylabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked mutual information scores
        sns.lineplot(x=range(len(self.mi_df)), y=self.mi_df['MI Score'],ax=my_ax[1])
        my_ax[1].set_title('Mutual Information Score in Descending Order')
        my_ax[1].set_ylabel('Importance - Mutual Information Score')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.grid(linestyle=':')
        plt.tight_layout()
        plt.show()

    def above_threshold_features_MI(self):
        # Retaining features with importance above threshold - MI Scores
        print(f"Features Importances List - MI score: {self.mi_df.shape}")
        print(f"Above Threshold Features Importances  List - MI score: {self.above_threshold_features_mi.shape}")
        above_threshold_features_MI=self.above_threshold_features_mi
        return above_threshold_features_MI

    def MI_feature_drop(self):
        x_train_reduced_post_pearson_MI=self.x_train_df.copy()
        for col in self.below_threshold_features:
            x_train_reduced_post_pearson_MI.drop([col],axis=1,inplace=True)
        
        print(f"Shape of dataset reduced through MI score: {x_train_reduced_post_pearson_MI.shape}")

        return x_train_reduced_post_pearson_MI

remove_post_MI_instance=MIFeatureAnalyzer(x_train_reduced_post_pearson, y_train_original_processed, label_col='label', rows_to_include=50000, threshold=0.05)
remove_post_MI_instance.MI_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around 0.05 MI score, hence we can keep the threshold at 0.05.
x_train_reduced_post_pearson_MI=remove_post_MI_instance.MI_feature_drop()
x_test_reduced_post_pearson_MI=x_test_reduced_post_pearson[x_train_reduced_post_pearson_MI.columns]


# Class for analyzing feature importance through Lasso - L1 Regression

class lasso_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, cv=None, random_state=None, max_iter=None):  
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.cv=cv
        self.random_state=random_state
        self.max_iter=max_iter
        self.custom_alphas = np.logspace(-3, 0, 100)
        self.lasso_cv_model = LassoCV(cv=self.cv,            
                                      random_state=self.random_state,
                                      max_iter=self.max_iter,
                                      alphas=self.custom_alphas
                                     ).fit(self.x_train_df,self.y_train_df)

        print(f"Optimal alpha found by LassoCV: {self.lasso_cv_model.alpha_}")

        # Get coefficients from the best Lasso model
        self.lasso_coefficients_df = pd.DataFrame({'Feature Coefficient':self.lasso_cv_model.coef_, 'Feature':self.x_train_df.columns})

        # Identify selected features (non-zero coefficients)
        self.selected_features_lasso_df = self.lasso_coefficients_df[self.lasso_coefficients_df['Feature Coefficient'] != 0].sort_values(by='Feature Coefficient', key=abs, ascending=False)

    def lasso_plotting(self):
        # --- Visualization ---
        # Top 50 Important Features
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Feature Coefficient', x='Feature', data=self.selected_features_lasso_df,ax=my_ax[0])
        my_ax[0].set_title('Top Features Basis Lasso - L1 Regression Feature Importances')
        my_ax[0].set_ylabel('Feature Coefficient Value')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.selected_features_lasso_df)), y=self.selected_features_lasso_df['Feature Coefficient'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis Lasso - L1 Regression in Descending Order')
        my_ax[1].set_ylabel('Feature Coefficient Value')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()

    def lasso_feature_drop(self):
        self.x_train_reduced_post_pearson_MI_lasso=self.x_train_df.copy()
        self.x_train_reduced_post_pearson_MI_lasso= self.x_train_reduced_post_pearson_MI_lasso[self.selected_features_lasso_df['Feature']]

        print(f"Shape of dataset reduced through Pearson, MI score & Lasso: {self.x_train_reduced_post_pearson_MI_lasso.shape}")

        return self.x_train_reduced_post_pearson_MI_lasso,self.selected_features_lasso_df,self.lasso_cv_model

lasso_FeatureAnalyzer_instance=lasso_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, cv=5, random_state=42, max_iter=10000)
lasso_FeatureAnalyzer_instance.lasso_plotting()
x_train_reduced_post_pearson_MI_lasso, Imp_features_lasso,model_lasso=lasso_FeatureAnalyzer_instance.lasso_feature_drop()
x_test_reduced_post_pearson_MI_lasso=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_lasso.columns]


#Training the lasso regressor on top features as per Lasso
lasso_FeatureAnalyzer_instance_two=lasso_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, cv=5, random_state=42, max_iter=10000)
x_train_reduced_post_pearson_MI_lasso_two, Imp_features_lasso_two, model_lasso_two=lasso_FeatureAnalyzer_instance_two.lasso_feature_drop()

#Predictions from lasso 
lasso_pred=model_lasso_two.predict(x_test_reduced_post_pearson_MI)
lasso_pred_df=pd.DataFrame(lasso_pred,columns=['pred_label from Lasso top features'])

m1 = lasso_pred_df['pred_label from Lasso top features'].round(3)
m2 = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas = m1.corr(m2)
print(f"Pearson Correlation for Lasso from Lasso's best features - original: {corr_pandas}")

#Accuracy as per RMSE
rmse_lasso = np.sqrt(mean_squared_error(m1,m2))
print(f"Root Mean Squared Error for Lasso from Lasso's best features - original: {rmse_lasso:.4f}")

#Accuracy as per R2
r2_lasso = r2_score(m1,m2)
print(f"R2 score for Lasso from Lasso's best features - original: {r2_lasso:.4f}")


# Class for analyzing feature importance through Ridge - L2 Regression

class ridge_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, cv=None, random_state=None, max_iter=None):  
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.cv=cv
        self.random_state=random_state
        self.max_iter=max_iter
        self.custom_alphas = np.logspace(-3, 0, 100)
        self.ridge_cv_model = RidgeCV(cv=self.cv,            
                                      alphas=self.custom_alphas
                                     ).fit(self.x_train_df,self.y_train_df)

        print(f"Optimal alpha found by RidgeCV: {self.ridge_cv_model.alpha_}")
        #print(self.ridge_cv_model.coef_)
        #print(self.x_train_df.columns)
        # Get coefficients from the best Lasso model
        self.ridge_coefficients_df = pd.DataFrame({'Feature Coefficient':self.ridge_cv_model.coef_.reshape(-1), 'Feature':self.x_train_df.columns})

        # Identify selected features (non-zero coefficients)
        self.selected_features_ridge_df = self.ridge_coefficients_df[self.ridge_coefficients_df['Feature Coefficient'] != 0].sort_values(by='Feature Coefficient', key=abs, ascending=False)

    def ridge_plotting(self):
        # --- Visualization ---
        # Top 50 Important Features
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Feature Coefficient', x='Feature', data=self.selected_features_ridge_df,ax=my_ax[0])
        my_ax[0].set_title('Top Features Basis Ridge - L2 Regression Feature Importances')
        my_ax[0].set_ylabel('Feature Coefficient Value')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.selected_features_ridge_df)), y=self.selected_features_ridge_df['Feature Coefficient'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis Ridge - L2 Regression in Descending Order')
        my_ax[1].set_ylabel('Feature Coefficient Value')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()

    def ridge_feature_drop(self):
        self.x_train_reduced_post_pearson_MI_ridge=self.x_train_df.copy()
        self.x_train_reduced_post_pearson_MI_ridge= self.x_train_reduced_post_pearson_MI_ridge[self.selected_features_ridge_df['Feature']]

        print(f"Shape of dataset reduced through Pearson, MI score & Ridge: {self.x_train_reduced_post_pearson_MI_ridge.shape}")

        return self.x_train_reduced_post_pearson_MI_ridge,self.selected_features_ridge_df,self.ridge_cv_model

ridge_FeatureAnalyzer_instance=ridge_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, cv=5)
ridge_FeatureAnalyzer_instance.ridge_plotting()
x_train_reduced_post_pearson_MI_ridge, Imp_features_ridge,model_ridge=ridge_FeatureAnalyzer_instance.ridge_feature_drop()
x_test_reduced_post_pearson_MI_ridge=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_ridge.columns]


#Ridge model trained on Best Features basis Ridge coefficients
ridge_FeatureAnalyzer_instance_best=ridge_FeatureAnalyzer(x_train_reduced_post_pearson_MI_ridge, y_train_original_processed, cv=5)
x_train_reduced_post_pearson_MI_ridge_best, Imp_features_ridge_best,model_ridge_best=ridge_FeatureAnalyzer_instance_best.ridge_feature_drop()

#Predictions from Ridge 
ridge_pred=model_ridge_best.predict(x_test_reduced_post_pearson_MI_ridge)
ridge_pred_df=pd.DataFrame(ridge_pred,columns=['pred_label from Ridge top features'])

m1_ridge = ridge_pred_df['pred_label from Ridge top features'].round(3)
m2_ridge = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_ridge = m1_ridge.corr(m2_ridge)
print(f"Pearson Correlation for Ridge from Ridge's best features - original: {corr_pandas_ridge}")

#Accuracy as per RMSE
rmse_ridge = np.sqrt(mean_squared_error(m1_ridge,m2_ridge))
print(f"Root Mean Squared Error for Ridge from Ridge's best features - original: {rmse_ridge:.4f}")

#Accuracy as per R2
r2_ridge = r2_score(m1_ridge,m2_ridge)
print(f"R2 score for Ridge from Ridge's best features - original: {r2_ridge:.4f}")


# Class for analyzing feature importance through Elastic Net Regression

class elastic_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, cv=None, max_iter=None, l1_ratio=None):  
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.cv=cv
        self.max_iter=max_iter
        self.l1_ratio=l1_ratio
        self.custom_alphas = np.logspace(-3, 0, 100)
        self.elastic_cv_model = ElasticNetCV(cv=self.cv,            
                                      alphas=self.custom_alphas,
                                      max_iter=self.max_iter,
                                      l1_ratio=self.l1_ratio
                                     ).fit(self.x_train_df,self.y_train_df)

        print(f"Optimal alpha found by ElasticCV: {self.elastic_cv_model.alpha_}")
        
        # Get coefficients from the best Elastic model
        self.elastic_coefficients_df = pd.DataFrame({'Feature Coefficient':self.elastic_cv_model.coef_.reshape(-1), 'Feature':self.x_train_df.columns})

        # Identify selected features (non-zero coefficients)
        self.selected_features_elastic_df = self.elastic_coefficients_df[self.elastic_coefficients_df['Feature Coefficient'] != 0].sort_values(by='Feature Coefficient', key=abs, ascending=False)

    def elastic_plotting(self):
        # --- Visualization ---
        # Top 50 Important Features
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Feature Coefficient', x='Feature', data=self.selected_features_elastic_df,ax=my_ax[0])
        my_ax[0].set_title('Top Features Basis Elastic Regression Feature Importances')
        my_ax[0].set_ylabel('Feature Coefficient Value')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.selected_features_elastic_df)), y=self.selected_features_elastic_df['Feature Coefficient'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis Elastic Regression in Descending Order')
        my_ax[1].set_ylabel('Feature Coefficient Value')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()

    def elastic_feature_drop(self):
        self.x_train_reduced_post_pearson_MI_elastic=self.x_train_df.copy()
        self.x_train_reduced_post_pearson_MI_elastic= self.x_train_reduced_post_pearson_MI_elastic[self.selected_features_elastic_df['Feature']]

        print(f"Shape of dataset reduced through Pearson, MI score & Elastic: {self.x_train_reduced_post_pearson_MI_elastic.shape}")

        return self.x_train_reduced_post_pearson_MI_elastic,self.selected_features_elastic_df,self.elastic_cv_model

elastic_FeatureAnalyzer_instance=elastic_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, cv=5, max_iter=1000, l1_ratio=0.5)
elastic_FeatureAnalyzer_instance.elastic_plotting()
x_train_reduced_post_pearson_MI_elastic, Imp_features_elastic,model_elastic=elastic_FeatureAnalyzer_instance.elastic_feature_drop()
x_test_reduced_post_pearson_MI_elastic=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_elastic.columns]


#Elastic Net model trained on Best Features basis Ridge coefficients
elastic_FeatureAnalyzer_instance_best=elastic_FeatureAnalyzer(x_train_reduced_post_pearson_MI_elastic, y_train_original_processed, cv=5, max_iter=1000, l1_ratio=0.5)
x_train_reduced_post_pearson_MI_elastic_best, Imp_features_elastic_best,model_elastic_best=elastic_FeatureAnalyzer_instance_best.elastic_feature_drop()

#Predictions from Ridge 
elastic_pred=model_elastic_best.predict(x_test_reduced_post_pearson_MI_elastic)
elastic_pred_df=pd.DataFrame(elastic_pred,columns=['pred_label from Elastic top features'])

m1_elastic = elastic_pred_df['pred_label from Elastic top features'].round(3)
m2_elastic = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_elastic = m1_elastic.corr(m2_elastic)
print(f"Pearson Correlation for Elastic from Elastic's best features - original: {corr_pandas_elastic}")

#Accuracy as per RMSE
rmse_elastic = np.sqrt(mean_squared_error(m1_elastic,m2_elastic))
print(f"Root Mean Squared Error for elastic from elastic's best features - original: {rmse_elastic:.4f}")

#Accuracy as per R2
r2_elastic = r2_score(m1_elastic, m2_elastic)
print(f"R2 score for elastic from elastic's best features - original: {r2_elastic:.4f}")


# Class for analyzing feature importance through Random Forest Regressor
class RF_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, rows_to_include=None,label_col=None, threshold=None, **rf_params):
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.rows_to_include = rows_to_include
        self.label = label_col
        self.threshold=threshold
        self.rf_params = rf_params
        
        # Initialize the Random Forest Regressor model
        self.rf_regressor = RandomForestRegressor(**self.rf_params)
        
        # Train the RF regressor
        self.rf_regressor.fit(self.x_train_df.tail(self.rows_to_include), self.y_train_df.tail(self.rows_to_include)['label'])

        # Get feature importances
        self.importances_reg_rf = self.rf_regressor.feature_importances_
        
        # Create a DataFrame for better readability and sorting
        self.feature_names_reg_rf = self.x_train_df.columns
        self.importance_reg_rf = pd.DataFrame({
            'Feature': self.feature_names_reg_rf,
            'Importance': self.importances_reg_rf
        }).sort_values(by='Importance',ascending=False)

        # Above Threshold features df
        self.above_threshold_features_rf=self.importance_reg_rf[self.importance_reg_rf['Importance']>=self.threshold]

    def RF_plotting(self):
        # --- Visualization ---
        # Top 50 Important Features
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Importance', x='Feature', data=self.importance_reg_rf.head(50),ax=my_ax[0])
        my_ax[0].set_title('Top 50 Features Basis Random Forest Regressor Feature Importances')
        my_ax[0].set_ylabel('Importance')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.importance_reg_rf)), y=self.importance_reg_rf['Importance'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis RF Regressor in Descending Order')
        my_ax[1].set_ylabel('Feature Importance')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()

    def above_threshold_features(self):
        # Retaining features with importance above threshold - RF Regressor
        print(f"Features Importances List - RF Regressor: {self.importance_reg_rf.shape}")
        print(f"Above Threshold Features Importances  List - RF Regressor: {self.above_threshold_features_rf.shape}")
        above_threshold_features_rf=self.above_threshold_features_rf
        self.x_train_reduced_post_pearson_MI=self.x_train_df.copy()
        self.above_threshold_features_rf=self.above_threshold_features_rf
        self.x_train_reduced_post_pearson_MI_RF=self.x_train_reduced_post_pearson_MI[self.above_threshold_features_rf['Feature']]
        return self.above_threshold_features_rf, self.x_train_reduced_post_pearson_MI_RF, self.rf_regressor

tuned_hyperparameters = {'n_estimators': 150,         # Number of trees in the forest. 
                        'max_depth': 15,              # Maximum depth of the tree. Prevents overfitting and speeds up.
                        'min_samples_split': 10,      # Minimum number of samples required to split an internal node.
                        'min_samples_leaf': 5,        # Minimum number of samples required to be at a leaf node.
                        'max_features': 0.7,          # The number of features to consider when looking for the best split.
                        'n_jobs': 2,                  # Number of jobs kept as 2 to optimize RAM
                        'random_state': 42            # Seed for reproducibility of results.
                        }

RF_FeatureAnalyzer_instance=RF_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, rows_to_include=50000,label_col='label', threshold=0.015, **tuned_hyperparameters)
RF_FeatureAnalyzer_instance.RF_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around 0.015 RF importance score, hence we can keep the threshold at 0.015.
above_threshold_features_RF,x_train_reduced_post_pearson_MI_RF, model_RF=RF_FeatureAnalyzer_instance.above_threshold_features()
x_test_reduced_post_pearson_MI_RF=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_RF.columns]


#Training the RF regressor on top features as per RF
RF_FeatureAnalyzer_instance_two=RF_FeatureAnalyzer(x_train_reduced_post_pearson_MI_RF, y_train_original_processed, rows_to_include=50000,label_col='label', threshold=0.015, **tuned_hyperparameters)
x_train_reduced_post_pearson_MI_RF_two, Imp_features_RF_two, model_RF_two=RF_FeatureAnalyzer_instance_two.above_threshold_features()

#Predictions from lasso 
RF_pred=model_RF_two.predict(x_test_reduced_post_pearson_MI_RF)
RF_pred_df=pd.DataFrame(RF_pred,columns=['pred_label from RF top features - Original'])

m1_RF = RF_pred_df['pred_label from RF top features - Original'].round(3)
m2_RF = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_RF = m1_RF.corr(m2_RF)
print(f"Pearson Correlation for Random Forest from RF's best features - original: {corr_pandas_RF}")

#Accuracy as per RMSE
rmse_RF = np.sqrt(mean_squared_error(m1_RF,m2_RF))
print(f"Root Mean Squared Error for Random Forest from RF's best features - original: {rmse_RF:.4f}")

#Accuracy as per R2
r2_RF = r2_score(m1_RF,m2_RF)
print(f"R2 score for RF from RF's best features - original: {r2_RF:.4f}")


# Class for analyzing feature importance through Lightweight GBM Regressor
class LGBM_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, label_col=None,threshold=None, random_state=None):
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.label = label_col
        self.threshold=threshold
        self.random_state = random_state
        
        # Initialize the LGBM Regressor model
        self.lgbm_regressor = lgb.LGBMRegressor(random_state=self.random_state)
        
        # Train the LGBM regressor
        self.lgbm_regressor.fit(self.x_train_df, self.y_train_df['label'])

        # Get feature importances
        self.importances_reg_lgbm = self.lgbm_regressor.feature_importances_
        
        # Create a DataFrame for better readability and sorting
        self.feature_names_reg_lgbm = self.x_train_df.columns
        self.importance_reg_lgbm = pd.DataFrame({
            'Feature': self.feature_names_reg_lgbm,
            'Importance': self.importances_reg_lgbm
        }).sort_values(by='Importance',ascending=False)

        # Above Threshold features df
        self.above_threshold_features_lgbm=self.importance_reg_lgbm[self.importance_reg_lgbm['Importance']>=self.threshold]


    def LGBM_plotting(self):
        # --- Visualization ---
        # Top 50 Important Features
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Importance', x='Feature', data=self.importance_reg_lgbm.head(50),ax=my_ax[0])
        my_ax[0].set_title('Top 50 Features Basis LGBM Regressor Feature Importances')
        my_ax[0].set_ylabel('Importance')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.importance_reg_lgbm)), y=self.importance_reg_lgbm['Importance'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis LGBM Regressor in Descending Order')
        my_ax[1].set_ylabel('Feature Importance')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()


    def above_threshold_features_LGBM(self):
        # Retaining features with importance above threshold - RF Regressor
        print(f"Features Importances List - LGBM Regressor: {self.importance_reg_lgbm.shape}")
        print(f"Above Threshold Features Importances  List - LGBM Regressor: {self.above_threshold_features_lgbm.shape}")

        self.x_train_reduced_post_pearson_MI=self.x_train_df.copy()
        self.above_threshold_features_lgbm=self.above_threshold_features_lgbm
        self.x_train_reduced_post_pearson_MI_LGBM=self.x_train_reduced_post_pearson_MI[self.above_threshold_features_lgbm['Feature']]
        return self.above_threshold_features_lgbm, self.x_train_reduced_post_pearson_MI_LGBM, self.lgbm_regressor

remove_post_LGBM_instance=LGBM_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, label_col='label', threshold=30, random_state=42)
remove_post_LGBM_instance.LGBM_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around LGBM importance score of 30, hence we can keep the threshold at 30.
above_threshold_features_lgbm, x_train_reduced_post_pearson_MI_LGBM, model_LGBM = remove_post_LGBM_instance.above_threshold_features_LGBM()
x_test_reduced_post_pearson_MI_LGBM=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_LGBM.columns]


#Training the LGBM regressor on top features as per LGBM
LGBM_FeatureAnalyzer_instance_two=LGBM_FeatureAnalyzer(x_train_reduced_post_pearson_MI_LGBM, y_train_original_processed, label_col='label', threshold=30, random_state=42)
x_train_reduced_post_pearson_MI_LGBM_two, Imp_features_LGBM_two, model_LGBM_two=LGBM_FeatureAnalyzer_instance_two.above_threshold_features_LGBM()

#Predictions from LGBM 
LGBM_pred=model_LGBM_two.predict(x_test_reduced_post_pearson_MI_LGBM)
LGBM_pred_df=pd.DataFrame(LGBM_pred,columns=['pred_label from LGBM top features - Original'])

m1_LGBM = LGBM_pred_df['pred_label from LGBM top features - Original'].round(3)
m2_LGBM = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_LGBM = m1_LGBM.corr(m2_LGBM)
print(f"Pearson Correlation for LGBM Regressor from LGBM's best features - original: {corr_pandas_LGBM}")

#Accuracy as per RMSE
rmse_LGBM = np.sqrt(mean_squared_error(m1_LGBM,m2_LGBM))
print(f"Root Mean Squared Error for LGBM Regressor from LGBM's best features - original: {rmse_RF:.4f}")

#Accuracy as per R2
r2_LGBM = r2_score(m1_LGBM,m2_LGBM)
print(f"R2 score for LGBM from LGBM's best features - original: {r2_LGBM:.4f}")


# Class for analyzing feature importance through XGB Regressor
class XG_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, label_col=None,threshold=None, random_state=None, n_estimators=None, learning_rate=None, max_depth=None, gamma=None, subsample=None, colsample_bytree=None, n_jobs=None):
        self.x_train_df = x_train_df
        self.y_train_df = y_train_df
        self.label = label_col
        self.threshold=threshold
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate      
        self.max_depth = max_depth               
        self.gamma = gamma                
        self.subsample = subsample             
        self.colsample_bytree = colsample_bytree      
        self.n_jobs = n_jobs
        
        # Initialize the XG Regressor model
        self.xg_regressor = xgb.XGBRegressor(random_state=self.random_state, 
                                             n_estimators=self.n_estimators,
                                             learning_rate=self.learning_rate,
                                             max_depth=self.max_depth,
                                             gamma=self.gamma, 
                                             subsample=self.subsample, 
                                             colsample_bytree=self.colsample_bytree,
                                             n_jobs=self.n_jobs)
        
        # Train the XG regressor
        self.xg_regressor.fit(self.x_train_df, self.y_train_df['label'])

        # Get feature importances
        self.importances_reg_xg = self.xg_regressor.feature_importances_
        
        # Create a DataFrame for better readability and sorting
        self.feature_names_reg_xg = self.x_train_df.columns
        self.importance_reg_xg = pd.DataFrame({
            'Feature': self.feature_names_reg_xg,
            'Importance': self.importances_reg_xg
        }).sort_values(by='Importance',ascending=False)

        # Above Threshold features df
        self.above_threshold_features_xg=self.importance_reg_xg[self.importance_reg_xg['Importance']>=self.threshold]

    def XG_plotting(self):
        # --- Visualization ---
        # Top 50 Important Features
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Importance', x='Feature', data=self.importance_reg_xg.head(50),ax=my_ax[0])
        my_ax[0].set_title('Top 50 Features Basis XG Regressor Feature Importances')
        my_ax[0].set_ylabel('Importance')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.importance_reg_xg)), y=self.importance_reg_xg['Importance'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis XG Regressor in Descending Order')
        my_ax[1].set_ylabel('Feature Importance')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()


    def above_threshold_features_XG(self):
        # Retaining features with importance above threshold - RF Regressor
        print(f"Features Importances List - XG Regressor: {self.importance_reg_xg.shape}")
        print(f"Above Threshold Features Importances  List - XG Regressor: {self.above_threshold_features_xg.shape}")

        self.x_train_reduced_post_pearson_MI=self.x_train_df.copy()
        self.above_threshold_features_xg=self.above_threshold_features_xg
        self.x_train_reduced_post_pearson_MI_XG=self.x_train_reduced_post_pearson_MI[self.above_threshold_features_xg['Feature']]
        return self.above_threshold_features_xg, self.x_train_reduced_post_pearson_MI_XG, self.xg_regressor

remove_post_XG_instance=XG_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, label_col='label', threshold=0.015, random_state=42, n_estimators=2000, learning_rate=0.05, max_depth=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7, n_jobs=-1)
remove_post_XG_instance.XG_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around XG importance score of 30, hence we can keep the threshold at 0.015.
above_threshold_features_xg, x_train_reduced_post_pearson_MI_XG, model_xg =remove_post_XG_instance.above_threshold_features_XG()
x_test_reduced_post_pearson_MI_XG=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_XG.columns]


#Training the XG regressor on top features as per XG
XG_FeatureAnalyzer_instance_two=XG_FeatureAnalyzer(x_train_reduced_post_pearson_MI_XG, y_train_original_processed, label_col='label', threshold=30, random_state=42)
x_train_reduced_post_pearson_MI_XG_two, Imp_features_XG_two, model_XG_two=XG_FeatureAnalyzer_instance_two.above_threshold_features_XG()

#Predictions from XG 
XG_pred=model_XG_two.predict(x_test_reduced_post_pearson_MI_XG)
XG_pred_df=pd.DataFrame(XG_pred,columns=['pred_label from XG top features - Original'])

m1_XG = XG_pred_df['pred_label from XG top features - Original'].round(3)
m2_XG = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_XG = m1_XG.corr(m2_XG)
print(f"Pearson Correlation for XG Regressor from XG's best features - original: {corr_pandas_RF}")

#Accuracy as per RMSE
rmse_XG = np.sqrt(mean_squared_error(m1_XG,m2_XG))
print(f"Root Mean Squared Error for XG Regressor from XG's best features - original: {rmse_RF:.4f}")

#Accuracy as per R2
r2_XG = r2_score(m1_XG,m2_XG)
print(f"R2 score for XG from XG's best features - original: {r2_XG:.4f}")


# Class for analyzing feature importance through SVM Regressor
class SVR_FeatureAnalyzer:
    # Constructor
    def __init__(self, x_train_df, y_train_df, x_test_df, y_test_df, label_col=None, timestamps_num=None, threshold=None, kernel=None, C=None, epsilon=None, cache_size=None, n_repeats=None, random_state=None, n_jobs=None):
        self.timestamps_num = timestamps_num
        self.x_train_df = x_train_df.tail(self.timestamps_num)
        self.y_train_df = y_train_df.tail(self.timestamps_num)
        self.x_test_df = x_test_df
        self.y_test_df = y_test_df
        self.label = label_col
        self.threshold = threshold
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize the SVM Regressor model
        self.svr_model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, cache_size=self.cache_size)
        
        # Train the SVM regressor
        self.svr_model.fit(self.x_train_df, self.y_train_df['label'])

        # Predict from SVM regressor
        self.y_pred_array = self.svr_model.predict(x_test_df)
        self.y_pred = pd.DataFrame(self.y_pred_array, columns=['Prediction from SVR'])

        # Evaluation metrics on SVM regressor - Pearson/RMSE/R2
        print(self.y_test_df)
        print(self.y_pred)
        
        self.SVR_pearson = self.y_pred['Prediction from SVR'].corr(self.y_test_df['label'])
        self.SVR_RMSE = np.sqrt(mean_squared_error(self.y_pred['Prediction from SVR'],self.y_test_df['label']))
        self.SVR_R2 = r2_score(self.y_pred['Prediction from SVR'],self.y_test_df['label'])

        print("\n--- SVR Model Results (RBF Kernel) ---")
        print(f"Test Set R-squared (R²): {self.SVR_R2:.4f}")
        print(f"Test Set RMSE: {self.SVR_RMSE:.4f}")
        print(f"Test Set Pearson Correlation (Actual vs. Predicted): {self.SVR_pearson:.4f}")
        
        # Feature Importance using Permutation Importance (Model Agnostic)
        print("\n--- 5. Feature Importance: Permutation Importance ---")
        
        # Use permutation importance on the held-out test set (X_test)
        # The scoring metric used here (default is r2) measures the decrease in score 
        # when a feature is randomly shuffled.
        self.result = permutation_importance(self.svr_model, self.x_test_df.tail(self.timestamps_num), 
                                        self.y_test_df.tail(self.timestamps_num), n_repeats=self.n_repeats, random_state=self.random_state, n_jobs=self.n_jobs)

        # Get the indices that sort the importance scores in descending order
        self.sorted_idx = self.result.importances_mean.argsort()[::-1]
        print(self.sorted_idx)

        # Create a DataFrame for better readability and sorting
        self.feature_names_reg_svr = self.x_train_df.columns
        self.importance_reg_svr = pd.DataFrame({
            'Feature': self.feature_names_reg_svr,
            'Importance': self.sorted_idx
        }).sort_values(by='Importance',ascending=False)

        # Above Threshold features df
        self.above_threshold_features_svr=self.importance_reg_svr[self.importance_reg_svr['Importance']>=self.threshold]

    def SVR_plotting(self):
        
        # Top 50 Important Features as per SVR
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Importance', x='Feature', data=self.importance_reg_svr.head(50),ax=my_ax[0])
        my_ax[0].set_title('Top 50 Features Basis SVM Regressor Feature Importances')
        my_ax[0].set_ylabel('Importance')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.importance_reg_svr)), y=self.importance_reg_svr['Importance'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis SVM Regressor in Descending Order')
        my_ax[1].set_ylabel('Feature Importance')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()

    def above_threshold_features_SVR(self):
        # Retaining features with importance above threshold - RF Regressor
        print(f"Features Importances List - SVM Regressor: {self.importance_reg_svr.shape}")
        print(f"Above Threshold Features Importances  List - SVM Regressor: {self.above_threshold_features_svr.shape}")

        self.x_train_reduced_post_pearson_MI=self.x_train_df.copy()
        self.above_threshold_features_svr=self.above_threshold_features_svr
        self.x_train_reduced_post_pearson_MI_SVR=self.x_train_reduced_post_pearson_MI[self.above_threshold_features_svr['Feature']]
        return self.above_threshold_features_svr, self.x_train_reduced_post_pearson_MI_SVR, self.svr_model


SVR_FeatureAnalyzer_instance=SVR_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, x_test_reduced_post_pearson_MI, y_test_original_processed, label_col='label', threshold=0, timestamps_num=10000, kernel='rbf', C=1, epsilon=0.1, cache_size=500, n_repeats=5, random_state=42, n_jobs=-1)
SVR_FeatureAnalyzer_instance.SVR_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around LGBM importance score of 30, hence we can keep the threshold at 30.

above_threshold_features_svr, x_train_reduced_post_pearson_MI_SVR, model_svr = SVR_FeatureAnalyzer_instance.above_threshold_features_SVR()
x_test_reduced_post_pearson_MI_SVR=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_SVR.columns]


#Training the SVR regressor on top features as per SVR                                             
SVR_FeatureAnalyzer_instance_two=SVR_FeatureAnalyzer(x_train_reduced_post_pearson_MI_SVR, y_train_original_processed, x_test_reduced_post_pearson_MI_SVR, y_test_original_processed, label_col='label', threshold=0, timestamps_num=10000, kernel='rbf', C=1, epsilon=0.1, cache_size=500, n_repeats=5, random_state=42, n_jobs=-1)
above_threshold_features_svr_two, x_train_reduced_post_pearson_MI_SVR_two, model_SVR_two=SVR_FeatureAnalyzer_instance_two.above_threshold_features_SVR()

#Predictions from SVR 
SVR_pred=model_SVR_two.predict(x_test_reduced_post_pearson_MI_SVR)
SVR_pred_df=pd.DataFrame(SVR_pred,columns=['pred_label from SVR top features - Original'])

m1_SVR = SVR_pred_df['pred_label from SVR top features - Original'].round(3)
m2_SVR = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_SVR = m1_SVR.corr(m2_SVR)
print(f"Pearson Correlation for SVR Regressor from SVR's best features - original: {corr_pandas_SVR}")

#Accuracy as per RMSE
rmse_SVR = np.sqrt(mean_squared_error(m1_SVR,m2_SVR))
print(f"Root Mean Squared Error for SVR Regressor from SVR's best features - original: {rmse_SVR:.4f}")

#Accuracy as per R2
r2_SVR = r2_score(m1_SVR,m2_SVR)
print(f"R2 score for SVR from SVR's best features - original: {r2_SVR:.4f}")


class ANN_FeatureAnalyzer:
    # --- Existing __init__ method here (omitted for brevity) ---
    def __init__(self, x_train_df, y_train_df, x_test_df, y_test_df, label_col=None, threshold=None, timestamps_num=None, activation=None, output_activation=None, optimizer=None, loss=None, metrics=None, monitor=None, patience=None, restore_best_weights=None, verbose=None, validation_split=None, epochs=None, batch_size=None, squared=None, n_repeats=None, random_state=None, n_jobs=None, scoring=None):
        self.timestamps_num = timestamps_num
        self.x_train_df = x_train_df.tail(self.timestamps_num)
        self.y_train_df = y_train_df.tail(self.timestamps_num)
        self.x_test_df = x_test_df
        self.y_test_df = y_test_df
        self.label = label_col
        self.input_dim = self.x_train_df.shape[1]
        self.threshold = threshold
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.squared = squared
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping = None # Will be instantiated in model_prep
        self.scoring = scoring
        
    # --- Your Keras Model Builder (with corrected return) ---
    def build_ann_model(self):
        self.ann_model = Sequential([
            Dense(64, activation=self.activation, input_shape=(self.input_dim,)),
            Dropout(0.2),
            Dense(32, activation=self.activation),
            Dropout(0.2),
            Dense(16, activation=self.activation),
            Dense(1, activation=self.output_activation)
        ])
        # CORRECTED: Return the Keras model object
        return self.ann_model 

    # --- Keras Compilation and Setup (Your provided method) ---
    def model_prep(self):
        # 1. Build the model
        self.build_ann_model()
        
        # 2. Compile the model
        self.ann_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        # 3. Define Early Stopping (instantiated here)
        self.early_stopping = EarlyStopping(
            monitor=self.monitor, 
            patience=self.patience, 
            restore_best_weights=self.restore_best_weights, 
            verbose=self.verbose
        )
        return self.ann_model # Return the compiled model

    # -----------------------------------------------------------------
    # SCALAR METHODS (MANDATORY FOR SCIKIT-LEARN COMPLIANCE)
    # These are used internally by permutation_importance
    # -----------------------------------------------------------------

    def fit(self, X_train_data=None, y_train_data=None):
        """
        [MANDATORY SCORING METHOD] Satisfies the Scikit-learn API check.
        Performs the actual Keras training by calling model_prep().
        """
        # The Canvas code correctly determines which data to use:
        X_data = X_train_data if X_train_data is not None else self.x_train_df
        y_data = y_train_data if y_train_data is not None else self.y_train_df

        self.input_dim = X_data.shape[1] 
        
        # Call model_prep to build, compile, and define callbacks
        self.model_prep() 
        
        # Perform the actual training (using user's variable name self.ann_fit)
        # NOTE: We use self.x_train_df and self.y_train_df for Keras training
        self.ann_fit = self.ann_model.fit(
            X_data, y_data, 
            validation_split=self.validation_split, 
            epochs=self.epochs, 
            batch_size=self.batch_size,
            callbacks=[self.early_stopping], # Ensure callbacks are passed as a list
            verbose=self.verbose
        )
        # MUST return the estimator instance itself
        return self 

    def predict(self, X):
        """
        [MANDATORY SCORING METHOD] Uses the trained Keras model for prediction.
        """
        if not hasattr(self, 'ann_model'):
            # Note: We rely on permutation_importance calling .fit() first.
            raise AttributeError("ANN model has not been built/trained. Call .fit() first.")
            
        # X is passed from permutation_importance as the data slice
        # Keras predict returns 2D, flatten to 1D for scikit-learn metrics
        return self.ann_model.predict(X, verbose=0).flatten()

    def score(self, X, y):
        """
        [MANDATORY SCORING METHOD] Calculates the R^2 score for the Scikit-learn API.
        """
        y_pred = self.predict(X)
        
        # Ensure target 'y' is also a flat array for r2_score comparison
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
             y_flat = y.values.flatten()
        else:
             y_flat = y.flatten()

        return r2_score(y_flat, y_pred)
    
    # -----------------------------------------------------------------
    # COMPREHENSIVE EVALUATION METHOD
    # -----------------------------------------------------------------
    
    def model_score(self):
        """
        Trains the model, runs predictions on the full test set, 
        and calculates all primary evaluation metrics (RMSE, R2, Pearson r).
        """
        # Call fit() with required arguments to train the model
        self.fit(self.x_train_df, self.y_train_df)
        
        # Evaluate ANN on the full X_test
        # Predictions from ANN - Use the full test set data stored in self.
        self.y_pred_ann = pd.DataFrame(
            self.ann_model.predict(self.x_test_df, verbose=self.verbose).flatten(), 
            columns=['Predictions from ANN']
        )
        
        # Calculate metrics (using self.label for robustness)
        target_col = self.label or 'label'
        
        self.ann_rmse = mean_squared_error(
            self.y_test_df[target_col], 
            self.y_pred_ann['Predictions from ANN'], 
            squared=self.squared
        )
        self.ann_r2 = r2_score(
            self.y_test_df[target_col], 
            self.y_pred_ann['Predictions from ANN']
        )
        self.ann_pearson, _ = pearsonr(
            self.y_test_df[target_col], 
            self.y_pred_ann['Predictions from ANN']
        )
        
        # Print results for quick check
        print(f"ANN Model Results: R2={self.ann_r2:.4f}, RMSE={self.ann_rmse:.4f}, Pearson={self.ann_pearson:.4f}")
        
        return self.ann_r2, self.ann_model
    
    # -----------------------------------------------------------------
    # Feature Analysis Method 
    # -----------------------------------------------------------------
    
    def model_permutation(self):
        """
        Calculates permutation importance on the fitted estimator.
        """
        # Ensure the model is trained before running feature importance
        if not hasattr(self, 'ann_model') or not hasattr(self, 'ann_fit'):
            self.fit(self.x_train_df, self.y_train_df)
            
        # 1. Calculate Permutation Importance
        self.result = permutation_importance(
            self, # The estimator (now Scikit-learn compliant)
            # Using the last portion of the test set for relevance
            self.x_test_df.tail(self.timestamps_num), 
            self.y_test_df.tail(self.timestamps_num), 
            n_repeats=self.n_repeats, 
            random_state=self.random_state, 
            n_jobs=self.n_jobs, 
            scoring=self.scoring
        )
        
        # 2. Get the indices that sort the importance scores in descending order
        self.sorted_idx = self.result.importances_mean.argsort()[::-1]
        
        # 3. Create correctly sorted arrays using the index array (self.sorted_idx)
        sorted_importance_means = self.result.importances_mean[self.sorted_idx]
        sorted_feature_names = self.x_train_df.columns[self.sorted_idx]

        # 4. Create the final DataFrame using the actual sorted scores
        self.importance_reg_ANN = pd.DataFrame({
            'Feature': sorted_feature_names,
            # Use the actual sorted scores, NOT the index array
            'Importance': sorted_importance_means 
        })
        
        # 5. Filter features above the threshold
        self.above_threshold_features_ANN_df = self.importance_reg_ANN[
            self.importance_reg_ANN['Importance'] >= self.threshold
        ]
        
        print("Feature Importances (ANN Regressor):\n", self.importance_reg_ANN)
        
        return self.importance_reg_ANN, self.above_threshold_features_ANN_df, self.ann_model
    
    # --- Placeholder for ANN_plotting (Ensuring it calls model_permutation) ---
    def ANN_plotting(self):
        # We need to import matplotlib and seaborn at the top of the file for this to work
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # The plotting method MUST call the permutation method first
        self.importance_reg_ANN, self.above_threshold_features_ANN_df, self.ann_model = self.model_permutation()
        
        # Now plot the results
        fig, my_ax = plt.subplots(nrows=1, ncols=2,figsize=(16, 7))
        sns.barplot(y='Importance', x='Feature', data=self.importance_reg_ANN.head(50),ax=my_ax[0])
        my_ax[0].set_title('Top 50 Features Basis ANN Regressor Feature Importances')
        my_ax[0].set_ylabel('Importance')
        my_ax[0].set_xlabel('Feature')
        my_ax[0].tick_params(axis='x', rotation=90)
        
        # Create a line plot of the ranked feature importances
        sns.lineplot(x=range(len(self.importance_reg_ANN)), y=self.importance_reg_ANN['Importance'],ax=my_ax[1])
        my_ax[1].set_title('Feature Importance Basis ANN Regressor in Descending Order')
        my_ax[1].set_ylabel('Feature Importance')
        my_ax[1].set_xlabel('Feature Rank')
        
        plt.tight_layout()
        plt.show()

    # --- Placeholder for above_threshold_features_ANN (Ensuring it calls ANN_plotting) ---
    def above_threshold_features_ANN(self):
        # Calling ANN_plotting ensures all calculations are run and saved to self.
        # This will indirectly ensure self.model_permutation() is run.
        self.ANN_plotting() 
        
        # Retaining features with importance above threshold
        print(f"Features Importances List - ANN Regressor: {self.importance_reg_ANN.shape}")
        
        # The following variables were created and saved inside model_permutation()
        x_train_reduced = self.x_train_df[self.above_threshold_features_ANN_df['Feature']]
        
        return self.above_threshold_features_ANN_df, x_train_reduced, self.ann_model

ANN_FeatureAnalyzer_instance=ANN_FeatureAnalyzer(x_train_reduced_post_pearson_MI, y_train_original_processed, x_test_reduced_post_pearson_MI, y_test_original_processed, label_col='label', threshold=-0.025, timestamps_num=10000, activation='relu', output_activation='linear', optimizer='adam', loss='mse', metrics=['mae'], monitor='val_loss', patience=15, restore_best_weights=True, verbose=0, validation_split=0.2, epochs=100, batch_size=256, squared=None, n_repeats=3, random_state=42, n_jobs=-1, scoring='r2')
above_threshold_features_ANN, x_train_reduced_post_pearson_MI_ANN, model_ANN = ANN_FeatureAnalyzer_instance.above_threshold_features_ANN()

x_test_reduced_post_pearson_MI_ANN=x_test_reduced_post_pearson_MI[x_train_reduced_post_pearson_MI_ANN.columns]
#Inference:
#1.The line chart depicts the elbow point at somewhere around ANN Permutation importance score of -0.025, hence we can keep the threshold at -0.025


#Training the ANN regressor on top features as per ANN
ANN_FeatureAnalyzer_instance_two=ANN_FeatureAnalyzer(x_train_reduced_post_pearson_MI_ANN, y_train_original_processed, x_test_reduced_post_pearson_MI_ANN, y_test_original_processed, label_col='label', threshold=-0.025, timestamps_num=10000, activation='relu', output_activation='linear', optimizer='adam', loss='mse', metrics=['mae'], monitor='val_loss', patience=15, restore_best_weights=True, verbose=0, validation_split=0.2, epochs=100, batch_size=256, squared=None, n_repeats=3, random_state=42, n_jobs=-1, scoring='r2')
above_threshold_features_ANN_two, x_train_reduced_post_pearson_MI_ANN_two, model_ANN_two = ANN_FeatureAnalyzer_instance_two.above_threshold_features_ANN()

#Training the ANN regressor on top features as per ANN 
#ANN_FeatureAnalyzer_instance.fit(X_train_data=x_train_reduced_post_pearson_MI_ANN, y_train_data=ANN_FeatureAnalyzer_instance.y_train_df)

# Extract the ANN model from the ANN feature analyzer class
#model_ANN_reduced = ANN_FeatureAnalyzer_instance.ann_model

#Predictions from ANN model
ANN_pred=model_ANN_two.predict(x_test_reduced_post_pearson_MI_ANN)
ANN_pred_df=pd.DataFrame(ANN_pred,columns=['Predictions from ANN top features - Original'])

m1_ANN = ANN_pred_df['Predictions from ANN top features - Original'].round(3)
m2_ANN = y_test_original_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation
corr_pandas_ANN = m1_ANN.corr(m2_ANN)
print(f"Pearson Correlation from ANN's best features - Original: {corr_pandas_ANN}")

#Accuracy as per RMSE
rmse_ANN = np.sqrt(mean_squared_error(m1_ANN, m2_ANN))
print(f"Root Mean Squared Error from ANN's best features - Original: {rmse_ANN:.4f}")

#Accuracy as per R2
r2_ANN = r2_score(m1_ANN,m2_ANN)
print(f"R2 score for ANN from ANN's best features - original: {r2_ANN:.4f}")


# Class for Combining the sets - x_train_reduced_post_pearson_MI/x_test_reduced_post_pearson_MI and y_train_original_processed/y_test_original_processed
# as feature engineering should be done on entire x_train_DRW and then further split to train test sets

class combine_df():
    # constructor
    def __init__(self, x_train_df, x_test_df, y_train_df, y_test_df):
        self.x_train_df = x_train_df.copy()
        self.x_test_df = x_test_df.copy()
        self.y_train_df = y_train_df.copy()
        self.y_test_df = y_test_df.copy()

    def combine(self):
        self.combined_df_x=pd.concat([self.x_train_df,self.x_test_df],axis=0).reset_index(drop=True)
        self.combined_df_y=pd.concat([self.y_train_df,self.y_test_df],axis=0).reset_index(drop=True)
        return self.combined_df_x, self.combined_df_y
        
# Combining the sets - x_train_reduced_post_pearson_MI/x_test_reduced_post_pearson_MI and y_train_original_processed/y_test_original_processed
combine_df_instance=combine_df(x_train_reduced_post_pearson_MI,x_test_reduced_post_pearson_MI,y_train_original_processed, y_test_original_processed)
combined_df_x, combined_df_y=combine_df_instance.combine()


# # Class to engineer new features - Rolling Averages, EWMA and Lagging features for all Original non-redundant data post initial pruning- x_train_reduced_post_pearson) ---
import gc

def batch_columns(col_list, batch_size):
    for i in range(0, len(col_list), batch_size):
        yield col_list[i:i+batch_size]

class feature_engg():
    def __init__(self, combined_df_x, combined_df_y):
        self.combined_df_x = combined_df_x.copy()
        self.combined_df_y = combined_df_y.copy()
        
    def rolling_avg(self, batch_size=10):
        all_batches = []
        for col_batch in batch_columns(list(self.combined_df_x.columns), batch_size):
            rolling_avg_1_wk_df=pd.DataFrame(index=self.combined_df_x.index)
            rolling_avg_1_month_df=pd.DataFrame(index=self.combined_df_x.index)
            rolling_avg_1_qtr_df=pd.DataFrame(index=self.combined_df_x.index)

            for col in col_batch:
                rolling_avg_1_wk_df[f'Rolling Avg - 15 mins Period {col}'] = self.combined_df_x[col].rolling(15).mean()
                rolling_avg_1_month_df[f'Rolling Avg - 30 mins Period {col}'] = self.combined_df_x[col].rolling(30).mean()
                rolling_avg_1_qtr_df[f'Rolling Avg - 60 mins Period {col}'] = self.combined_df_x[col].rolling(60).mean()

            batch_df = pd.concat([rolling_avg_1_wk_df, rolling_avg_1_month_df, rolling_avg_1_qtr_df], axis=1)
            all_batches.append(batch_df)
            del rolling_avg_1_wk_df, rolling_avg_1_month_df, rolling_avg_1_qtr_df, batch_df
            gc.collect()

        return pd.concat(all_batches, axis=1)

    def EWMA_features(self, batch_size=10):
        all_batches = []
        for col_batch in batch_columns(list(self.combined_df_x.columns), batch_size):
            EWM_avg_1_wk_df=pd.DataFrame(index=self.combined_df_x.index)
            EWM_avg_1_month_df=pd.DataFrame(index=self.combined_df_x.index)
            EWM_avg_1_qtr_df=pd.DataFrame(index=self.combined_df_x.index)
            
            for col in col_batch:
                EWM_avg_1_wk_df[f'EWM Avg - 15 mins Period {col}'] = self.combined_df_x[col].ewm(span=15,min_periods=15).mean()
                EWM_avg_1_month_df[f'EWM Avg - 30 mins Period {col}'] = self.combined_df_x[col].ewm(span=30,min_periods=30).mean()
                EWM_avg_1_qtr_df[f'Rolling Avg - 60 mins Period {col}'] = self.combined_df_x[col].ewm(span=60,min_periods=60).mean()
            
            batch_df = pd.concat([EWM_avg_1_wk_df, EWM_avg_1_month_df, EWM_avg_1_qtr_df], axis=1)
            all_batches.append(batch_df)
            del EWM_avg_1_wk_df, EWM_avg_1_month_df, EWM_avg_1_qtr_df, batch_df
            gc.collect()

        return pd.concat(all_batches, axis=1)

    def lagging_features(self, batch_size=10):
        all_batches = []
        for col_batch in batch_columns(list(self.combined_df_x.columns), batch_size):
            Lag_10mins_df=pd.DataFrame(index=self.combined_df_x.index)
            Lag_30mins_df=pd.DataFrame(index=self.combined_df_x.index)
            Lag_60mins_df=pd.DataFrame(index=self.combined_df_x.index)
            Lag_720mins_df=pd.DataFrame(index=self.combined_df_x.index)
            Lag_1440mins_df=pd.DataFrame(index=self.combined_df_x.index)
            
            for col in col_batch:
                Lag_10mins_df[f'Lag - 15 Minutes {col}'] = self.combined_df_x[col].shift(periods=15)
                Lag_30mins_df[f'Lag - 30 Minutes {col}'] = self.combined_df_x[col].shift(periods=30)
                Lag_60mins_df[f'Lag - 60 Minutes {col}'] = self.combined_df_x[col].shift(periods=60)
                # Longer lags commented out in original
                # Lag_720mins_df[f'Lag - 720 Minutes {col}'] = self.combined_df_x[col].shift(periods=720)
                # Lag_1440mins_df[f'Lag - 1440 Minutes {col}'] = self.combined_df_x[col].shift(periods=1440)

            batch_df = pd.concat([Lag_10mins_df, Lag_30mins_df, Lag_60mins_df, Lag_720mins_df, Lag_1440mins_df], axis=1)
            all_batches.append(batch_df)
            del Lag_10mins_df, Lag_30mins_df, Lag_60mins_df, Lag_720mins_df, Lag_1440mins_df, batch_df
            gc.collect()

        return pd.concat(all_batches, axis=1)

    def final_engg_features(self):
        rolling_features_df = self.rolling_avg()
        ewm_features_df = self.EWMA_features()
        lag_features_df = self.lagging_features()

        self.Final_Engg_df_x = pd.concat([self.combined_df_x, rolling_features_df, ewm_features_df, lag_features_df], axis=1)
        self.Final_Engg_df_x = self.Final_Engg_df_x.loc[:, ~self.Final_Engg_df_x.columns.duplicated()]
        self.Final_Engg_df_x = self.Final_Engg_df_x.dropna().reset_index(drop=True)

        self.y_engg = self.combined_df_y.loc[self.Final_Engg_df_x.index]

        # Convert to float32
        for df in (self.Final_Engg_df_x, self.y_engg):
            for col in df.columns:
                if df[col].dtype != 'float32':
                    df[col] = df[col].astype('float32')

        print(f"Shape of Final Engineered Dataset: {self.Final_Engg_df_x.shape}")
        print(f"Shape of aligned y_train_engg: {self.y_engg.shape}")
        
        return self.Final_Engg_df_x, self.y_engg

feature_engg_instance = feature_engg(combined_df_x, combined_df_y)
x_engg, y_engg = feature_engg_instance.final_engg_features()


DataSplitter_instance_engg=DataSplitter(x_engg, y_engg,test_size=0.3, random_state=42, shuffle=False)
x_train_engg, x_test_engg, y_train_engg, y_test_engg=DataSplitter_instance_engg.train_test_set()


processed_dataframes_final_engg = {}

# outlier_treatement class methods call on x_train_engg exclusively on engineered features
x_processor_engg = outlier_treatement(dataframe=x_train_engg.iloc[:,50:], scaling_fit_dataset=x_train_engg.iloc[:,50:], threshold_for_std_dev=1e-9)
x_train_engg_processed = x_processor_engg.memory_optimize()
processed_dataframes_final_engg['x_train_engg_processed'] = x_train_engg_processed

# outlier_treatement class methods call on x_test_engg exclusively on engineered features
x_processor_engg.dataframe = x_test_engg.iloc[:,50:].copy()
x_test_engg_processed = x_processor_engg.memory_optimize()
processed_dataframes_final_engg['x_test_engg_processed'] = x_test_engg_processed

# outlier_treatement class methods call on y_train_engg exclusively on engineered features
y_processor_engg = outlier_treatement(dataframe=y_train_engg, scaling_fit_dataset=y_train_engg, threshold_for_std_dev=1e-9)
y_train_engg_processed = y_processor_engg.memory_optimize()
processed_dataframes_final_engg['y_train_engg_processed'] = y_train_engg_processed

# outlier_treatement class methods call on y_test_engg exclusively on engineered features
y_processor_engg.dataframe = y_test_engg.copy()
y_test_engg_processed = y_processor_engg.memory_optimize()
processed_dataframes_final_engg['y_test_engg_processed'] = y_test_engg_processed

# Storing results in new variables
x_train_engg_processed=processed_dataframes_final_engg['x_train_engg_processed']
x_test_engg_processed=processed_dataframes_final_engg['x_test_engg_processed']
y_train_engg_processed=processed_dataframes_final_engg['y_train_engg_processed']
y_test_engg_processed=processed_dataframes_final_engg['y_test_engg_processed']

#Combining the original features back with transformed engineered features
x_train_engg_processed=pd.concat([x_train_engg.iloc[:,0:50],x_train_engg_processed],axis=1)
x_test_engg_processed=pd.concat([x_test_engg.iloc[:,0:50],x_test_engg_processed],axis=1)


###### free the dictionary not needed anymore
del processed_dataframes_final_engg


#Final dropping of features with extreme std deviation features in x_test_engg_processed to increase signal to noise
x_test_engg_processed_reduced = x_test_engg_processed.drop(columns=x_test_engg_processed.columns[x_test_engg_processed.std() > 3])
x_train_engg_processed_reduced=x_train_engg_processed[x_test_engg_processed_reduced.columns]


# Class to analyse variances for all features
class feature_variance_analyzer():
    #Constructor
    def __init__(self, x_train_engg_df):
        self.x_train_engg_df = x_train_engg_df

    # Function to plot feature variances
    def feature_variance_plotting(self):
        # Calculate variances
        self.variances = self.x_train_engg_df.var().sort_values(ascending=True) 
        print("Top 10 features with lowest variance:\n", self.variances.head(10))
        print("\nTop 10 features with highest variance:\n", self.variances.tail(10))
        
        # Visualize the distribution of variances
        plt.figure(figsize=(12, 6))
        sns.histplot(self.variances, bins=50, kde=True)
        plt.title('Distribution of Feature Variances (Scaled Data)')
        plt.xlabel('Variance')
        plt.ylabel('Number of Features')
        plt.grid(True)
        plt.show()
        
        # Plot cumulative sum of variances to see how many features contribute to variance
        # This can help decide a cutoff if you're looking for a sharp drop-off
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(self.variances)), np.cumsum(self.variances.values))
        plt.title('Cumulative Sum of Variances')
        plt.xlabel('Number of Features (sorted by variance)')
        plt.ylabel('Cumulative Variance')
        plt.grid(True)
        plt.show()
        return self.variances

    # Function to drop features zero threshold
    def dropping_below_variance_threshold(self):
        self.feature_variance_plotting()
        more_than_zero_variance_cols = list(self.variances[self.variances > 0].index)
        self.x_train_engg_processed_post_variance=self.x_train_engg_df[more_than_zero_variance_cols]
        print(f"The shape of engineered dataset post dropping zero variances features: {self.x_train_engg_processed_post_variance.shape}")
        return self.x_train_engg_processed_post_variance
        
feature_variance_analyzer_instance_engg=feature_variance_analyzer(x_train_engg_processed_reduced)
x_train_engg_processed_dropped_zero_variance=feature_variance_analyzer_instance_engg.dropping_below_variance_threshold()
x_test_engg_processed_dropped_zero_variance=x_test_engg_processed_reduced[x_train_engg_processed_dropped_zero_variance.columns]


remove_multicollinearity_instance_engg=remove_multicollinearity(x_train_engg_processed_dropped_zero_variance, y_train_engg_processed,threshold=0.80,rows_to_include=40000,label='label',method='pearson')
x_train_engg_dropped_variance_pearson=remove_multicollinearity_instance_engg.multicollinear_treatment()
x_test_engg_dropped_variance_pearson=x_test_engg_processed_dropped_zero_variance[x_train_engg_dropped_variance_pearson.columns]


MIFeatureAnalyzer_instance_engg=MIFeatureAnalyzer(x_train_engg_dropped_variance_pearson, y_train_engg_processed, label_col='label', rows_to_include=50000, threshold=0.1)
MIFeatureAnalyzer_instance_engg.MI_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around 0.1 MI score, hence we can keep the threshold at 0.1.
x_train_engg_dropped_variance_pearson_MI=MIFeatureAnalyzer_instance_engg.MI_feature_drop()
x_test_engg_dropped_variance_pearson_MI=x_test_engg_dropped_variance_pearson[x_train_engg_dropped_variance_pearson_MI.columns]


lasso_FeatureAnalyzer_instance_engg=lasso_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, cv=5, random_state=42, max_iter=10000)
lasso_FeatureAnalyzer_instance_engg.lasso_plotting()
x_train_engg_dropped_variance_pearson_MI_lasso, Imp_features_lasso_engg,model_lasso_engg=lasso_FeatureAnalyzer_instance_engg.lasso_feature_drop()
x_test_engg_dropped_variance_pearson_MI_lasso=x_test_engg_dropped_variance_pearson_MI[x_train_engg_dropped_variance_pearson_MI_lasso.columns]

#Training the lasso regressor on top engg features as per Lasso
lasso_FeatureAnalyzer_instance_two_engg=lasso_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI_lasso, y_train_engg_processed, cv=5, random_state=42, max_iter=10000)
x_train_engg_dropped_variance_pearson_MI_lasso_two, Imp_features_lasso_two_engg, model_lasso_two_engg=lasso_FeatureAnalyzer_instance_two_engg.lasso_feature_drop()

#Predictions from lasso best features - engineered
lasso_pred_engg=model_lasso_two_engg.predict(x_test_engg_dropped_variance_pearson_MI_lasso)
lasso_pred_df_engg=pd.DataFrame(lasso_pred_engg,columns=['Pred_label from Lasso top features - Engineered'])

m1_lasso_engg = lasso_pred_df_engg['Pred_label from Lasso top features - Engineered'].round(3)
m2_lasso_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_lasso_engg = m1_lasso_engg.corr(m2_lasso_engg)
print(f"Pearson Correlation for Lasso from Lasso's best features - Engineered: {corr_pandas_lasso_engg}")

#Accuracy as per RMSE
rmse_pandas_lasso_engg = np.sqrt(mean_squared_error(m1_lasso_engg,m2_lasso_engg))
print(f"Root Mean Squared Error for Lasso from Lasso's best features - Engineered: {rmse_pandas_lasso_engg:.4f}")

#Accuracy as per R2 Score
r2_lasso_engg = r2_score(m1_lasso_engg,m2_lasso_engg)
print(f"R2 for Lasso from Lasso's best features - Engineered: {r2_lasso_engg:.4f}")



ridge_FeatureAnalyzer_instance_engg=ridge_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, cv=5)
ridge_FeatureAnalyzer_instance_engg.ridge_plotting()
x_train_engg_dropped_variance_pearson_MI_ridge, Imp_features_ridge_engg,model_ridge_engg=ridge_FeatureAnalyzer_instance_engg.ridge_feature_drop()
x_test_engg_dropped_variance_pearson_MI_ridge=x_test_engg_dropped_variance_pearson_MI[x_train_engg_dropped_variance_pearson_MI_ridge.columns]

#Ridge model trained on Best Features basis Ridge coefficients
ridge_FeatureAnalyzer_instance_two_engg=ridge_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI_ridge, y_train_engg_processed, cv=5)
x_train_engg_dropped_variance_pearson_MI_ridge_two, Imp_features_ridge_engg_two, model_ridge_engg_two=ridge_FeatureAnalyzer_instance_two_engg.ridge_feature_drop()

#Predictions from Ridge from Best features - engineered
ridge_pred_engg=model_ridge_engg_two.predict(x_test_engg_dropped_variance_pearson_MI_ridge)
ridge_pred_df_engg=pd.DataFrame(ridge_pred_engg,columns=['Pred_label from Ridge Top features - Engineered'])

m1_ridge_engg = ridge_pred_df_engg['Pred_label from Ridge Top features - Engineered'].round(3)
m2_ridge_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_ridge_engg = m1_ridge_engg.corr(m2_ridge_engg)
print(f"Pearson Correlation for Ridge from Ridge's best features - Engineered: {corr_pandas_ridge_engg}")

#Accuracy as per RMSE
rmse_ridge_engg = np.sqrt(mean_squared_error(m1_ridge_engg,m2_ridge_engg))
print(f"Root Mean Squared Error for Ridge from Ridge's best features - Engineered: {rmse_ridge_engg:.4f}")

#Accuracy as per R2 Score
r2_ridge_engg = r2_score(m1_ridge_engg,m2_ridge_engg)
print(f"R2 for Ridge from Ridge's best features - Engineered: {r2_ridge_engg:.4f}")


elastic_FeatureAnalyzer_instance_engg=elastic_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, cv=5, max_iter=1000, l1_ratio=0.5)
elastic_FeatureAnalyzer_instance_engg.elastic_plotting()
x_train_engg_dropped_variance_pearson_MI_elastic, Imp_features_elastic_engg,model_elastic_engg=elastic_FeatureAnalyzer_instance_engg.elastic_feature_drop()
x_test_engg_dropped_variance_pearson_MI_elastic=x_test_engg_dropped_variance_pearson_MI[x_train_engg_dropped_variance_pearson_MI_elastic.columns]

#Elastic Net model trained on Best Features - engineered basis Ridge coefficients 
elastic_FeatureAnalyzer_instance_two_engg=elastic_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI_elastic, y_train_engg_processed, cv=5, max_iter=1000, l1_ratio=0.5)
x_train_engg_dropped_variance_pearson_MI_elastic_two, Imp_features_elastic_engg_two, model_elastic_engg_two=elastic_FeatureAnalyzer_instance_two_engg.elastic_feature_drop()

#Predictions from Elastic Net - Best features engineered 
elastic_pred_engg=model_elastic_engg_two.predict(x_test_engg_dropped_variance_pearson_MI_elastic)
elastic_pred_df_engg=pd.DataFrame(elastic_pred_engg,columns=['Pred_label from Elastic top features - Engineered'])

m1_elastic_engg = elastic_pred_df_engg['Pred_label from Elastic top features - Engineered'].round(3)
m2_elastic_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_elastic_engg = m1_elastic_engg.corr(m2_elastic_engg)
print(f"Pearson Correlation for Elastic from Elastic's best features - Engineered: {corr_pandas_elastic_engg}")

#Accuracy as per RMSE
rmse_elastic_engg = np.sqrt(mean_squared_error(m1_elastic_engg,m2_elastic_engg))
print(f"Root Mean Squared Error for elastic from elastic's best features - Engineered: {rmse_elastic_engg:.4f}")

#Accuracy as per R2 Score
r2_elastic_engg = r2_score(m1_elastic_engg,m2_elastic_engg)
print(f"R2 for Elastic from Elastic's best features - Engineered: {r2_elastic_engg:.4f}")


remove_post_LGBM_instance_engg=LGBM_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, label_col='label', threshold=30, random_state=42)
remove_post_LGBM_instance_engg.LGBM_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around LGBM importance score of 30, hence we can keep the threshold at 30.
above_threshold_features_LGBM_engg,x_train_engg_dropped_variance_pearson_MI_LGBM, model_LGBM_engg = remove_post_LGBM_instance_engg.above_threshold_features_LGBM()
x_test_engg_dropped_variance_pearson_MI_LGBM=x_test_engg_dropped_variance_pearson_MI[x_train_engg_dropped_variance_pearson_MI_LGBM.columns]

#Training the LGBM regressor on top features as per LGBM
LGBM_FeatureAnalyzer_instance_engg_two=LGBM_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI_LGBM, y_train_engg_processed, label_col='label', threshold=30, random_state=42)
x_train_engg_dropped_variance_pearson_MI_LGBM_two, Imp_features_LGBM_two, model_LGBM_engg_two=LGBM_FeatureAnalyzer_instance_engg_two.above_threshold_features_LGBM()

#Predictions from LGBM 
LGBM_pred_engg=model_LGBM_engg_two.predict(x_test_engg_dropped_variance_pearson_MI_LGBM)
LGBM_pred_df_engg=pd.DataFrame(LGBM_pred_engg,columns=['Pred_label from LGBM top features - Engineered'])

m1_LGBM_engg = LGBM_pred_df_engg['Pred_label from LGBM top features - Engineered'].round(3)
m2_LGBM_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation on Best Features - LGBM:
corr_pandas_LGBM_engg = m1_LGBM_engg.corr(m2_LGBM_engg)
print(f"Pearson Correlation for LGBM Regressor from LGBM's best features - Engineered: {corr_pandas_LGBM_engg}")

#Accuracy as per RMSE
rmse_LGBM_engg = np.sqrt(mean_squared_error(m1_LGBM_engg,m2_LGBM_engg))
print(f"Root Mean Squared Error for LGBM Regressor from LGBM's best features - Engineered: {rmse_LGBM_engg:.4f}")

#Accuracy as per R2 score
r2_LGBM_engg = r2_score(m1_LGBM_engg,m2_LGBM_engg)
print(f"R2 Score for LGBM Regressor from LGBM's best features - Engineered: {r2_LGBM_engg:.4f}")

#Accuracy as per R2 Score
r2_LGBM_engg = r2_score(m1_LGBM_engg,m2_LGBM_engg)
print(f"R2 for LGBM from LGBM's best features - Engineered: {r2_LGBM_engg:.4f}")


RF_FeatureAnalyzer_instance_engg=RF_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, rows_to_include=50000,label_col='label', threshold=0.015, **tuned_hyperparameters)
RF_FeatureAnalyzer_instance_engg.RF_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around 0.015 RF importance score, hence we can keep the threshold at 0.015.
above_threshold_features_RF_engg,x_train_engg_dropped_variance_pearson_MI_RF, model_RF_engg=RF_FeatureAnalyzer_instance_engg.above_threshold_features()
x_test_engg_dropped_variance_pearson_MI_RF=x_test_engg_dropped_variance_pearson_MI[x_train_engg_dropped_variance_pearson_MI_RF.columns]

#Training the RF regressor on top features - engineered as per RF
RF_FeatureAnalyzer_instance_engg_two=RF_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI_RF, y_train_engg_processed, rows_to_include=50000,label_col='label', threshold=0.015, **tuned_hyperparameters)
x_train_engg_dropped_variance_pearson_MI_RF_two, Imp_features_RF_engg_two, model_RF_engg_two=RF_FeatureAnalyzer_instance_engg_two.above_threshold_features()

#Predictions from lasso 
RF_pred_engg=model_RF_engg_two.predict(x_test_engg_dropped_variance_pearson_MI_RF)
RF_pred_engg_df=pd.DataFrame(RF_pred_engg,columns=['Pred_label from RF top features - Engineered'])

m1_RF_engg = RF_pred_engg_df['Pred_label from RF top features - Engineered'].round(3)
m2_RF_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_RF_engg = m1_RF_engg.corr(m2_RF_engg)
print(f"Pearson Correlation for Random Forest from RF's best features - Engineered: {corr_pandas_RF_engg}")


#Accuracy as per RMSE
rmse_RF_engg = np.sqrt(mean_squared_error(m1_RF_engg,m2_RF_engg))
print(f"Root Mean Squared Error for Random Forest from RF's best features - Engineered: {rmse_RF_engg:.4f}")

#Accuracy as per R2 Score
r2_RF_engg = r2_score(m1_RF_engg,m2_RF_engg)
print(f"R2 Score for Random Forest from RF's best features - Engineered: {r2_RF_engg:.4f}")


XG_FeatureAnalyzer_instance_engg=XG_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, label_col='label', threshold=0.016, random_state=42, n_estimators=2000, learning_rate=0.05, max_depth=5, gamma=0.1, subsample=0.8, colsample_bytree=0.7, n_jobs=-1)
XG_FeatureAnalyzer_instance_engg.XG_plotting()
#Inference:
#1.The line chart depicts the elbow point at somewhere around LGBM importance score of 30, hence we can keep the threshold at 0.015.
above_threshold_features_XG_engg,x_train_engg_dropped_variance_pearson_MI_XG, model_XG_engg =XG_FeatureAnalyzer_instance_engg.above_threshold_features_XG()
x_test_engg_dropped_variance_pearson_MI_XG=x_test_engg_dropped_variance_pearson_MI[x_train_engg_dropped_variance_pearson_MI_XG.columns]

#Training the XG regressor on top features as per XG - Engineered
XG_FeatureAnalyzer_instance_engg_two=XG_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI_XG, y_train_engg_processed, label_col='label', threshold=0.015, random_state=42)
x_train_engg_dropped_variance_pearson_MI_XG_two, Imp_features_XG_engg_two, model_XG_engg_two=XG_FeatureAnalyzer_instance_engg_two.above_threshold_features_XG()

#Predictions from XG top features - Engineered
XG_pred_engg=model_XG_engg_two.predict(x_test_engg_dropped_variance_pearson_MI_XG)
XG_pred_engg_df=pd.DataFrame(XG_pred_engg,columns=['Pred_label from XG top features - Engineered'])

m1_XG_engg = XG_pred_engg_df['Pred_label from XG top features - Engineered'].round(3)
m2_XG_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:
corr_pandas_XG_engg = m1_XG_engg.corr(m2_XG_engg)
print(f"Pearson Correlation for XG Regressor from XG's best features - Engineered: {corr_pandas_XG_engg}")

#Accuracy as per RMSE
rmse_XG_engg = np.sqrt(mean_squared_error(m1_XG_engg,m2_XG_engg))
print(f"Root Mean Squared Error for XG Regressor from XG's best features - original: {rmse_XG_engg:.4f}")

#Accuracy as per R2
r2_XG_engg = r2_score(m1_XG_engg,m2_XG_engg)
print(f"R2 Score for XG Regressor from XG's best features - original: {r2_XG_engg:.4f}")


ANN_FeatureAnalyzer_instance_engg=ANN_FeatureAnalyzer(x_train_engg_dropped_variance_pearson_MI, y_train_engg_processed, x_test_engg_dropped_variance_pearson_MI, y_test_engg_processed, label_col='label', threshold=-0.025, timestamps_num=10000, activation='relu', output_activation='linear', optimizer='adam', loss='mse', metrics=['mae'], monitor='val_loss', patience=15, restore_best_weights=True, verbose=0, validation_split=0.2, epochs=100, batch_size=256, squared=None, n_repeats=3, random_state=42, n_jobs=-1, scoring='r2')
above_threshold_features_ANN_engg, x_train_reduced_post_pearson_MI_ANN_engg, model_ANN_engg = ANN_FeatureAnalyzer_instance_engg.above_threshold_features_ANN() 


x_test_reduced_post_pearson_MI_ANN_engg=x_test_engg_dropped_variance_pearson_MI[x_train_reduced_post_pearson_MI_ANN_engg.columns]

ANN_FeatureAnalyzer_instance_engg.fit(X_train_data=x_train_reduced_post_pearson_MI_ANN_engg, y_train_data=ANN_FeatureAnalyzer_instance_engg.y_train_df)

# Get the new, correctly-sized model object. 
# This variable (model_ANN_reduced_engg) is guaranteed to be the 39-feature model.
model_ANN_reduced_engg = ANN_FeatureAnalyzer_instance_engg.ann_model

#Predictions from ANN
ANN_pred_engg=model_ANN_reduced_engg.predict(x_test_reduced_post_pearson_MI_ANN_engg)
ANN_pred_engg_df=pd.DataFrame(ANN_pred_engg,columns=['Predictions from ANN top features - Engineered'])

m1_ANN_engg = ANN_pred_engg_df['Predictions from ANN top features - Engineered'].round(3)
m2_ANN_engg = y_test_engg_processed['label'].reset_index(drop=True).round(3)

#Accuracy as per Pearson Correlation:ed
corr_pandas_ANN_engg = m1_ANN_engg.corr(m2_ANN_engg)
print(f"Pearson Correlation from ANN's best features - Engineered: {corr_pandas_ANN_engg}")

#Accuracy as per RMSE
rmse_ANN_engg = np.sqrt(mean_squared_error(m1_ANN_engg, m2_ANN_engg))
print(f"Root Mean Squared Error from ANN's best features - Engineered: {rmse_ANN_engg:.4f}")

#Accuracy as per R2
r2_ANN_engg = r2_score(m1_ANN_engg, m2_ANN_engg)
print(f"R2 Score from ANN's best features - Engineered: {r2_ANN_engg:.4f}")



# Final Best Features 
final_best_df = pd.concat([x_train_engg_dropped_variance_pearson_MI_LGBM, y_train_engg_processed],axis=1)


final_engg_df.to_csv('submission.csv', index=False)


x_test_DRW, y_test_DRW


# Class for preprocessing of unseen test - x_test_DRW
class test_preprocessing_pipeline():
    # Constructor
    def __init__(self, x_test_df, y_test_df, x_processor, x_train_reduced, feature_engg_instance, x_processor_engg, x_train_eng_final):
        self.x_test_df = x_test_df
        self.y_test_df = y_test_df
        self.x_processor = x_processor
        self.x_train_reduced = x_train_reduced
        self.feature_engg_instance = feature_engg_instance
        self.x_processor_engg = x_processor_engg
        self.x_train_eng_final = x_train_eng_final

   # Function for transformation pipeline on final test data - 'x_test_DRW'
    def transformation(self):
        self.x_processor.dataframe = self.x_test_df.copy()
        self.x_test_DRW_processed = self.x_processor.memory_optimize()

        # Convert to float32 for RAM optimization
        #for col in self.x_test_DRW_processed.select_dtypes(include=['float64']).columns:
             #self.x_test_DRW_processed[col] = self.x_test_DRW_processed[col].astype('float32')
        
        del self.x_test_df  # free memory
        return self.x_test_DRW_processed

     # Function for feature dropping pipeline on final processed test data - 'x_test_DRW_processed'
    def feature_drop(self):
        self.transformation()
        self.x_test_DRW_processed_reduced = self.x_test_DRW_processed[self.x_train_reduced.columns]
         
        del self.x_test_DRW_processed  # free memory
        return self.x_test_DRW_processed_reduced

     # Function for engineering pipeline on final reduced test data - 'x_test_DRW_processed_reduced'
    def engineering(self):  
        self.feature_drop()
        self.feature_engg_instance.combined_df_x = self.x_test_DRW_processed_reduced
        self.feature_engg_instance.combined_df_y = self.y_test_df

        self.x_test_DRW_engineered, self.y_test_DRW_engineered = self.feature_engg_instance.final_engg_features()

        del self.x_test_DRW_processed_reduced, self.y_test_df  # free memory
        return self.x_test_DRW_engineered, self.y_test_DRW_engineered
    
    # Function for transformation pipeline on final engineered test data - 'x_test_DRW_engineered'
    def transformation_engg(self): 
        self.engineering()
        self.x_processor_engg.dataframe = self.x_test_DRW_engineered.copy()
        self.x_test_DRW_engg_processed = self.x_processor_engg.memory_optimize()

        # Convert engineered features to float32
        for col in self.x_test_DRW_engg_processed.select_dtypes(include=['float64']).columns:
            self.x_test_DRW_engg_processed[col] = self.x_test_DRW_engg_processed[col].astype('float32')

        # Combine first 51 original features back
        self.x_test_DRW_engg_processed = pd.concat([self.x_test_DRW_engineered.iloc[:, 0:51], self.x_test_DRW_engg_processed],axis=1)

        del self.x_test_DRW_engineered  # free memory
        return self.x_test_DRW_engg_processed
    
    # Function for feature dropping pipeline on final engineered and processed test data - 'x_test_DRW_engg_processed'
    def feature_drop_engg(self):
        self.transformation_engg()
        self.x_test_final = self.x_test_DRW_engg_processed[self.x_train_eng_final.columns]

        del self.x_test_DRW_engg_processed  # free memory
        return self.x_test_final


test_preprocessing_pipeline_instance=test_preprocessing_pipeline(x_test_DRW, y_test_DRW, x_processor, x_train_reduced_post_pearson_MI, feature_engg_instance, x_processor_engg, x_train_engg_dropped_variance_pearson_MI_LGBM)
x_test_DRW_final = test_preprocessing_pipeline_instance.feature_drop_engg()


DataSplitter_instance_final=DataSplitter(x_train_engg_dropped_variance_pearson_MI_LGBM,y_train_engg_processed,test_size=0.3, random_state=42, shuffle=False)

#Train Test split on train.parquet dataset
x_train_final, x_test_final, y_train_final, y_test_final=DataSplitter_instance_final.train_test_set()


#Predictions from Final Best Estimator - LGBM (trained on best top performing engineered features)
pred_final=model_LGBM_engg_two.predict(x_test_DRW_final)
pred_final_df=pd.DataFrame(pred_final,columns=['Pred_label from Top Best features'])




