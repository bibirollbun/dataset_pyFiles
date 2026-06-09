pip install Levenshtein pgmpy


import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

import statsmodels.api as sm


import os
import random
import warnings
import networkx as nx
import numpy as np 
import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2


import re
from itertools import combinations
import Levenshtein
from collections import defaultdict

from scipy.interpolate import LSQUnivariateSpline
from scipy.io import arff

from concurrent.futures import ThreadPoolExecutor, as_completed

import pgmpy.estimators as ests
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.metrics import structure_score 
from pgmpy.inference import BeliefPropagation
from pgmpy.inference import VariableElimination

import matplotlib.pyplot as plt


class generic_Utilities:
    '''Generic Utilities on Lists and Dataframes'''
    def __init__(self):
        self.allFunctions = {}
        self.allFunctions['List_Operartion_1'] = str("Function: intersection_of_lists(list1, list2), Returns : list3")
        self.allFunctions['List_Operartion_2'] = str("Function: difference_of_lists(list1, list2), Returns : list4")
        
        self.allFunctions['Folder_Operartion_1'] = str("Function: get_csv_column_names(folder_path, num_workers=4), Returns : column_names_dict")
        self.allFunctions['Folder_Operartion_2'] = str("Function: walk_through_folder(folder_path), Returns : None") 
        
        self.allFunctions['Plot_Operartion_1'] = str("Function: plot_NumericVscumSum(x, y), Returns : None")
        self.allFunctions['Plot_Operartion_2'] = str("Function: logistic_regression_with_roc(X, y), Returns: test_roc_auc")
        self.allFunctions['Plot_Operartion_3'] = str("Function: cart_with_roc(X, y), Returns: test_roc_auc")
        self.allFunctions['Plot_Operartion_4'] = str("Function: bayesian_network_with_roc(train, test, target_column), Returns: test_roc_auc")
        
        self.allFunctions['Dictionary_Operartion_1'] = str("Function: filter_and_sort_subsets(subset_counts, threshold), Returns: sorted_subsets")
        
        self.allFunctions['Dataframe_Operartion_1'] = str("Function: get_numeric_and_non_numeric_columns(df), Returns: list4, list5")
        self.allFunctions['Dataframe_Operartion_2'] = str("Function: remove_single_unique_or_all_nans(df), Returns: df")
        self.allFunctions['Dataframe_Operartion_3'] = str("Function: columns_with_missing_values(df), Returns: list6")
        self.allFunctions['Dataframe_Operartion_4'] = str("Function: fill_col_with_median(df, colNames), Returns: df")
        self.allFunctions['Dataframe_Operartion_5'] = str("Function: columns_with_more_than_X_percent_unique(df, colNames, perc), Returns: list7")
        self.allFunctions['Dataframe_Operartion_6'] = str("Function: convert_and_create_factorizedColumns(df, colNames), Returns: df, mapDict")
        self.allFunctions['Dataframe_Operartion_7'] = str("Function: fillMissing_predictFactorizedColumns(df, usable_cols, colName), Returns: df, mapDict")
        self.allFunctions['Dataframe_Operartion_9'] = str("Function: oneHotEncoded(df, columns_to_oneHot), Returns: new_df")
        
        self.allFunctions['FeatureEngineering_Operartion_1'] = str("Function: train_subset_counts_oneHot(df, target, value, options= COUNT, WIG, empericalProb, n_jobs=1), Returns: subset_counts")
        self.allFunctions['FeatureEngineering_Operartion_2'] = str("Function: train_LeastSquareSpline_fit(df, target, variable, degree), Returns: breakpoints_original")
        self.allFunctions['FeatureEngineering_Operartion_3'] = str("Function: train_UnivariateSpline_fit(df, target, variable, threshold), Returns: breakpoints_original")
        self.allFunctions['FeatureEngineering_Operartion_4'] = str("Function: train_cart_bins_with_plot(df, variableCol, targetCol, max_n_bins, n_jobs=1), Returns: breakpoints_original")
        
        self.allFunctions['FeatureCreation_Operartion_1'] = str("Function: create_combinedFeatures_df(df, required_cols, wanted_subsets,  options = INT, FRACTION), Returns: new_df")
        self.allFunctions['FeatureCreation_Operartion_2'] = str("Function: create_one_hot_encode_ranges(df, colName, required_columns, breakpoints), Returns: new_df")
       
        self.loadedFunctions = {}
        return None
        

    def walk_through_folder(self, folder_path):
        self.loadedFunctions['Folder_Operartion_2'] = str("Function: walk_through_folder(folder_path), Returns : None") 
        for root, dirs, files in os.walk(folder_path):
            print(f"Current directory: {root}")
            print("Subdirectories:", dirs)
            print("Files:", files)
            print()

        
    def get_csv_column_names(self, folder_path, num_workers=4):
        self.loadedFunctions['Folder_Operartion_1'] = str("Function: get_csv_column_names(folder_path, num_workers=4), Returns : column_names_dict")
        column_names_dict = {}

        # Function to read column names from a CSV file
        def read_columns(file_path):
            try:
                df = pd.read_csv(file_path, nrows=0)
                return file_path, df.columns.tolist()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return file_path, []

        # Traverse the directory and get all CSV file paths
        csv_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(read_columns, file): file for file in csv_files}

            for future in as_completed(future_to_file):
                file_path, columns = future.result()
                file_name = os.path.basename(file_path)
                column_names_dict[file_name] = columns

        return column_names_dict    
        
        
    def intersection_of_lists(self, list1, list2):
        self.loadedFunctions['List_Operartion_1'] = str("Function: intersection_of_lists(list1, list2), Returns : list3")
        return list(set(list1) & set(list2))


    def difference_of_lists(self, list1, list2):
        self.loadedFunctions['List_Operartion_2'] = str("Function: difference_of_lists(list1, list2), Returns : list4")
        return [item for item in list1 if item not in list2]
    
    
    def plot_NumericVscumSum(self, x, y):
        self.loadedFunctions['Plot_Operartion_1'] = str("Function: plot_NumericVscumSum(x, y), Returns : None")
        modified_y = np.cumsum(y)/(np.sum(y)+0.0000000000001)
        plt.plot(x, modified_y, marker='o', linestyle='-', color='b')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.show()
    
    
    def get_numeric_and_non_numeric_columns(self, df):
        self.loadedFunctions['Dataframe_Operartion_1'] = str("Function: get_numeric_and_non_numeric_columns(df), Returns: list4, list5")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Numeric columns: {numeric_cols}")
        print(f"Non-numeric columns: {non_numeric_cols}")
        return numeric_cols, non_numeric_cols


    def remove_single_unique_or_all_nans(self, df):
        self.loadedFunctions['Dataframe_Operartion_2'] = str("Function: remove_single_unique_or_all_nans(df), Returns: df")
        removed_columns = []
        for column in df.columns:
            if df[column].nunique() <= 1 or df[column].isna().all():
                removed_columns.append(column)
                df = df.drop(columns=[column])
        print(f"Removed columns due to all NaN or only 1 unique value: {removed_columns}")
        return df


    def columns_with_missing_values(self, df):
        self.loadedFunctions['Dataframe_Operartion_3'] = str("Function: columns_with_missing_values(df), Returns: list6")
        missing_cols = [col for col in df.columns if df[col].isna().any()]
        print(f"Missing data columns: {missing_cols}")
        return missing_cols


    def fill_col_with_median(self, df, colNames):
        self.loadedFunctions['Dataframe_Operartion_4'] = str("Function: fill_col_with_median(df, colNames), Returns: df")
        try:
            for col in colNames:
                median_value = df[col].median()
                df.fillna({col: median_value}, inplace=True)
                print("Done inputing missing numeric values with median for column :" + str(col))
            return df
        except:
            print("Columns that are not numeric might be included! Please Check. Returning original dataframe." )
            return df
            
        

    def columns_with_more_than_X_percent_unique(self, df, colNames, perc):
        self.loadedFunctions['Dataframe_Operartion_5'] = str("Function: columns_with_more_than_X_percent_unique(df, colNames, perc), Returns: list7")
        total_rows = len(df)
        threshold = total_rows * 0.01 * perc  
        cols_with_high_uniques = [col for col in colNames if df[col].nunique() > threshold]
        print(f"Columns with high uniques , >= {perc} %  of number of rows in the data: {cols_with_high_uniques}")
        return cols_with_high_uniques
    


    def convert_and_create_factorizedColumns(self, df, colNames):
        self.loadedFunctions['Dataframe_Operartion_6'] = str("Function: convert_and_create_factorizedColumns(df, colNames), Returns: df, mapDict")
        mapDict = {}
        try:
            for colName in colNames:
                df[colName] = df[colName].astype('object')
                df[colName], unique_values = pd.factorize(df[colName])
                # Create a mapping dictionary for the column
                mapDict[colName] = {value: i for i, value in enumerate(unique_values)}
            return df, mapDict
        except:
            print("Columns with missing values might be included! Please Check. Returning original dataframe and a empty dictionary" )
            return df, mapDict

        
    def fillMissing_predictFactorizedColumns(self, df, usable_cols, colName):
        self.loadedFunctions['Dataframe_Operartion_7'] = str("Function: fillMissing_predictFactorizedColumns(df, usable_cols, colName), Returns: df, mapDict")
        mapDict = {}
        try:
            df[colName] = df[colName].astype('object')
            df[colName], unique_values = pd.factorize(df[colName])
            mapDict[colName] = {value: i for i, value in enumerate(unique_values)}
            # Train the model to predict missing values
            non_missing_idx = df[colName] != -1  # Using -1 for factorized NaNs
            missing_idx = df[colName] == -1
            if missing_idx.sum() > 0:
                X_train = df.loc[non_missing_idx, usable_cols]
                y_train = df.loc[non_missing_idx, colName]
                print(y_train)
                X_test = df.loc[missing_idx,  usable_cols]
                model = LogisticRegression(max_iter=1000, solver ='lbfgs',  multi_class='auto')
                model.fit(X_train, y_train)
                # Predict the missing values
                predicted = model.predict(X_test)
                print(predicted)
                # Replace the missing values with the predicted values
                df.loc[missing_idx, colName] = predicted
            return df, mapDict
        except:
            print("Columns with non-numeric values might be included! Please Check. Returning original dataframe and a empty dictionary" )
            return df, mapDict
    
    
    def apply_meanDistance(self, df, colName, string_list):
        self.loadedFunctions['Dataframe_Operartion_8'] = str("Function: apply_meanDistance(df, colName, string_list), Returns: df")
        def calculate_meanDistanceFromAList(input_string, string_list):
            def sorensen_dice(a, b):
                def get_bigrams(string):
                # Generate bigrams from a string
                    return [string[i:i+2] for i in range(len(string)-1)]
            # Sørensen-Dice coefficient for two sets
                a_bigrams = set(get_bigrams(a))
                b_bigrams = set(get_bigrams(b))
                overlap = len(a_bigrams & b_bigrams)
                total = len(a_bigrams) + len(b_bigrams)
                if total == 0:
                    return 1.0 if a == b else 0.0  # Handle identical empty strings
                return 2 * overlap / total
            sum_Levenshtein = 0
            sum_sorensen_dice = 0
            for string in string_list:
                sum_Levenshtein = sum_Levenshtein + Levenshtein.distance(input_string, string)
                sum_sorensen_dice = sum_sorensen_dice + sorensen_dice(input_string, string)
            return float(sum_Levenshtein/len(string_list)),float(sum_sorensen_dice/len(string_list))
        # Calculate mean distances for each row and add a new column
        df[['mean_Levenshtein', 'mean_sorensen_dice']] = df[colName].apply(
            lambda x: pd.Series(calculate_meanDistanceFromAList(x, string_list))
        )
        return df
    
    
    def oneHotEncoded(self, df, columns_to_oneHot):
        self.loadedFunctions['Dataframe_Operartion_9'] = str("Function: oneHotEncoded(df, columns_to_oneHot), Returns: new_df")
        # Perform one-hot encoding on specified columns
        df_encoded = pd.get_dummies(df, columns=columns_to_oneHot, drop_first=True, dtype=int)
        return df_encoded

    
    def filter_and_sort_subsets(self, subset_counts, threshold):
        self.loadedFunctions['Dictionary_Operartion_1'] = str("Function: filter_and_sort_subsets(subset_counts, threshold), Returns: sorted_subsets")
        # Filter subsets based on the given threshold
        filtered_subsets = {subset: count for subset, count in subset_counts.items() if count > threshold}
        # Sort the filtered subsets based on their counts in descending order
        sorted_subsets = sorted(filtered_subsets.items(), key=lambda item: item[1], reverse=True)
        return sorted_subsets
    
    
    def create_combinedFeatures_df(self, df, required_cols, wanted_subsets, options):
        self.loadedFunctions['FeatureCreation_Operartion_1'] = str("Function: create_combinedFeatures_df(df, required_cols, wanted_subsets,  options = INT, FRACTION), Returns: new_df")
        orig_df = df.copy()
        for subset, _ in wanted_subsets:
            new_col_name = "_".join(subset) + "_combined"
            if(options=='FRACTION'):
                orig_df[new_col_name] = df[list(subset)].mean(axis=1)
            else:
                orig_df[new_col_name] = df[list(subset)].min(axis=1)

        # Return a dataframe with the id, target, and new fraction columns
        new_combined_columns = [("_".join(subset) + "_combined") for subset, _ in wanted_subsets]
        selected_columns = required_cols + new_combined_columns
        return orig_df[selected_columns]
    
    
    def create_one_hot_encode_ranges(self, df, colName, required_columns, breakpoints):
        self.loadedFunctions['FeatureCreation_Operartion_2'] = str("Function: create_one_hot_encode_ranges(df, colName, required_columns, breakpoints), Returns: new_df")
        # Ensure breakpoints are sorted
        breakpoints = sorted(breakpoints)

        # Create a new DataFrame with the required columns
        new_df = df[required_columns].copy()

        # Create one-hot encoded columns based on breakpoints
        for i in range(len(breakpoints) - 1):
            lower_bound = breakpoints[i]
            upper_bound = breakpoints[i + 1]
            col_name = f"{colName}_{lower_bound:.2f}to{upper_bound:.2f}"
            new_df[col_name] = np.where((df[colName] > lower_bound) & (df[colName] <= upper_bound), 1, 0)

        return new_df
    
    
    def train_subset_counts_oneHot(self, df, target, value, options, n_jobs=1):
        self.loadedFunctions['FeatureEngineering_Operartion_1'] = str("Function: train_subset_counts_oneHot(df, target, value, options= COUNT, WIG, empericalProb , n_jobs=1), Returns: subset_counts")
        # Filter rows where the target equals value
        df_target_value = df[df[target] == value].copy()
        valSum = df_target_value[target].sum()

        # Drop target column to get only the one-hot encoded columns
        one_hot_columns = df_target_value.drop(columns=[target]).columns

        # Dictionary to store subsets and their counts
        subset_counts = defaultdict(int)

        # Helper function to check if a subset has more than two one-hot columns from the same original column
        def valid_subset(subset):
            original_cols = [col.split('_')[0] for col in subset]
            return all(original_cols.count(col) <= 1 for col in original_cols)

        # Function to calculate counts for a subset
        def calculate_count(subset):
            subset_df = df_target_value[list(subset)]
            count = (subset_df.sum(axis=1) == len(subset)).sum()

            if options == 'COUNT':
                return subset, count
            elif options == 'WIG':
                wig_value = (1 / (len(subset) + 1)) * (count / len(df_target_value)) - (1 / 2 ** len(subset))
                return subset, wig_value
            elif options == 'empericalProb':
                emp_prob = count / valSum
                return subset, emp_prob
            else:
                return subset, count

        all_combinations = []
        for r in range(1, len(one_hot_columns) + 1):
            for subset in combinations(one_hot_columns, r):
                if valid_subset(subset):
                    all_combinations.append(subset)

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(calculate_count, subset) for subset in all_combinations]
            for future in as_completed(futures):
                subset, count = future.result()
                subset_counts[subset] = count

        return dict(subset_counts)
    
    
    def train_LeastSquareSpline_fit(self, df, target, variable, degree):
        self.loadedFunctions['FeatureEngineering_Operartion_2'] = str("Function: train_LeastSquareSpline_fit(df, target, variable, degree), Returns: breakpoints_original")
        # 1. Sort the dataframe based on the variable column.
        df_sorted = df.sort_values(by=variable)

        # 2. Convert all values in the sorted variable column to values between [0,1].
        df_sorted['TranformedVariable'] = (df_sorted[variable] - df_sorted[variable].min()) / (df_sorted[variable].max() - df_sorted[variable].min())

        # 3. Convert the target column to Cumulative Sum divided By Total Sum so that it is also between [0,1].
        df_sorted[target] = df_sorted[target].cumsum() / df_sorted[target].sum()

        # 4. Fit the best linear spline on the modified target based on the modified variable.
        # Define knot points (as degree + 1 points excluding the endpoints)
        num_knots = degree 
        knots = np.linspace(0, 1, num_knots + 2)[1:-1]  # exclude 0 and 1 as knots

        spline = LSQUnivariateSpline(df_sorted['TranformedVariable'], df_sorted[target], t=knots, k=degree)

        # 5. Return a list which contains all the break points of the fitted spline based on the original variable column
        breakpoints = spline.get_knots()
        breakpoints_original = df_sorted[variable].min() + breakpoints * (df_sorted[variable].max() - df_sorted[variable].min())
        return breakpoints_original
    
    
    def train_UnivariateSpline_fit(self, df, target, variable, threshold):
        self.loadedFunctions['FeatureEngineering_Operartion_3'] = str("Function: train_UnivariateSpline_fit(df, target, variable, threshold), Returns: breakpoints_original")

        # 1. Sort the dataframe based on the variable column.
        df_sorted = df.sort_values(by=variable)

        # 2. Convert all values in the sorted variable column to values between [0,1].
        df_sorted['TranformedVariable'] = (df_sorted[variable] - df_sorted[variable].min()) / (df_sorted[variable].max() - df_sorted[variable].min())

        # 3. Convert the target column to Cumulative Sum divided By Total Sum so that it is also between [0,1].
        df_sorted[target] = df_sorted[target].cumsum() / df_sorted[target].sum()

        spline = UnivariateSpline(df_sorted['TranformedVariable'], df_sorted[target], s=threshold)

        # 5. Return a list which contains all the break points of the fitted spline based on the original variable column
        breakpoints = spline.get_knots()
        breakpoints_original = df_sorted[variable].min() + breakpoints * (df_sorted[variable].max() - df_sorted[variable].min())

        return breakpoints_original
    
    
    def train_cart_bins_with_plot(self, df, variableCol, targetCol, max_n_bins, n_jobs=1):
        self.loadedFunctions['FeatureEngineering_Operartion_4'] = str("Function: train_cart_bins_with_plot(df, variableCol, targetCol, max_n_bins, n_jobs=1), Returns: breakpoints_original")
        best_auc = 0
        best_bins = []
        best_model = None

        def fit_cart_model(leaf_nodes):
            cart_model = DecisionTreeClassifier(max_leaf_nodes=leaf_nodes, random_state=42)
            cart_model.fit(df[[variableCol]], df[targetCol])
            predictions = cart_model.predict_proba(df[[variableCol]])[:, 1]
            auc = roc_auc_score(df[targetCol], predictions)
            thresholds = cart_model.tree_.threshold
            thresholds = thresholds[thresholds != -2]  # Remove dummy thresholds
            bins = sorted(thresholds)
            bins = [df[variableCol].min()] + bins + [df[variableCol].max()]
            return auc, cart_model, bins

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(fit_cart_model, leaf_nodes): leaf_nodes for leaf_nodes in range(max_n_bins, 1, -1)}

            for future in futures:
                try:
                    auc, cart_model, bins = future.result()
                    if auc > best_auc:
                        best_auc = auc
                        best_model = cart_model
                        best_bins = bins
                    else:
                        break
                except Exception as e:
                    print(f"An error occurred with leaf_nodes {futures[future]}: {e}")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(df[variableCol], df[targetCol], color='blue', label='Actual Values')

        for i in range(len(best_bins) - 1):
            df_pred = pd.DataFrame({variableCol: [best_bins[i], best_bins[i+1]]})
            plt.plot([best_bins[i], best_bins[i+1]], [best_model.predict(df_pred)[0], best_model.predict(df_pred)[1]], color='red', linewidth=2)

        plt.xlabel(variableCol)
        plt.ylabel(targetCol)
        plt.title('CART Fitted Model vs Actual Values')
        plt.legend()
        plt.show()
        return best_bins
    
    
    def logistic_regression_with_roc(self, X, y):
        self.loadedFunctions['Plot_Operartion_2'] = str("Function: logistic_regression_with_roc(X, y), Returns: test_roc_auc")
        # 1. Split the data into X, y train and X, y test with a proportion of test 0.2 randomly
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Fit the data on the training set using sm.Logit, print the summary of the fit using lgbfs
        logit_model = sm.Logit(y_train, X_train).fit(method='lbfgs')
        print(logit_model.summary())

        # 3. Predict on the Test Data
        y_train_pred = logit_model.predict(X_train)
        y_test_pred = logit_model.predict(X_test)

        # 4. Print The Test and Train roc_auc
        train_roc_auc = roc_auc_score(y_train, y_train_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)
        print(f"Train ROC AUC: {train_roc_auc}")
        print(f"Test ROC AUC: {test_roc_auc}")

        # 5. Plot the ROC_AUC for the model
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr_train, tpr_train, label=f"Train ROC AUC = {train_roc_auc:.2f}", color='blue')
        plt.plot(fpr_test, tpr_test, label=f"Test ROC AUC = {test_roc_auc:.2f}", color='red')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        # 6. Return the test roc_auc as a float
        return test_roc_auc
    
    
    def cart_with_roc(self, X, y):
        self.loadedFunctions['Plot_Operartion_3'] = str("Function: cart_with_roc(X, y), Returns: test_roc_auc")
        # 1. Split the data into X, y train and X, y test with a proportion of test 0.2 randomly
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # 2. Fit the data on the training set using CART
        cart_model = DecisionTreeClassifier(max_leaf_nodes=max(2*X_train.shape[1],10), random_state=42)
        cart_model.fit(X_train, y_train)

        # 3. Predict on the Test Data
        y_train_pred = cart_model.predict_proba(X_train)[:, 1]
        y_test_pred = cart_model.predict_proba(X_test)[:, 1]

        # 4. Print The Test and Train roc_auc
        train_roc_auc = roc_auc_score(y_train, y_train_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)
        print(f"Train ROC AUC: {train_roc_auc}")
        print(f"Test ROC AUC: {test_roc_auc}")

        # 5. Plot the ROC_AUC for the model
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr_train, tpr_train, label=f"Train ROC AUC = {train_roc_auc:.2f}", color='blue')
        plt.plot(fpr_test, tpr_test, label=f"Test ROC AUC = {test_roc_auc:.2f}", color='red')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        # 6. Return the test roc_auc as a float
        return test_roc_auc

    def bayesian_network_with_roc(self, train, test, target_column):
        self.loadedFunctions['Plot_Operation_4'] = str("Function: bayesian_network_with_roc(train, test, target_column), Returns: test_roc_auc")

        
        # 1. Initialize the TreeSearch object and estimate the structure using Chow-Liu algorithm
        est = TreeSearch(train, root_node=target_column)
        dag = est.estimate(estimator_type='chow-liu')
        
        # 2. Visualize the DAG
        pos = nx.spiral_layout(dag)
        nx.draw(dag, pos=pos, with_labels=True, node_color='b', font_size=7, arrowstyle='fancy', alpha=0.8)
        plt.savefig('plt.png', dpi=700)
        
        # 3. Convert DAG to Pandas Edgelist
        edges = nx.to_pandas_edgelist(dag)
        
        # 4. Initialize the Bayesian Network and fit the model
        model = BayesianNetwork(dag)
        model.fit(train)
        
        # 5. Get the Conditional Probability Distributions (CPDs)
        cpds = model.get_cpds()
        for cpd in cpds:
            print(cpd)
        
        # 6. Predict on the Test Data
        y_pred = model.predict(test.drop(columns=[target_column]))
        
        # 7. Calculate and Print the Test ROC AUC
        test_roc_auc = roc_auc_score(test[target_column], y_pred[target_column])
        print(f"Test ROC AUC: {test_roc_auc}")
        
        # 8. Plot the ROC AUC for the model
        fpr_test, tpr_test, _ = roc_curve(predict_data[target_column], y_pred[target_column])
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_test, tpr_test, label=f"Test ROC AUC = {test_roc_auc:.2f}", color='red')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        # 9. Return the test roc_auc as a float
        return test_roc_auc



IsItGoingtoRain_utils = generic_Utilities()
IsItGoingtoRain_utils.allFunctions


data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
# concatenating first train data and test data, helpfull for applying same imputation and feature enginnearing.
data = pd.concat([data, test], ignore_index=True)
data


Target_Col = ['rainfall']
Identifier_Cols = ['id','day']
X_Cols = ['pressure', 'maxtemp', 'temparature',
          'mintemp','dewpoint', 'humidity',
          'cloud', 'sunshine', 'winddirection',
          'windspeed']


Numeric_Cols, Non_Numeric_Cols = IsItGoingtoRain_utils.get_numeric_and_non_numeric_columns(data[X_Cols])
MissingData_Cols = IsItGoingtoRain_utils.columns_with_missing_values(data[X_Cols])
GreaterThanTENpercUniQ_Cols = IsItGoingtoRain_utils.columns_with_more_than_X_percent_unique(data, Numeric_Cols, 10)
GreaterThanEIGHTpercUniQ_Cols = IsItGoingtoRain_utils.columns_with_more_than_X_percent_unique(data, Numeric_Cols, 8)
GreaterThanFIVEpercUniQ_Cols = IsItGoingtoRain_utils.columns_with_more_than_X_percent_unique(data, Numeric_Cols, 5)
GreaterThanONEpercUniQ_Cols = IsItGoingtoRain_utils.columns_with_more_than_X_percent_unique(data, Numeric_Cols, 1)
LessUniqueNA_Cols = IsItGoingtoRain_utils.intersection_of_lists(MissingData_Cols
    , IsItGoingtoRain_utils.difference_of_lists(GreaterThanONEpercUniQ_Cols, GreaterThanFIVEpercUniQ_Cols))
LessUniqueNA_Cols


ProcessedData_0 = IsItGoingtoRain_utils.remove_single_unique_or_all_nans(data)
ProcessedData_1_Median = IsItGoingtoRain_utils.fill_col_with_median(ProcessedData_0, ['winddirection'])
ProcessedData_1_Median


train_data = ProcessedData_1_Median[~ProcessedData_1_Median.rainfall.isnull()]
test_data = ProcessedData_1_Median[ProcessedData_1_Median.rainfall.isnull()]
target = 'rainfall'


baseline_logistic_ROCAUC = IsItGoingtoRain_utils.logistic_regression_with_roc(train_data[X_Cols], train_data[Target_Col])


baseline_CART_ROCAUC = IsItGoingtoRain_utils.cart_with_roc(train_data[X_Cols], train_data[Target_Col])


RawData = ProcessedData_1_Median[Target_Col + Identifier_Cols]


breakpoints_temparature_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'temparature', 'rainfall', 10, n_jobs=1)
breakpoints_maxtemp_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'maxtemp', 'rainfall', 9, n_jobs=1)
breakpoints_mintemp_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'mintemp', 'rainfall', 10, n_jobs=1)

breakpoints_winddirection_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'winddirection', 'rainfall', 5, n_jobs=1)
breakpoints_windspeed_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'windspeed', 'rainfall', 15, n_jobs=1)

breakpoints_Pressure_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'pressure', 'rainfall', 5, n_jobs=1)
breakpoints_sunshine_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'sunshine', 'rainfall', 10, n_jobs=1)
breakpoints_cloud_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'cloud', 'rainfall', 10, n_jobs=1)
breakpoints_humidity_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'humidity', 'rainfall', 10, n_jobs=1)
breakpoints_dewpoint_CART = IsItGoingtoRain_utils.train_cart_bins_with_plot(train_data, 'dewpoint', 'rainfall', 10, n_jobs=1)


breakpoints_temparatureCART = [5,10,18,20,30,35]
breakpoints_maxtempCART = [10,12,17,28,32,36]
breakpoints_mintempCART = [0,8,16,18,20,25,30]

breakpoints_winddirectionCART = [10.0, 22.5, 85.0, 235.0, 300.0]
breakpoints_windspeedCART = [4, 8.5, 10, 15, 20, 25, 60]

breakpoints_PressureCART = [999,1009,1021,1023,1050]
breakpoints_sunshineCART = [0.0,5,7.5,9,11,12.5]
breakpoints_cloudCART = [0, 64, 67, 73, 76, 84, 100]
breakpoints_humidityCART = [35, 57, 59.5, 63.5, 74.5, 80.5, 100.0]
breakpoints_dewpointCART = [-1,10,20, 30]

temparatureCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'temparature', ['id','day'], breakpoints_temparatureCART)
maxtempCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'maxtemp', ['id','day'], breakpoints_maxtempCART)
mintempCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'mintemp', ['id','day'], breakpoints_mintempCART)

winddirectionCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'winddirection', ['id','day'], breakpoints_winddirectionCART)
windspeedCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'windspeed', ['id','day'], breakpoints_windspeedCART)

pressureCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'pressure', ['id','day'], breakpoints_PressureCART)
sunshineCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'sunshine', ['id','day'], breakpoints_sunshineCART)
cloudCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'temparature', ['id','day'], breakpoints_cloudCART)
humidityCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'humidity', ['id','day'], breakpoints_humidityCART)
dewpointCART = IsItGoingtoRain_utils.create_one_hot_encode_ranges(ProcessedData_1_Median, 'dewpoint', ['id','day'], breakpoints_dewpointCART)


Step0_Data = pd.merge(RawData, 
                      temparatureCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      maxtempCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      mintempCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      winddirectionCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      windspeedCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      pressureCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      sunshineCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                      cloudCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                     humidityCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data = pd.merge(Step0_Data, 
                     dewpointCART,  
                      how='left', left_on=['id','day'], right_on = ['id','day'])
Step0_Data


Train0 = Step0_Data[Step0_Data.rainfall>-1] 
Train0 = Train0.drop(columns=['id','day'])


Train0


list(Train0.columns)


wanted_subsets = IsItGoingtoRain_utils.train_subset_counts_oneHot(Train0.head(100), 'rainfall' , 1, 'COUNT', n_jobs=1)


# Helper function to check if a subset has more than two one-hot columns from the same original column
def valid_subset(subset):
    original_cols = [col.split('_')[0] for col in subset]
    return all(original_cols.count(col) <= 1 for col in original_cols)

# Function to calculate counts for a subset
def calculate_count(subset):
    subset_df = df_target_value[list(subset)]
    count = (subset_df.sum(axis=1) == len(subset)).sum()

    if options == 'COUNT':
        return subset, count
    elif options == 'WIG':
        wig_value = (1 / (len(subset) + 1)) * (count / len(df_target_value)) - (1 / 2 ** len(subset))
        return subset, wig_value
    elif options == 'empericalProb':
        emp_prob = count / valSum
        return subset, emp_prob
    else:
        return subset, count


df_target_value = df[df[target] == value].copy()
valSum = df_target_value[target].sum()

# Drop target column to get only the one-hot encoded columns
one_hot_columns = df_target_value.drop(columns=[target]).columns

# Dictionary to store subsets and their counts
subset_counts = defaultdict(int)
all_combinations = []
for r in range(1, len(one_hot_columns) + 1):
    for subset in combinations(one_hot_columns, r):
        if valid_subset(subset):
            all_combinations.append(subset)

with ThreadPoolExecutor(max_workers=n_jobs) as executor:
    futures = [executor.submit(calculate_count, subset) for subset in all_combinations]
    for future in as_completed(futures):
        subset, count = future.result()
        subset_counts[subset] = count

return dict(subset_counts)


wanted_subsets = IsItGoingtoRain_utils.train_subset_counts_oneHot(, , ,)
sorted_subsets = IsItGoingtoRain_utils.filter_and_sort_subsets(wanted_subsets, threshold)
InteractionDF = IsItGoingtoRain_utils.create_combinedFeatures_df(df, ['id','day'], sorted_subsets,  'INT')

