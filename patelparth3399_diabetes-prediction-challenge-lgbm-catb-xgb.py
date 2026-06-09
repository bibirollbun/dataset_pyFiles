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


!nvidia-smi


import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")


# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


Train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
Test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


Train_df.head()


Test_df.head()


Train_df.shape


Test_df.shape


Train_df.columns


Test_df.columns


# Check for the null values 
Train_df.isnull().sum()


Test_df.isnull().sum()


# Check for the duplicated values 
Train_df.duplicated().sum()


Test_df.duplicated().sum()


Train_df.info()


# Store target column into a one variable 
Target_col = 'diagnosed_diabetes'


# Seperate categorical and numerical data from the main datasets.

Cats = Train_df.select_dtypes('object').columns.to_list()

Nums = [col for col in Train_df.columns if col not in Cats + ['id', 'diagnosed_diabetes'] ]


Cats 


Nums


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from IPython.display import display, Markdown


# Configure plot styles for better visualization
sns.set_style('whitegrid')

class DiabetesEDA:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) for the diabetes prediction dataset,
    with an emphasis on structured and clear output for notebook presentation.
    """
    def __init__(self, data: pd.DataFrame, target_col: str):
        self.data = data.copy()
        self.target_col = target_col
        
        # Define feature lists based on your input 
        self.numerical_cols = [
            'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day',
            'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol_total',
            'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'
        ]
        self.categorical_cols = [
            'gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status'
        ]
        # Ensure target is recognized as categorical (0 or 1)
        self.data[target_col] = self.data[target_col].astype('category')

    def run_all_eda(self):
        """Executes all stages of the EDA with clear separation."""
        display(Markdown("## ğŸ“Š Comprehensive EDA Report"))
        
        display(Markdown("---"))
        display(Markdown("## 1. Initial Data Inspection & Quality"))
        self.initial_inspection()
        
        display(Markdown("---"))
        display(Markdown("## 2. Numerical Feature Analysis"))
        self.analyze_numerical_features()
        
        display(Markdown("---"))
        display(Markdown("## 3. Categorical Feature Analysis"))
        self.analyze_categorical_features()
        
        display(Markdown("---"))
        display(Markdown("## 4. Target Variable Analysis"))
        self.analyze_target()


    # ------------ Stage 1: Initial Inspection & Data Quality -------------
    def initial_inspection(self):
        """Display data shape, info, and checks for missing values."""
        
        display(Markdown("### 1.1 Dataset Structure"))
        print(f"Dataset Shape: Rows = {self.data.shape[0]}, Columns = {self.data.shape[1]}")
        
        display(Markdown("### 1.2 Data Types and Non-Null Counts"))
        self.data.info()
        
        display(Markdown("### 1.3 Missing Values Analysis âš ï¸�"))
        missing_summary = self.data.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
        
        if not missing_summary.empty:
            missing_df = pd.DataFrame({
                'Count': missing_summary.values,
                'Percentage': (missing_summary.values / len(self.data) * 100).round(2)
            }, index=missing_summary.index)
            display(Markdown("**Features with Missing Values:**"))
            display(missing_df)
        else:
            print("âœ… No missing values found in the dataset.")


    # ------------ Stage 2: Numerical Features Analysis ----------------
    def analyze_numerical_features(self):
        """Performs univariate and bivariate analysis on numerical features."""
        
        display(Markdown("### 2.1 Descriptive Statistics & Outliers (Univariate)"))
        desc_stats = self.data[self.numerical_cols].describe().T
        display(desc_stats) # Display as a formatted table

        # Visualize distribution and outliers (Box Plots)
        display(Markdown("#### Box Plots (Visualizing Distribution and Outliers)"))
        plt.figure(figsize=(18, 12))
        for i, col in enumerate(self.numerical_cols):
            plt.subplot(4, 5, i + 1)
            sns.boxplot(y=self.data[col])
            plt.title(f'Box Plot: {col}', fontsize=10)
            plt.ylabel('')
        plt.tight_layout()
        plt.show()
        plt.close()

        
        display(Markdown("### 2.2 Feature-Feature Correlation Analysis"))
        corr_matrix = self.data[self.numerical_cols].corr()
        plt.figure(figsize=(12, 10))
        # Use annot=True sparingly for large matrix; False is cleaner for the full set
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Heatmap of Numerical Features', fontsize=14)
        plt.show()
        plt.close()


        display(Markdown("### 2.3 Bivariate Analysis (Numerical Feature vs Target)"))
        
        # Visualize difference in mean/medians (Target Group Comparison)
        display(Markdown("#### Distribution Comparison by Target Group (Box Plots)"))
        plt.figure(figsize=(18, 12))
        for i, col in enumerate(self.numerical_cols):
            plt.subplot(5, 4, i + 1)
            # Ensure target is plotted on x-axis
            sns.boxplot(x=self.target_col, y=col, data=self.data) 
            plt.title(f'{col} by Target', fontsize=10)
            plt.xlabel(f'Diabetes (0/1)')
        plt.tight_layout()
        plt.show()
        plt.close()


        # Statistical test (t-test) for mean difference 
        display(Markdown("#### Two-Sample T-test Results (Comparing Means for Target=0 vs Target=1)"))
        display(Markdown("_**Hâ‚€: No significant difference in means.** A p-value **< 0.05** suggests a statistically significant difference._"))
        
        ttest_result = {}
        
        for col in self.numerical_cols:
            clean_data = self.data[[col, self.target_col]].dropna()

            group_0 = clean_data[clean_data[self.target_col] == 0][col]
            group_1 = clean_data[clean_data[self.target_col] == 1][col]

            if len(group_0) > 1 and len(group_1) > 1:
                # Use Welch's t-test (equal_var=False) as a robust choice
                statistics, p_value= ttest_ind(group_0, group_1, equal_var=False)
                ttest_result[col] = f"{p_value:.4f}"
            else:
                ttest_result[col] = "Insufficient data"
        
        # Convert to Series and display as a formatted table, sorted by p-value
        ttest_series = pd.Series(ttest_result).sort_values()
        ttest_df = pd.DataFrame({'P-Value': ttest_series})
        display(ttest_df)


    # ---------- Stage 3. Categorical Feature Analysis ------------
    def analyze_categorical_features(self):
        """Performs univariate and bivariate analysis on categorical features."""

        display(Markdown("### 3.1 Frequency Counts (Univariate)"))
        for col in self.categorical_cols:
            display(Markdown(f"#### Feature: **{col}**"))
            # Calculate counts and percentages
            counts = self.data[col].value_counts(dropna=False)
            percentages = (self.data[col].value_counts(normalize=True, dropna=False) * 100).round(2)
            
            freq_df = pd.DataFrame({
                'Count': counts,
                'Percentage (%)': percentages
            })
            display(freq_df)


        display(Markdown("### 3.2 Bivariate Analysis (Categorical Feature vs Target)"))
        display(Markdown("#### Distribution of Target (Diabetes) within each Category"))
        
        plt.figure(figsize=(18, 12))
        for i, col in enumerate(self.categorical_cols):
            plt.subplot(2, 3, i + 1)
            # Use countplot with hue to see the target distribution
            sns.countplot(x=col, hue=self.target_col, data=self.data)
            plt.title(f"{col} Distribution by Target", fontsize=12)
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.legend(title=self.target_col)
        plt.tight_layout()
        plt.show()
        plt.close()
        
        
        # Statistical test (Chi-squared) for independence
        display(Markdown("#### Chi-squared Test for Independence (Categorical Feature vs Target)"))
        display(Markdown("_**Hâ‚€: Feature and Target are independent.** A p-value **< 0.05** suggests a dependency (association)._"))
        
        chi2_result = {}
        for col in self.categorical_cols:
            # Create the contingency table
            contingency_tabel = pd.crosstab(self.data[col], self.data[self.target_col])
        
            # Chi-squared test
            try:
                # Check for small expected frequencies (common issue)
                chi2, p, dof, expected = chi2_contingency(contingency_tabel)
                
                # A common heuristic check: warn if any expected value is too low (e.g., < 5)
                if np.min(expected) < 5 and np.sum(expected < 5) > 0.2 * np.size(expected):
                     chi2_result[col] = f"Warning: Small expected frequency ({p:.4f})"
                else:
                    chi2_result[col] = f"{p:.4f}"
            except ValueError:
                chi2_result[col] = "Error: Contingency table issue" # e.g., if a column is all NaN
        
        # Convert to Series and display as a formatted table, sorted by p-value
        chi2_series = pd.Series(chi2_result).sort_values()
        chi2_df = pd.DataFrame({'P-Value': chi2_series})
        display(chi2_df)


    # ---------- Stage 4: Target Analysis --------------
    def analyze_target(self):
        """Analyze the balance of the target variable"""
        
        display(Markdown("### 4.1 Target Variable Balance"))

        # Calculate counts and percentages
        target_counts = self.data[self.target_col].value_counts()
        target_percentages = (self.data[self.target_col].value_counts(normalize=True) * 100).round(2)
        
        target_df = pd.DataFrame({
            'Count': target_counts,
            'Percentage (%)': target_percentages
        })
        target_df.index.name = f'Target ({self.target_col})'
        display(target_df)

        
        # Visualization
        display(Markdown("#### Target Distribution Plot"))
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.target_col, data=self.data)
        plt.title(f"Target Variable Distribution ({self.target_col})", fontsize=12)
        plt.xlabel(f'Target Variable (0 vs 1)')
        plt.ylabel('Count')
        plt.show()
        plt.close()

        # Imbalance warning
        if target_percentages.min() < 25.0: # Using 25% as a heuristic for warning
            display(Markdown("\n### âš ï¸� Imbalance Warning"))
            display(Markdown(f"The target variable is **imbalanced** (Min class: {target_percentages.min()}%) This requires careful handling (e.g., **SMOTE**, **weighted loss**, or metrics like **AUC-ROC**)."))
        else:
            print("\nTarget variable is reasonably balanced.")


eda = DiabetesEDA(data = Train_df, target_col = 'diagnosed_diabetes')
eda.run_all_eda()


Train_df


Test_df


Cats


Cats = [
    'gender',
     'ethnicity',
     'education_level',
     'income_level',
     'smoking_status',
     'employment_status'
]

def show_value_counts(df: pd.DataFrame, df_name: str, cols: list):
    """Calculates and displays value counts for specified columns."""
    display(Markdown(f'### Value counts for **{df_name}**'))

    for col in cols:
        print(f"\n--------- Counts for: '{col}' ---------")
        counts = df[col].value_counts(dropna = False)
        print(counts)
        print(f"Total Unique Categories: {len(counts)}")
    print("\n" + "=" * 50)


show_value_counts(Train_df, "Train_df", Cats)
show_value_counts(Test_df, "Test_df", Cats)


from sklearn.preprocessing import LabelEncoder
from IPython.display import display



# List of categories columns to encode 
Cats = [
    'gender',
     'ethnicity',
     'education_level',
     'income_level',
     'smoking_status',
     'employment_status'
]


label_encoders = {}

print("--------- Applying Label Encoding based on Value Counts ------------")


# 1. Fit and Transform the Train Data (Train_df)
print("\n Fitting and transforming Train_df.....")
for col in Cats:
    le = LabelEncoder()

    Train_df[col] = Train_df[col].astype(str)

    Train_df[col] = le.fit_transform(Train_df[col])

    label_encoders[col] = le
    print(f"âœ… Encoded '{col}' with {len(le.classes_)} classes.")



# 2. Transform the Test Data (Test_df) using the Fitted Encoders
print("\n Transforming Test_df.....")
for col in Cats:
    le = label_encoders[col]

    Test_df[col] = Test_df[col].astype(str)


    # --- Robust Transformation for the Test set -------

    # 1. Find the mapping of known classes from the fitted encoder 
    known_labels_map = {name: i for i, name in enumerate(le.classes_)}

    # 2. Define a default code for any potential unseen labels (which is safe for tree models)
    default_code = len(le.classes_)

    # 3. Apply mapping: Use the known map, or the default code for unseen labels
    Test_df[col] = Test_df[col].apply(lambda x: known_labels_map.get(x, default_code))


    print(f"âœ… Transformed '{col}'.")



print("\n-------- Transformed Train_df Head (Categorical features are now integers) ----------")
display(Train_df[Cats].head())


# Check 

Train_df['gender'].value_counts()


Train_df['ethnicity'].value_counts()


Train_df['education_level'].value_counts()


Train_df['income_level'].value_counts()


Train_df['smoking_status'].value_counts()


Train_df['employment_status'].value_counts()


# --- Configuration ---
Seed = 42
N_splits = 10
Target_col = 'diagnosed_diabetes'


Cats_col = [
    'gender',
     'ethnicity',
     'education_level',
     'income_level',
     'smoking_status',
     'employment_status'
]

# ------- Categorical Features Analysis Function ---
def analyze_categorical_features(df, cat_cols, target_col):
    """
    Plots the distribution (Pie Chart) and positive rate (Bar Chart) for each category.
    """
    print("\n--- Categorical Feature Analysis with Visualization ---")
    
    # Setup subplot structure: 2 columns, one row per feature
    n_cols = 2
    n_rows = len(cat_cols)
    
    # Create the figure with appropriate size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        # Calculate positive rate and distribution for plotting
        target_analysis = df.groupby(col)[target_col].agg(['mean', 'count']).reset_index()
        target_analysis = target_analysis.rename(columns={'mean': 'Positive Rate', 'count': 'Count'})
        target_analysis = target_analysis.sort_values(by='Positive Rate', ascending=False)
        
        # --- Plot 1: Pie Chart for Distribution (Value Counts) ---
        ax_pie = axes[i * n_cols]
        value_counts = df[col].value_counts()
        
        ax_pie.pie(
            value_counts,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10},
            wedgeprops={'edgecolor': 'black'}
        )
        ax_pie.set_title(f'Distribution of {col} (Count)', fontsize=14)
        ax_pie.axis('equal') # Ensures pie is circular

        # --- Plot 2: Bar Chart for Positive Rate (Target Mean) ---
        ax_bar = axes[i * n_cols + 1]
        
        sns.barplot(
            x=col, 
            y='Positive Rate', 
            data=target_analysis, 
            ax=ax_bar,
            palette='viridis',
            order=target_analysis[col] # Ensure sorted order
        )
        
        ax_bar.set_title(f'Risk by {col} (Positive Rate)', fontsize=14)
        ax_bar.set_ylabel(f'Mean({target_col})', fontsize=12)
        ax_bar.set_xlabel(col, fontsize=12)
        
        # Rotate x-labels for better readability
        ax_bar.tick_params(axis='x', rotation=45)
        
        # Add sample size labels on top of the bars
        for k, p in enumerate(ax_bar.patches):
            ax_bar.annotate(f"N={target_analysis['Count'].iloc[k]}", 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points', 
                            fontsize=10)
        
    plt.tight_layout()
    plt.show()


analyze_categorical_features(Train_df, Cats_col, Target_col)


Train_df.head()


Test_df.head()





# Store IDs and drop the 'id' column from features
test_ids = Test_df['id']
Train_df = Train_df.drop(columns=['id'])
Test_df = Test_df.drop(columns=['id'])


def map_ordinals(df):
    # Smoking: Current > Former > Never
    smoke_map = {
        'Never': 0, 'No': 0, 
        'Former': 1, 
        'Current': 2, 'Smoker': 2, 'Yes': 2
    }
    if 'smoking_status' in df.columns:
        # Fills NaN/other unmapped values with 0 (Never/No)
        df['smoking_status_risk'] = df['smoking_status'].map(smoke_map).fillna(0)
    return df

Train_df = map_ordinals(Train_df)
Test_df = map_ordinals(Test_df)


# --------- Medical Feature Engineering --------------
def engineer_medical_features(df):
    
    # --- BMI Categories (Standard Medical Ranges) ---
    # Underweight < 18.5, Normal 18.5-25, Overweight 25-30, Obese > 30
    # Use right=False to include the lower bound but exclude the upper bound
    df['BMI_Cat'] = pd.cut(df['bmi'], 
                           bins=[-1, 18.5, 25, 30, df['bmi'].max() + 1], 
                           labels=[0, 1, 2, 3], right=False).astype(int)
    
    # --- Blood Pressure Categories ---
    # Normal: Sys < 120, Elevated: 120-130, Stage 1: 130-140, Stage 2: >= 140
    df['BP_Risk_Level'] = pd.cut(df['systolic_bp'], 
                                 bins=[-1, 120, 130, 140, df['systolic_bp'].max() + 1], 
                                 labels=[0, 1, 2, 3], right=False).astype(int)
    
    # --- Interaction Features ---
    # Visceral Fat Indicator (approximation)
    df['Visceral_Fat'] = df['bmi'] * df['waist_to_hip_ratio']
    
    # Lipid Risk (Atherogenic Index of Plasma, Log AIP)
    # Using np.log1p for robustness against small/zero values, assuming they represent a value close to 0
    df['AIP'] = np.log1p(df['triglycerides']) - np.log1p(df['hdl_cholesterol'])
    
    # Mean Arterial Pressure (MAP)
    df['MAP'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3
    
    return df

Train_df = engineer_medical_features(Train_df)
Test_df = engineer_medical_features(Test_df)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 5. Unsupervised Clustering (Patient Profiling) ----------------
print("\nGenerating Clusters for Patient Profiling...")
cluster_cols = ['age', 'bmi', 'systolic_bp', 'cholesterol_total', 'diet_score', 'physical_activity_minutes_per_week']
scaler = StandardScaler()

# Fit on combined data to ensure consistent clusters
combined_data = pd.concat([Train_df[cluster_cols], Test_df[cluster_cols]], axis=0)
combined_scaled = scaler.fit_transform(combined_data)

# Create 7 clusters
kmeans = KMeans(n_clusters=7, random_state=Seed, n_init=10)
kmeans.fit(combined_scaled)

# Separate back into train and test
n_train = len(Train_df)
Train_df['Cluster_ID'] = pd.Categorical(kmeans.predict(combined_scaled[:n_train]))
Test_df['Cluster_ID'] =pd.Categorical(kmeans.predict(combined_scaled[n_train:]))

print("Clustering complete.")


# ------- Categorical Setup -------
X = Train_df.drop(columns=[Target_col])
y = Train_df[Target_col]
X_test = Test_df.copy()

# New features created: 'smoking_status_risk', 'BMI_Cat', 'BP_Risk_Level', 'Cluster_ID'
# We treat 'education_level' and 'income_level' as Categorical now, 
# letting LightGBM/CatBoost handle the ranking internally.
cat_cols = X.select_dtypes(include=['object']).columns.tolist() + ['BMI_Cat', 'BP_Risk_Level', 'Cluster_ID']

print(f"\nCategorical Features used: {cat_cols}")

# Ensure category dtype for LGBM/XGB (CatBoost can handle 'object' as well)
for col in cat_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')


print("Post-engineering shapes:")
print(f"Train: {Train_df.shape}, Test: {Test_df.shape}")
# After X/X_test setup:
print(f"X shape: {X.shape}, X_test shape: {X_test.shape}")
# Columns diff:
print("X columns:", X.columns.tolist())
print("X_test columns:", X_test.columns.tolist())





from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# ------------- Modeling: Stratified K-Fold Ensemble --------------------
skf = StratifiedKFold(n_splits=N_splits, shuffle=True, random_state=Seed)
out_of_fold_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Scale position weight for XGBoost to handle class imbalance
positive_count = y.sum()
negative_count = len(y) - positive_count
scale_pos_weight_value = negative_count / positive_count

print("\n--- Starting Stratified K-Fold Ensemble Training ---")
for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # ---------------- Model 1: LightGBM (Primary Model) -------------------
    model_lgb = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.012,
        num_leaves=31,
        max_depth=8,
        subsample=0.7,
        colsample_bytree=0.5,
        class_weight='balanced', # Handles imbalance
        reg_alpha=1.0,          # L1 Regularization
        reg_lambda=2.0,         # L2 Regularization
        device='gpu',             # NEW: Enable GPU (auto-uses available GPUs)
        gpu_platform_id=0,        # NEW: CUDA platform (default)
        gpu_device_id=0,          # NEW: Primary GPU (0); CatBoost will use 1
        random_state= Seed + fold,
        n_jobs=1,
        verbose=-1
    )
    
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
        categorical_feature=[col for col in cat_cols if col in X_train.columns]
    )
    
    # -------------- Model 2: CatBoost (Secondary Model) -----------------
    model_cb = cb.CatBoostClassifier(
        iterations=3000,
        learning_rate=0.012,
        depth=6,
        l2_leaf_reg=4,
        auto_class_weights='Balanced', # Handles imbalance
        cat_features=[col for col in cat_cols if col in X_train.columns],
        random_seed= Seed + fold,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU",            # NEW: Enable GPU
        devices='0:1'               # NEW: Use both GPUs (0 and 1) for T4 x2
    )
    
    model_cb.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=150
    )
    
    # -------------- Model 3: XGBoost (Tertiary Model) --------------
    # We use the scale_pos_weight you defined earlier
    model_xgb = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.012,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.5,
        scale_pos_weight=scale_pos_weight_value, # Handles imbalance
        enable_categorical=True, # Native support for 'category' dtype
        tree_method='gpu_hist',
        reg_lambda=2.0,
        random_state= Seed + fold,
        n_jobs=1,
        early_stopping_rounds=150,
        verbosity=0 
    )
    
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # --------------- Ensemble (Weighted Averaging) ---------------------
    
    p_lgb = model_lgb.predict_proba(X_val)[:, 1]
    p_cb = model_cb.predict_proba(X_val)[:, 1]
    p_xgb = model_xgb.predict_proba(X_val)[:, 1]
    
    out_of_fold_preds[val_idx] = (0.4 * p_lgb) + (0.4 * p_cb) + (0.2 * p_xgb) # Weighting: LGBM (40%), CatBoost (40%), XGB (20%)
    
    # Test predictions
    t_lgb = model_lgb.predict_proba(X_test)[:, 1]
    t_cb = model_cb.predict_proba(X_test)[:, 1]
    t_xgb = model_xgb.predict_proba(X_test)[:, 1]
    
    test_preds += ((0.4 * t_lgb) + (0.4 * t_cb) + (0.2 * t_xgb)) / N_splits
    
    print(f"Fold {fold+1}/{N_splits} AUC: {roc_auc_score(y_val, out_of_fold_preds[val_idx]):.5f}")


# Overall Out Of Fold AUC
oof_auc = roc_auc_score(y, out_of_fold_preds)
print(f"\nOverall OOF AUC: {oof_auc:.5f}")


from sklearn.metrics import roc_auc_score, roc_curve

# ----------------- Final Ensemble Model Performance Summary -----------------

# Assuming 'y' is the full target vector and 'out_of_fold_preds' are the ensemble OOF predictions.

# Calculate the overall Out-of-Fold (OOF) AUC for the ensemble
final_ensemble_auc = roc_auc_score(y, out_of_fold_preds)
print(f"âœ… Final Ensemble Out-of-Fold AUC: {final_ensemble_auc:.4f}")

# Generate ROC Curve data for visualization
fpr, tpr, thresholds = roc_curve(y, out_of_fold_preds)

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_ensemble_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Ensemble Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


Train_df


Test_df


# To load "id" column to compare the submission output with particular id
Test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


# ----------------- Submission Code -----------------

# Assuming Test_df (the original loaded test data) and 
# test_preds (the final averaged test predictions) are available.

submission_df = pd.DataFrame({
    'id': Test_df['id'],
    'diagnosed_diabetes': test_preds # The predicted probability for the target
})

# Save the submission file to a CSV
submission_df.to_csv('submission.csv', index=False)

print("âœ… Kaggle submission file 'submission.csv' successfully created!")
print("\nFirst 5 rows of the submission file:")
print(submission_df.head())




