import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def process_file(filename, dirname):
    """Process a single parquet file and extract time-based features"""
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    
    # Drop 'step' column if it exists
    if 'step' in df.columns:
        df.drop('step', axis=1, inplace=True)
    
    # Convert time_of_day to hours
    df["hours"] = df["time_of_day"] // (3_600 * 1_000_000_000)
    
    # Define time periods
    night = ((df["hours"] >= 22) | (df["hours"] <= 5))
    day = ((df["hours"] <= 20) & (df["hours"] >= 7))
    
    # Initialize features dictionary
    features = {}
    
    # Basic activity features
    features['non_wear_mean'] = df["non-wear_flag"].mean()
    features['active_enmo_sum'] = df["enmo"][df["enmo"] >= 0.05].sum()
    
    # Process each column for different time periods
    for col in ['enmo', 'anglez', 'light', 'battery_voltage']:
        # Full day statistics
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_std"] = df[col].std()
        features[f"{col}_max"] = df[col].max()
        features[f"{col}_min"] = df[col].min()
        features[f"{col}_diff_mean"] = df[col].diff().mean()
        features[f"{col}_diff_std"] = df[col].diff().std()
        
        # Night time statistics
        night_data = df.loc[night, col]
        features[f"{col}_night_mean"] = night_data.mean()
        features[f"{col}_night_std"] = night_data.std()
        features[f"{col}_night_max"] = night_data.max()
        features[f"{col}_night_min"] = night_data.min()
        
        # Day time statistics
        day_data = df.loc[day, col]
        features[f"{col}_day_mean"] = day_data.mean()
        features[f"{col}_day_std"] = day_data.std()
        features[f"{col}_day_max"] = day_data.max()
        features[f"{col}_day_min"] = day_data.min()
    
    return features, filename.split('=')[1]

def load_data_parquet(dirname) -> pd.DataFrame:
    """Load and process time series data from directory in parallel"""
    ids = os.listdir(dirname)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    
    features_list, indexes = zip(*results)
    
    # Create DataFrame with extracted features and IDs
    df = pd.DataFrame(features_list)
    df['id'] = indexes
    
    return df


from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

featuresCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
                'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
                'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
                'PreInt_EduHx-computerinternet_hoursday','sii']

# chia thành 3 nhóm features chính (Bộ dữ liệu khách quan csv)
demographicFeatures = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'sii']

phycsicsFeatures = ['CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
                'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                'BIA-BIA_TBW','sii' ]

behaviorFeatures = ['PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
                    'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
                'PreInt_EduHx-computerinternet_hoursday','sii']

cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
          'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

train_ts = load_data_parquet('/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet')
test_ts = load_data_parquet(r'/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet')

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')


behaviorFeatures += time_series_cols

train_df = train[behaviorFeatures]


# Loại bỏ các feature liên quan đến "season"
filtered_features = [feature for feature in behaviorFeatures if feature not in cat_c and feature != 'sii']

# Loại bỏ các hàng có giá trị NaN trong y
train_df = train_df[behaviorFeatures].dropna(subset=['sii'])

# Chuẩn bị dữ liệu X và y
X = train_df[filtered_features]
y = train_df['sii']

# Định nghĩa pipeline xử lý dữ liệu số
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Định nghĩa ColumnTransformer để áp dụng pipeline cho các cột số
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, filtered_features)
])

# Fit và transform X
preprocessor.fit(X)
X_transformed = pd.DataFrame(preprocessor.transform(X), columns=filtered_features)

# Kiểm tra các dòng đầu tiên của dữ liệu đã transform
print("Transformed X DataFrame:")
print(X_transformed.head())


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,  # Increased from 0.1
    'reg_lambda': 5,  # Increased from 1
    'random_state': 2023,
    'tree_method': 'gpu_hist',
}
RF_Params = {
    'n_estimators': 200,         # Increased from 100 to match better performing RandomForest_Default
    'max_depth': 15,             # Added specific depth to control complexity
    'min_samples_split': 5,      # Increased to reduce overfitting
    'min_samples_leaf': 2,       # Increased to ensure more robust leaf nodes
    'max_features': 'sqrt',      # Keep sqrt as it works well for classification
    'bootstrap': True,           # Keep bootstrapping enabled
    'random_state': 2023,        # Keep same random state for reproducibility
    'n_jobs': -1,                # Keep using all cores
    'class_weight': 'balanced',  # Keep balanced class weights
    'criterion': 'gini'          # Keep gini criterion
}



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

# Ensemble model imports
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
seed = 2023
np.random.seed(seed)

# Base models for stacking/voting
base_rf = RandomForestClassifier(n_estimators=100, random_state=seed)
base_gb = GradientBoostingClassifier(n_estimators=100, random_state=seed)
base_xgb = XGBClassifier(n_estimators=100, random_state=seed, eval_metric='mlogloss')

# Create ensemble models
models = [
    RandomForestClassifier(n_estimators=100, random_state=seed),
    GradientBoostingClassifier(n_estimators=100, random_state=seed),
    ExtraTreesClassifier(n_estimators=100, random_state=seed),
    AdaBoostClassifier(random_state=seed),
    BaggingClassifier(random_state=seed),
    XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=seed),
    LGBMClassifier(n_estimators=100, random_state=seed),
    CatBoostClassifier(n_estimators=100, random_state=seed, verbose=0),
    VotingClassifier(estimators=[
        ('rf', base_rf),
        ('gb', base_gb),
        ('xgb', base_xgb)
    ], voting='soft'),
    StackingClassifier(estimators=[
        ('rf', base_rf),
        ('gb', base_gb),
        ('xgb', base_xgb)
    ], final_estimator=GradientBoostingClassifier(random_state=seed))
]

model_names = [
    'RandomForest',
    'GradientBoosting',
    'ExtraTrees',
    'AdaBoost',
    'Bagging',
    'XGBoost',
    'LightGBM',
    'CatBoost',
    'VotingEnsemble',
    'StackingEnsemble'
]

# Function to generate baseline results0
def generate_baseline_results(models, model_names, X, y, metrics='kappa', cv=5, plot_results=False):
    """
    Evaluate multiple ensemble models with cross-validation and optionally plot results.
    
    Parameters:
    -----------
    models: list
        List of initialized model objects
    model_names: list
        List of model names (should match length of models)
    X: DataFrame or array
        Feature matrix
    y: Series or array
        Target variable
    metrics: str
        Scoring metric to use
    cv: int
        Number of cross-validation folds
    plot_results: bool
        Whether to display a boxplot of results
        
    Returns:
    --------
    DataFrame with mean and std dev of model performance
    """
    # Define k-fold
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    entries = []
    
    # Loop through each model
    for model, model_name in zip(models, model_names):
        print(f"Training: {model_name}")
        try:
            scores = cross_val_score(model, X, y, scoring=metrics, cv=kfold)
            # Save results for all models
            entries.extend([(model_name, fold_idx, score) for fold_idx, score in enumerate(scores)])
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Create DataFrame
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_id', 'cohen_kappa_score'])
    
    # Optional: Plot results if specified
    if plot_results and len(cv_df) > 0:
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='model_name', y='cohen_kappa_score', data=cv_df, color='lightblue', showmeans=True)
        plt.title("Ensemble Models Performance using 5-fold Cross-Validation", fontsize=14)
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Accuracy Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    # Summary result
    if len(cv_df) > 0:
        mean = cv_df.groupby('model_name')['cohen_kappa_score'].mean()
        std = cv_df.groupby('model_name')['cohen_kappa_score'].std()

        baseline_results = pd.concat([mean, std], axis=1)
        baseline_results.columns = ['Mean', 'Standard Deviation']

        # Sort results
        baseline_results.sort_values(by='Mean', ascending=False, inplace=True)
        return baseline_results
    else:
        return pd.DataFrame(columns=['Mean', 'Standard Deviation'])

# Run evaluation and display results
print("\n===== Ensemble Models Evaluation =====\n")
cv_results = generate_baseline_results(models, model_names, X_transformed, y, metrics='kappa', cv=5, plot_results=True)

# Print full results
print("\nEnsemble Model Performance Summary:")
print(cv_results)

# Visualize top performers with a bar plot
plt.figure(figsize=(14, 6))
top_results = cv_results.head(5)
bar = plt.bar(
    top_results.index,
    top_results['Mean'],
    yerr=top_results['Standard Deviation'],
    capsize=10,
    color='skyblue',
    edgecolor='navy'
)

# Add values on top of bars
for i, b in enumerate(bar):
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 0.005,
        f"{top_results['Mean'].iloc[i]:.4f}",
        ha='center',
        fontweight='bold'
    )

plt.title('Top 5 Ensemble Models by Accuracy', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Accuracy Score', fontsize=12)
plt.ylim(top=1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()





# Preprocess the test data
X_test = test[filtered_features]
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=filtered_features)

# Use the trained model to make predictions
best_model =  GradientBoostingClassifier(n_estimators=100, random_state=seed)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'sii': y_test_pred
})

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)



print("Submission file created successfully.")  # In ra thông báo khi hoàn thành


submiss = pd.read_csv('/kaggle/working/submission.csv')




