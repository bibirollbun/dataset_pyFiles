import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def process_file(filename, dirname):
    """Process a single parquet file and extract features"""
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    
    # Drop 'step' column if it exists
    if 'step' in df.columns:
        df.drop('step', axis=1, inplace=True)
    
    # Initialize dictionary to store features
    features = {}
    
    # Extract basic statistics for each column
    for col in df.columns:
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_std"] = df[col].std()
        features[f"{col}_min"] = df[col].min()
        features[f"{col}_max"] = df[col].max()
        features[f"{col}_sum"] = df[col].sum()
    
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

cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
          'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
          'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

train_ts = load_data_parquet("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_data_parquet("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')


featuresCols += time_series_cols

train_df = train[featuresCols]


# Loại bỏ các feature liên quan đến "season"
filtered_features = [feature for feature in featuresCols if feature not in cat_c and feature != 'sii']

# Loại bỏ các hàng có giá trị NaN trong y
train_df = train_df[featuresCols].dropna(subset=['sii'])

# Chuẩn bị dữ liệu X và y
X = train_df[filtered_features]
y = train_df['sii']

# Định nghĩa pipeline xử lý dữ liệu số
num_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=3)),
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


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Random seed
seed = 2023

# List of models
models = [
    LinearSVC(max_iter=10000, random_state=seed),
    SVC(random_state=seed),
    KNeighborsClassifier(metric='minkowski', p=2),
    LogisticRegression(solver='liblinear', max_iter=1000),
    DecisionTreeClassifier(random_state=seed),
    RandomForestClassifier(random_state=seed),
    ExtraTreesClassifier(random_state=seed),
    AdaBoostClassifier(random_state=seed),
    XGBClassifier(eval_metric='logloss', random_state=seed)    
]

# Function to generate baseline results
def generate_baseline_results(models, X, y, metrics='accuracy', cv=5, plot_results=False):
    # Define k-fold
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    entries = []
    
    # Loop through each model
    for model in models:
        model_name = model.__class__.__name__
        print(f"Training: {model_name}")
        scores = cross_val_score(model, X, y, scoring=metrics, cv=kfold)
        # Lưu kết quả của tất cả các mô hình vào entries
        entries.extend([(model_name, fold_idx, score) for fold_idx, score in enumerate(scores)])
    
    # Create DataFrame
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_id', 'accuracy_score'])
    
    # Optional: Plot results if specified
    if plot_results:
        sns.boxplot(x='model_name', y='accuracy_score', data=cv_df, color='lightblue', showmeans=True)
        plt.title("Boxplot of baseline Model Accuracy using 5-fold cross-validation")
        plt.xticks(rotation=45)
        plt.show()
    
    # Summary result
    mean = cv_df.groupby('model_name')['accuracy_score'].mean()
    std = cv_df.groupby('model_name')['accuracy_score'].std()

    baseline_results = pd.concat([mean, std], axis=1)
    baseline_results.columns = ['Mean', 'Standard Deviation']

    # Sort results
    baseline_results.sort_values(by='Mean', ascending=False, inplace=True)

    return baseline_results

# Chạy hàm và hiển thị kết quả
cv_results = generate_baseline_results(models, X_transformed, y, metrics='accuracy', cv=5, plot_results=False)

# In toàn bộ kết quả
print(cv_results)



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

# Preprocess the test data
X_test = test[filtered_features]
X_test = pd.DataFrame(preprocessor.transform(X_test), columns=filtered_features)

# Use the trained model to make predictions
best_model = XGBClassifier(eval_metric='logloss', **XGB_Params)
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




