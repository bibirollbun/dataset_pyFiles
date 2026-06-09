import numpy as np 
import pandas as pd 
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv").set_index('id')
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv").set_index('id')
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.head()


test.head()


train.info()


train.describe()


print(train.isnull().sum())
print(test.isnull().sum())


train['Sex'].value_counts().plot(kind='pie',autopct='%1.1f%%' )


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Select numerical columns
numerical_df = train.select_dtypes(include='number')
num_cols = len(numerical_df.columns)

# Set up the plot grid: vertical (n rows, 1 column)
plt.figure(figsize=(6, 4 * num_cols))  # Adjust height for each plot

# Plot each numerical column vertically
for i, col in enumerate(numerical_df.columns, 1):
    plt.subplot(num_cols, 1, i)
    sns.histplot(numerical_df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()



def engineer_features(df):
    # 1. BMI (Body Mass Index)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # 2. Workout Intensity (Heart Rate × Duration)
    df['Workout_Intensity'] = df['Heart_Rate'] * df['Duration']
    
    # 3. Age Group Binning
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 20, 35, 50, 65, 100],
        labels=['Teen', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
    )
    
    # 4. Is_Male binary encoding
    df['Is_Male'] = (df['Sex'] == 'male').astype(int)
    
    # 5. Temp Above Normal
    df['Temp_Above_Normal'] = df['Body_Temp'] - 37.0
    
    
    # 7. Heart Rate Zone
    def heart_rate_zone(hr):
        if hr < 90:
            return 'Low'
        elif hr < 120:
            return 'Moderate'
        else:
            return 'High'
    
    df['HR_Zone'] = df['Heart_Rate'].apply(heart_rate_zone)
    
    return df



train = engineer_features(train)
test = engineer_features(test)


from sklearn.preprocessing import LabelEncoder

label_cols = ['Sex', 'Age_Group', 'HR_Zone']

for col in label_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])  # assumes all test categories exist in train



def downcast_df(df):
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
        
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
        
    return df



train = downcast_df(train)
test = downcast_df(test)



def add_pairwise_combinations(df, columns=None, operations=['sum', 'diff', 'prod', 'ratio'], row_stats=None):
    """
    Creates pairwise combinations and optional row-wise statistics.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        columns (list): Columns to use. If None, all numeric columns except 'Calories'.
        operations (list): Pairwise operations: 'sum', 'diff', 'prod', 'ratio'
        row_stats (list): Row-wise stats to calculate: 'mean', 'std', 'min', 'max', 'median', 'count'

    Returns:
        df (pd.DataFrame): Updated DataFrame
    """
    if columns is None:
        columns = [col for col in df.select_dtypes(include=[np.number]).columns if col != 'Calories']

    # Pairwise combinations
    for col1, col2 in tqdm(combinations(columns, 2), total=len(columns)*(len(columns)-1)//2):
        if 'sum' in operations:
            df[f'{col1}_{col2}_sum'] = df[col1] + df[col2]
        if 'diff' in operations:
            df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
            df[f'{col2}_{col1}_diff'] = df[col2] - df[col1]
        if 'prod' in operations:
            df[f'{col1}_{col2}_prod'] = df[col1] * df[col2]
        if 'ratio' in operations:
            df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-6)
            df[f'{col2}_{col1}_ratio'] = df[col2] / (df[col1] + 1e-6)

    # Row-wise statistics
    if row_stats:
        row_data = df[columns]
        if 'mean' in row_stats:
            df['row_mean'] = row_data.mean(axis=1)
        if 'median' in row_stats:
            df['row_median'] = row_data.median(axis=1)

    return df



train = add_pairwise_combinations(train)
test = add_pairwise_combinations(test)


import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np


def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# Custom RMSLE eval function for XGBoost (for early stopping)
def rmsle_xgb_eval(preds, dtrain):
    labels = dtrain.get_label()
    score = rmsle(labels, preds)
    return 'rmsle', score


# Split features and target
X = train.drop('Calories', axis=1)
y = train['Calories']

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store validation scores
scores = []

# Loop over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Initialize model
    model = XGBRegressor(
        tree_method='hist',
        enable_categorical=True,
        device='cuda',
        max_depth=9,
        colsample_bynode=0.3,
        subsample=0.8,
        n_estimators=50_000,
        learning_rate=0.01,
        min_child_weight=10,
    )
    
    # Fit with early stopping using custom RMSLE
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=rmsle_xgb_eval,
        early_stopping_rounds=500,
        verbose=False
    )
    
    # Predict and evaluate using RMSLE
    y_pred = model.predict(X_val)
    score = rmsle(y_val, y_pred)
    print(f"RMSLE: {score:.4f}")
    scores.append(score)

# Average score across folds
print(f"\nAverage RMSLE: {np.mean(scores):.4f}")



test_pred = model.predict(test)
sample_sub['Calories'] = test_pred
sample_sub.to_csv('submission.csv',index = False)
sample_sub.head()


# Get importance scores from booster
importance_dict = model.get_booster().get_score(importance_type='gain')

# Create DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': list(importance_dict.keys()),
    'Importance': list(importance_dict.values())
}).sort_values(by='Importance', ascending=False).head(50)

# Plot
plt.figure(figsize=(12, 20))
plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='skyblue')
plt.xlabel("Gain")
plt.title("Top 50 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()





