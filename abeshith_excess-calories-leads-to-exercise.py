import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.head()


from termcolor import colored
import pandas as pd

def pretty_info(df, name="DataFrame", show_sample=True):
    print(colored(f"\n Summary for: {name}", "cyan", attrs=["bold", "underline"]))
    
    # Shape
    print(colored(f"\n Shape: {df.shape[0]} rows Ã— {df.shape[1]} columns", "green"))

    # Duplicates
    dup_count = df.duplicated().sum()
    print(colored(f" Duplicated Rows: {dup_count}", "red" if dup_count else "green"))

    # Data Types
    print(colored("\n Column Data Types:", "blue"))
    print(df.dtypes.value_counts())

    # Nulls
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        print(colored("\n Missing Values:", "yellow"))
        print(nulls.sort_values(ascending=False))
    else:
        print(colored("\n No Missing Values Found", "green"))

    # Unique values
    print(colored("\n Unique Values per Column (<=20 only):", "magenta"))
    for col in df.columns:
        uniqs = df[col].nunique()
        if uniqs <= 20:
            print(f"{col}: {df[col].unique()}")

    # Describe
    print(colored("\n Summary Statistics (numerical):", "blue"))
    print(df.describe().T)

    # Head
    if show_sample:
        print(colored("\n Sample Data (Top 5 rows):", "cyan"))
        print(df.head())

    print(colored("\n Analysis Complete", "green", attrs=["bold"]))


pretty_info(train, name="Training Data")


pretty_info(test, name="Test Data")


numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']
z_scores = np.abs(stats.zscore(train[numeric_cols]))
train = train[(z_scores < 3).all(axis=1)] 


def feature_engineering(df):
    df = df.copy()
    
    # BMI - Added to train data
    if 'Weight' in df.columns and 'Height' in df.columns:
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Calories per minute - Added to train data
    if 'Calories' in df.columns and 'Duration' in df.columns:
        df['Calories_per_min'] = df['Calories'] / df['Duration'].replace(0, 1)
    
    # Sex as numeric
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Interaction features
    if 'Age' in df.columns and 'Weight' in df.columns:
        df['Age_Weight'] = df['Age'] * df['Weight']
    
    if 'Heart_Rate' in df.columns and 'Body_Temp' in df.columns:
        df['Heart_Temp'] = df['Heart_Rate'] * df['Body_Temp']

    return df


train = feature_engineering(train)
test = feature_engineering(test)


for col in ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Calories_per_min','Age_Weight','Heart_Temp']:
    corr = np.corrcoef(train[col], train['Calories'])[0, 1]
    print(f"{col} correlation with Calories: {corr:.3f}")


anova = stats.f_oneway(train[train['Sex'] == 0]['Calories'],
                       train[train['Sex'] == 1]['Calories'])
print(f"ANOVA test for Sex: F={anova.statistic:.3f}, p={anova.pvalue:.3f}")


features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
            'BMI', 'Age_Weight', 'Heart_Temp']
target = 'Calories'


X = train[features]
y = np.log1p(train[target])
X_test = test[features]


model = xgb.XGBRegressor(device='cuda',
        max_depth=8,
        colsample_bytree=0.9,
        subsample=0.9,
        n_estimators=1500,
        learning_rate=0.007,
        eval_metric="rmse")
model.fit(X, y)


log_preds = model.predict(X_test)


preds = np.expm1(log_preds)


submission = pd.DataFrame({'id': submission['id'], 'Calories': preds})
submission.to_csv('submission.csv', index=False)


submission




