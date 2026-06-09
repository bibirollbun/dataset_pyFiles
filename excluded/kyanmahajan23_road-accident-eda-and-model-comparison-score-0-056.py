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


# ===============================
# ğŸ“¦ Core Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from termcolor import colored
import itertools
import re

# ===============================
# âš™ï¸� Data Preprocessing
# ===============================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score

# ===============================
# ğŸ§  Ensemble Models
# ===============================
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    StackingClassifier, StackingRegressor,
    VotingClassifier, VotingRegressor
)

# ===============================
# ğŸ§© XGBoost, LightGBM, CatBoost
# ===============================
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# ===============================
# ğŸ“ˆ Regression Models
# ===============================
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LogisticRegression
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# ===============================
# ğŸ§ª Evaluation Metrics
# ===============================
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)



#setting up the seed and the environment
import os
import random
import numpy as np
import tensorflow as tf


seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


import pandas as pd
import numpy as np 

data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")


def describe_data(data):
    print(colored("ğŸ“� DATA OVERVIEW", "cyan", attrs=["bold"]))
    print(colored(f"ğŸ”¹ Shape: {data.shape[0]} rows Ã— {data.shape[1]} columns", "green"))

    # Column Names
    print("\n" + colored("ğŸ“‹ Column Names:", "yellow", attrs=["bold"]))
    for col in data.columns.tolist():
        print(f"   â€¢ {col}")

    # Data Types
    print("\n" + colored("ğŸ”� Data Types:", "yellow", attrs=["bold"]))
    dtypes_df = data.dtypes.reset_index()
    dtypes_df.columns = ["Column", "Type"]
    print(tabulate(dtypes_df, headers="keys", tablefmt="fancy_grid"))

    # Numerical Columns
    num_cols = data.select_dtypes(include="number").columns.tolist()
    print("\n" + colored("Numerical Columns:", "yellow", attrs=["bold"]))
    print(num_cols if num_cols else "None")
    cat_cols = data.select_dtypes(include="object").columns.tolist()
    print("\n" + colored(" Categorical Columns:", "yellow", attrs=["bold"]))
    print(cat_cols if cat_cols else "None")

    # Random Sample
    print("\n" + colored("ğŸ�² Random 5 Rows:", "yellow", attrs=["bold"]))
    print(data.sample(5))

    # Missing Values
    missing_values = data.isnull().sum()
    missing_df = missing_values.reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    print("\n" + colored("âš ï¸� Missing Values:", "red", attrs=["bold"]))
    print(tabulate(missing_df, headers="keys", tablefmt="fancy_grid"))


def analyse_column(data):
    # Categorical Columns (only with < 10 unique values)
    categorical_columns = data.select_dtypes(include="object").columns.tolist()
    print(colored(" CATEGORICAL COLUMNS (Unique < 10)", "cyan", attrs=["bold"]))
    for col in categorical_columns:
        if data[col].nunique() < 10:   
            print(colored(f"\nâ–¶ Column: {col}", "yellow", attrs=["bold"]))
            vc = data[col].value_counts().reset_index()
            vc.columns = ["Category", "Count"]
            print(tabulate(vc, headers="keys", tablefmt="fancy_grid"))

    # Numerical Summary
    print("\n" + colored("ğŸ”¢ NUMERICAL SUMMARY", "cyan", attrs=["bold"]))
    desc = data.describe().reset_index()
    print(tabulate(desc, headers="keys", tablefmt="fancy_grid"))


#function to plot all the numerical columns hisplots and boxplot
def numerical_plots(data):
    numeric_cols = data.select_dtypes(include="number").columns
    n = len(numeric_cols)
    
    fig, axes = plt.subplots(n, 2, figsize=(10, 5*n))  
    axes = axes.ravel()

    for i, col in enumerate(numeric_cols):
        
        sns.histplot(data[col].dropna(), bins=30, ax=axes[2*i], kde=True)
        axes[2*i].set_title(f"Histogram of {col}")
        
        # Boxplot
        sns.boxplot(x=data[col], ax=axes[2*i+1])
        axes[2*i+1].set_title(f"Boxplot of {col}")
    
    plt.tight_layout()
    plt.show()


#function to plot all the categorical columns
def plot_categorical_columns(df):
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        # only plot if number of unique categories <= 10
        if df[col].nunique() <= 10:
            plt.figure(figsize=(6,4))
            sns.countplot(x=col, data=df, order=df[col].value_counts().index)
            plt.title(f"Countplot of {col}")
            plt.xticks(rotation=45)
            plt.show()


def scatter_plots(df):
    
    num_cols = df.select_dtypes(include='number').columns
    
   
    pairs = list(itertools.combinations(num_cols, 2))
    
    for col1, col2 in pairs:
        
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[col1], y=df[col2], alpha=0.6)
        plt.title(f"Scatter Plot: {col1} vs {col2}")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()


describe_data(data);
scatter_plots(data);
plot_categorical_columns(data);
numerical_plots(data);
analyse_column(data);





corr = data.select_dtypes(include="number").corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Columns")
plt.show()


y = data['accident_risk']
x = data.drop('accident_risk', axis = 1)
test_Data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
ids = test_Data["id"]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)




#This function converts scales the numerical columns, add one hot encoding to categorical, and Apply Tfidf to 
#text

def preprocess_data(X_train, X_test, y_train, y_test, test_data):
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse=False, drop='first'), categorical_cols),
         
        ],
        remainder="passthrough"
    )
    
    X_train_arr = transformer.fit_transform(X_train)
    X_test_arr = transformer.transform(X_test)
    test_data_arr = transformer.transform(test_data)
    
    # Get feature names
    
    num_cols = numerical_cols
    cat_cols = transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)

    
    # passthrough columns = everything not in numerical + categorical + text_col
    passthrough_cols = [c for c in X_train.columns if c not in numerical_cols + categorical_cols ]
    
    all_cols = num_cols + list(cat_cols)  + passthrough_cols
    
    # Convert to DataFrame
    X_train_processed = pd.DataFrame(
        X_train_arr.toarray() if hasattr(X_train_arr, "toarray") else X_train_arr,
        columns=all_cols,
        index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        X_test_arr.toarray() if hasattr(X_test_arr, "toarray") else X_test_arr,
        columns=all_cols,
        index=X_test.index
    )
    test_Data_= pd.DataFrame(
        test_data_arr,
        columns = all_cols,
        index = test_data.index
    )


    

    return X_train_processed, X_test_processed, y_train, y_test,  test_Data_



X_train_, X_test_, y_train_, y_test_,  test_data_ = preprocess_data(X_train, X_test, y_train, y_test,test_Data);


from sklearn.metrics import mean_squared_error
import numpy as np
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # acc = accuracy_score(y_test, y_pred)
    # prec = precision_score(y_test, y_pred, average="weighted")
    # rec = recall_score(y_test, y_pred, average="weighted")
    # f1 = f1_score(y_test, y_pred, average="weighted")
    print("rmse", rmse)
    # print("Accuracy:", acc)
    # print("Precision:", prec)
    # print("Recall:", rec)
    # print("F1 Score:", f1)
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    




from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Define models to compare
models = {
    # "Linear Regression": LinearRegression(),
    # "Ridge Regression": Ridge(),
    # "Lasso Regression": Lasso(),
    # "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    # "XGBoost": XGBRegressor(random_state=42),
    # "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

# Store results
results = []

# Loop through models
for name, model in models.items():
    model.fit(X_train_, y_train_)
    y_pred = model.predict(X_test_)
    rmse = np.sqrt(mean_squared_error(y_test_, y_pred))
    results.append((name, rmse))
    print(f"{name}: RMSE = {rmse:.4f}")

# Convert to DataFrame for comparison
rmse_df = pd.DataFrame(results, columns=["Model", "RMSE"]).sort_values(by="RMSE")
print("\nModel Comparison:")
print(rmse_df)



output = models['CatBoost'].predict(test_data_)

final_data = pd.DataFrame({
    "id": ids,
    "accident_risk": output
})


final_data.to_csv("submission3.csv", index = False)

