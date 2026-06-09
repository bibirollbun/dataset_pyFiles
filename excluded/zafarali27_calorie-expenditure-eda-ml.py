import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble  import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline,make_pipeline


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df.head().style.background_gradient(cmap = "PuBu")


from colorama import Fore, Style

# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n") 

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")


numerical_f =  ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
for feature in numerical_f:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()
    print("\n")


train_df["Sex"].value_counts().plot(kind="pie", autopct='%.0f%%')
plt.title("Distribution of Sex")
plt.show()


sns.pairplot(train_df.drop(columns='Sex'))
plt.show()


colors = sns.color_palette('husl', len(numerical_f))
rows = -(-len(numerical_f) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_f, colors), 1):
    plt.subplot(rows, 4, i)
    sns.violinplot(data=train_df, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


sns.heatmap(train_df[numerical_f].corr(),annot = True)
plt.title('Correlation Matrix of Numerical Features')
plt.show()


for numerical in numerical_f[:-1]:
    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(data=train_df, x=numerical, y="Calories", hue="Sex", palette="Set2")
    plt.title(f"{numerical} VS Calories")
    plt.show()
    


# Feature and target split
x = train_df.drop(columns=["Calories"])
y = train_df["Calories"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature identification
numerical_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = x.select_dtypes(include="object").columns.tolist()

# Correct transformers
numerical_transformer = make_pipeline(StandardScaler())
categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Bagging Regressor
bag = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    n_estimators=500,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

# Pipeline
pipe = make_pipeline(preprocessor, bag)

# Fit and predict
pipe.fit(x_train, y_train)
predictions = pipe.predict(x_test)

# Evaluate
rms = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rms:.2f}')



# from sklearn.model_selection import GridSearchCV

# # Define parameter grid
# param_grid = {
#     'baggingregressor__n_estimators': [100, 200],
#     'baggingregressor__max_samples': [0.5, 0.7, 1.0],
#     'baggingregressor__base_estimator__max_depth': [None, 5, 10],
# }

# # Create GridSearchCV object
# grid_search = GridSearchCV(
#     pipe,
#     param_grid,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
#     verbose=1
# )

# # Fit grid search
# grid_search.fit(x_train, y_train)

# # Predict using best model
# best_model = grid_search.best_estimator_
# predictions = best_model.predict(x_test)

# # Evaluation
# rms = np.sqrt(mean_squared_error(y_test, predictions))
# print(f'Best Params: {grid_search.best_params_}')
# print(f'Best RMSE: {rms:.2f}')


