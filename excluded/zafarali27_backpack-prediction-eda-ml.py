import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,mean_squared_error


df_train = pd.read_csv(r"/kaggle/input/playground-series-s5e2/train.csv")
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv')
df_sample_submission = pd.read_csv(r"/kaggle/input/playground-series-s5e2/sample_submission.csv")


df_train.head()


# Train df
print("df train shape: ")
print(f"{df_train.shape}\n")

print("df train info: ")
print(f"{df_train.info()}\n")

print("df train isnull sum: ")
print(f"{df_train.isnull().sum()}\n")

print("df train describe: ")
print(f"{df_train.describe()}\n")


# Test df
print("df test shape: ")
print(f"{df_test.shape}\n")

print("df test info: ")
print(f"{df_test.info()}\n")

print("df test isnull sum: ")
print(f"{df_test.isnull().sum()}\n")

print("df test describe: ")
print(f"{df_test.describe()}\n")


# categorical tran df
cate_df_train = df_train.select_dtypes(include="object")
for cate_df_train in df_train.columns:
    df_train[cate_df_train].fillna(df_train[cate_df_train].mode()[0],inplace=True)


# categorical test df
cate_df_test = df_test.select_dtypes(include="object")
for cate_df_test in df_test.columns:
    df_test[cate_df_test].fillna(df_test[cate_df_test].mode()[0],inplace=True)


for dit_columns in df_train.select_dtypes(include=["float64"]).columns:
    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    sns.distplot(df_train[dit_columns])
    plt.title(f'Distribution of {dit_columns}')
    plt.show()

    plt.subplot(122)
    import scipy.stats as stats
    stats.probplot(df_train[dit_columns], dist="norm", plot=plt)
    plt.title(f'Distribution of {dit_columns}')

    plt.show()


df_train["Brand"].value_counts().plot(kind="pie", autopct='%.0f%%', shadow=True)
plt.title('Brand Distribution')
plt.show()

print("____________________________________________________________________\n")

# Group by 'Brand' and get the maximum 'Price', then sort the values in descending order
brand_price_max = df_train.groupby("Brand")["Price"].max().sort_values(ascending=False)

# Create a bar plot
brand_price_max.plot(kind='bar', figsize=(6, 4), color=sns.color_palette("viridis"))

# Add title and labels
plt.title('Maximum Price by Brand')
plt.xlabel('Brand')
plt.ylabel('Price')

# Show the plot
plt.show()


df_train["Material"].value_counts().plot(kind="pie", autopct='%.0f%%', shadow=True)
plt.title('Distribution of Materials')
plt.show()


df_train["Size"].value_counts().plot(kind="pie", autopct='%.0f%%', shadow=True)
plt.title("Distribution of Size")
plt.show()


# Scatter plot for Size vs Compartments with hue as Laptop Compartment
sns.scatterplot(data=df_train, x="Size", y="Compartments", hue="Laptop Compartment")
plt.title('Scatter plot of Size vs Compartments')
plt.show()

# Boxen plot for Size vs Compartments with hue as Laptop Compartment
sns.boxenplot(data=df_train, x="Size", y="Compartments", hue="Laptop Compartment")
plt.title('Boxen plot of Size vs Compartments')
plt.show()


df_train["Style"].value_counts().plot(kind = "bar",color=sns.color_palette("viridis"))
plt.title("Counts of Style")
plt.show()


label_encoder = LabelEncoder()
standar_scaler = StandardScaler()
for col in df_train.select_dtypes(include="object").columns:
    df_train[col] = label_encoder.fit_transform(df_train[col])
    df_test[col] = label_encoder.fit_transform(df_test[col])


x = df_train.drop(columns=["Price"])
y = df_train["Price"]


stander = StandardScaler()
x = pd.DataFrame(stander.fit_transform(x),columns=x.columns)


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# from sklearn.model_selection import GridSearchCV
# d_tree = DecisionTreeRegressor()
# # Define the parameter grid
# param_grid = {
#     'max_depth': [3, 5, 7, 10,2,15,20],
#     'max_features': [0.2, 0.5, 0.8],
#     'min_samples_split': [2, 5, 10,8,9,],
#     'max_leaf_nodes':[6,2,5,10,13,7,1,20],
#     'min_samples_leaf':[1.4,2,6,8,9,10,15]
# }

# # Initialize the GridSearchCV with the DecisionTreeRegressor
# d_tree_grid = GridSearchCV(estimator=d_tree, param_grid=param_grid, cv=5, verbose=2)

# # Fit the model
# d_tree_grid.fit(x_train, y_train)


# print(d_tree_grid.best_params_)
# print(d_tree_grid.best_score_)


d_tree = DecisionTreeRegressor(max_depth= 5, max_features = 0.8, max_leaf_nodes = 10, min_samples_leaf = 8, min_samples_split =2)
d_tree.fit(x_train, y_train)
y_pred = d_tree.predict(x_test)
print("RMEN",np.sqrt(mean_squared_error(y_test,y_pred)))


# RandomSearchCV

# Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features = [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]

# Bootstrap samples
bootstrap = [True,False]

# Minimum number of samples required to split a node
min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]


param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
              'max_samples':max_samples,
              'bootstrap':bootstrap,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf
             }


from sklearn.model_selection import RandomizedSearchCV
random = RandomForestRegressor()

param_grid = RandomizedSearchCV(estimator = random, 
                       param_distributions = param_grid, 
                       cv = 5, 
                       verbose=2)


param_grid.fit(x_train,y_train)


# print(param_grid.best_params_)
# print(param_grid.best_score_)


random = RandomForestRegressor(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_samples=1.0, max_features=1.0, max_depth=8, bootstrap=True)
random.fit(x_train, y_train)
y_pred = random.predict(x_test)
print("RMEN",np.sqrt(mean_squared_error(y_test,y_pred)))


df_test.head()


sub = random.predict(df_test)

submission = pd.DataFrame({
    "id": df_test["id"],  
    "Price": sub
})

submission.to_csv("submission.csv", index=False)
print("Ok")


