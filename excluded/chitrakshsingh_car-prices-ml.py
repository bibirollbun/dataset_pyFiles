import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


train_df = pd.read_csv("/kaggle/input/playground-series-s4e9/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s4e9/test.csv")


train_df.shape


train_df.head()


train_df.info()


cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
num_cols = [col for col in train_df.columns if train_df[col].dtype == 'int64']


for col in cat_cols:
    print(col, " : ", train_df[col].isnull().sum())


for col in cat_cols:
    mode_value = train_df[col].mode()[0] 
    train_df[col].fillna(mode_value, inplace=True) 

for col in cat_cols:
    print(col, " : ", train_df[col].isnull().sum())


for col in cat_cols:
    print(col, " : ", train_df[col].nunique(), train_df[col].unique())


print(train_df['fuel_type'].unique())


ns = train_df[train_df['fuel_type'].str.contains('not supported', na=False)]

print("\nRows with not supported in 'fuel type':")
print(len(ns))


hyphen_rows = train_df[train_df['fuel_type'].str.contains('-', na=False)]

print("\nRows with hyphen in 'fuel type':")
print(len(hyphen_rows))


train_df['fuel_type'] = train_df['fuel_type'].str.replace('-', 'not supported', regex=False)


ns = train_df[train_df['fuel_type'].str.contains('not supported', na=False)]

print("\nRows with not supported in 'fuel type':")
print(len(ns))


for i in range(30):
    print(train_df['engine'][i])


# Initialize lists to store parsed values
hp_list = []
liters_list = []
cylinders_list = []

# Loop through the entries in the DataFrame
for i in range(len(train_df)):
    engine_string = train_df['engine'][i]
    
    # Split the string into parts
    parts = engine_string.split()
    
    # Initialize variables to None
    hp = None
    liters = None
    cylinders = None
    
    # Extract and parse the values with error handling
    try:
        # Loop through parts to find HP, liters, and cylinders
        for part in parts:
            if 'HP' in part:
                hp = int(float(part.replace('HP', '')))  # Convert HP to int
            elif 'L' in part:
                liters = int(float(part.replace('L', '')))  # Convert L to int
            elif 'V' in part:
                cylinders = int(part[1:])  # Extract the number after 'V'
            elif part.isdigit():  # Check if the part is a digit
                cylinders = int(part)  # Convert directly to int if it's a digit

    except ValueError:
        # If parsing fails for a specific value, set that value to None
        if hp is not None and 'HP' in part:
            hp = hp  # Keep the previously parsed HP
        if liters is not None and 'L' in part:
            liters = liters  # Keep the previously parsed liters
        if cylinders is not None and 'V' in part:
            cylinders = cylinders  # Keep the previously parsed cylinders

    # Append the parsed values to the lists
    hp_list.append(hp)
    liters_list.append(liters)
    cylinders_list.append(cylinders)

# Create a DataFrame to store the parsed values
parsed_data = {
    'HP': hp_list,
    'L': liters_list,
    'Cylinders': cylinders_list
}

# Convert to DataFrame
parsed_df = pd.DataFrame(parsed_data)

# Display the parsed DataFrame
print(parsed_df.head(30))


parsed_df.isnull().sum()


parsed_df['HP'].fillna(parsed_df['HP'].mean(), inplace=True)
parsed_df['L'].fillna(parsed_df['L'].mode()[0], inplace=True)  # mode returns a Series, take the first element
parsed_df['Cylinders'].fillna(parsed_df['Cylinders'].mode()[0], inplace=True)


parsed_df.isnull().sum()


train_df = train_df.drop(columns=['engine'])
train_df = pd.concat([train_df, parsed_df], axis=1)


for col in num_cols:
    print(col, " : ", train_df[col].isnull().sum())


cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
num_cols = [col for col in train_df.columns if train_df[col].dtype == 'int64']
cat_cols, num_cols


for col in cat_cols:
    print(col, " : ", train_df[col].nunique())


cat_cols = cat_cols[2:]
cat_cols


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        plt.subplot(1, 2, 1)
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title("Frequency of " + col_name)
        plt.xticks(rotation=90)

        plt.subplot(1, 2, 2)
        values = dataframe[col_name].value_counts()
        plt.pie(x=values, labels=values.index, autopct=lambda p: '{:.2f}% ({:.0f})'.format(p, p/100 * sum(values)))
        plt.title("Frequency of " + col_name)
        plt.legend(labels=['{} - {:.2f}%'.format(index, value/sum(values)*100) for index, value in zip(values.index, values)],
                   loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)
        plt.show(block=True)



for col in cat_cols:
    cat_summary(train_df, col, True)


def my_distplot(df, col, ax):
    sns.distplot(df[col], ax=ax)
    ax.set_title(f'Distribution Plot of {col}')
 

def my_boxplot(df, col, ax):
    sns.boxplot(y=df[col], ax=ax)

    
# Matrix Plots:
def my_heatmap(df, size):
    if size: plt.figure(figsize=size)
    sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap='Blues', annot_kws={"size": 12})
    plt.title('Correlation Heatmap')
    plt.show()
    
#vsplot
def my_vsplot(df, normal_col, label_col):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.bar(list(dict(df[normal_col].value_counts()).keys()), dict(df[normal_col].value_counts()).values(), color='r')
    plt.bar(list(dict(df[normal_col][df[label_col] == 1].value_counts()).keys()), dict(df[normal_col][df[label_col] == 1].value_counts()).values(), color='b')

    plt.xlabel(normal_col)
    plt.ylabel('Count')
    plt.legend(['All', label_col])
    # plt.title('The number of requests from different protocols')
    
def plot_charts_grid_single_feature(df, plot_func, size=(12, 4), n_col=1):
    if len(df.columns) == 0:
        return
    n_rows = (len(df.columns) + n_col-1) // n_col
    fig, axes = plt.subplots(n_rows, n_col, figsize=(size[0]*n_col, size[1]*n_rows))
    if len(df.columns) == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, label in enumerate(df.columns):
        plot_func(df, label, axes[i])
        axes[i].set_xlabel(label)

    for j in range(i+1, n_rows*n_col):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


plot_charts_grid_single_feature(train_df[num_cols], my_distplot)


plot_charts_grid_single_feature(train_df[num_cols], my_boxplot, size=(2, 4), n_col=6)








n_numeric_cols = len(train_df.select_dtypes(include=[np.number]).columns) // 3 * 2
my_heatmap(train_df.select_dtypes(include=[np.number]), size=(n_numeric_cols+1, n_numeric_cols+1))





def skewDis(df, col):
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_bridge = df[col].quantile(0.25)-(iqr*3)
    upper_bridge = df[col].quantile(0.75)+(iqr*3)
    
#     print(lower_bridge, upper_bridge)
    
    df.loc[df[col] >= upper_bridge, col] = upper_bridge
    df.loc[df[col] <= lower_bridge, col] = lower_bridge

for i in num_cols:
    skewDis(train_df, i)


plot_charts_grid_single_feature(train_df[num_cols], my_boxplot, size=(2, 4), n_col=6)


train_df.head()


# Define features and target variable
X = train_df.drop(['price', 'clean_title'], axis=1)
y = train_df['price']

# Define categorical columns for label encoding and ordinal encoding
label_cols = ['brand', 'model', 'ext_col', 'int_col', 'accident']
ordinal_cols = ['model_year', 'fuel_type', 'transmission']

# Create a label encoder for label encoding
label_encoder = LabelEncoder()

# Apply label encoding to specified columns
for col in label_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Create an ordinal encoder for ordinal encoding
ordinal_encoder = OrdinalEncoder()

# Apply ordinal encoding to specified columns
X[ordinal_cols] = ordinal_encoder.fit_transform(X[ordinal_cols])

# Display the processed DataFrame
X.head()


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


%%time
xgb_params = {
    'lambda': [0.03880258557285165],
    'alpha': [0.02129832295514386],
    'colsample_bytree': [0.4],
    'subsample': [0.7],
    'learning_rate': [0.014],
    'max_depth': [17],
    'min_child_weight': [85],
    'n_estimators': [10000],
}


# Use KFold for regression
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV with XGBRegressor
xgb = GridSearchCV(estimator=XGBRegressor(random_state=42),
                   param_grid=xgb_params, cv=kf, n_jobs=4)

# Fit the model
xgb_model = xgb.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, xgb_pred)
mse = mean_squared_error(y_test, xgb_pred)
r2 = r2_score(y_test, xgb_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


%%time

lgbm_params={
    'num_leaves': [426],
    'max_depth': [20],
    'learning_rate': [0.011353178352988012],
    'n_estimators': [10000],
    'metric': ['rmse'],
    'subsample': [0.5772552201954328],
    'colsample_bytree': [0.9164865430101521],
    'reg_alpha': [1.48699088003429e-06],
    'reg_lambda': [0.41539458543414265],
    'min_data_in_leaf': [73],
    'feature_fraction': [0.751673655170548],
    'bagging_fraction': [0.5120415391590843],
    'bagging_freq': [2],
    'random_state': [42],  # Note: random_state is usually not varied in grid search
    'min_child_weight': [0.017236362383443497],
    'cat_smooth': [54.81317407769262],
}

# Use KFold for regression
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV with LGBMRegressor
lgbm = GridSearchCV(estimator=LGBMRegressor(random_state=42),
                    param_grid=lgbm_params, cv=kf, n_jobs=4)

# Fit the model
lgbm_model = lgbm.fit(X_train, y_train)

# Make predictions
lgbm_pred = lgbm_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, lgbm_pred)
mse = mean_squared_error(y_test, lgbm_pred)
r2 = r2_score(y_test, lgbm_pred)

# Print evaluation metrics
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

