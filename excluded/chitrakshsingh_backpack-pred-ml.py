import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, StratifiedKFold


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train_df.shape


train_df.head()


train_df.info()





cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
num_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']


cat_cols, num_cols


for col in cat_cols:
    print(col, " : ", train_df[col].isnull().sum())


for col in cat_cols:
    print(col, " : ", train_df[col].nunique(), train_df[col].unique())


for col in cat_cols:
    mode_value = train_df[col].mode()[0] 
    train_df[col].fillna(mode_value, inplace=True) 

for col in cat_cols:
    print(col, " : ", train_df[col].isnull().sum())





for col in num_cols:
    print(col, " : ", train_df[col].isnull().sum())


train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean(), inplace=True) 


for col in num_cols:
    print(col, " : ", train_df[col].isnull().sum())





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


from scipy.stats import chi2_contingency

# Function to calculate Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - 1
    kcorr = k - 1
    return np.sqrt(phi2corr / min((kcorr, rcorr)))

# Create a correlation matrix for categorical variables
corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

for col1 in cat_cols:
    for col2 in cat_cols:
        corr_matrix.loc[col1, col2] = cramers_v(train_df[col1], train_df[col2])

# Convert the correlation matrix to numeric
corr_matrix = corr_matrix.astype(float)

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title("Cramér's V Correlation Matrix for Categorical Variables")
plt.show()


n_numeric_cols = len(train_df.select_dtypes(include=[np.number]).columns) // 3 * 2
my_heatmap(train_df.select_dtypes(include=[np.number]), size=(n_numeric_cols+1, n_numeric_cols+1))


label_cols = ['Brand', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
ordinal_cols = ['Size']

label_encoder = LabelEncoder()

for col in label_cols:
    train_df[col] = label_encoder.fit_transform(train_df[col])

# Manually map the Size values
size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
train_df['Size'] = train_df['Size'].map(size_mapping)

train_df.head()


for col in cat_cols:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=col, y='Price', data=train_df, estimator='mean')
    plt.title(f'Average Price by {col}')
    plt.xlabel(col)
    plt.ylabel('Average Price')
    plt.grid(axis='y')
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.show()


sns.scatterplot(x='Weight Capacity (kg)', y='Price', data=train_df, hue='Size')


X = train_df.drop(['Price'], axis=1)
y = train_df['Price']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





%%time

#sample parameter
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

kf = KFold(n_splits=3, shuffle=True, random_state=42)

xgb = GridSearchCV(estimator=XGBRegressor(random_state=42),
                   param_grid=xgb_params, cv=kf, n_jobs=4)

xgb_model = xgb.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, xgb_pred)
mse = mean_squared_error(y_test, xgb_pred)
r2 = r2_score(y_test, xgb_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# %%time

# lgbm_params = {
#     'num_leaves': [426],
#     'max_depth': [20],
#     'learning_rate': [0.011353178352988012],
#     'n_estimators': [10000],
#     'metric': ['rmse'],
#     'subsample': [0.5772552201954328],
#     'colsample_bytree': [0.9164865430101521],
#     'reg_alpha': [1.48699088003429e-06],
#     'reg_lambda': [0.41539458543414265],
#     'min_data_in_leaf': [73],
#     'feature_fraction': [0.751673655170548],
#     'bagging_fraction': [0.5120415391590843],
#     'bagging_freq': [2],
#     'random_state': [42],
#     'min_child_weight': [0.017236362383443497],
#     'cat_smooth': [54.81317407769262],
# }

# kf = KFold(n_splits=3, shuffle=True, random_state=42)

# lgbm = GridSearchCV(estimator=LGBMRegressor(random_state=42),
#                     param_grid=lgbm_params, cv=kf, n_jobs=4)

# lgbm_model = lgbm.fit(X_train, y_train)

# lgbm_pred = lgbm_model.predict(X_test)

# mae = mean_absolute_error(y_test, lgbm_pred)
# mse = mean_squared_error(y_test, lgbm_pred)
# r2 = r2_score(y_test, lgbm_pred)

# print(f'Mean Absolute Error: {mae}')
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

