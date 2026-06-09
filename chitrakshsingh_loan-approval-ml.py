import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as fgr
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


train_df = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")


train_df.shape


train_df.head()


train_df.info()


cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
num_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64' or train_df[col].dtype == 'int64']


cat_cols, num_cols


for col in cat_cols:
    print(col, " : ", train_df[col].isnull().sum())


for col in cat_cols:
    print(col, " : ", train_df[col].nunique(), train_df[col].unique())





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


def skewDis(df, col):
    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
    lower_bridge = df[col].quantile(0.25)-(iqr*3)
    upper_bridge = df[col].quantile(0.75)+(iqr*3)
    
#     print(lower_bridge, upper_bridge)
    
    df.loc[df[col] >= upper_bridge, col] = upper_bridge
    df.loc[df[col] <= lower_bridge, col] = lower_bridge


for i in num_cols:
    if i != 'loan_status':
        skewDis(train_df, i)


plot_charts_grid_single_feature(train_df[num_cols], my_boxplot, size=(2, 4), n_col=6)


n_numeric_cols = len(train_df.select_dtypes(include=[np.number]).columns) // 3 * 2
my_heatmap(train_df.select_dtypes(include=[np.number]), size=(n_numeric_cols+1, n_numeric_cols+1))


print(train_df.columns)


train_df.head()


label_cols = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
ordinal_cols = ['loan_grade']

label_encoder = LabelEncoder()

for col in label_cols:
    train_df[col] = label_encoder.fit_transform(train_df[col])

loan_grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
train_df['loan_grade'] = train_df['loan_grade'].map(loan_grade_mapping)

train_df.head()


%%time

# Correlation matrix
correlation_matrix = train_df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(train_df, hue='loan_status', vars=['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'])
plt.title('Pairplot of Features vs Loan Status')
plt.show()

# Boxplot for categorical features against loan amount
plt.figure(figsize=(12, 6))
sns.boxplot(x='loan_grade', y='loan_amnt', data=train_df)
plt.title('Loan Amount Distribution by Loan Grade')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='loan_intent', y='loan_amnt', data=train_df)
plt.title('Loan Amount Distribution by Loan Intent')
plt.show()

# Scatter plot for income vs loan amount colored by loan status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='person_income', y='loan_amnt', hue='loan_status', s=100, data=train_df)
plt.title('Income vs Loan Amount')
plt.xlabel('Person Income')
plt.ylabel('Loan Amount')
plt.legend(title='Loan Status')
plt.grid(True)
plt.show


def stacked_bar_plot(df, feature, target='loan_status'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='coolwarm')
    plt.title(f'Stacked Bar Plot of {feature} vs {target}')
    plt.ylabel('Proportion')
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.show()


for col in cat_cols:
    stacked_bar_plot(train_df, col)


# def feature_engineering(df):
    
#     df['loan_to_income_ratio'] = df['loan_amnt'] / df['person_income']  
#     df['financial_burden'] = df['loan_amnt'] * df['loan_int_rate'] 
#     df['income_per_year_emp'] = df['person_income'] / (df['person_emp_length'])
#     df['cred_hist_to_age_ratio'] = df['cb_person_cred_hist_length'] / df['person_age']
#     df['int_to_loan_ratio'] = df['loan_int_rate'] / df['loan_amnt']
#     df['loan_int_emp_interaction'] = df['loan_int_rate'] * df['person_emp_length']
#     df['debt_to_credit_ratio'] = df['loan_amnt'] / df['cb_person_cred_hist_length'] 
#     df['int_to_cred_hist'] = df['loan_int_rate'] / df['cb_person_cred_hist_length']  
#     df['int_per_year_emp'] = df['loan_int_rate'] / (df['person_emp_length'])
#     df['loan_amt_per_emp_year'] = df['loan_amnt'] / (df['person_emp_length'])      
#     df['income_to_loan_ratio'] = df['person_income'] / df['loan_amnt'] 
    
#     return df

# #median_income = df_train['person_income'].median()
# train_df = feature_engineering(train_df)
# test_df = feature_engineering(test_df)


# Separate features and labels for classification
X = train_df.drop('loan_status', axis=1)
y = train_df['loan_status']


scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split the data into train and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


%%time

lgbm_params = {
    'objective': ['binary'],
    'n_estimators': [3000],
    'metric': ['binary_logloss'],
    'boosting_type': ['gbdt'],
    'random_state': [42],
    'learning_rate': [0.0322942967545754],
    'num_leaves': [24],
    'max_depth': [15],
    'min_data_in_leaf': [25],  
    'feature_fraction': [0.6236144085285287], 
    'bagging_fraction': [0.9596685778433888], 
    'bagging_freq': [3],  
    'verbose': [-1]
}

skf = StratifiedKFold(n_splits=3)

lgbm = GridSearchCV(estimator=LGBMClassifier(random_state=42, verbosity=-1),
                              param_grid=lgbm_params, cv=skf, n_jobs=3, verbose=-1)
                      
lgbm_model = lgbm.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_test)
accuracy_score(y_test, lgbm_pred)


%%time

catboost_params = {
    'depth': 7,
    'learning_rate': 0.19893301995319765,
    'bagging_temperature': 0.7979373495258176,
    'l2_leaf_reg': 5,
    'loss_function': 'Logloss',
    'iterations': 400,
    'grow_policy': 'Lossguide',
    'eval_metric': 'AUC',
}

cb = CatBoostClassifier()
cb_model = cb.fit(X_train, y_train)
cb_pred = cb_model.predict(X_test)
accuracy_score(y_test, cb_pred)




