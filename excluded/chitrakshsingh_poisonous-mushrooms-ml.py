import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import IsolationForest

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


train_df = pd.read_csv("/kaggle/input/playground-series-s4e8/train.csv", index_col='id')
test_df = pd.read_csv("/kaggle/input/playground-series-s4e8/test.csv")
sample_df = pd.read_csv("/kaggle/input/playground-series-s4e8/sample_submission.csv")


train_df.shape


train_df.head()


train_df.info()


train_df.isnull().sum() / train_df.shape[0] * 100


cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
num_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']
cat_cols, num_cols


for col in cat_cols:
    print(col, train_df[col].unique())


for col in cat_cols:
    print(col, " : ", train_df[col].nunique())


for column in cat_cols:
    print(f"\nTop value counts in '{column}':\n{train_df[column].value_counts().head(10)}")


threshold = 70

for column in cat_cols:
    value_counts = train_df[column].value_counts()
    infrequent_categories = value_counts[value_counts < threshold].index
    train_df[column] = train_df[column].replace(infrequent_categories, 'unknown')
    print(f"\nUpdated value counts in '{column}':\n{train_df[column].value_counts().head(10)}")


test_df.info()


for column in cat_cols:
    if column != 'class':
        value_counts = test_df[column].value_counts()
        infrequent_categories = value_counts[value_counts < threshold].index
        test_df[column] = test_df[column].replace(infrequent_categories, 'unknown')
        print(f"\nUpdated value counts in '{column}':\n{test_df[column].value_counts().head(10)}")


for col in cat_cols:
    print(col, " : ", train_df[col].isnull().sum() / train_df.shape[0] * 100)


drop_cols = ['stem-root', 'stem-surface' , 'veil-type', 'veil-color', 'spore-print-color']

train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)


cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']

for column in cat_cols:
    train_df[column].fillna(train_df[column].mode().iloc[0], inplace=True)

for column in cat_cols:
    if column != 'class':
        test_df[column].fillna(test_df[column].mode().iloc[0], inplace=True)
    

for col in cat_cols:
    if col != 'class':
        print(col, " : ", train_df[col].isnull().sum() / train_df.shape[0] * 100)


train_df = train_df[~train_df.isin(['unknown']).any(axis=1)]
test_df = test_df[~test_df.isin(['unknown']).any(axis=1)]


for column in cat_cols:
    print(f"\nTop value counts in '{column}':\n{train_df[column].value_counts().head(10)}")





train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)


for col in num_cols:
    print(col, " : ", train_df[col].isnull().sum())


for col in num_cols:
    print(col, " : ", test_df[col].isnull().sum())


for column in num_cols:
    train_df[column].fillna(train_df[column].mean(), inplace=True)

for column in num_cols:
    test_df[column].fillna(test_df[column].mean(), inplace=True)


for col in num_cols:
    print(col, " : ", train_df[col].isnull().sum())


for col in num_cols:
    print(col, " : ", test_df[col].isnull().sum())


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





# Initialize the LabelEncoder
encoder = LabelEncoder()

# Encode all categorical columns
for column in train_df.select_dtypes(include=['object']).columns:
    train_df[column] = encoder.fit_transform(train_df[column])


X = train_df.drop('class', axis=1)
y = train_df['class']


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


%%time

isolation_forest = IsolationForest(contamination=0.024, random_state=42)
outlier_labels = isolation_forest.fit_predict(X_train)


non_outliers_mask = outlier_labels != -1
X_train = X_train[non_outliers_mask]
y_train = y_train[non_outliers_mask]


%%time 

xgb_params = {
    'colsample_bytree': [0.43786552283911356], 
    'learning_rate': [0.027640232910206706], 
    'max_depth': [15], 
    'min_child_weight': [8], 
    'n_estimators': [456], 
    'subsample': [0.9379640997273687]}


skf = StratifiedKFold(n_splits=3)

xgb = GridSearchCV(estimator=XGBClassifier(random_state=42),
                              param_grid=xgb_params, cv=skf, n_jobs=4)

# lgbm = GridSearchCV(estimator=LGBMClassifier(random_state=42, verbosity=-1),
#                               param_grid=lgbm_params, cv=skf, n_jobs=3, verbose=-1)
                    
xgb_model = xgb.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
accuracy_score(y_test, xgb_pred)


for column in test_df.select_dtypes(include=['object']).columns:
    test_df[column] = encoder.fit_transform(test_df[column])


test_df_copy = test_df.drop(columns=['id'])


test_preds = xgb_model.predict(test_df_copy)
test_preds = encoder.inverse_transform(test_preds)


output = pd.DataFrame({'id': test_df['id'],
                       'class': test_preds})

output.to_csv('submission.csv', index=False)

output.head()

