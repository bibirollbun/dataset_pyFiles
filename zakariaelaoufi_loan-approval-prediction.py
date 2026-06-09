import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


loan_df = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')


loan_df.head()


loan_df.describe().T


loan_df.info()


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


loan_df.isnull().sum()


def plot_multiple_boxplots(data):
    # Select only numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data[numeric_columns])
    plt.title('Outlier Detection')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


plot_multiple_boxplots(loan_df)


def remove_outliers_iqr(data):
    # Select only numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.drop(['loan_status', 'id'])
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_cleaned = data.copy()
    
    for col in numeric_columns:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remove rows with values outside the bounds
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned


clean_loan_df = remove_outliers_iqr(loan_df)


print(f'{loan_df.shape[0]-clean_loan_df.shape[0]} rows removed')


plot_multiple_boxplots(clean_loan_df)


X = clean_loan_df.drop(columns=['loan_status','id'])
y = clean_loan_df['loan_status']


X.head()


# Get unique values of loan status
numeric_col = X.select_dtypes(include=['int64','float64']).columns.tolist()
non_ordinal = X.select_dtypes(include='object').drop('loan_grade', axis=1).columns.tolist()
ordinal_cat = [X['loan_grade'].name]
loan_grade_cat = [sorted(loan_df['loan_grade'].unique().tolist())]

tranformer = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(sparse_output=False, drop='if_binary'), non_ordinal),
    ('ordinal', OrdinalEncoder(categories=loan_grade_cat), ordinal_cat),
    ('standardize', StandardScaler(), numeric_col)
    ],
    remainder='passthrough', verbose_feature_names_out=False
).set_output(transform="pandas")

X_transformed = tranformer.fit_transform(X)


X_transformed.head()


plot_multiple_boxplots(X_transformed[numeric_col])


cols_to_delete = ['person_home_ownership_OTHER','loan_intent_PERSONAL']
X_transformed.drop(columns=cols_to_delete,inplace=True)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X_transformed,y,test_size=0.1, random_state=42)
random_forest = RandomForestClassifier(n_estimators=40, random_state=42,oob_score=True)
knn = KNeighborsClassifier(n_neighbors=2)

best_rf = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    criterion='gini',   
    max_depth=20,       
    max_features='sqrt',
    min_samples_split=10,
    n_estimators=120
)


random_forest.fit(X_train, y_train)
knn.fit(X_train, y_train)
best_rf.fit(X_train, y_train)


print(f'Random Forest Accuracy: {random_forest.score(X_test, y_test)}')
print(f'KNN Accuracy: {knn.score(X_test, y_test)}')


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [40, 80, 120],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt','log2'],
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 5, 10, 20]
}


# grid_search = GridSearchCV(estimator=rf,param_grid=param_grid, cv=5, n_jobs=-1)


# grid_search.fit(X_train, y_train) 
# After applying grid search 
# Here is: 
# Best parameters: {'criterion': 'gini', 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_split': 10, 'n_estimators': 120}
# Best score: 0.9518402057573633


# best_params = grid_search.best_params_
# best_rf = grid_search.best_estimator_


# print(f'Best parameters: {best_params}')
# print(f'Best score: {grid_search.best_score_}')


# print(f'Fine tuned random forest model score: {best_rf.score(X_test, y_test)}')
# print(f'free tuned random forest model score in oob: {best_rf.oob_score_}')


df_test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
df_test.head()


X_df_test = df_test.drop(columns=['id'])
df_test_transformed = tranformer.transform(X_df_test)


df_test_transformed.drop(columns=cols_to_delete,inplace=True)


df_test_transformed.head()


y_df_test_pred = best_rf.predict(df_test_transformed)


# sub = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')
# sub['loan_status'] = y_df_test_pred
# sub.to_csv('submission.csv', index=False)

