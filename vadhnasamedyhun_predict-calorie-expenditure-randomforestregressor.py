import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from scipy.stats import normaltest, boxcox
from sklearn.ensemble import IsolationForest, RandomForestRegressor


calories = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

calories.head(5)


calories.drop('id', axis=1, inplace=True)


calories.shape


calories.info()


calories.dtypes


calories.isnull().sum()


calories[calories.duplicated() == True]


calories['BMI'] = calories.Weight / ((calories.Height / 100)**2)


len(calories[calories.Sex == 'male']) / calories.shape[0]


num_cols = calories.loc[:, ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']]


preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), num_cols.columns),
        ('cat', OneHotEncoder(), ['Sex'])
    ]
)


X = calories.drop('Calories', axis=1)
y = calories.Calories


X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.3, random_state=42
)


preprocessor.fit(X_train)


feature_names = np.concatenate((preprocessor.named_transformers_['num'].get_feature_names_out(num_cols.columns), preprocessor.named_transformers_['cat'].get_feature_names_out(['Sex'])))


X_train_processed = pd.DataFrame(preprocessor.transform(X_train),columns=feature_names)


X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


rf_clf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


rf_clf.fit(X_train_processed, y_train)


rf_preds = rf_clf.predict(X_test_processed)


rmsle(y_test, rf_preds)


train_preds = rf_clf.predict(X_train_processed)

print("Training RMSLE:", rmsle(y_train, train_preds))
print("Test RMSLE:", rmsle(y_test, rf_preds))


metrics = ['Training', 'Test']
values = [rmsle(y_train, train_preds), rmsle(y_test, rf_preds)]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=['blue', 'orange'])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom')

plt.title('Model Performance: Training vs Test RMSLE', pad=20)
plt.ylabel('RMSLE')
plt.ylim(0, max(values) * 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


def dataset_processing(path):
    # Read the file
    df = pd.read_csv(path)
    
    # Make a copy with ID preserved
    df_copy = df.copy()
    df_copy = df_copy.drop('id', axis=1)
    df_copy_id = df[['id']]
    
    # Create BMI column
    df_copy['BMI'] = df_copy['Weight'] / ((df_copy['Height'] / 100)**2)
    
    # Define numeric columns
    num_cols = df_copy.loc[:, ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']]
    
    feature_names = np.concatenate((
        preprocessor.named_transformers_['num'].get_feature_names_out(num_cols.columns),
        preprocessor.named_transformers_['cat'].get_feature_names_out(['Sex'])
    ))
    
    df_test_processed = pd.DataFrame(preprocessor.transform(df_copy), columns=feature_names)
    
    # Prediction
    rf_preds = rf_clf.predict(df_test_processed)
    
    # Get the test set indices to match with IDs
    df_indices = df_test_processed.index
    
    # Create predictions DataFrame with corresponding IDs
    rf_preds_df = pd.DataFrame({
        'id': df_copy_id.loc[df_indices, 'id'].values,
        'Calories': rf_preds
    })
    
    return rf_preds_df


test_preds = dataset_processing('/kaggle/input/playground-series-s5e5/test.csv')


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

submission['Calories'] = np.round(test_preds.Calories, 2)
submission.to_csv("/kaggle/working/submission.csv", index=False)
submission.head()




