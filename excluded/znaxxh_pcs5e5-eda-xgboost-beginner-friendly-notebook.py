import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Display basic info about the training data
train.describe()
train.info()
train.head()


# Check for unique values and nulls in 'Sex' column
distinct_sex_values = train['Sex'].unique()
train.isnull().sum().sort_values(ascending=False)


# Print unique values in 'Sex'
print(train['Sex'].unique())


# firstly we'll binary encode the 'Sex' column since it have only 2 distinct values 'Male' and 'Female'
# Binary encode the 'Sex' column using LabelEncoder
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.fit_transform(test['Sex'])

# Binary Encoding Manually using map function
# train['Sex'] = train['Sex'].map({'male': 0 , 'female': 1}) 


sns.barplot(x=train['Duration'],y=train['Calories'])


sns.barplot(x=train['Heart_Rate'],y=train['Calories'])


sns.barplot(x=train['Body_Temp'],y=train['Calories'])


# Sample 100 rows for scatter plots
sample_train_df = train.sample(n=100, random_state=42)


# Scatter plot with regression line for 'Sex' vs 'Calories'
sns.lmplot(x='Sex', y='Calories', data=sample_train_df, scatter_kws={'alpha':0.5}, line_kws={"color":"red"})
plt.title("Scatter Plot with Regression Line")


# Correlation heatmaps for various features
sex_correlation = train[['Sex','Calories']].corr()
sns.heatmap(sex_correlation,  annot=True)
plt.title('Correlation Matrix of Sex and Calories')


Age_correlation = train[['Age','Calories']].corr()
sns.heatmap(Age_correlation , annot= True)
plt.title("Correlation Between Age and Calories")


sns.lmplot(x='Age', y='Calories', data=sample_train_df, scatter_kws={"alpha":0.5}, line_kws={"color":"red"})


Height_correlation = train[['Height','Calories']].corr()
sns.heatmap(Height_correlation, annot = True)
plt.title('Correlation between Height and Calories')


sns.lmplot(x='Height',y='Calories', data=sample_train_df, scatter_kws={'alpha': 0.5}, line_kws={"color":"red"})


Weight_correlation = train[['Weight','Calories']].corr()
sns.heatmap(Weight_correlation, annot = True)
plt.title('Correlation between Weight and Calories')


sns.lmplot(x='Weight',y='Calories', data=sample_train_df, scatter_kws={'alpha': 0.5}, line_kws={"color":"red"})


Duration_correlation = train[['Duration','Calories']].corr()
sns.heatmap(Duration_correlation, annot = True)
plt.title('Correlation between Duration and Calories')


sns.lmplot(x='Duration',y='Calories', data=sample_train_df, scatter_kws={'alpha': 0.5}, line_kws={"color":"red"})


Heart_Rate_correlation = train[['Heart_Rate','Calories']].corr()
sns.heatmap(Heart_Rate_correlation, annot = True)
plt.title('Correlation between Heart_Rate and Calories')
sns.lmplot(x='Heart_Rate',y='Calories', data=sample_train_df, scatter_kws={'alpha': 0.5}, line_kws={"color":"red"})


Body_Temp_correlation = train[['Body_Temp','Calories']].corr()
sns.heatmap(Body_Temp_correlation, annot = True)
plt.title('Correlation between Body_Temp and Calories')
sns.lmplot(x='Body_Temp',y='Calories', data=sample_train_df, scatter_kws={'alpha': 0.5}, line_kws={"color":"red"})


# Check for duplicate rows in train and test sets
Duplicated_train_rows = train.duplicated()
Duplicated_test_rows = test.duplicated()
print("Number of duplicate rows in training set:", Duplicated_train_rows.sum())
print("Number of duplicate rows in test set:", Duplicated_test_rows.sum())


# Duration per kg
train['Duration_per_kg'] = train['Duration'] / train['Weight']
test['Duration_per_kg'] = test['Duration'] / test['Weight']


# Heart Rate x Duration (workout intensity)
train['HRxDuration'] = train['Heart_Rate'] * train['Duration']
test['HRxDuration'] = test['Heart_Rate'] * test['Duration']
# this can be the intensity of the workout


# BMI calculation
train['Height_m'] = train['Height'] / 100  # if height is in cm
train['BMI'] = train['Weight'] / (train['Height_m'] ** 2)
train=train.drop('Height_m', axis= 1)

test['Height_m'] = test['Height'] / 100  # if height is in cm
test['BMI'] = test['Weight'] / (test['Height_m'] ** 2)
test=test.drop('Height_m', axis= 1)


# Visualize Calories by Age
sns.barplot(x=train['Age'] , y=train['Calories'])


# Age group feature
def age_group(age):
    if age < 20:
        return 'Teen'
    elif age < 30:
        return 'Young Adult'
    elif age < 45:
        return 'Adult'
    elif age < 60:
        return 'Middle Aged'
    else:
        return 'Senior'


train['Age_Group'] = train['Age'].apply(age_group)
duration_by_age_group = train.groupby('Age_Group')['Duration'].mean().to_dict()
train['Exercise_Duration_By_Age'] = train['Age_Group'].map(duration_by_age_group)

test['Age_Group'] = test['Age'].apply(age_group)
duration_by_age_group = test.groupby('Age_Group')['Duration'].mean().to_dict()
test['Exercise_Duration_By_Age'] = test['Age_Group'].map(duration_by_age_group)


# Map Age_Group to numeric values
Age_Group_map = {'Teen': 1, 'Young Adult': 0, 'Adult': 2, 'Middle Aged': 3, 'Senior': 4}
train["Age_Group"] = train["Age_Group"].map(Age_Group_map)
test["Age_Group"] = test["Age_Group"].map(Age_Group_map)


# Check for nulls after feature engineering
train.isnull().sum()


# Feature list for modeling
Features = ['Sex', 'Age','Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Duration_per_kg', 'HRxDuration', 'BMI' ,'Exercise_Duration_By_Age' ]


# Prepare data for model
X = train[Features]
y = train['Calories']


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Train XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)


# Train the model on the training set
xgb_model.fit(X_train, y_train)


# Predict on the validation set
y_val_pred = xgb_model.predict(X_val)
y_val_pred = np.maximum(0, y_val_pred)  # Ensure no negative predictions


rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))
print(f"XGBRegressor RMSLE (validation): {rmsle:.5f}")


# Now train on the full data for final submission
xgb_model.fit(X, y)
y_pred = xgb_model.predict(test[Features])
y_pred = np.maximum(0, y_pred)  # Ensure no negative predictions


submission = pd.DataFrame({
    'id': test.id,
    'Calories': y_pred
})
submission.to_csv('submission.csv', index=False)

