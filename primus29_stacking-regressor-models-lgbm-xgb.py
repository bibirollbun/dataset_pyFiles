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


df=pd.read_csv(r"/kaggle/input/playground-series-s5e5/train.csv")
df.head()


import matplotlib.pyplot as plt
import seaborn as sns


df.info


df.sample(5)


df['Sex'].value_counts()


df.describe()


df.hist(bins=50, figsize=(20, 15))
plt.show()


attributes = ["Age",	"Height", "Weight",	"Duration",	"Heart_Rate",	"Body_Temp",	"Calories"]


# Loop through each attribute in the 'attributes' list
for i in attributes:
    # Create a new figure and axis with tight layout to prevent overlapping content
    fig, ax = plt.subplots(tight_layout=True)

    # Create a 2D histogram to show the joint distribution of 'Body_Temp' and the current attribute
    # 'bins=25' specifies the number of bins in each dimension
    # 'cmap="plasma"' sets the color map for the plot
    hist = ax.hist2d(df['Body_Temp'], df[i], bins=25, cmap='plasma')

    # Add a color bar to indicate the count of data points in each bin
    cbar = plt.colorbar(hist[3], ax=ax)
    cbar.set_label('Count', fontsize='14')

    # Set labels and title with formatting for better readability
    ax.set_xlabel('Body Temp', fontsize='14')
    ax.set_ylabel(i, fontsize='14')
    ax.set_title(f'2D Histogram of Body Temp vs {i}', color='#FE420F', fontsize='16')

    # Display the plot
    plt.show()


# Set the figure size for better visibility of the heatmap
plt.figure(figsize=(10, 6))

# Calculate the correlation matrix for the specified attributes in the dataframe
corr_matrix = df[attributes].corr()

# Plot a heatmap to visualize the correlation matrix
# 'annot=True' displays the correlation values on the heatmap
# 'cmap="viridis"' sets the color scheme for the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='viridis')

# Add a title to the heatmap with specified font size
plt.title('Correlation Matrix of Features', fontsize=14)

# Show the heatmap plot
plt.show()


# Import OrdinalEncoder from scikit-learn for encoding categorical features
from sklearn.preprocessing import OrdinalEncoder

# Initialize the encoder
ordinal_encoder = OrdinalEncoder()

# Encode the 'Sex' column from categorical (e.g., 'Male', 'Female') to numerical values (e.g., 0, 1)
# This is necessary for many machine learning models that require numerical input
df['Sex'] = ordinal_encoder.fit_transform(df[['Sex']])


df['Sex'][:10]


# Calculate Body Mass Index (BMI) from weight (kg) and height (cm)
df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)

# Estimate exercise intensity as heart rate per minute of exercise
df['exercise_intensity'] = df['Heart_Rate'] / df['Duration']

# Total cardiovascular load (heart rate multiplied by duration)
df['heart_rate_duration'] = df['Heart_Rate'] * df['Duration']

# Thermal load: body temperature multiplied by duration
df['temp_duration'] = df['Body_Temp'] * df['Duration']

# Ratio of heart rate to body temperature — may indicate stress or exertion
df['hr_to_temp'] = df['Heart_Rate'] / df['Body_Temp']

# Ratio of heart rate to age — normalized cardiovascular activity
df['hr_to_age'] = df['Heart_Rate'] / df['Age']

# Combined measure of age and body mass index — possible health risk indicator
df['age_bmi'] = df['Age'] * df['bmi']

# Estimate of maximum heart rate based on age (common fitness formula)
df['max_heart_rate'] = 220 - df['Age']

# Proportion of current heart rate to estimated max heart rate — exertion level
df['heart_rate_intensity'] = df['Heart_Rate'] / df['max_heart_rate']


X = df.drop(['Calories', 'id'], axis = 1).copy()
y = df['Calories']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state = 42)


# Import the standard scaler for feature normalization
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler (scales features to mean=0 and std=1)
scaler = StandardScaler()
scaler_test = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler_test.fit_transform(y_train.values.reshape(-1, 1))

# Convert the scaled training data back into a DataFrame with original column names
import pandas as pd
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=[y_train.name])
y_train_log = np.log1p(y_train)

# Use the same scaler (fitted on training data) to transform the test data
X_test_scaled = scaler.transform(X_test)
y_test_scaled = scaler_test.transform(y_test.values.reshape(-1, 1))

# Convert the scaled test data back into a DataFrame with original column names
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
t_test_scaled_df = pd.DataFrame(y_test_scaled, columns=[y_test.name])


# from sklearn.tree import DecisionTreeRegressor
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(X_train, y_train)


# X_test.head()


# y_pred = tree_reg.predict(X_test)


# y_pred


# y_test


# from sklearn.metrics import mean_squared_log_error
# rmlse = np.sqrt(mean_squared_log_error(y_test, y_pred))
# print(rmlse)


# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(y_test, y_pred)


# from sklearn.metrics import mean_squared_error
# mean_squared_error(y_test, y_pred)


# X_train_scaled_df[:10]


# # Import the RandomForestRegressor from scikit-learn
# from sklearn.ensemble import RandomForestRegressor

# # Initialize the Random Forest Regressor with default parameters
# for_reg = RandomForestRegressor()

# # Fit the model on the training data
# # X_train: features, y_train: target variable
# for_reg.fit(X_train_scaled_df, y_train)


# # Predicting on the scaled test data
# y_pred = for_reg.predict(X_test_scaled_df)


# from sklearn.metrics import mean_squared_log_error
# rmlse = np.sqrt(mean_squared_log_error(y_test, y_pred))
# print(rmlse)


from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Base estimators
xgb_model = XGBRegressor(
    n_estimators=10000,
    learning_rate=0.04,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbosity=0,
    random_state=42,
)
lgb_model = LGBMRegressor(
    n_estimators=10000,
    learning_rate=0.04,
    max_depth=10,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)
# Stacking with Ridge as meta-model
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    final_estimator=Ridge(),
    cv=3,
    n_jobs=-1
)
# Fit model
stack_model.fit(X_train_scaled_df, y_train_log)


from sklearn.metrics import mean_squared_error, mean_squared_log_error
y_pred_log = stack_model.predict(X_test_scaled_df)
y_pred = np.expm1(y_pred_log)

mse = mean_squared_error(y_test, y_pred)
print(mse)
rmlse = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(rmlse)


df_test=pd.read_csv(r"/kaggle/input/playground-series-s5e5/test.csv")
df_test.head()


# Calculate Body Mass Index (BMI) from weight (kg) and height (cm)
df_test['bmi'] = df_test['Weight'] / ((df_test['Height'] / 100) ** 2)

# Estimate exercise intensity as heart rate per minute of exercise
df_test['exercise_intensity'] = df_test['Heart_Rate'] / df_test['Duration']

# Total cardiovascular load (heart rate multiplied by duration)
df_test['heart_rate_duration'] = df_test['Heart_Rate'] * df_test['Duration']

# Thermal load: body temperature multiplied by duration
df_test['temp_duration'] = df_test['Body_Temp'] * df_test['Duration']

# Ratio of heart rate to body temperature — may indicate stress or exertion
df_test['hr_to_temp'] = df_test['Heart_Rate'] / df_test['Body_Temp']

# Ratio of heart rate to age — normalized cardiovascular activity
df_test['hr_to_age'] = df_test['Heart_Rate'] / df_test['Age']

# Combined measure of age and body mass index — possible health risk indicator
df_test['age_bmi'] = df_test['Age'] * df_test['bmi']

# Estimate of maximum heart rate based on age (common fitness formula)
df_test['max_heart_rate'] = 220 - df_test['Age']

# Proportion of current heart rate to estimated max heart rate — exertion level
df_test['heart_rate_intensity'] = df_test['Heart_Rate'] / df_test['max_heart_rate']



# Import OrdinalEncoder from scikit-learn for encoding categorical features
from sklearn.preprocessing import OrdinalEncoder

# Initialize the encoder
ordinal_encoder = OrdinalEncoder()

# Encode the 'Sex' column from categorical (e.g., 'Male', 'Female') to numerical values (e.g., 0, 1)
# This is necessary for many machine learning models that require numerical input
df_test['Sex'] = ordinal_encoder.fit_transform(df_test[['Sex']])



X = df_test.drop('id', axis = 1).copy()
ids = df_test['id']


X.head()


X.head()


# Import the standard scaler for feature normalization
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler (scales features to mean=0 and std=1)
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_scaled = scaler.fit_transform(X)

# Convert the scaled training data back into a DataFrame with original column names
import pandas as pd
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


X_scaled_df.head()


y_test_pred = stack_model.predict(X_scaled_df)


y_test_pred


y_pred = np.expm1(y_test_pred)


y_pred


sub = pd.DataFrame({
    "id": ids,
    "Calories": y_pred
})
sub.to_csv("sub.csv", index=False)
# Save to /kaggle/working/ so it appears in Output tab
sub.to_csv("/kaggle/working/sub.csv", index=False)







