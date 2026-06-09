import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error





train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

# Drop 'id' column from train (it's not a useful feature)
train_df = train_df.drop(columns=['id'])
# Remove duplicates (to avoid bias in training)
train_df.drop_duplicates(inplace=True)

# Encode 'Sex' as numeric 
train_df['Sex'] = train_df['Sex'].map({'male': 1, 'female': 0})
test_df['Sex'] = test_df['Sex'].map({'male': 1, 'female': 0})


# Scatter plots to understand relationships
custom_palette = {1: '#89CFF0', 0: '#FFB6C1'}
sns.scatterplot(data=train_df, x='Height', y='Weight', hue='Sex',palette=custom_palette)
plt.title('Height vs Weight by Sex')
plt.show()

sns.scatterplot(data=train_df, x='Duration', y='Calories', hue='Sex',palette=custom_palette)
plt.title('Duration vs Calories by Sex')
plt.show()

sns.scatterplot(data=train_df, x='Body_Temp', y='Calories', hue='Sex',palette=custom_palette)
plt.title('Body Temperature vs Calories by Sex')
plt.show()



numerical_features = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Calories"
    ]
# KDE plots for all numerical features to understand their distributions

# Generate a distinct color for each feature using the pastel palette
kde_colors = sns.color_palette("pastel", len(numerical_features))

# Auto-calculate the required number of subplot rows (4 plots per row)
n_cols = 4
n_rows = -(-len(numerical_features) // n_cols)  # Ceiling division

plt.figure(figsize=(20, 5 * n_rows))

# Loop through each numerical feature and create a KDE plot
for idx, (feature, color) in enumerate(zip(numerical_features, kde_colors), 1):
    plt.subplot(n_rows, n_cols, idx)
    sns.kdeplot(data=train_df, x=feature, fill=True, color=color, linewidth=1.8)
    plt.title(f"Distribution of '{feature}'", fontsize=14, color=color)
    plt.xlabel(feature, fontsize=11)
    plt.ylabel("Density")
    plt.grid(alpha=0.2)

plt.tight_layout()
plt.suptitle("KDE Plots for Numerical Features", fontsize=18, y=1.02)
plt.show()



def feature_engineering(df):
    # Basic features
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['BMR'] = np.where(df['Sex'] == 1,
                         10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] + 5,
                         10 * df['Weight'] + 6.25 * df['Height'] - 5 * df['Age'] - 161)
    df['HRmax'] = 208 - (0.7 * df['Age'])

    df['HeartRate_Duration'] = df['Heart_Rate'] * df['Duration']
    df['BodyTemp_Duration'] = df['Body_Temp'] * df['Duration']
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['Age_HeartRate'] = df['Age'] * df['Heart_Rate']
    df['BMI_Duration'] = df['BMI'] * df['Duration']
    df['Age_BodyTemp'] = df['Age'] * df['Body_Temp']
    df['Weight_to_Age'] = df['Weight'] / df['Age']


    # Ratio features
    df['HeartRate_to_Temp'] = df['Heart_Rate'] / df['Body_Temp']
    df['HeartRate_to_Age'] = df['Heart_Rate'] / df['Age']
    df['HeartRate_to_MaxHR'] = df['Heart_Rate'] / df['HRmax']
    df['Exercise_Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Age_BMI'] = df['Age'] * df['BMI']

    # Log-transformed features to handle skewness
    for col in ['Age', 'Weight', 'Body_Temp', 'Height', 'Duration', 'Heart_Rate']:
        df[f'log_{col}'] = np.log1p(df[col])

    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)


male_train = train_df[train_df['Sex'] == 1]
female_train = train_df[train_df['Sex'] == 0]

x_male = male_train.drop(columns=['Calories'])
x_female = female_train.drop(columns=['Calories'])
y_male = np.log1p(male_train['Calories'])
y_female = np.log1p(female_train['Calories'])



scaler_m = StandardScaler()
scaler_f = StandardScaler()

x_male_scaled = scaler_m.fit_transform(x_male)
x_female_scaled = scaler_f.fit_transform(x_female)

model_male = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.8,
    min_child_weight=5,
    gamma=0.2,
    early_stopping_rounds=100,
    eval_metric="rmse",
    random_state=42
)

model_female = XGBRegressor(**model_male.get_params())


# Split training data (male)
x_train_m, x_valid_m, y_train_m, y_valid_m = train_test_split(
    x_male_scaled, y_male, test_size=0.2, random_state=42
)

model_male.fit(
    x_train_m, y_train_m,
    eval_set=[(x_valid_m, y_valid_m)],
    verbose=100
)

# Split training data (female)
x_train_f, x_valid_f, y_train_f, y_valid_f = train_test_split(
    x_female_scaled, y_female, test_size=0.2, random_state=42
)

model_female.fit(
    x_train_f, y_train_f,
    eval_set=[(x_valid_f, y_valid_f)],
    verbose=100
)



# Train predictions
y_male_pred = np.expm1(model_male.predict(x_male_scaled))
y_female_pred = np.expm1(model_female.predict(x_female_scaled))

# True values (not logged!)
true_male = np.expm1(y_male)
true_female = np.expm1(y_female)

# RMSLE
male_rmsle = np.sqrt(mean_squared_log_error(true_male, y_male_pred))
female_rmsle = np.sqrt(mean_squared_log_error(true_female, y_female_pred))

print(f"Male RMSLE: {male_rmsle:.5f}")
print(f"Female RMSLE: {female_rmsle:.5f}")



#Safe prediction to ensure no negative predictions
def safe_predict(model, X):
    preds = model.predict(X)
    preds = np.expm1(preds)  # Invert log1p
    return np.maximum(preds, 0)


def predict_test(test_df, scaler_m, scaler_f, model_m, model_f):
    df = test_df.copy()
    df.set_index('id', inplace=True)

    male_data = df[df['Sex'] == 1]
    female_data = df[df['Sex'] == 0]

    if not male_data.empty:
        X_m = scaler_m.transform(male_data)
        df.loc[male_data.index, 'Calories'] = safe_predict(model_m, X_m)
    if not female_data.empty:
        X_f = scaler_f.transform(female_data)
        df.loc[female_data.index, 'Calories'] = safe_predict(model_f, X_f)

    return df.reset_index()[['id', 'Calories']]

submission = predict_test(test_df, scaler_m, scaler_f, model_male, model_female)
submission.to_csv('submission_Predict Calorie Expenditure.csv', index=False)
print("Submission file saved as 'submission_Predict Calorie Expenditure.csv'")


sns.histplot(submission['Calories'], kde=True)
plt.title('Calories Prediction Distribution')
plt.xlabel('Predicted Calories')
plt.ylabel('Frequency')
plt.show()

