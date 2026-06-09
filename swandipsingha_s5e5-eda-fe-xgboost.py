import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


print(f"Dataset Shape: {train_df.shape}")

print("\nData Info:")
train_df.info()

print("\nNumerical Features Summary:")
display(train_df.describe())

print("\nFirst 10 Rows of the Dataset:")
display(train_df.head(10))


numerical_features = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Calories"
    
]

for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_df[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train_df[feature].skew():.2f}")
    print(f"Number of Missing Values: {train_df[feature].isnull().sum()}")



sex_counts = train_df["Sex"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Sex")
plt.axis("equal")
plt.show()

print(f"Number of Unique {feature}: {train_df[feature].nunique()}")
print(f"Missing Values in {feature}: {train_df[feature].isnull().sum()}")


import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = train_df.select_dtypes(include='number')

sns.pairplot(numeric_df, corner=True, plot_kws={'alpha': 0.5})
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()



for feature in numerical_features[:-1]:  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=train_df[feature], y=train_df["Calories"], alpha=0.5
    )
    plt.title(f"{feature} vs. Calories")
    plt.xlabel(feature)
    plt.ylabel("Calories")
    plt.show()

correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x=train_df["Sex"], y=train_df["Calories"])
plt.title("Sex vs. Calories")
plt.xlabel("Sex")
plt.ylabel("Calories")
plt.xticks(rotation=45)
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.lineplot(data=train_df[col], color=color)
    plt.title(f'Trend Plot of {col}', fontsize=14, color=color)
    plt.xlabel('Index')
    plt.ylabel(col)

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)
    sns.lineplot(data=train_df[col].sort_values().reset_index(drop=True), color='black', linewidth=1)
    plt.title(f'KDE + Trend of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.violinplot(data=train_df, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()



test_df=pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)



import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer

# Step 1: Select numerical columns
numeric_cols = [col for col in numerical_features if col != "Calories"]



# Step 2: Calculate original skewness on train
original_skewness = train_df[numeric_cols].skew().sort_values(ascending=False)

# Step 3: Initialize transformed DataFrames
train_df_transformed = train_df.copy()
test_df_transformed = test_df.copy()

# Store transformers for each column
transformers = {}

# Step 4: Apply skewness correction based on train_df
for col in numeric_cols:
    if train_df[col].nunique() <= 1:
        continue
    
    if original_skewness[col] > 0.5:  # Right skew
        if (train_df[col] > 0).all():
            # Log transform
            train_df_transformed[col] = np.log1p(train_df[col])
            test_df_transformed[col] = np.log1p(test_df[col])
        else:
            # Yeo-Johnson (handles zero/neg)
            pt = PowerTransformer(method='yeo-johnson')
            train_df_transformed[col] = pt.fit_transform(train_df[[col]])
            test_df_transformed[col] = pt.transform(test_df[[col]])
            transformers[col] = pt
    elif original_skewness[col] < -0.5:  # Left skew
        pt = PowerTransformer(method='yeo-johnson')
        train_df_transformed[col] = pt.fit_transform(train_df[[col]])
        test_df_transformed[col] = pt.transform(test_df[[col]])
        transformers[col] = pt

# Step 5: Calculate skewness after transformation
transformed_skewness = train_df_transformed[numeric_cols].skew().sort_values(ascending=False)

# Step 6: Print comparison
skew_df = pd.DataFrame({
    'Original Skew': original_skewness,
    'Transformed Skew': transformed_skewness
}).sort_values(by='Original Skew', ascending=False)

print(skew_df)



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Make copies to avoid changing the original
cleaned_train_df =train_df_transformed.copy()
cleaned_test_df = test_df_transformed.copy()

# Select numerical columns only (excluding any non-numeric or non-relevant columns)
numeric_cols = [col for col in numerical_features if col != "Calories"]

# Remove outliers using IQR for both train_df and test_df
for col in numeric_cols:
    # Train set
    Q1_train = cleaned_train_df[col].quantile(0.25)
    Q3_train = cleaned_train_df[col].quantile(0.75)
    IQR_train = Q3_train - Q1_train
    lower_bound_train = Q1_train - 1.5 * IQR_train
    upper_bound_train = Q3_train + 1.5 * IQR_train
    cleaned_train_df = cleaned_train_df[(cleaned_train_df[col] >= lower_bound_train) & (cleaned_train_df[col] <= upper_bound_train)]
    
    
# Plot boxplots showing up to 75% of the data for both train_df 
plt.figure(figsize=(16, 10))

# Plot for cleaned train_df
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=cleaned_train_df[col], fliersize=3)
    plt.title(f'{col} Boxplot (Train)')
    plt.ylim(cleaned_train_df[col].quantile(0), cleaned_train_df[col].quantile(0.75))  # Show up to 75%
    plt.tight_layout()

plt.show()




# Binary classification based on temperature for train_df and test_df
cleaned_test_df=test_df
cleaned_train_df['Temp_Binary'] = np.where(cleaned_train_df['Body_Temp'] <= 39.5, 0, 1)
cleaned_test_df['Temp_Binary'] = np.where(cleaned_test_df['Body_Temp'] <= 39.5, 0, 1)

# Binary classification based on heart rate for train_df and test_df
cleaned_train_df['HeartRate_Binary'] = np.where(cleaned_train_df['Heart_Rate'] <= 99.5, 0, 1)
cleaned_test_df['HeartRate_Binary'] = np.where(cleaned_test_df['Heart_Rate'] <= 99.5, 0, 1)

# Encode sex column for train_df and test_df
cleaned_train_df['Sex'] = cleaned_train_df['Sex'].map({'male': 1, 'female': 0})
cleaned_test_df['Sex'] = cleaned_test_df['Sex'].map({'male': 1, 'female': 0})



import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # Import tqdm for progress tracking

X = cleaned_train_df.drop(columns=['Calories', 'id'])
y = cleaned_train_df['Calories']

# Step 2: Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the XGBoost model
model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Regression task
    colsample_bytree=0.3,          # Subsample ratio of columns
    learning_rate=0.1,             # Step size at each iteration
    max_depth=5,                   # Maximum depth of a tree
    alpha=10,                      # L2 regularization term
    n_estimators=1000,             # Number of trees
    random_state=42,
    verbose=200  # Set verbose to get more detailed output
)

# Step 4: Train the model with tqdm for progress tracking
for _ in tqdm(range(1), desc="Training Model", ncols=100):
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
              eval_metric="mae", early_stopping_rounds=50, verbose=200)

# Step 5: Prepare test data (remove 'id' column from test_df)
X_test_df = cleaned_test_df.drop(columns=['id'])  # Exclude id from features
y_test_pred = model.predict(X_test_df)  # Predictions for test_df

# Step 6: Prepare the submission file
submission = pd.DataFrame({
    'id': test_df['id'],  # 'id' column from test_df
    'Calories': y_test_pred.clip(0)   # Predictions for 'Calories'
})

# Step 7: Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' has been created.")





