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


import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


print("Shape of Train Data", train_df.shape)


display(train_df.head(10))


train_df.info()


train_df.isnull().sum()


train_df.nunique()


train_df.describe().transpose()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Age', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Height', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Weight', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Duration', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Heart_Rate', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Body_Temp', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


# Set a larger figure size
plt.figure(figsize=(15, 8))  

# Create the countplot
sns.countplot(x='Calories', data=train_df, hue='Sex')

plt.xticks(rotation=45)  # Rotate x-axis labels if they overlap
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()


numerical_df = train_df.select_dtypes(include=['int64', 'float64'])


from scipy import stats
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# Get all pairs of numerical columns
column_pairs = combinations(['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories'], 2)

# Set style
sns.set(style="whitegrid")

# Loop through each pair and plot
for col1, col2 in column_pairs:
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with regression line
    sns.regplot(x=col1, y=col2, data=numerical_df, scatter_kws={'alpha':0.6})
    
    # Calculate statistics
    corr_coef, p_value = stats.pearsonr(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    slope, intercept, _, _, _ = stats.linregress(numerical_df[col1].dropna(), numerical_df[col2].dropna())
    
    # Add statistics to plot
    stats_text = (f"Pearson r = {corr_coef:.2f}\n"
                  f"p-value = {p_value:.4f}\n"
                  f"Regression: y = {slope:.2f}x + {intercept:.2f}")
    
    plt.gcf().text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Titles and labels
    plt.title(f'{col1} vs {col2}', fontsize=14)
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Automated interpretation
    abs_r = abs(corr_coef)
    
    # Interpret Pearson r
    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak or no"
    
    direction = "positive" if corr_coef > 0 else "negative" if corr_coef < 0 else "no"
    
    # Interpret p-value
    if p_value < 0.001:
        sig_text = "highly statistically significant (p < 0.001)"
    elif p_value < 0.05:
        sig_text = "statistically significant (p < 0.05)"
    else:
        sig_text = "not statistically significant (p ≥ 0.05)"
    
    # Print interpretation
    print(f"\nInterpretation for {col1} vs {col2}:")
    print(f"- {strength} {direction} linear relationship")
    print(f"- The correlation is {sig_text}\n")
    print("-" * 60)  # Separator line


corr = abs(numerical_df.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,8))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
plt.xticks(rotation = 50)   # Aesthetic purposes
plt.yticks(rotation = 20)   # Aesthetic purposes
plt.show()


from scipy.stats import skew  # For skewness calculation

# Set up subplots
n_cols = 3  # Number of columns in the grid
n_rows = (len(numerical_df.columns) // n_cols) + 1

# Create a figure with subplots
plt.figure(figsize=(15, 5 * n_rows))  # Adjust size as needed

# Loop through numerical columns and plot KDE + skewness
for i, column in enumerate(numerical_df.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.kdeplot(data=numerical_df, x=column, fill=True)
    
    # Calculate skewness
    skewness = skew(numerical_df[column].dropna())  # Handle NaN if needed
    skew_text = f'Skewness: {skewness:.2f}'
    
    # Add skewness as text in the plot
    plt.text(0.05, 0.9, skew_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f'KDE of {column}')
    plt.xlabel(column)

plt.tight_layout()
plt.show()


def create_safe_features(df):
    """Calculate features without using Calories"""
    df = df.copy()

    # Body Mass Index (BMI)
    df['BMI'] = df['Weight'] / (df['Height']/100)**2

    # Body Surface Area (BSA)
    df['BSA'] = 0.007184 * (df['Height']**0.725) * (df['Weight']**0.425)

    # Weight-to-Height Ratio
    df['Weight_Height_Ratio'] = df['Weight'] / df['Height']
    
    # Basic transformations
    df['Max_HR_Estimate'] = 220 - df['Age']
    df['HR_Reserve'] = df['Heart_Rate'] / (220 - df['Age'])
    df['Temp_HR_Index'] = df['Body_Temp'] * df['Heart_Rate'] / 1000
    df['Age_Duration_Ratio'] = df['Age'] / df['Duration']
    df['Weighted_Duration'] = df['Duration'] * df['Weight'] / 100
    df['HR_squared'] = df['Heart_Rate'] ** 2
    df['Age_squared'] = df['Age']**2
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']

    # Age groups
    df['Age_Group'] = pd.cut(df['Age'], 
                                  bins=[0,20,30,40,50,100], 
                                  labels=['Teen','20s','30s','40s','50+'])

    # Heart Rate Intensity
    df['HR_Zone'] = pd.cut(df['Heart_Rate'],
                                bins=[0,120,150,180,300],
                                labels=['Light','Moderate','Vigorous','Max'])
    
    # MET calculation without Calories
    df['MET_safe'] = (df['Heart_Rate'] * 0.0175) / df['Weight']
    
    # Estimated calories/min (hypothetical scaling)
    df['Calories_per_min_est'] = df['Heart_Rate'] * df['Weight'] * 0.0005
    
    # CV_Load 
    df['CV_Load'] = (df['Heart_Rate'] * df['Duration']) / (df['Age'] * df['Body_Temp'])
    
    return df


# Apply to your data
train_df_safe = create_safe_features(train_df)


train_df_safe.columns


categorical_cols = train_df_safe.select_dtypes(include=['object', 'category']).columns.tolist()
print("Categorical columns to encode:", categorical_cols)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def encode_cat(df):
    # Copy the original DataFrame to preserve it
    df_encoded = df.copy()

    # A. Label Encoding for Ordinal Categories
    ordinal_cols = ['Age_Group', 'HR_Zone']  # If these exist
    le = LabelEncoder()
    for col in ordinal_cols:
        if col in df_encoded.columns:
            df_encoded[col+'_encoded'] = le.fit_transform(df_encoded[col])
            # Optional: Save the mapping for reference
            print(f"{col} mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # B. One-Hot Encoding for Nominal Categories
    nominal_cols = ['Sex']  # Add others if needed
    df_encoded = pd.get_dummies(
        df_encoded, 
        columns=nominal_cols, 
        prefix=nominal_cols,
        drop_first=True  # Avoid dummy variable trap
    )

    # C. Drop original categorical columns (optional)
    cols_to_drop = categorical_cols
    df_encoded = df_encoded.drop(columns=cols_to_drop, errors='ignore')
    return df_encoded


train_df_encoded = encode_cat(train_df_safe)


cols = [col for col in train_df_encoded.columns if col != 'Calories'] + ['Calories']
train_df_encoded = train_df_encoded[cols]
train_df_encoded.head()


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlations with Calories
corr_matrix = train_df_encoded.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix[['Calories']].sort_values('Calories', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation with Calories")
plt.show()

# Select features with |correlation| > threshold (e.g., 0.3)
strong_corr = corr_matrix['Calories'][abs(corr_matrix['Calories']) > 0.3].index.tolist()
print("Strongly correlated features:", strong_corr)


from sklearn.feature_selection import mutual_info_regression

X = train_df_encoded.drop(columns=['Calories', 'id'])
y = train_df_encoded['Calories']

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores}) \
       .sort_values('MI_Score', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='MI_Score', y='Feature', data=mi_df)
plt.title("Mutual Information Scores")
plt.show()

# Select top N features
top_mi_features = mi_df.head(8)['Feature'].tolist()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Selected features from correlation analysis
selected_features = [
    'Age', 'Height', 'Weight',
    'Duration', 'Heart_Rate', 'Body_Temp', 
    'HR_Reserve', 'Temp_HR_Index', 'Age_Duration_Ratio',
    'Weighted_Duration', 'HR_squared', 'HR_Duration_Interaction',
    'CV_Load' ,'MET_safe', 'Calories_per_min_est',
    'BSA', 'BMI', 'Weight_Height_Ratio', 'Max_HR_Estimate', 'Age_Group_encoded' ,'Sex_male'
]
target = 'Calories'

# 1. Data Preparation (SAFE VERSION)
X = train_df_encoded[selected_features]
y = train_df_encoded[target]

# Create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create train/validation split (no subsampling yet)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42
)

# 2. Use KFold 
from sklearn.model_selection import KFold
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# 3. SAFE Subsample (100K or less if data is smaller)
sample_size = min(100000, len(X_train_sub))
tune_idx = np.random.choice(X_train_sub.index, size=sample_size, replace=False)
X_tune = X_train_sub.loc[tune_idx]
y_tune = y_train_sub.loc[tune_idx]

# 4. RMSLE Scorer
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# 5. LightGBM Tuning (SAFE VERSION)
lgbm_params = {
    'n_estimators': [150, 200, 250],
    'num_leaves': [30, 40, 50],
    'learning_rate': [0.08, 0.1, 0.12],
    'subsample': uniform(0.7, 0.2),
    'colsample_bytree': uniform(0.7, 0.2),
}

search = RandomizedSearchCV(
    LGBMRegressor(random_state=42, verbose=-1),
    param_distributions=lgbm_params,
    n_iter=10,
    cv=cv,  # Using KFold instead
    scoring=rmsle_scorer,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 6. Run the search
search.fit(X_tune, y_tune)

# 7. Retrain best model on full data
best_lgbm = LGBMRegressor(**search.best_params_, random_state=42)
best_lgbm.fit(X_train, y_train)  # Using ALL training data

# 8. Evaluate
test_pred = best_lgbm.predict(X_test)
print(f"Final Test RMSLE: {rmsle(y_test, test_pred):.4f}")


# Compare train vs test performance
train_pred = best_lgbm.predict(X_train)
test_pred = best_lgbm.predict(X_test)

print(f"Train RMSLE: {rmsle(y_train, train_pred):.4f}") 
print(f"Test RMSLE: {rmsle(y_test, test_pred):.4f}")


importance = pd.Series(best_lgbm.feature_importances_, index=selected_features)
keep_features = importance[importance > 10].index  # Adjust threshold
X_train_reduced = X_train[keep_features]


keep_features


test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


print("Shape of Test Data", test_df.shape)


display(test_df.head(10))


test_df.isnull().sum()


# Apply to your data
test_df_safe = create_safe_features(test_df)


test_df_encoded = encode_cat(test_df_safe)


test_df_encoded.columns


test_pred = best_lgbm.predict(test_df_encoded[selected_features])

# Clip negative predictions (if any)
test_pred = np.clip(test_pred, 0, None)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'], 
    'Calories': test_pred
})

# Save predictions
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")


submission.head()

