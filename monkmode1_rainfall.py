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


rainfall=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


rain=rainfall.copy()


rain.shape


rain.info()


rain.describe()


rain.isnull().sum()


rain.duplicated().sum()


import pandas as pd

# Assuming you already have the 'rain' DataFrame loaded
# Define a function to classify weather conditions
def classify_weather(row):
    if row['rainfall'] > 5:  # Adjust threshold as needed
        return "Rainy"
    elif row['cloud'] > 6:  # High cloud cover (assuming scale 0-8)
        return "Cloudy"
    elif row['humidity'] > 80 and row['dewpoint'] > 20:
        return "Humid"
    elif row['sunshine'] > 7:  # More sunshine means clear weather
        return "Sunny"
    elif row['maxtemp'] > 30:  # Adjust based on region
        return "Hot"
    elif row['mintemp'] < 10:
        return "Cold"
    else:
        return "Mild"

# Apply function to create a new column in the 'rain' dataset
rain['weather'] = rain.apply(classify_weather, axis=1)

# Display the first few rows to check the results
print(rain[['day', 'weather']])



import pandas as pd

# Assuming you already have the 'rain' DataFrame loaded
# Define a function to classify weather conditions
def classify_weather(row):
    if row['cloud'] > 6:  # High cloud cover (assuming scale 0-8)
        return "Cloudy"
    elif row['humidity'] > 80 and row['dewpoint'] > 20:
        return "Humid"
    elif row['sunshine'] > 7:  # More sunshine means clear weather
        return "Sunny"
    elif row['maxtemp'] > 30:  # Adjust based on region
        return "Hot"
    elif row['mintemp'] < 10:
        return "Cold"
    else:
        return "Mild"

# Apply function to create a new column in the 'rain' dataset
test['weather'] = test.apply(classify_weather, axis=1)

# Display the first few rows to check the results
print(test[['day', 'weather']])



def classify_season(day):
    if 1 <= day <= 59 or 335 <= day <= 365:
        return "Winter"
    elif 60 <= day <= 151:
        return "Spring"
    elif 152 <= day <= 243:
        return "Summer"
    elif 244 <= day <= 334:
        return "Autumn"
    else:
        return "Unknown"  # Just in case

# Apply function to create the 'season' column
rain['season'] = rain['day'].apply(classify_season)

# Display first few rows
print(rain[['day', 'season']])



test['season'] = test['day'].apply(classify_season)

# Display first few rows
print(test[['day', 'season']])


rain


import seaborn as sns
import matplotlib.pyplot as plt

# Automatically select numerical columns
num_cols = rain.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude 'host_id' or other ID-like columns if needed
num_cols = [col for col in num_cols if col not in ['id']]

# Descriptive statistics
print(rain[num_cols].describe())

# Loop through numerical columns and generate histograms, box plots, and density plots
for col in num_cols:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    sns.histplot(rain[col], bins=30, kde=True, ax=axes[0])
    axes[0].set_title(f'{col} - Histogram')

    # Box Plot (for outliers)
    sns.boxplot(x=rain[col], ax=axes[1])
    axes[1].set_title(f'{col} - Box Plot')

    # KDE Plot (density)
    sns.kdeplot(rain[col], ax=axes[2])
    axes[2].set_title(f'{col} - Density Plot')

    plt.show()

# Check skewness of numerical columns
print("\nSkewness of Numerical Features:")
print(rain[num_cols].skew())


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(rain[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Get numerical columns (excluding 'rainfall' itself)
numerical_features = rain.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'rainfall' in numerical_features:
    numerical_features.remove('rainfall')

# Plot settings
plt.figure(figsize=(15, 20))
num_cols = 3  # Number of columns in the plot grid
num_rows = (len(numerical_features) + num_cols - 1) // num_cols

# Generate scatter plots for all numerical features vs rainfall
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.scatterplot(x=rain[feature], y=rain['rainfall'], alpha=0.5, color='blue')
    
    # Calculate Pearson and Spearman correlations
    pearson_corr, pearson_p = pearsonr(rain[feature], rain['rainfall'])
    spearman_corr, spearman_p = spearmanr(rain[feature], rain['rainfall'])
    
    # Add titles and annotations
    plt.title(f"{feature} vs Rainfall", fontsize=10)
    plt.xlabel(feature, fontsize=8)
    plt.ylabel("Rainfall", fontsize=8)
    plt.grid(alpha=0.3)
    
    # Highlight significant correlations (p < 0.05)
    annotation = (
        f"Pearson: {pearson_corr:.2f} (p={pearson_p:.3f})\n"
        f"Spearman: {spearman_corr:.2f} (p={spearman_p:.3f})"
    )
    plt.annotate(annotation, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8)

plt.tight_layout()
plt.show()

# Summary table of correlations
correlation_summary = []
for feature in numerical_features:
    pearson_corr, _ = pearsonr(rain[feature], rain['rainfall'])
    spearman_corr, _ = spearmanr(rain[feature], rain['rainfall'])
    correlation_summary.append({
        "Feature": feature,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr
    })

correlation_df = pd.DataFrame(correlation_summary)
print("\nCorrelation Summary:")
print(correlation_df.sort_values(by="Spearman", ascending=False))



import seaborn as sns
import matplotlib.pyplot as plt

categorical_features = rain.select_dtypes(include='object')  # Replace with actual categorical features

for cat in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=rain[cat], y=rain["rainfall"])
    plt.title(f"Boxplot of Rainfall vs {cat}")
    plt.xticks(rotation=45)
    plt.show()



for cat in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=rain[cat], y=rain["rainfall"])
    plt.title(f"Violin Plot of Rainfall vs {cat}")
    plt.xticks(rotation=45)
    plt.show()



from scipy.stats import f_oneway

for cat in categorical_features:
    groups = [rain[rain[cat] == value]["rainfall"] for value in rain[cat].unique()]
    f_stat, p_value = f_oneway(*groups)
    print(f"ANOVA Test for {cat}: p-value = {p_value:.5f}")



import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your dataset (replace df with your actual DataFrame)
df = rain.copy()  # Make sure to replace this with your dataset

# Drop categorical variables (VIF is only for numerical features)
df_numeric = df.select_dtypes(include=[np.number]).dropna()

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = df_numeric.columns
vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(len(df_numeric.columns))]

# Display results
print(vif_data)



import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your dataset (replace 'df' with your actual DataFrame)
# df = pd.read_csv('your_data.csv')  # Uncomment if loading from CSV

# Define a function to calculate VIF
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# Set a threshold for VIF
VIF_THRESHOLD = 10

# Drop non-numeric columns (if any)
df_numeric = df.select_dtypes(include=[np.number])

# Iteratively remove features with high VIF
while True:
    vif_df = calculate_vif(df_numeric)
    max_vif = vif_df["VIF"].max()

    if max_vif < VIF_THRESHOLD:
        break  # Stop if all VIFs are below the threshold

    # Drop the feature with the highest VIF
    feature_to_drop = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
    print(f"Dropping '{feature_to_drop}' with VIF={max_vif:.2f}")
    
    df_numeric = df_numeric.drop(columns=[feature_to_drop])

# Final VIF report
print("\nFinal Features and their VIFs:")
print(calculate_vif(df_numeric))



rain.rainfall


X_full = rain.drop(columns=["id","rainfall"])  # Features
y = rain["rainfall"]  # Target variable (0 or 1)


from sklearn.preprocessing import StandardScaler, LabelEncoder

# Identify categorical columns
categorical_cols = X_full.select_dtypes(include=['object']).columns

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_full[col] = le.fit_transform(X_full[col])  # Convert to numeric
    label_encoders[col] = le  # Save encoder for later use

# Now apply StandardScaler
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_full, y, test_size=0.2, random_state=42, stratify=y)



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# Define classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(),
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# Perform cross-validation again with all features
results_full = {}
cv = 5  # Define cross-validation folds
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    results_full[name] = (cv_scores.mean(), cv_scores.std())

# Print updated results
print("Classifier Performance with All Features (Mean Accuracy ± Std Dev):")
for model, (mean_acc, std_acc) in results_full.items():
    print(f"{model}: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")



from sklearn.model_selection import GridSearchCV

# Define hyperparameter grids
param_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "criterion": ["gini", "entropy"]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "Support Vector Machine": {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    },
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
}

best_models = {}

for name, params in param_grids.items():
    model = models[name]
    grid_search = GridSearchCV(model, params, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_:.4f}\n")



from sklearn.ensemble import RandomForestClassifier

# Best hyperparameters found earlier
best_rf = RandomForestClassifier(criterion="gini", max_depth=10, n_estimators=50, random_state=42)

# Train on the full training dataset
best_rf.fit(X_train, y_train)




test


test=test.drop(columns='id')


import pandas as pd

# Find missing values
missing_values = pd.DataFrame(test.isnull().sum(), columns=["Missing Values"])
print(missing_values[missing_values["Missing Values"] > 0])



test.winddirection.fillna(test.winddirection.mean(), inplace=True)



from sklearn.preprocessing import StandardScaler, LabelEncoder

# Identify categorical columns
categorical_cols = test.select_dtypes(include=['object']).columns

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])  # Convert to numeric
    label_encoders[col] = le  # Save encoder for later use

# Now apply StandardScaler
scaler = StandardScaler()
test_scaled_full = scaler.fit_transform(test)


test_predictions = best_rf.predict(test_scaled_full)
print("Predictions:", test_predictions[:10])



submission=pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


submission['rainfall'] = test_predictions
submission.to_csv("submission.csv", index=False)
print("\nSubmission file saved successfully!")







