import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import pandas as pd

sns.set_palette("husl")
pd.option_context('mode.use_inf_as_na', True)


def numerical_distrib_analysis(data, numerical_features):
    """
    Analyze the distribution of numerical variables using histograms and boxplots.
    
    :param data: Pandas DataFrame containing the data
    :param numerical_features: List of names of numerical column features
    """
    
    for feature in numerical_features:
        plt.figure(figsize=(12, 5))

        # Histogramme avec KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        # Box plot pour dÃ©tecter les outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[feature])
        plt.title(f"Box Plot of {feature}")

        plt.tight_layout()
        plt.show()

        # Statistiques supplÃ©mentaires
        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {data[feature].skew():.2f}")
        print(f"Number of Missing Values: {data[feature].isnull().sum()}")

def categorical_distrib_analysis(data, categorical_features, top_n=10):
    """
    Analysis and visualization of categorical variables.

    :param data: Pandas DataFrame containing the data.
    :param categorical_features: List of categorical columns to analyze.
    :param top_n: Number of most frequent categories to display for variables with many unique values.
    """
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))

        # Check the number of unique categories
        unique_count = data[feature].nunique()

        if unique_count > top_n:  
            # If many unique values, show only top_n most common categories
            top_categories = data[feature].value_counts().nlargest(top_n)
            sns.barplot(x=top_categories.index, y=top_categories.values, palette="pastel")
            plt.title(f"Top {top_n} {feature} Categories")
        else:
            # If not, show all categories
            sns.countplot(x=data[feature], order=data[feature].value_counts().index, palette="pastel")
            plt.title(f"Distribution of {feature}")

        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

        # Show statistics
        print(f"Feature: {feature}")
        print(f"Number of Unique Values: {unique_count}")
        print(f"Missing Values: {data[feature].isnull().sum()}\n")

def numerical_correlation_analysis(data, numerical_features, target):
    """
    Analysis and visualization of the relationships between numerical variables and the target variable.

    :param data: Pandas DataFrame containing the data.
    :param numerical_features: List of numerical columns to analyze.
    :param target: Name of the target variable.
    """
    
    # Scatter plots for each numerical variable (without target)
    for feature in numerical_features:
        if feature != target:  # Exclude target
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[feature], y=data[target], alpha=0.5)
            plt.title(f"{feature} vs. {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.show()

    # Correlation matrice
    correlation_matrix = data[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

def categorical_correlation_analysis(data, categorical_features, target, high_cardinality_threshold=10):
    """
    Visualization of categorical variables with respect to the target variable using box plots.

    :param data: Pandas DataFrame containing the data.
    :param categorical_features: List of categorical columns to analyze.
    :param target: Name of the target variable.
    :param high_cardinality_threshold: Cardinality threshold to ignore variables with too many categories.
    """

    for feature in categorical_features:
        if data[feature].nunique() <= high_cardinality_threshold:  # Ignorer les variables Ã  haute cardinalitÃ©
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[feature], y=data[target], palette='husl')
            plt.title(f"{feature} vs. {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Skipping {feature}: too many unique values ({data[feature].nunique()})\n")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


original_df = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')

# Concatenate original data with synthetics ones
train_df = pd.concat([train_df, original_df], axis=0, ignore_index=True)
train_df.drop_duplicates()

# path = '/kaggle/input/playground-series-s5e4/poadcast_dataset.csv'.replace('\u00A0', ' ')
sample_submission = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")

print("\nData Info:")
train_df.info()

print("\nNumerical Features Summary:")
display(train_df.describe())

print("\nFirst 10 rows of Dataset:")
train_df.head(10)



# Analysing distributions of numerical features
numerical_features = [
    'Episode_Length_minutes', 
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 
    'Number_of_Ads',
    'Listening_Time_minutes',
]

numerical_distrib_analysis(train_df, numerical_features)



categorical_features = [
    'Podcast_Name', 
    'Episode_Title', 
    'Genre', 
    'Publication_Day',
    'Publication_Time', 
    'Episode_Sentiment'
]

categorical_distrib_analysis(train_df, categorical_features)


numerical_correlation_analysis(train_df, numerical_features, "Listening_Time_minutes")


categorical_correlation_analysis(train_df, categorical_features, 'Listening_Time_minutes')


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Encoder for categorical data
label_encoders = {col: LabelEncoder() for col in categorical_features}

# Apply LabelEncoder to each categorical column
for col in categorical_features:
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])


print("Missing Values per Column:")
print(train_df.isnull().sum())

print("Missing Values per Column:")
print(test_df.isnull().sum())


# Replacing null values by median
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)


# Null values could mean no guest 
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
train_df.dropna(inplace=True)

test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)

# Deleting outliers 
train_df = train_df[train_df['Number_of_Ads']<10]



print("Missing Values per Column:")
print(train_df.isnull().sum())

print("Missing Values per Column:")
print(test_df.isnull().sum())


from sklearn.model_selection import train_test_split

X = train_df.drop(['Listening_Time_minutes'], axis=1)
y = train_df['Listening_Time_minutes']
# test_df = test_df.drop('id', axis=1)

# Split Training data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from lightgbm import LGBMRegressor
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Configuration du modÃ¨le avec les meilleurs paramÃ¨tres
best_params_lgbm = {
    'num_leaves': 589,
    'max_depth': 12,
    'learning_rate': 0.023111458265010466,
    'min_child_samples': 10,
    'subsample': 0.8725177917814763,
    'colsample_bytree': 0.8584398483559579,
    'reg_alpha': 0.2571489639295177,
    'reg_lambda': 6.2143374851920505,
    'max_bin': 183,
}

model_lgbm = LGBMRegressor(
    **best_params_lgbm,
    objective='regression',
    metric='rmse',
    n_estimators=5000,
    early_stopping_rounds=250,
    random_state=42,
    device='cpu',
    n_jobs=-1,
    verbose=-1
)

# EntraÃ®nement avec early stopping
print("DÃ©but de l'entraÃ®nement...")
model_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)
print("EntraÃ®nement terminÃ©!")

# Ã‰valuation sur le test set
y_pred = model_lgbm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRMSE on test set: {rmse:.4f}")

# Importance des features
lgb.plot_importance(model_lgbm)
plt.show()


pred_lgbm = model_lgbm.predict(test_df)
pred_lgbm

submission_lgbm = pd.DataFrame({'id': sample_submission.id, 'Listening_Time_minutes' : pred_lgbm})
submission_lgbm.to_csv('submission.csv', index=False)


