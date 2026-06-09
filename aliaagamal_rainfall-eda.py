import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc , accuracy_score , precision_score ,recall_score ,f1_score,roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")


data.sample(7)


data.info()


data.isnull().sum()


data.duplicated().sum()


numerical_cols = data.drop(['id', 'day' , 'rainfall'], axis=1).columns


data[numerical_cols].describe()


# Set dark mode
plt.style.use("dark_background")

# Create subplots for each numerical feature
for feature in numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]})

    # Histogram with KDE
    sns.histplot(data[feature], bins="auto", kde=True, color="#1f77b4", ax=axes[0], edgecolor="black", alpha=0.8)

    # Mean & Median lines
    mean_val = data[feature].mean()
    median_val = data[feature].median()
    axes[0].axvline(mean_val, color="yellow", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    axes[0].axvline(median_val, color="red", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}")

    # Titles & Labels
    axes[0].set_title(f"Histogram & KDE: {feature}", fontsize=12)
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Box Plot
    sns.boxplot(x=data[feature], color="#1f77b4", ax=axes[1], width=0.4)
    axes[1].set_title(f"Box Plot: {feature}", fontsize=12)

    # Layout adjustment
    plt.tight_layout()
    plt.show()



# Count of each category (0 = No Rain, 1 = Rain)
rainfall_counts = data['rainfall'].value_counts()
rainfall_counts


rainfall_counts = {'Rain': 1650, 'No Rain': 540}
labels = list(rainfall_counts.keys())
sizes = list(rainfall_counts.values())
colors = sns.color_palette("pastel", len(labels))

plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p * sum(sizes) / 100)})',
                                   colors=colors, startangle=140, wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

for autotext in autotexts:
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
    autotext.set_color('black')

plt.title("Rainfall Distribution", fontsize=14, fontweight='bold')
plt.show()



def box_plot(data , col1 , col2):
  plt.figure(figsize=(7, 5))
  sns.boxplot(data=data, x=col1, y=col2,hue=col1, palette="pastel")
  plt.title(f"{col2} Distribution by {col1}", fontsize=14, fontweight='bold')
  plt.xlabel(f"{col1}")
  plt.ylabel(f"{col2}")
  plt.show()


box_plot(data , 'rainfall' , 'pressure')


def select_and_perform_test(df, col1, col2):
    """
    Selects and performs the appropriate statistical test based on normality and variance checks.

    Parameters:
    df (pd.DataFrame): The dataset.
    col1 (str): The first column (independent variable, usually categorical/binary).
    col2 (str): The second column (dependent variable, numerical).

    Returns:
    str: Recommended statistical test and its result.
    """

    # Split the data into two groups based on the independent variable
    unique_values = df[col1].unique()
    if len(unique_values) != 2:
        return "Error: The independent variable must have exactly two unique values."

    group1 = df[df[col1] == unique_values[0]][col2].dropna()
    group2 = df[df[col1] == unique_values[1]][col2].dropna()

    # Check normality using Shapiro-Wilk test (or KS test for large samples)
    norm_p1 = stats.shapiro(group1).pvalue if len(group1) < 5000 else stats.kstest(group1, 'norm').pvalue
    norm_p2 = stats.shapiro(group2).pvalue if len(group2) < 5000 else stats.kstest(group2, 'norm').pvalue

    normality1 = norm_p1 > 0.05  # True if normally distributed
    normality2 = norm_p2 > 0.05  # True if normally distributed

    print(f"Normality Test Results:")
    print(f"  Group {unique_values[0]}: p-value = {norm_p1:.4f} ({'Normal' if normality1 else 'Not Normal'})")
    print(f"  Group {unique_values[1]}: p-value = {norm_p2:.4f} ({'Normal' if normality2 else 'Not Normal'})")

    # Check variance equality using Leveneâ€™s test
    var_p = stats.levene(group1, group2).pvalue
    equal_variance = var_p > 0.05

    print(f"Variance Test (Levene's test) p-value = {var_p:.4f} ({'Equal Variance' if equal_variance else 'Unequal Variance'})")

    # Select and perform the appropriate test
    if normality1 and normality2:
        if equal_variance:
            test_name = "t-test (parametric)"
            stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
        else:
            test_name = "Welchâ€™s t-test (parametric, unequal variance)"
            stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    else:
        test_name = "Mann-Whitney U Test (non-parametric)"
        stat, p_value = stats.mannwhitneyu(group1, group2)

    print(f"Selected Test: {test_name}")
    print(f"Test Statistic = {stat:.4f}, p-value = {p_value:.4f}")

    return test_name, stat, p_value


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'pressure')


box_plot(data , 'rainfall' , 'maxtemp')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'maxtemp')


data.columns


box_plot(data , 'rainfall' , 'temparature')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'temparature')


box_plot(data , 'rainfall' , 'mintemp')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'mintemp')


box_plot(data , 'rainfall' , 'dewpoint')


Test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'dewpoint')


box_plot(data , 'rainfall' , 'humidity')


select_and_perform_test(data, 'rainfall', 'humidity')


box_plot(data , 'rainfall' , 'cloud')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'cloud')


box_plot(data, 'rainfall', 'sunshine')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'sunshine')


box_plot(data, 'rainfall', 'winddirection')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'winddirection')


box_plot(data, 'rainfall', 'windspeed')


test_name, stat, p_value = select_and_perform_test(data, 'rainfall', 'windspeed')


plt.figure(figsize=(8, 6))
sns.regplot(x='cloud', y='sunshine', data=data, scatter_kws={'alpha':0.5})
plt.title('Regression Plot: Cloud Cover vs. Sunshine Hours', fontsize=14)
plt.xlabel('Cloud Cover (%)')
plt.ylabel('Sunshine Hours')
plt.show()


# Spearman's Rank Correlation Test
correlation, p_value = stats.spearmanr(data['cloud'], data['sunshine'])

print(f"Spearman's Rank Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("There is a statistically significant correlation between cloud cover and sunshine hours.")
    if correlation < 0:
        print("The correlation is negative, indicating an inverse relationship.")
    else:
        print("The correlation is positive, indicating a direct relationship.")
else:
    print("There is no statistically significant correlation between cloud cover and sunshine hours.")


data.columns


plt.figure(figsize=(8, 6))
sns.regplot(x='temparature', y='humidity', data=data, scatter_kws={'alpha':0.5})
plt.title('Regression Plot: Temperature vs. Humidity', fontsize=14)
plt.xlabel('Temperature (Â°C)')
plt.ylabel('Humidity (%)')
plt.show()


# Spearman's Rank Correlation Test
correlation, p_value = stats.spearmanr(data['temparature'], data['humidity'])

print(f"Spearman's Rank Correlation: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("There is a statistically significant correlation between temperature and humidity.")
    if correlation < 0:
        print("The correlation is negative, indicating an inverse relationship (humidity increases as temperature decreases).")
    else:
        print("The correlation is positive, indicating a direct relationship (humidity increases as temperature increases).")
else:
    print("There is no statistically significant correlation between temperature and humidity.")


def get_season(day):
    if day in range(80, 172):  # Spring (March 20 - June 20)
        return "Spring"
    elif day in range(172, 264):  # Summer (June 21 - Sept 22)
        return "Summer"
    elif day in range(264, 355):  # Fall (Sept 23 - Dec 20)
        return "Fall"
    else:  # Winter (Dec 21 - March 19)
        return "Winter"

data["season"] = data["day"].apply(get_season)


plt.figure(figsize=(8, 6))
sns.countplot(x='season', hue='rainfall', data=data, palette="pastel")
plt.title('Rainfall Distribution by Season')
plt.xlabel('Season')
plt.ylabel('Count')
plt.legend(title='Rainfall', labels=['No Rain', 'Rain'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Create a contingency table
contingency_table = pd.crosstab(data['season'], data['rainfall'])

# Perform Chi-squared test with Yates' correction
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table, correction=True)

print("Chi-squared Statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)

# Interpretation
alpha = 0.05  # Significance level
if p_value < alpha:
    print("There is a statistically significant relationship between season and rainfall.")
else:
    print("There is no statistically significant relationship between season and rainfall.")


def preprocess_data(df):
    df = df.copy()  # Avoid modifying original data

    # Apply cyclic encoding for 'day'
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 365)
    df.drop(columns=["day"], inplace=True)  # Drop original 'day' column

    # Encode only the 'season' column
    encoder = OneHotEncoder(drop='first', sparse_output=False)  # Drop first category
    season_encoded = encoder.fit_transform(df[['season']])  # Encode 'season'

    # Get new column names for encoded features
    encoded_feature_names = encoder.get_feature_names_out(['season'])

    # Convert to DataFrame and merge with original data (excluding the original 'season' column)
    season_encoded_df = pd.DataFrame(season_encoded, columns=encoded_feature_names, index=df.index)
    df = df.drop(columns=['season']).reset_index(drop=True)  # Remove original column
    df = pd.concat([df, season_encoded_df], axis=1)  # Merge encoded features

    return df, encoder  # Return transformed data and encoder



data, encoder = preprocess_data(data)


# Split Data
X = data.drop(columns=['rainfall'])
y = data['rainfall']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)


def balance_data(X_train, y_train):
    """
    Balances the dataset using SMOTE + Random Undersampling.
    """
    smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversampling
    undersample = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Undersampling

    balancer = ImbPipeline([
        ('smote', smote),
        ('undersample', undersample)
    ])

    return balancer.fit_resample(X_train, y_train)


def train_models(X_train, y_train):
    """
    Trains multiple models using GridSearchCV.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    dict: Dictionary containing trained models and best parameters.
    """
    models = {
        "SVM": (Pipeline([('scaler', StandardScaler()), ('classifier', SVC())]),
                {"classifier__C": [0.1, 1, 10], "classifier__kernel": ["linear", "rbf"]}),

        "KNN": (Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),
                {"classifier__n_neighbors": [3, 5, 7]}),

        "LogisticRegression": (Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())]),
                               {"classifier__C": [0.1, 1, 10]}),

        "PCA + LogisticRegression": (Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=5)), ('classifier', LogisticRegression())]),
                                     {"classifier__C": [0.1, 1, 10]}),

        "RandomForest": (RandomForestClassifier(),
                         {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}),

        "GradientBoosting": (GradientBoostingClassifier(),
                             {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]})
    }

    best_models = {}

    for model_name, (model, params) in models.items():
        print(f"Training {model_name}...")
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[model_name] = {"model": grid.best_estimator_, "best_params": grid.best_params_}

    return best_models


best_models = train_models(X_train, y_train)


def evaluate_models(best_models, X_val, y_val):
    results = {}

    for model_name, info in best_models.items():
        model = info["model"]
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
       # roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        results[model_name] = {
            "accuracy": acc, "precision": precision, "recall": recall,
            "f1_score": f1,
        }

        print(f"\nğŸ”¹ **{model_name}**")
        print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

    return results


results = evaluate_models(best_models, X_val, y_val)


def compare_models(results):
 
    model_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in model_names]

    plt.figure(figsize=(10,5))
    sns.barplot(x=model_names, y=accuracies)
    plt.xticks(rotation=45)
    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.show()


compare_models(results)


# Find Best Model
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = best_models[best_model_name]["model"]
print(f"\nğŸ�† Best Model: {best_model_name}")


def plot_confusion_matrix(best_model, X_val, y_val):
 
    y_pred = best_model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


plot_confusion_matrix(best_model, X_val, y_val)


test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test.head()


test.isnull().sum()


# rØ«place nan value in winddirection with mdien
test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())


test["season"] = test["day"].apply(get_season)


test, encoder = preprocess_data(test)


test.head()


# make submission
predictions = best_model.predict(test)
submission = pd.DataFrame({'id': test.id, 'rainfall': predictions})
submission.to_csv('submission.csv', index=False)




