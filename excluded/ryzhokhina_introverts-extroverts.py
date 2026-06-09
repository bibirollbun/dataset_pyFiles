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


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


personality_datasert = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
personality_dataset = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv')



print(personality_datasert.shape)
personality_datasert.head(10)


personality_datasert.isna().sum()


personality_datasert.Personality.value_counts()


print(personality_dataset.shape)
personality_dataset.head(10)


personality_dataset.Personality.value_counts()


personality_dataset.isna().sum()


print(train.shape)
train.head()


train.Personality.value_counts()


# concatenate datasets

full_train = pd.concat([train, personality_dataset])
print(full_train.shape)


full_train.head()





print("Shape:", full_train.shape)
print("Columns:\n", full_train.columns.tolist())
print("Data types:\n", full_train.dtypes)
print("Missing values:\n", full_train.isnull().sum())
full_train.info()


# Calculate missing values
missing = full_train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

# Plot
plt.figure(figsize=(8, 5))
missing.plot(kind='bar', color='salmon')
plt.title("Missing Values per Column")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


def plot_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percent = df.isnull().mean()*100

    missing_data = pd.DataFrame({
        'Missing Counts': missing_count,
        'Missing Percent':missing_percent
    })
    
    missing_data = missing_data[missing_data['Missing Counts'] >0].sort_values(by='Missing Percent')
    
    if (missing_data.empty) :
        print('Data Farame has no missing data')

    fig, ax = plt.subplots(figsize=(10,8))
    
    bars = ax.barh(missing_data.index, missing_data['Missing Percent'], color = 'skyblue')
    bar_labels = []
    for count, pct in zip(missing_data['Missing Counts'],  missing_data['Missing Percent']):
        bar_labels.append(f"{int(count)} ({pct:.1f}%)")   

    ax.bar_label(bars, labels=bar_labels)
    ax.set_xlabel("Missing Values (%)")
    ax.set_title("Missing Data per Column ")
    ax.grid(axis = 'x', linestyle= '--', alpha=0.5)
    plt.show()




plot_missing_values(full_train)


plot_missing_values(test)


categorical_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
for col in categorical_cols:
    full_train[col] = full_train[col].astype('category')


def plot_category(df, col):
    counts = df[col].value_counts(dropna=False)
    
    # Count values including NaN
    counts = df[col].value_counts(dropna=False)
    counts.index = counts.index.astype(str).str.replace('nan', 'Missing')
    
    # Plot with container
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color='skyblue')
    
    # Add labels on top
    ax.bar_label(bars, labels=counts.values, padding=2)
    
    # Formatting
    ax.set_title(f"Value Counts for '{col}'")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


# Explore categories
for col in categorical_cols:
    plot_category(full_train, col)


numeric_cols = full_train.select_dtypes(include=['float64', 'int64']).columns.drop('id')

# Histograms
full_train[numeric_cols].hist(figsize=(12, 8), bins=15)
plt.suptitle("Histograms of Numeric Features")
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=full_train[col])
    plt.title(col)
plt.tight_layout()
plt.show()



# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(full_train[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.show()



import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*observed=False.*", category=FutureWarning)

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(train.select_dtypes(include='number').columns, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Personality', y=col, data=full_train)
        plt.title(col)
        plt.title(f'{col} by Personality')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



categorical_cols = full_train.select_dtypes(include='category').columns.drop('Personality')

for col in categorical_cols:
    ct = pd.crosstab(full_train[col], full_train['Personality'], normalize='index', dropna=False)
    ct = ct *100
    ax = ct.plot(kind='bar', stacked=True, figsize=(8, 4), colormap='tab20')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='center')
    plt.title(f'Distribution of Personality by {col}')
    plt.ylabel('Proportion')
    plt.legend(title='Personality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import lightgbm as lgb


full_train['Personality'].value_counts()


train['Personality'].value_counts()


personality_dataset['Personality'].value_counts()


features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing',
    'Friends_circle_size', 'Post_frequency'
]
target = 'Personality'

# Define X and Y
X = train[features] #full_train[features]
y = train[target] #full_train[target]
# X = full_train[features]
# y = full_train[target]

# 2. Encode target labels (Introvert = 0, Extrovert = 1)
target_le = LabelEncoder()
y = target_le.fit_transform(y)


X.isna().sum()


def impute_numeric_scaled(df, method="knn"):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    if method == "mean":
        imputer = SimpleImputer(strategy="mean")
    elif method == "iterative":
        imputer = IterativeImputer(random_state=0)
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
    else:
        raise ValueError("Unsupported method for numeric imputation")

    df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)
    df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns)
    return df_imputed_scaled


def impute_categorical(df, method="most_frequent"):
    if method == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    elif method == "constant0":
        imputer = SimpleImputer(strategy="constant", fill_value='No')
    else:
        raise ValueError("Unsupported method for categorical imputation")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)



num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']
target_col = 'Personality'


def encode_binary_cats(df):
    fear_le = LabelEncoder()
    drained_le = LabelEncoder()
    df.loc[:, 'Stage_fear'] = fear_le.fit_transform(df['Stage_fear'])
    df.loc[:, 'Drained_after_socializing']= drained_le.fit_transform(df['Drained_after_socializing'])
    df['Stage_fear'] = df['Stage_fear'].astype('category')
    df['Drained_after_socializing'] = df['Drained_after_socializing'].astype('category')
    return df


def create_rf_model():
    best_params = {
      'n_estimators': 296,
      'max_depth': 17,
      'min_samples_split': 4,
      'min_samples_leaf': 6,
      'max_features': 'log2'
    }
    RF_model = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    return RF_model


# Calculate class imbalance ratio
counts = np.bincount(y)
neg, pos = counts[0], counts[1]
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


def create_xgb_model(scale_pos_weight = 2.44 ):
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    return xgb_model


def create_lgb_model(scale_pos_weight = 2.44 ):
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        random_state=42, 
        verbose=-1
    )
    return lgb_model


def evaluate_imputation_cv(x, y, num_method, cat_method, model, n_splits=5, random_state=42):
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    cat_cols = ['Stage_fear', 'Drained_after_socializing']
    
    # Impute once for full dataset
    num_df = impute_numeric_scaled(x[num_cols], method=num_method)
    cat_df = impute_categorical(x[cat_cols], method=cat_method)
    final_df = pd.concat([num_df, cat_df], axis=1)
    final_df = encode_binary_cats(final_df)
    #print(final_df.shape)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores = []

    for train_idx, test_idx in skf.split(final_df, y):
        X_train, X_test = final_df.iloc[train_idx], final_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))

    return np.mean(acc_scores)



# Try combinations
methods = [
    ("mean", "most_frequent"),
    ("knn", "most_frequent"),
    ("iterative", "most_frequent"),
    ("mean", "constant0"),
    ("knn", "constant0")
]

models = {"xgb":create_xgb_model , "rf": create_rf_model, "lgb": create_lgb_model}

results = {}
for num_method, cat_method in methods:
    for model_name, creator in models.items():
        model = creator()
        acc = evaluate_imputation_cv(X.copy(), y.copy(), num_method, cat_method, model)
        key = f"num: {num_method}, cat: {cat_method}, model: {model_name}"
        results[key] = acc

# Convert results to DataFrame
import pandas as pd
results_df = pd.DataFrame([
    {
        "num_method": num,
        "cat_method": cat,
        "model": model,
        "accuracy": acc
    }
    for (key, acc) in results.items()
    for num, cat, model in [key.replace("num: ", "").replace("cat: ", "").replace("model: ", "").split(", ")]
])

# Display nicely
results_df = results_df.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
results_df


def fill_missing_values(train_df, test_df):
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
    cat_cols = ['Stage_fear', 'Drained_after_socializing']

    # numerical features
    train_nums = train_df[num_cols]
    test_nums = test_df[num_cols]
    
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df[num_cols]), columns=num_cols)
    test_scaled = pd.DataFrame(scaler.transform(test_df[num_cols]), columns=num_cols)

    num_imputer = IterativeImputer(random_state=0)
    
    train_scaled = pd.DataFrame(num_imputer.fit_transform(train_scaled), columns=num_cols)
    test_scaled = pd.DataFrame(num_imputer.transform(test_scaled), columns=num_cols)

    # categorical feature
    
    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_cat = pd.DataFrame(cat_imputer.fit_transform(train_df[cat_cols]), columns=cat_cols)
    test_cat = pd.DataFrame(cat_imputer.transform(test_df[cat_cols]), columns=cat_cols)
    
    final_train_df = pd.concat([train_scaled, train_cat], axis=1)
    final_test_df = pd.concat([test_scaled, test_cat], axis=1)

    fear_le = LabelEncoder()
    drained_le = LabelEncoder()
    
    final_train_df.loc[:, 'Stage_fear'] = fear_le.fit_transform(final_train_df['Stage_fear'])
    final_train_df.loc[:, 'Drained_after_socializing']= drained_le.fit_transform(final_train_df['Drained_after_socializing'])
    final_train_df['Stage_fear'] = final_train_df['Stage_fear'].astype('category')
    final_train_df['Drained_after_socializing'] = final_train_df['Drained_after_socializing'].astype('category')
    
    final_test_df.loc[:, 'Stage_fear'] = fear_le.transform(final_test_df['Stage_fear'])
    final_test_df.loc[:, 'Drained_after_socializing']= drained_le.transform(final_test_df['Drained_after_socializing'])
    final_test_df['Stage_fear'] = final_test_df['Stage_fear'].astype('category')
    final_test_df['Drained_after_socializing'] = final_test_df['Drained_after_socializing'].astype('category')

    return final_train_df, final_test_df


train_df, test_df = fill_missing_values(X, test.copy())


train_df.shape, test_df.shape


y.shape


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
acc_scores = []
fold_preds = []
model = create_rf_model()
for train_idx, test_idx in skf.split(train_df, y):
    X_train, X_test = train_df.iloc[train_idx], train_df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_pred))
    
    probas = model.predict_proba(test_df)
    fold_preds.append(probas)
# Average the probabilities
avg_probas = np.mean(fold_preds, axis=0)

# Convert to predicted classes
final_preds = (avg_probas[:, 1] >= 0.5).astype(int)  # for binary classification


# Output the average accuracy across all folds
print(f"Average Accuracy across {n_splits} folds: {np.mean(acc_scores):.4f}")

# Final Classification Report (using the whole dataset, but typically, you may compute on the final model)
model.fit(train_df, y)  # Fit on the whole data for final model
final_predictions = model.predict(train_df)
print("Final Classification Report on Entire Dataset:\n", classification_report(y, final_predictions))


submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


submission.head()


submission.loc[:, 'Personality'] = target_le.inverse_transform(final_preds)


submission.head()


submission.to_csv('submission.csv', index=False)

