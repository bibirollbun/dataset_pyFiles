import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore, Style
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgboost as xgb



train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


train_df.head()


# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n") 

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")


print ('ğŸ”� Check for missing values\n')
missing_values = train_df.isnull().sum()
missing_percent = (missing_values / len(train_df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
missing_df = missing_df[missing_df['Missing Values'] > 0]
missing_df


# Define numerical & categorical  columns
numerical_features = [feature for feature in train_df.columns if train_df[feature].dtype != "O"]
categorical_features = [feature for feature in train_df.columns if train_df[feature].dtype == "O"]

print(f" We have features: {len(numerical_features)} numerical features {numerical_features}")
print("\n")
print(f" We have features: {len(categorical_features)} categorical features {categorical_features}")


for feature in numerical_features:
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)

    sns.histplot(data = train_df, x = feature , kde = True, bins = 30,hue = "Personality",palette="inferno")
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    sns.boxplot(train_df[feature])
    plt.title(f"Boxplt of {feature}")
    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {train_df[feature].skew():.2f}")
    print(f"Number of Missing Values: {train_df[feature].isnull().sum()}")


# Define numerical & categorical  columns
numerical_features = [feature for feature in train_df.columns if train_df[feature].dtype != "O"]
categorical_features = [feature for feature in train_df.columns if train_df[feature].dtype == "O"]

print(f" We have features: {len(numerical_features)} numerical features {numerical_features}")
print("\n")
print(f" We have features: {len(categorical_features)} categorical features {categorical_features}")


for categorical in categorical_features:
    Count = train_df[categorical].value_counts()

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data = train_df, x = categorical,palette="Set2")
    plt.title(f"Count of {categorical}")
    plt.xticks(rotation=90)
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.pie(Count, labels=Count.index, autopct="%1.1f%%", startangle=90)
    plt.title(f"Percentage of {categorical}")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()


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


for feature in numerical_features[:-1]:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=train_df[feature], y=train_df["Personality"], alpha=0.5,color = "red"
    )
    plt.title(f"{feature} vs. Personality")
    plt.xlabel(feature)
    plt.ylabel("Personality")
    plt.show()

correlation_matrix = train_df[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data = train_df, x = "Stage_fear", hue = "Personality", palette="inferno")
plt.title("Stage_fear VS Personality")
plt.xlabel("Stage_fear")
plt.ylabel("Counts")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data = train_df, x = "Drained_after_socializing", hue =  "Personality",palette = "inferno")
plt.title("Drained_after_socializing VS Personality")
plt.xlabel("Drained_after_socializing")
plt.ylabel("Counts")
plt.tight_layout()
plt.show()


# Select numerical features (columns that are not of object type)
numerical_features = [col for col in train_df.columns if train_df[col].dtype != "O"]

# Select categorical features (columns that are of object type)
categorical_features = [col for col in train_df.columns if train_df[col].dtype == "O"]

# Impute missing values in numerical columns with the mean of the training set
for col in numerical_features:
    train_df.loc[:, col] = train_df[col].fillna(train_df[col].mean())
    test_df.loc[:, col] = test_df[col].fillna(train_df[col].mean())

# Impute missing values in categorical columns with the string 'Missing'
for col in categorical_features:
    if col in train_df.columns:
        train_df.loc[:, col] = train_df[col].fillna('Missing')
    if col in test_df.columns:
        test_df.loc[:, col] = test_df[col].fillna('Missing')


# List of categorical columns to encode
cat_value = ["Stage_fear", "Drained_after_socializing"]

# Encode specified categorical columns
for col in cat_value:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.fit_transform(test_df[col])



# Encode Target
label = LabelEncoder()
train_df['Personality_encoded'] = label.fit_transform(train_df["Personality"])  # 0=Extrovert, 1=Introvert

# Drop the original target column to get features
drop_train = train_df.drop(columns=["Personality", "Personality_encoded"])

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(drop_train), columns=drop_train.columns)
X_test = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

# Target variable
y = train_df["Personality_encoded"]


#  Define the DecisionTreeClassifier
d_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth =10,
    min_samples_leaf= 4,
    min_samples_split = 2
)

d_tree.fit(X,y)

# Predict on the training set
y_pred = d_tree.predict(X)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state = 42)

# Evaluate using cross_val_score
scores = cross_val_score(d_tree,X,y, cv = skf,scoring = "accuracy")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=d_tree.classes_)
disp.plot(cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"\nâœ… Decision Tree CV Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")


# Define the  RandomForestClassifier 
random = RandomForestClassifier(
    max_depth=10,
    min_samples_leaf=4,
    min_samples_split = 2,
    random_state=42
)

# Fit the model
random.fit(X,y)

# Predict on the training set
y_pred = random.predict(X)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits = 5,shuffle = True, random_state = 42)

# Evaluate using cross_val_score
scores = cross_val_score(random, X,y, cv = skf,scoring = "accuracy")

# 7) Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=random.classes_)
disp.plot(cmap='Blues')
plt.title("RandomForest Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"\nâœ… RandomForest CV Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")


# # Define parameter grid for GridSearchCV
# param_grid = {
#     'max_depth': [5, 7, 9],
#     'learning_rate': [0.01, 0.0055],
#     'n_estimators': [500, 1000],
#     'subsample': [0.8, 0.9],
#     'colsample_bytree': [0.6, 0.65],
#     'reg_lambda': [1.0, 2.0],
#     'reg_alpha': [2.0, 4.5],
#     'gamma': [0.0, 0.02],
#     'min_child_weight': [1, 1.5]
# }

# # Set up Stratified K-Fold cross-validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Define the XGBoost model (CPU version)
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='logloss',
#     enable_categorical=False,
#     use_label_encoder=False,
#     random_state=42,
#     verbosity=0
# )

# # Set up GridSearchCV
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=skf,
#     n_jobs=-1,
#     verbose=1
# )

# # Fit GridSearchCV
# grid_search.fit(X, y)

# print(f"\nâœ… Best XGBoost CV Accuracy: {grid_search.best_score_:.5f}")
# print("Best Parameters:", grid_search.best_params_)


import xgboost as xgb
# Define the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    enable_categorical=False,
    use_label_encoder=False,
    random_state=42,
    n_estimators=2000,             
    learning_rate=0.0055,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.65,
    reg_lambda=1.0,
    reg_alpha=4.5,
    gamma=0.02,
    min_child_weight=1.5,
    verbosity=0
)

# Fit the model
xgb_model.fit(X, y)

# Predict on the training set
y_pred = xgb_model.predict(X)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using cross_val_score
scores = cross_val_score(xgb_model, X, y, cv=skf, scoring="accuracy")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = xgb_model.classes_)
disp.plot(cmap='Blues')
plt.title(" xgboost Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"\nâœ… XGBoost CV Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")


# from catboost import CatBoostClassifier
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

# # Define parameter grid for GridSearchCV
# param_grid = {
#     'iterations': [200, 300,100],
#     'learning_rate': [0.05, 0.1],
#     'depth': [3, 4, 5]
# }

# # Set up Stratified K-Fold cross-validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Define the CatBoostClassifier (no need to set params here, GridSearchCV will do it)
# cat_model = CatBoostClassifier(
#     random_seed=42,
#     verbose=False
# )

# # Set up GridSearchCV
# grid_search = GridSearchCV(
#     estimator=cat_model,
#     param_grid=param_grid,
#     scoring='accuracy',
#     cv=skf,
#     n_jobs=-1,
#     verbose=1
# )

# # Fit GridSearchCV
# grid_search.fit(X, y)

# print(f"\nâœ… Best CatBoost CV Accuracy: {grid_search.best_score_:.5f}")
# print("Best Parameters:", grid_search.best_params_)


# Define theCatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=5,
    random_seed=42,
    verbose=False
)

# Fit the model
cat_model.fit(X, y)

# Predict on the training set
y_pred = cat_model.predict(X)

# Set up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using cross_val_score
scores = cross_val_score(cat_model, X, y, cv=skf, scoring="accuracy")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = cat_model.classes_)
disp.plot(cmap='Blues')
plt.title(" CatBoostClassifier Confusion Matrix")
plt.tight_layout()
plt.show()

print(f"\nâœ… CatBoost CV Accuracy: {scores.mean():.5f} (+/- {scores.std():.5f})")


#  Predict on test data
preds = cat_model.predict(X_test)

#  Decode back to original labels
pred_labels = label.inverse_transform(preds)  # Converts 0 â†’ 'Extrovert', 1 â†’ 'Introvert'

#   submission file

submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': pred_labels
})
submission.to_csv("submission.csv", index=False)


 #  Check
print(submission.head())


