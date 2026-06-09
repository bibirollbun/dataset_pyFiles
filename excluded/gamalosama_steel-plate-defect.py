import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import os
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier



from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score,classification_report,f1_score,mean_squared_error,roc_auc_score,precision_score,recall_score,roc_curve,ConfusionMatrixDisplay,confusion_matrix,auc
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder,RobustScaler,MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression,SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator,TransformerMixin
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier


# Define a premium dark elegant color palette
dark_elegant_palette = [
    "#8ecae6",  # Light Blue Accent
    "#219ebc",  # Medium Blue
    "#023047",  # Deep Navy
    "#ffb703",  # Warm Yellow
    "#fb8500",  # Warm Orange
    "#e63946",  # Muted Red
    "#a8dadc",  # Soft Mint
    "#457b9d",  # Slate Blue
]

# Set a dark theme for Seaborn
sns.set_theme(
    style="darkgrid",              # Dark grid background
    palette=dark_elegant_palette,  # Custom color palette
    font="DejaVu Sans",
    rc={
        "axes.facecolor": "#212529",     # Dark background for axes
        "figure.facecolor": "#212529",   # Dark background for figure
        "axes.edgecolor": "#f8f9fa",     # Light axis lines
        "axes.labelcolor": "#f8f9fa",    # Light axis labels
        "text.color": "#f8f9fa",         # Light text
        "xtick.color": "#f8f9fa",        # Light tick labels
        "ytick.color": "#f8f9fa",
        "grid.color": "#495057",         # Muted gridlines
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (8, 5),
        "axes.linewidth": 1.2,
    }
)

# Apply the palette to Matplotlib
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=dark_elegant_palette)

# Preview the palette
sns.palplot(dark_elegant_palette)
plt.title("Premium Dark Elegant Palette", fontsize=14, color="#f8f9fa", backgroundcolor="#212529")
plt.show()


df_sample_sub = pd.read_csv(r'/kaggle/input/playground-series-s4e3/sample_submission.csv')
df_train = pd.read_csv(r'/kaggle/input/playground-series-s4e3/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s4e3/test.csv')


df_sample_sub.sample(7)


df_train.sample(7)


df_test.sample(7)


print(f"the shape of train = {df_train.shape}")
print(f"the shape of test = {df_test.shape}")


df_train.info()


df_test.info()


print(f"number of duplicated in train: {df_train.duplicated().sum()}")
print(f"number of duplicated in test: {df_test.duplicated().sum()}")


list(df_train.columns)


train_id = df_train['id']
test_id = df_test['id']

df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


X = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
       'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
       'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
       'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
       'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
       'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
       'SigmoidOfAreas']

Y = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']




X_train, Y_train, X_test = df_train[X], df_train[Y], df_test[X]


X_train.hist(figsize=(20, 20), bins=30,)

plt.tight_layout()
plt.show()


cat_columns = ['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Outside_Global_Index']
num_columns = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']


# Calculate grid size
num_plots = num_cols = 3
num_rows = (num_plots + num_cols - 1) // num_cols  


# Create subplot grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()  # Flatten for easy iteration

# Plot each categorical column
for idx, col in enumerate(cat_columns):
    ax = axes[idx]
    
    df_train[cat_columns].groupby(col).sum().plot(kind='bar', stacked=True, ax=ax)
    
    ax.set_title(f"Stacked Bar Plot of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Sum")
    ax.tick_params(axis='x', rotation=0)


plt.tight_layout()
plt.show()


target_melted = Y_train.melt(var_name='Class', value_name='Label')

# Create a count plot (barplot of 0s and 1s per class)
plt.figure(figsize=(10, 6))
sns.countplot(data=target_melted, x='Class', hue='Label')

plt.title("Distribution of 0s and 1s in Each Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.legend(title="Label", labels=["0", "1"])
plt.tight_layout()
plt.show()


X_train[['Pixels_Areas', 'X_Perimeter', 'Y_Perimeter']].boxplot(figsize=(10, 5))
plt.show()


# def remove_outliers(df, column):
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# train = remove_outliers(train, 'Pixels_Areas')


corr = X_train.corr()
corr


cols = corr.index
cm = np.corrcoef(X_train.values.T)

plt.figure(figsize=(15, 15))

hm = sns.heatmap(cm, cbar=False, annot=True, fmt='.2f', annot_kws={'size': 9}, 
                 yticklabels=cols.values, xticklabels=cols.values)

hm.xaxis.tick_top()
plt.xticks(rotation=45, ha='left')
plt.show()


def feature_engineering(df):
    df['X_range'] = abs(df['X_Maximum'] - df['X_Minimum'])
    df['Y_range'] = abs(df['Y_Maximum'] - df['Y_Minimum'])

    df['Luminosity_range'] = abs(df['Maximum_of_Luminosity'] - df['Minimum_of_Luminosity'])

    # df['Areas'] = np.exp(df['LogOfAreas'])
    # df['X_Index'] = np.exp(df['Log_X_Index'])
    # df['Y_Index'] = np.exp(df['Log_Y_Index'])

    # df.drop(columns=['LogOfAreas', 'Log_X_Index', 'Log_Y_Index'], inplace=True)
    
    return df


# X_train = feature_engineering(X_train)
# X_test = feature_engineering(X_test)


# for col in ['LogOfAreas', 'Log_X_Index', 'Log_Y_Index']:
#         num_columns.remove(col)
# num_columns.extend(['X_range', 'Y_range', 'Luminosity_range', 'Areas', 'X_Index', 'Y_Index'])



X_train.sample(5)


class OutlierThresholdTransformer(BaseEstimator, TransformerMixin): 
    def __init__(self, columns, q1=0.25, q3=0.75): 
        self.columns = columns 
        self.q1 = q1 
        self.q3 = q3
        self.thresholds_ = {}

    def fit(self, X, y=None): 
        for col in self.columns: 
            Q1 = X[col].quantile(self.q1)
            Q3 = X[col].quantile(self.q3)
            iqr = Q3 - Q1
            up_limit = Q3 + 1.5 * iqr
            low_limit = Q1 - 1.5 * iqr
            self.thresholds_[col] = (low_limit, up_limit)
        return self

    def transform(self, X): 
        X_copy = X.copy() 
        for col in self.columns:
            low_limit, up_limit = self.thresholds_[col]
            X_copy.loc[X_copy[col] < low_limit, col] = low_limit
            X_copy.loc[X_copy[col] > up_limit, col] = up_limit
        return X_copy



outlier_clipper = OutlierThresholdTransformer(columns=num_columns)
X_train = outlier_clipper.fit_transform(X_train)
X_test = outlier_clipper.transform(X_test)


print("The Shape of X_train is :",X_train.shape)

print("The Shape of y_train is :",Y_train.shape)

print("The Shape of X_test is :",X_test.shape)


def detect_skewed_columns(dataframe, threshold=0.5):
    """
    Identifies skewed numerical columns in a pandas DataFrame based on a given skewness threshold.

    Parameters:
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing numerical features to be evaluated.
        
    threshold : float, optional (default=0.5)
        The absolute skewness value above which a column is considered skewed.

    Returns:
    -------
    List[str]
        A list of column names where the absolute skewness exceeds the given threshold.
    """
    return [col for col in dataframe.columns if abs(dataframe[col].skew()) > threshold]



skewed_columns = detect_skewed_columns(X_train[num_columns])
print("Skewed columns:", skewed_columns)



# Define columns
log_cols = skewed_columns


class OrdinalEncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder_ = None

    def fit(self, X, y=None):
        ordinal_data = X[self.columns].values
        self.encoder_ = OrdinalEncoder()
        self.encoder_.fit(ordinal_data)
        return self

    def transform(self, X):
        X_copy = X.copy()
        ordinal_data = X_copy[self.columns].values
        encoded_data = self.encoder_.transform(ordinal_data)
        X_copy[self.columns] = encoded_data
        return X_copy
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, shift=1):
        self.columns = columns
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = X_copy[self.columns].apply(lambda x: np.log(x + self.shift))
        return X_copy
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.columns])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = self.scaler_.transform(X_copy[self.columns])
        return X_copy
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class FullPipeline:
    def __init__(self, log_cols, num_columns, cat_columns):
        self.log_cols = log_cols
        self.num_columns = num_columns
        self.cat_columns = cat_columns

        self.full_pipeline = Pipeline(steps=[
            ('log_transformer', LogTransformer(columns=self.log_cols)),
            ('scaler', StandardScalerTransformer(columns=self.num_columns)),
            ('ordinal_encoder', OrdinalEncodeColumns(columns=self.cat_columns)),
        ])

    def fit(self, X, y=None):
        X_copy = X.copy()
        self.full_pipeline.fit(X_copy, y)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = self.full_pipeline.transform(X_copy)
        return X_copy
    
    def fit_transform(self, X, y=None):
        X_copy = X.copy()
        self.fit(X_copy, y)
        X_copy = self.full_pipeline.transform(X_copy)
        return X_copy



fl_pipe=FullPipeline(log_cols=log_cols, num_columns=num_columns, cat_columns=cat_columns)

X_train=fl_pipe.fit_transform(X_train)

X_test=fl_pipe.transform(X_test)


# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, 
    Y_train, 
    test_size=0.3, 
    random_state=42
)

# Initialize and train the Random Forest model
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_split, y_train_split)

# Extract feature importances
importances = rf_clf.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

importance_df



# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

# Target categories for evaluation
target_labels = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

# List of classifiers to test
candidate_models = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    LogisticRegression(max_iter=200, random_state=42),
    SVC(probability=True, random_state=42),
    KNeighborsClassifier(),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    CatBoostClassifier(verbose=0, random_state=42),
    LGBMClassifier(random_state=42)
]

# Store model names and average ROC-AUC values
model_labels = []
mean_auc_scores = []

# Loop through each classifier
for estimator in candidate_models:
    auc_per_label = []
    
    for defect in target_labels:
        estimator.fit(X_train_split, y_train_split[defect])
        proba_predictions = estimator.predict_proba(X_val_split)[:, 1]
        auc_value = roc_auc_score(y_val_split[defect], proba_predictions) * 100
        auc_per_label.append(auc_value)
    
    model_labels.append(estimator.__class__.__name__)
    mean_auc_scores.append(np.mean(auc_per_label))

# Create DataFrame of results
evaluation_results = pd.DataFrame({
    'Classifier': model_labels,
    'Avg ROC-AUC (%)': mean_auc_scores
}).sort_values(by='Avg ROC-AUC (%)', ascending=False)

print(evaluation_results)



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.pointplot(
    data=evaluation_results,
    x='Classifier',
    y='Avg ROC-AUC (%)',
    color='tab:blue',
    markers='o'
)
plt.xticks(rotation=75)
plt.title('Classifier Performance: Mean ROC-AUC', fontsize=14, fontweight='bold')
plt.xlabel('Model')
plt.ylabel('ROC-AUC (%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import numpy as np

# Define the parameter grid
param_dist = {
    'num_leaves': [20, 31, 40, 50, 60],
    'max_depth': [-1, 5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [0, 0.01, 0.1, 0.5]
}

best_params_per_label = {}

# Loop over each target label
for defect in target_labels:
    lgbm = LGBMClassifier(random_state=42)
    search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=30,  # Number of random combos to try
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, Y_train[defect])
    best_params_per_label[defect] = search.best_params_
    print(f"{defect} - Best Params: {search.best_params_}")

# Store the results
best_params_per_label



from lightgbm import LGBMClassifier

# Dictionary to store tuned models for each defect category
best_models = {}

for defect in target_labels:
    # Use the parameters found during tuning
    tuned_params = best_params_per_label[defect]
    
    lgbm = LGBMClassifier(random_state=42, **tuned_params)
    lgbm.fit(X_train, Y_train[defect])
    
    best_models[defect] = lgbm



submission_df = pd.DataFrame({'id': test_id})

for defect in target_labels:
    # Predict probabilities (for ROC-AUC evaluation)
    submission_df[defect] = best_models[defect].predict_proba(X_test)[:, 1]


submission_df.head()


submission_df.to_csv("submission.csv", index=False)




