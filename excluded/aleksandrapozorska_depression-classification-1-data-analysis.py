import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn import metrics

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.simplefilter("ignore", category=FutureWarning)



!pip install -q category_encoders

import category_encoders as ce


# Load the data
train_path = '/kaggle/input/playground-series-s4e11/train.csv' # Kaggle
test_path = '/kaggle/input/playground-series-s4e11/test.csv' # Kaggle 
# train_path = 'data/raw/full_data.csv' # code
# test_path = 'data/external/external_data.csv' # code
columns_names =['id', 'name', 'gender', 'age', 'city', 'occupation_status', 'profession', 
                'academic_pressure', 'work_pressure', 'cgpa', 'study_satisfaction', 'job_satisfaction', 
                'sleep_duration', 'dietary_habits', 'degree', 'suicidal_thoughts', 'work_study_hours', 
                'financial_stress', 'family_history_mental_illness', 'depression']



df = pd.read_csv(train_path, names=columns_names, header=0)
df.head()


external_df = pd.read_csv(test_path, names=columns_names[:-1], header=0) 
external_df.head()


# summary of the data
df.info(), external_df.info()


df.shape, external_df.shape


# Summary statistics for numerical columns
df.describe()


missing_count= df.isnull().sum().sort_values(ascending=False)
missing_count= missing_count[missing_count > 0]
missing_percent = (missing_count/len(df)) * 100

df_missing = pd.DataFrame({'Missing train values: counts': missing_count, "Missing train values: percentage": missing_percent.round(3)})
df_missing


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="Oranges")
plt.title("Missing Values Heatmap")
plt.show()


np.mean(df.depression)
#imbalance of class , starting point 82% 


counts = df.depression.value_counts()
percentages = df.depression.value_counts(normalize=True) * 100

custom_palette = {1: "#fdbb84", 0: "#fee8c8"}

plt.figure(figsize=(8, 6))
sns.countplot(x=df.depression, data=df, hue='depression', palette=custom_palette)
plt.title('Distribution of the target (depression) with percentages', fontsize=14)
plt.xlabel(' ')
plt.ylabel('Count', fontsize=12)

for i, (count, pct) in enumerate(zip(counts, percentages)):
    plt.text(i, count/2, f'Count: {count} \n Percentage: {pct:.2f}%', ha='center', va='bottom', fontsize=10)
plt.xticks(ticks=range(len(counts)), labels=counts.index, fontsize=12)
plt.show()


num_features = df.select_dtypes(exclude=['object']).columns.tolist()[1:-1] # without id and depression columns
print(num_features)


continuous = ['age', 'cgpa']
discrete = [
    'academic_pressure', 'work_pressure', 'study_satisfaction',
    'job_satisfaction', 'work_study_hours', 'financial_stress'
]

fig = plt.figure(figsize=(20, 9))

for idx, feature in enumerate(num_features, 1):
    plt.subplot(2, 4, idx)
    if feature in continuous:
        plt.hist(df[feature], color='#fdbb84', bins=20, rwidth=0.75)
    else:
        vals = df[feature].dropna().unique()
        bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1)
        plt.hist(df[feature], bins=bins, color='#fdbb84', rwidth=0.5)
        plt.xticks(np.arange(vals.min(), vals.max() + 1))
    plt.title(f'Distribution of {feature}', pad=10, fontsize=12)

plt.tight_layout(pad=2.0)
plt.show()


df.academic_pressure.value_counts(), df.study_satisfaction.value_counts()


df.work_pressure.value_counts(), df.job_satisfaction.value_counts()


df.work_study_hours.value_counts()


df.financial_stress.value_counts()


df.cgpa.value_counts()


fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

for idx, feature in enumerate(num_features):
    means = df.groupby(feature)['depression'].mean().reset_index()
    x = means[feature]
    y = means['depression']
    axes[idx].scatter(x, y, color='#fdbb84')
    
    if len(x) > 1:
        z1 = np.polyfit(x, y, 1)
        p1 = np.poly1d(z1)
        x_seq = np.linspace(x.min(), x.max(), 100)
        axes[idx].plot(x_seq, p1(x_seq), "r--", linewidth=1, label='Linear trend (deg 1)')
    
    if len(x) > 2:
        z2 = np.polyfit(x, y, 2)
        p2 = np.poly1d(z2)
        x_seq = np.linspace(x.min(), x.max(), 100)
        axes[idx].plot(x_seq, p2(x_seq), "b--", linewidth=1, label='Quadratic trend (deg 2)')
    
    axes[idx].set_title(f'Depression Rate by {feature}', fontsize=14, pad=12)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Mean Depression Rate', fontsize=10)
    axes[idx].grid(alpha=0.2)
    axes[idx].legend()

plt.tight_layout(pad=2.0)
plt.show()


features = num_features
fig = plt.figure(figsize=(20, 9))
for idx, feature in enumerate(features, 1):
    plt.subplot(2,4, idx)
    ax = sns.boxplot(x='depression', y=feature, data=df, hue='depression', palette=custom_palette)
    plt.title(f'Depression Distribution by {feature}', pad=10, fontsize=12)
    plt.xlabel('Depression (0: No, 1: Yes)', fontsize=10)
    plt.ylabel(feature)
    medians = df.groupby('depression')[feature].median()
    for i, median in enumerate(medians):
        ax.text(i, median, f'{median:.2f}', horizontalalignment='center', size='medium')
plt.tight_layout(pad=2.0)
plt.show()


fig = plt.figure(figsize=(20, 9))
features = num_features

for idx, feature in enumerate(features, 1):
    plt.subplot(2, 4, idx)
    sns.histplot(data=df, x=feature, hue='depression', element='step', stat='density', common_norm=False)
    plt.title(f'Depression Distribution by {feature}', pad=10, fontsize=12)
    plt.xlabel(feature, fontsize=10)
    plt.ylabel('Density')
    plt.legend(labels=['Depressed','Not Depressed'])

plt.tight_layout(pad=2.0)
plt.show()


pd.crosstab(df.academic_pressure, df.occupation_status)


pd.crosstab(df.work_pressure, df.occupation_status)


pd.crosstab(df.study_satisfaction, df.occupation_status)


pd.crosstab(df.job_satisfaction, df.occupation_status)


plt.figure(figsize=(12, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()


cat_features = df.select_dtypes(include=['object']).columns.tolist()
print(cat_features)


unique_counts = df[cat_features].nunique().reset_index()
unique_counts.columns = ['Feature', 'Unique Values Count']
unique_counts = unique_counts.sort_values(by='Unique Values Count', ascending=False)
unique_counts


fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.barplot(
    x=df[cat_features].nunique().sort_values(ascending=False).values,
    y=df[cat_features].nunique().sort_values(ascending=False).index,
    ax=ax,
    color='#fdbb84'
)
ax.set_title('Unique values in categorical columns')

for i, value in enumerate(df[cat_features].nunique().sort_values(ascending=False).values):
    ax.text(value, i, f'{value}', va='center')

plt.show()


df.gender.value_counts(), df.occupation_status.value_counts(), df.suicidal_thoughts.value_counts(), df.family_history_mental_illness.value_counts()


df.degree.value_counts().head(30), df.city.value_counts().head(30), df.profession.value_counts().head(30), df.sleep_duration.value_counts().head(5), df.dietary_habits.value_counts().head(15)


import matplotlib.patches as mpatches

palette = {1: "#fdbb84", 0: "#fee8c8"}


def percent_within_x(df, feature, target='depression'):
    g = df.groupby([feature, target]).size().rename('n').reset_index()
    g['total'] = g.groupby(feature)['n'].transform('sum')
    g['percent'] = 100 * g['n'] / g['total']
    return g

features = ['gender', 'occupation_status', 'suicidal_thoughts', 'family_history_mental_illness']

plt.figure(figsize=(24,6))
for i, feat in enumerate(features, 1):
    ax = plt.subplot(1,4,i)
    g = percent_within_x(df, feat, 'depression')
    sns.barplot(data=g, x=feat, y='percent', hue='depression',
                palette=palette, dodge=True, ax=ax)
    ax.set_title(f'Within-group share: {feat}')
    ax.set_ylabel('Percent within category')
    ax.set_ylim(0, 100)
    handles = [
        mpatches.Patch(color=palette[0], label='Not Depressed'),
        mpatches.Patch(color=palette[1], label='Depressed'),
    ]
    ax.legend(handles=handles, title=None, loc='upper right')     
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', padding=2, fontsize=9)
plt.tight_layout(pad=3.0)
plt.show()


def countplot_percent_topn(ax, df, feature, top_n, target='depression', other='Other'):
    top = df[feature].value_counts().nlargest(top_n).index
    gname = f'{feature} grouped'
    dfx = df.copy()
    dfx[gname] = dfx[feature].where(dfx[feature].isin(top), other=other)
    g = percent_within_x(dfx, gname, target)
    sns.barplot(data=g, x=gname, y='percent', hue=target,
                palette=palette, dodge=True, ax=ax)
    ax.set_title(f'Within-group share: {feature}')
    ax.set_ylabel('Percent within category')
    ax.set_ylim(0, 100)
    handles = [
        mpatches.Patch(color=palette[0], label='Not Depressed'),
        mpatches.Patch(color=palette[1], label='Depressed'),
    ]
    ax.legend(handles=handles, title=None, loc='upper right')       
    ax.tick_params(axis='x', rotation=90)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', padding=2, fontsize=9)

plt.figure(figsize=(24,15))
axes = [plt.subplot(2,3,i) for i in range(1,6)]
countplot_percent_topn(axes[0], df, 'city', 10)
countplot_percent_topn(axes[1], df, 'profession', 10)
countplot_percent_topn(axes[2], df, 'degree', 10)
countplot_percent_topn(axes[3], df, 'sleep_duration', 4, other='7-8 hours')
countplot_percent_topn(axes[4], df, 'dietary_habits', 3, other='Moderate')
plt.tight_layout(pad=3.0)
plt.show()



duplicates = df[df.duplicated(keep=False)]
duplicates


# Removing unnecessary columns: Name, id
df = df.drop(["name", "id"], axis=1)
df.head()


X = df.drop(["depression"], axis=1)
y = df.depression


y.value_counts()
# Reminder from analysis above - data is imbalanced - stratify=y while splitting the data to preserve class balance 


X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)


X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.1, random_state=42, stratify=y_full_train)


print(f"Class distribution in training set: {y_train.value_counts()} \nClass distribution in validation set: {y_val.value_counts()}\nClass distribution in test set: {y_test.value_counts()}")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape} \nX_val shape: {X_val.shape}, y_val shape: {y_val.shape} \nX_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


class InconsistencyFlagger(BaseEstimator, TransformerMixin):
    """
    Adds an 'inconsistency_flag' column to the DataFrame, marking rows with logical inconsistencies
    based on pressure, satisfaction, and occupation/profession fields.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X['inconsistency_flag'] = (
            ((X['work_pressure'] > 0) & (X['academic_pressure'] > 0)).astype(int) +
            ((X['study_satisfaction'] > 0) & (X['job_satisfaction'] > 0)).astype(int) +
            (
                ((X['occupation_status'] == 'Working Professional') & (X['profession'] == 'Student')) |
                ((X['occupation_status'] == 'Student') & (X['profession'] != 'Student') & (X['profession'].notna()))
            ).astype(int)
        )
        return X


class ConditionalFlagger(BaseEstimator, TransformerMixin):
    """
    Adds binary flag columns for missing values in specified columns, using user-defined conditional rules.
    Flags are set as 'not applicable' or 'imputed' depending on another column's value, as defined in the flagging_map dictionary.
    """
    def __init__(self, flagging_map, suffix_map=None):
        self.flagging_map = flagging_map
        self.suffix_map = suffix_map or {'not_applicable': '_not_applicable', 'imputed': '_imputed'}

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, conditions in self.flagging_map.items():
            for flag_type, cond in conditions.items():
                cond_col, cond_val = list(cond.items())[0]
                mask = (X[cond_col] == cond_val) & (X[col].isna())
                flag_col = f"{col}{self.suffix_map[flag_type]}"
                X[flag_col] = 0
                X.loc[mask, flag_col] = 1
        return X


X_train_flagged = X_train.copy()
X_test_flagged = X_test.copy()


inconsistency_flagger = InconsistencyFlagger()
inconsistency_flagger.fit(X_train_flagged)
X_train_flagged = inconsistency_flagger.transform(X_train_flagged)
X_test_flagged = inconsistency_flagger.transform(X_test_flagged)
X_train_flagged[X_train_flagged.inconsistency_flag != 0]


flagging_map = {
    'work_pressure': {
        'not_applicable': {'occupation_status': 'Student'},
        'imputed': {'occupation_status': 'Working Professional'}
    },
    'job_satisfaction': {
        'not_applicable': {'occupation_status': 'Student'},
        'imputed': {'occupation_status': 'Working Professional'}
    },
    'academic_pressure': {
        'not_applicable': {'occupation_status': 'Working Professional'},
        'imputed': {'occupation_status': 'Student'}
    },
    'study_satisfaction': {
        'not_applicable': {'occupation_status': 'Working Professional'},
        'imputed': {'occupation_status': 'Student'}
    },
    'cgpa': {
        'not_applicable': {'occupation_status': 'Working Professional'},
        'imputed': {'occupation_status': 'Student'}
    }
}


conditional_flagger = ConditionalFlagger(flagging_map)
conditional_flagger.fit(X_train_flagged)
X_train_flagged = conditional_flagger.transform(X_train_flagged)
X_test_flagged = conditional_flagger.transform(X_test_flagged)


pipeline = Pipeline([
    ("inconsistency_flagger", InconsistencyFlagger()),
    ("conditional_flagger", ConditionalFlagger(flagging_map)),
])

pipeline.fit(X_train)


class RelationalImputer(BaseEstimator, TransformerMixin):
    """
    Conditionally imputes missing values in specified columns based on another column's value.
    Args:
        cols_to_impute (list of str): Columns to impute missing values in.
        condition_col (str): Column used to define the imputation condition.
        condition_value (str): Value in `condition_col` that triggers imputation
        strategy (str, default='constant'): Imputation strategy ('constant', 'median', or 'most_frequent').
        fill_value (any, optional): Value to use for 'constant' strategy.
    Raises:
        ValueError: If strategy is invalid or required columns are missing.
    """

    def __init__(self,
                 cols_to_impute: list["str"],
                 condition_col: str,
                 condition_value: str,
                 strategy: str = 'constant',
                 fill_value: Any = None
                 ):
        self.cols_to_impute = cols_to_impute
        self.condition_col = condition_col
        self.condition_value = condition_value
        self.strategy = strategy
        self.fill_value = fill_value
        self.medians_ = {}
        self.modes_ = {}
        if strategy not in ['constant', 'median', 'most_frequent']:
            raise ValueError("Unknown strategy. Choose: 'constant', 'median' or 'most_frequent'")
        if strategy == 'constant' and fill_value is None:
            raise ValueError("fill_value must be set for 'constant' strategy.")

    def fit(self, X: pd.DataFrame, y=None) -> "RelationalImputer":
        """Learn imputation values from data where condition is met."""
        required_cols = set(self.cols_to_impute) | {self.condition_col}
        missing_cols = required_cols - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        mask = X[self.condition_col] == self.condition_value
        if self.strategy == 'median':
            for col in self.cols_to_impute:
                self.medians_[col] = X.loc[mask, col].median()
        elif self.strategy == 'most_frequent':
            for col in self.cols_to_impute:
                mode_series = X.loc[mask, col].mode(dropna=True)
                self.modes_[col] = mode_series.iloc[0] if not mode_series.empty else self.fill_value
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values where condition is met."""
        X = X.copy()
        for col in self.cols_to_impute:
            mask = (X[self.condition_col] == self.condition_value) & (X[col].isna())
            if self.strategy == 'median':
                X.loc[mask, col] = self.medians_[col]
            elif self.strategy == 'constant':
                X.loc[mask, col] = self.fill_value
            elif self.strategy == 'most_frequent':
                X.loc[mask, col] = self.modes_[col]
        return X

    def set_output(self, transform="pandas"):
        return self



X_train_transformed = X_train_flagged.copy()
X_test_transformed = X_test_flagged.copy()


cat_imputer_students = RelationalImputer(cols_to_impute=['profession'], 
                                         condition_col='occupation_status', 
                                         condition_value='Student',
                                         fill_value='Student')
cat_imputer_students.fit(X_train_transformed)
X_train_transformed = cat_imputer_students.transform(X_train_transformed)
X_test_transformed = cat_imputer_students.transform(X_test_transformed)


class FeatureCategoryMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mappings, other_value="other", clean_func=None):
        """
        Maps categorical feature values to main categories using provided mappings.
        Args:
        mappings : Column to category mappings.
        other_value : Label for unmapped categories.
        clean_func : Function to normalize values before mapping.
        """
        self.mappings = mappings
        self.other_value = other_value
        self.clean_func = clean_func
        self.value_to_main_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.value_to_main_ = {}
        for col, mapping in self.mappings.items():
            value_to_main_col = {}
            for main_cat, variants in mapping.items():
                for v in variants:
                    key = self.clean_func(v) if self.clean_func else v
                    value_to_main_col[key] = main_cat
            self.value_to_main_[col] = value_to_main_col
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.mappings.keys():
            def map_func(x):
                if pd.isna(x):
                    return x
                key = self.clean_func(x) if self.clean_func else x
                return self.value_to_main_[col].get(key, self.other_value)

            X[col] = X[col].apply(map_func)
        return X

    def set_output(self, transform="pandas"):
        return self


# Cleaning strings
def clean_category(value):
    if pd.isna(value):
        return value
    if isinstance(value, str):
        return value.replace('.', '').replace(' ', '').replace('_', '').lower()
    return str(value).lower()


# Categories and mappings
dietary_mapping = {
    'healthy': ['healthy'],
    'unhealthy': ['unhealthy'],
    'moderate': ['moderate']
}

sleep_mapping = {
    '<5': ['lessthan5hours', '1-2hours', '1-3hours', '1-6hours', '2-3hours', '3-4hours', '3-6hours', '4-5hours'],
    '5-7': ['5-6hours', '4-6hours', '6-7hours'],
    '7-8': ['7-8hours', '6-8hours', '8hours'],
    '>8': ['morethan8hours', '9-11hours', '10-11hours', '8-9hours']
}

degree_mapping = {
    "secondary_education": ["Class12", "Class11"],
    "undergraduate": ["bed", "barch", "bcom", "bpharm", "bca", "bba", "bsc", "btech", "llb", "bhm", "ba", "be", "barch"],
    "postgraduate": ["med", "mca", "msc", "llm", "mpharm", "mtech", "mba", "me", "md", "mhm", "mcom", "mbbs", "ma", "march"],
    "doctorate": ["phd"]
}

mappings = {
    'dietary_habits': dietary_mapping,
    'sleep_duration': sleep_mapping,
    'degree': degree_mapping
}


X_train_transformed.degree.value_counts().head(5), X_train_transformed.sleep_duration.value_counts().head(5), X_train_transformed.dietary_habits.value_counts().head(5), 


grouper = FeatureCategoryMapper(mappings=mappings,
                                other_value='other',
                                clean_func=clean_category)
grouper.fit(X_train_transformed)
X_train_transformed = grouper.transform(X_train_transformed)
X_test_transformed = grouper.transform(X_test_transformed)


X_train_transformed.degree.value_counts(), X_train_transformed.sleep_duration.value_counts(), X_train_transformed.dietary_habits.value_counts(), 


X_train_transformed = X_train_transformed.replace({
    "degree": {"other": "undergraduate"},
    "sleep_duration": {"other": "7-8"},
    "dietary_habits": {"other": "moderate"},
})
X_test_transformed = X_test_transformed.replace({
    "degree": {"other": "undergraduate"},
    "sleep_duration": {"other": "7-8"},
    "dietary_habits": {"other": "moderate"},
})



columns = ['dietary_habits', 'sleep_duration', 'degree']
fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 5))

for ax, col in zip(axes, columns):
    counts = X_train_transformed[col].value_counts()
    wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.2f%%', startangle=180, colors=plt.cm.Pastel2.colors, pctdistance=0.7)
    ax.set_title(f'{col.replace("_", " ").title()} Distribution')
    ax.axis('equal')
    ax.legend(counts.index, title=col.replace('_', ' ').title(), loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# Schema to the new version of the class that assign rare categories to apropriate category of the feature (rather most frequent) 
SCHEMA = {
    "dietary_habits": {
        "map": {
            "healthy":   ["healthy"],
            "unhealthy": ["unhealthy"],
            "moderate":  ["moderate"],
        },
        "other": "moderate",
    },
    "sleep_duration": {
        "map": {
            "<5":  ["lessthan5hours","1-2hours","1-3hours","1-6hours","2-3hours","3-4hours","3-6hours","4-5hours"],
            "5-7": ["5-6hours","4-6hours","6-7hours"],
            "7-8": ["7-8hours","6-8hours","8hours"],
            ">8":  ["morethan8hours","9-11hours","10-11hours","8-9hours"],
        },
        "other": "<5",
    },
    "degree": {
        "map": {
            "secondary_education": ["Class12","Class11"],
            "undergraduate": ["bed","barch","bcom","bpharm","bca","bba","bsc","btech","llb","bhm","ba","be","barch"],
            "postgraduate": ["med","mca","msc","llm","mpharm","mtech","mba","me","md","mhm","mcom","mbbs","ma","march"],
            "doctorate": ["phd"],
        },
        "other": "undergraduate",
    },
}

clean = lambda s: str(s).strip().lower()


# new version of the class 
# class FeatureCategoryMapper(BaseEstimator, TransformerMixin):
#     def __init__(self, schema: dict, global_other: str = "other", clean_func=None):
#         """
#         schema: dict[column -> {"map": {main -> [variants]}, "other": <label optional>}]
#         global_other: fallback if per-column "other" not provided
#         clean_func: optional callable normalizing inputs (e.g., str.lower.strip)
#         """
#         self.schema = schema
#         self.global_other = global_other
#         self.clean_func = clean_func

#         # fitted state
#         self.value_to_main_ = {}      # col -> dict[value -> main_cat]
#         self.other_by_col_ = {}       # col -> other label

#     def fit(self, X: pd.DataFrame, y=None):
#         self.value_to_main_.clear()
#         self.other_by_col_.clear()

#         for col, spec in self.schema.items():
#             mapping = spec.get("map", {})
#             lut = {}
#             for main_cat, variants in mapping.items():
#                 for v in variants:
#                     key = self.clean_func(v) if self.clean_func else v
#                     lut[key] = main_cat
#             self.value_to_main_[col] = lut
#             self.other_by_col_[col] = spec.get("other", self.global_other)
#         return self

#     def transform(self, X: pd.DataFrame):
#         X = X.copy()
#         out = pd.DataFrame(index=X.index)
#         for col in self.schema.keys():
#             lut = self.value_to_main_[col]
#             other_label = self.other_by_col_[col]

#             def map_func(x):
#                 if pd.isna(x):
#                     return x
#                 key = self.clean_func(x) if self.clean_func else x
#                 return lut.get(key, other_label)

#             out[col] = X[col].apply(map_func)
#         return out

#     def set_output(self, transform="pandas"):
#         return self



# mapper = FeatureCategoryMapper(schema=SCHEMA, global_other="other", clean_func=clean_category)

# mapper.fit(X_train_transformed)
# X_train_transformed2 = mapper.transform(X_train_transformed)
# X_test_transformed2 = mapper.transform(X_test_transformed)


# X_train_transformed2.degree.value_counts(), X_train_transformed2.sleep_duration.value_counts(), X_train_transformed2.dietary_habits.value_counts(), 


X_train[(X_train.profession.isna() & (X_train.occupation_status == "Working Professional"))]


class RareCategoryCombiner(BaseEstimator, TransformerMixin):
    """
    Combines rare categories in specified categorical columns into an 'other' label.
    Args:
    columns : Columns to process.
    min_count : Minimum count to retain a category.
    other_label : Label to assign to rare categories.
    """
    def __init__(self, columns, min_count=10, other_label='other'):
        self.columns = columns
        self.min_count = min_count
        self.other_label = other_label
        self.rare_cats_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            counts = X[col].value_counts(dropna=True)
            self.rare_cats_[col] = counts[counts < self.min_count].index.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.columns:
            mask = X[col].isin(self.rare_cats_[col])
            X.loc[mask, col] = self.other_label
        return X

    def set_output(self, transform="pandas"):
        return self



rare_grouper = RareCategoryCombiner(columns=['profession', 'city'], min_count=10, other_label='other')
rare_grouper.fit(X_train_transformed)
X_train_transformed = rare_grouper.transform(X_train_transformed)
X_test_transformed = rare_grouper.transform(X_test_transformed)


custom_cat_pipe = Pipeline([
    ('cat_imputer_students', RelationalImputer(
        cols_to_impute=['profession'], 
        condition_col='occupation_status', 
        condition_value='Student',
        fill_value='Student')),
    ('cat_rare_combiner', RareCategoryCombiner(
        columns=['profession', 'city'],
        min_count=10,
        other_label='other')),
    ('cat_mapper', FeatureCategoryMapper(
        mappings=mappings,
        other_value='other',
        clean_func=clean_category)),
])

custom_cat_pipe.fit(X_train)


X_train_binning = X_train_transformed.copy()
X_test_binning = X_test_transformed.copy()


sns.histplot(data=X_train_binning[X_train_binning['cgpa'] != -1], x='cgpa', kde=True, color='#fdbb84')
plt.show()


class UniformBinner(BaseEstimator, TransformerMixin):

    def __init__(self, col, n_bins=3, labels=None):
        self.col = col
        self.n_bins = n_bins
        self.labels = labels if labels is not None else {i: f'bin_{i}' for i in range(n_bins)}
        self.est = None

    def fit(self, X: pd.DataFrame, y=None):
        mask = X[self.col].notna()
        self.est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        self.est.fit(X.loc[mask, [self.col]])
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        mask_valid = X[self.col].notna()
        if mask_valid.any():
            X.loc[mask_valid, self.col] = (self.est.transform(X.loc[mask_valid, [self.col]]).astype(int).flatten())
            X[self.col] = X[self.col].astype(object)
            X.loc[mask_valid, self.col] = X.loc[mask_valid, self.col].map(self.labels)
        return X



categorizer = UniformBinner(col='cgpa', n_bins=3, labels={0: 'low', 1: 'medium', 2: 'high'})
categorizer.fit(X_train_binning)
X_train_binning = categorizer.transform(X_train_binning)
X_test_binning = categorizer.transform(X_test_binning)
X_train_binning.cgpa.value_counts()


cgpa_with_binning = Pipeline([
    ('binner', UniformBinner(col='cgpa', n_bins=4, labels={0: 'low', 1: 'medium', 2: 'high', 3: 'excellent'})),
    ('imputer_students_freq', RelationalImputer(
        cols_to_impute=['cgpa'],
        condition_col='occupation_status',
        condition_value='student',
        strategy='most_frequent',
    )),
    ('imputer_workers_const', RelationalImputer(
        cols_to_impute=['cgpa'],
        condition_col='occupation_status',
        condition_value='Working Professional',
        strategy='constant',
        fill_value='not_applicable',
    )),
])


cgpa_without_binning = Pipeline([
    ("imputer_students_median", RelationalImputer(
        cols_to_impute=['cgpa'],
        condition_col='occupation_status',
        condition_value='student',
        strategy='median',
    )),
    ('imputer_workers_const', RelationalImputer(
        cols_to_impute=['cgpa'],
        condition_col='occupation_status',
        condition_value='Working Professional',
        strategy='constant',
        fill_value=-1,
    )),
])
cgpa_without_binning.fit(X_train)


cgpa_with_binning.fit(X_train)


X_train_binned = cgpa_with_binning.transform(X_train)


value_counts = X_train_binned.cgpa.value_counts()
value_counts_no_na = value_counts.drop('not_applicable')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot including 'not_applicable'
ax1 = value_counts.plot(kind='bar', color='#fdbb84', ax=axes[0])
ax1.set_xlabel('CGPA Bin')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of binned CGPA categories | Including "not_applicable"')
for i, v in enumerate(value_counts):
    ax1.text(i, v + value_counts.max()*0.01, str(v), ha='center', va='bottom')

# Plot excluding 'not_applicable'
ax2 = value_counts_no_na.plot(kind='bar', color='#fdbb84', ax=axes[1])
ax2.set_xlabel('CGPA Bin')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of binned CGPA categories | Excluding "not_applicable"')
for i, v in enumerate(value_counts_no_na):
    ax2.text(i, v + value_counts_no_na.max()*0.01, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()


X_train_imputed = X_train_transformed.copy()
X_test_imputed = X_test_transformed.copy()


num_imputer_students_const = RelationalImputer(
            cols_to_impute=['work_pressure', 'job_satisfaction'],
            condition_col='occupation_status',
            condition_value='Student',
            strategy='constant',
            fill_value=-1,
)
num_imputer_students_const.fit(X_train_imputed)
X_train_imputed = num_imputer_students_const.transform(X_train_imputed)
X_test_imputed = num_imputer_students_const.transform(X_test_imputed)
# X_transformed [X_transformed.job_satisfaction == -1]
# X_train[(imputer.transform(X_train).job_satisfaction == -1)]


num_imputer_workers_const = RelationalImputer(
            cols_to_impute=['academic_pressure', 'study_satisfaction', 'cgpa'],
            condition_col='occupation_status',
            condition_value='Working Professional',
            strategy='constant',
            fill_value=-1,
)
num_imputer_workers_const.fit(X_train_imputed)
X_train_imputed = num_imputer_workers_const.transform(X_train_imputed)
X_test_imputed = num_imputer_workers_const.transform(X_test_imputed)


num_imputer_students_median = RelationalImputer(
            cols_to_impute=['academic_pressure', 'study_satisfaction', 'cgpa'],
            condition_col='occupation_status',
            condition_value='Student',
            strategy='median')
num_imputer_students_median.fit(X_train_imputed)
X_train_imputed = num_imputer_students_median.transform(X_train_imputed)
X_test_imputed = num_imputer_students_median.transform(X_test_imputed)



num_imputer_workers_median = RelationalImputer(
            cols_to_impute=['work_pressure', 'job_satisfaction'],
            condition_col='occupation_status',
            condition_value='Working Professional',
            strategy='median')
num_imputer_workers_median.fit(X_train_imputed)
X_train_imputed = num_imputer_workers_median.transform(X_train_imputed)
X_test_imputed = num_imputer_workers_median.transform(X_test_imputed)


X_train_imputed.work_pressure.value_counts()


custom_num_pipe = Pipeline([
    ('num_imputer_students_const', RelationalImputer(
        cols_to_impute=['work_pressure', 'job_satisfaction'],
        condition_col='occupation_status',
        condition_value='Student',
        strategy='constant',
        fill_value=-1,
    )),
    ('num_imputer_workers_const', RelationalImputer(
        cols_to_impute=['academic_pressure', 'study_satisfaction', 'cgpa'],
        condition_col='occupation_status',
        condition_value='Working Professional',
        strategy='constant',
        fill_value=-1,
    )),
    ('num_imputer_students_median', RelationalImputer(
        cols_to_impute=['academic_pressure', 'study_satisfaction', 'cgpa'],
        condition_col='occupation_status',
        condition_value='Student',
        strategy='median'
    )),
    ('num_imputer_workers_median', RelationalImputer(
        cols_to_impute=['work_pressure', 'job_satisfaction'],
        condition_col='occupation_status',
        condition_value='Working Professional',
        strategy='median'
    )),
])


custom_num_pipe.fit(X_train)


class FeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Combines two numerical columns using a specified aggregation strategy.
    Args
    col1: First column name.
    col2: Second column name.
    strategy: Aggregation method, {'max', 'min', 'sum', 'mean'}, default='max'
    new_col_name: Name of the new combined column.
    """
    def __init__(self, col1: str, col2: str, strategy='max', new_col_name=None):
        self.col1 = col1
        self.col2 = col2
        self.strategy = strategy
        self.new_col_name = new_col_name or f"{col1}_{strategy}_{col2}"

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if self.strategy == 'max':
            X[self.new_col_name] = X[[self.col1, self.col2]].max(axis=1)
        elif self.strategy == 'min':
            X[self.new_col_name] = X[[self.col1, self.col2]].min(axis=1)
        elif self.strategy == 'sum':
            X[self.new_col_name] = X[self.col1] + X[self.col2]
        elif self.strategy == 'mean':
            X[self.new_col_name] = (X[self.col1] + X[self.col2]) / 2
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return X

    def set_output(self, transform="pandas"):
        return self


X_train_fe = X_train_imputed.copy()
X_test_fe = X_test_imputed.copy()


num_feature_combiner_pressure = FeatureCombiner('work_pressure', 'academic_pressure', strategy='max', new_col_name='work_academic_stress')
num_feature_combiner_pressure.fit(X_train_fe)
X_train_fe = num_feature_combiner_pressure.transform(X_train_fe)
X_test_fe = num_feature_combiner_pressure.transform(X_test_fe)


X_train_fe


num_feature_combiner_satisfaction = FeatureCombiner(col1='job_satisfaction', col2='study_satisfaction', strategy='max', new_col_name='job_study_satisfaction')
num_feature_combiner_satisfaction.fit(X_train_fe)
X_train_fe = num_feature_combiner_satisfaction.transform(X_train_fe)
X_test_fe = num_feature_combiner_satisfaction.transform(X_test_fe)


feature_combiner_pipe = Pipeline([
    ('num_feature_combiner_satisfaction', FeatureCombiner(col1='job_satisfaction', col2='study_satisfaction', 
                                                      strategy='max', 
                                                      new_col_name='job_study_satisfaction')),
    ('num_feature_combiner_pressure', FeatureCombiner('work_pressure', 'academic_pressure',
                                                     strategy='max', 
                                                     new_col_name='work_academic_stress')),
])

feature_combiner_pipe.fit(X_train)


custom_preprocessor = Pipeline([
        ('custom_cat', custom_cat_pipe),
        ('custom_num', custom_num_pipe),
        ('feature_combiner', feature_combiner_pipe)
    ])

pipeline = Pipeline([
    ("inconsistency_flagger", InconsistencyFlagger()),
    ("conditional_flagger", ConditionalFlagger(flagging_map)),
    ("custom_preprocessor", custom_preprocessor)
])

pipeline.fit(X_train)


X_train_cleaned = X_train_fe.copy()
X_test_cleaned = X_test_fe.copy()


# missing values of numerical features
cat_features = ['gender', 'city', 'occupation_status', 'profession', 'sleep_duration', 'dietary_habits', 'degree', 'suicidal_thoughts', 'family_history_mental_illness']
missing_count= X_train_cleaned[cat_features].isnull().sum().sort_values(ascending=False)
missing_count= missing_count[missing_count > 0]
missing_percent = (missing_count/len(X_train_cleaned[cat_features])) * 100
df_missing = pd.DataFrame({'Missing train values: counts': missing_count, "Missing train values: percentage": missing_percent.round(3)})
df_missing


cat_binary_features = ["gender", "occupation_status", "suicidal_thoughts", "family_history_mental_illness"]
cat_multiclass_features = ["dietary_habits", "sleep_duration", "degree"]
cat_highcard_features = ["profession", "city"]


imputer_freq = SimpleImputer(strategy='most_frequent')
imputer_freq.fit(X_train_cleaned[cat_binary_features])
X_train_cleaned[cat_binary_features] = imputer_freq.transform(X_train_cleaned[cat_binary_features])
X_test_cleaned[cat_binary_features] = imputer_freq.transform(X_test_cleaned[cat_binary_features])


imputer_freq = SimpleImputer(strategy='constant', fill_value="other")
imputer_freq.fit(X_train_cleaned[cat_multiclass_features+cat_highcard_features])
X_train_cleaned[cat_multiclass_features+cat_highcard_features] = imputer_freq.transform(X_train_cleaned[cat_multiclass_features+cat_highcard_features])
X_test_cleaned[cat_multiclass_features+cat_highcard_features] = imputer_freq.transform(X_test_cleaned[cat_multiclass_features+cat_highcard_features])


# Checking if there are remaining missing values 
X_train_cleaned[cat_features].isna().sum(), X_test_cleaned[cat_features].isna().sum()


num_features = (X_train_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist())
print(num_features)


# missing values of numerical features
missing_count= X_train_cleaned[num_features].isnull().sum().sort_values(ascending=False)
missing_count= missing_count[missing_count > 0]
missing_percent = (missing_count/len(X_train_cleaned[num_features])) * 100

df_missing = pd.DataFrame({'Missing train values: counts': missing_count, "Missing train values: percentage": missing_percent.round(3)})
df_missing


num_standard_features = ['age', 'work_study_hours', 'financial_stress', 'work_academic_stress', 'job_study_satisfaction']
imputer_median = SimpleImputer(strategy='median')
imputer_median.fit(X_train_cleaned[num_standard_features])
X_train_cleaned[num_standard_features] = imputer_median.transform(X_train_cleaned[num_standard_features])
X_test_cleaned[num_standard_features] = imputer_median.transform(X_test_cleaned[num_standard_features])


# Checking if there are remaining missing values 
X_train_cleaned[num_standard_features].isnull().sum().sort_values(ascending=False), X_test_cleaned[num_standard_features].isnull().sum().sort_values(ascending=False)


X_train_encoded= X_train_cleaned.copy()
X_test_encoded = X_test_cleaned.copy()


encoder_oe = OrdinalEncoder()
encoder_oe.fit(X_train_encoded[cat_binary_features])
X_train_encoded[cat_binary_features] = encoder_oe.transform(X_train_encoded[cat_binary_features])
X_test_encoded[cat_binary_features] = encoder_oe.transform(X_test_encoded[cat_binary_features])


encoder_te = ce.TargetEncoder()
encoder_te.fit(X_train_encoded[cat_highcard_features], y_train)
X_train_encoded[cat_highcard_features] = encoder_te.transform(X_train_encoded[cat_highcard_features])


X_train_en = pd.get_dummies(X_train_encoded[cat_multiclass_features])
X_test_en = pd.get_dummies(X_test_encoded[cat_multiclass_features])

X_train_encoded_full = pd.concat([X_train_encoded[num_features+cat_binary_features], X_train_en], axis=1)
X_test_encoded_full = pd.concat([X_test_encoded[num_features+cat_binary_features], X_test_en], axis=1)
X_train_encoded_full.columns


encoder_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder_ohe.fit(X_train_encoded[cat_multiclass_features])

X_train_ohe = encoder_ohe.transform(X_train_encoded[cat_multiclass_features])
X_test_ohe = encoder_ohe.transform(X_test_encoded[cat_multiclass_features])


X_train_encoded_df = pd.DataFrame(X_train_ohe, columns=encoder_ohe.get_feature_names_out(cat_multiclass_features), index=X_train_encoded[cat_multiclass_features].index)
X_test_encoded_df = pd.DataFrame(X_test_ohe, columns=encoder_ohe.get_feature_names_out(cat_multiclass_features), index=X_test_encoded[cat_multiclass_features].index)

X_train_encoded_full = pd.concat([X_train_encoded[num_features+cat_binary_features+cat_highcard_features], X_train_encoded_df], axis=1)
X_test_encoded_full = pd.concat([X_test_encoded[num_features+cat_binary_features+cat_highcard_features], X_test_encoded_df], axis=1)
X_train_encoded_full.columns


print(num_features)


X_train_scaled = X_train_encoded_full.copy()
X_test_scaled = X_test_encoded_full.copy()


scaler = StandardScaler()
scaler.fit(X_train_scaled[num_features])
X_train_scaled[num_features] = scaler.transform(X_train_scaled[num_features])
X_test_scaled[num_features] = scaler.transform(X_test_scaled[num_features])


num_custom_features = ['academic_pressure', 'work_pressure', 'cgpa', 'study_satisfaction', 'job_satisfaction']
num_standard_features = ['age', 'work_study_hours', 'financial_stress', 'work_academic_stress', 'job_study_satisfaction']

cat_binary_features = ["gender", "occupation_status", "suicidal_thoughts", "family_history_mental_illness"]
cat_multiclass_features = ["dietary_habits", "sleep_duration", "degree"]
cat_highcard_features = ["profession", "city"]


num_standard_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

num_custom_pipe = Pipeline([
    ("scaler", StandardScaler()),
])

num_preprocessor = ColumnTransformer([
    ("num_standard", num_standard_pipe, num_standard_features),
    ("num_custom", num_custom_pipe, num_custom_features),
])


cat_multiclass_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
    ("encoder", OneHotEncoder(drop='first',handle_unknown="ignore", sparse_output=False)),

])

cat_highcard_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
    ("encoder", ce.TargetEncoder()),
])


cat_binary_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("encoder", OrdinalEncoder()),
])


cat_preprocessor = ColumnTransformer([
    ("cat_multiclass", cat_multiclass_pipe, cat_multiclass_features),
    ("cat_highcard", cat_highcard_pipe, cat_highcard_features),
    ("cat_binary", cat_binary_pipe, cat_binary_features),
])

standard_preprocessor = ColumnTransformer([
    ("numerical", num_preprocessor, num_standard_features+num_custom_features),
    ("categorical", cat_preprocessor, cat_binary_features+cat_multiclass_features+cat_highcard_features),
],remainder='passthrough')



preprocess_pipeline = Pipeline([
    ("inconsistency_flagger", InconsistencyFlagger()),
    ("conditional_flagger", ConditionalFlagger(flagging_map)),
    ("custom_preprocessor", custom_preprocessor),
    ("standard_preprocessor", standard_preprocessor),
])

preprocess_pipeline.fit(X_train,y_train)


baseline = Pipeline([
    ("preprocess", preprocess_pipeline),
    ("model", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
        random_state=42
    ))
]) 


baseline.fit(X_train, y_train)


y_val_prob = baseline.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_prob >= 0.5).astype(int)


acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
roc = roc_auc_score(y_val, y_val_prob)

print("Logistic Regression Baseline (Validation)")
print(f"Accuracy: {acc:.4f}")
print(f"F1:       {f1:.4f}")
print(f"ROC-AUC:  {roc:.4f}")
print("\nClassification report (threshold=0.5):")
print(classification_report(y_val, y_val_pred, digits=3))





external_df = pd.read_csv(test_path, names=columns_names[:-1], header=0) 
external_df.head()


# ids is needed for the prediction (later)
ids = external_df['id'].copy()
external_df = external_df.drop(["name", "id"], axis=1)
external_df.head()


probs = baseline.predict_proba(external_df)[:, 1]
predictions = (probs >= 0.5).astype(int)
output = pd.DataFrame({'id': ids, 'Depression': predictions})
output


output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")







