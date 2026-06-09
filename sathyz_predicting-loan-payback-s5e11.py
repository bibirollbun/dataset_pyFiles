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


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)





from pandas.api.types import CategoricalDtype
edu_order = ["Other", "High School", "Bachelor's", "Master's", "PhD"]


grade_subgrade_order = [ '{}{}'.format(ch,i) for ch in ('A', 'B', 'C', 'D', 'E', 'F') for i in range(1, 6) ]

def load_data(file_path):
    edu_dtype = CategoricalDtype(categories=edu_order, ordered=True)    
    grade_subgrade_dtype = CategoricalDtype(categories=grade_subgrade_order, ordered=True)
    # grade_dtype = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', ], ordered=True)
    
    # ["gender", "marital_status", "employment_status", "loan_purpose"]
    gender_dtype = CategoricalDtype(categories=['Male', 'Female', 'Other'])
    marital_status_dtype = CategoricalDtype(categories=['Single', 'Married', 'Widowed', 'Divorced', ])
    employment_status_dtype = CategoricalDtype(categories=['Student', 'Unemployed', 'Employed', 'Self-Employed', 'Retired' ])
    loan_purpose_dtype = CategoricalDtype(categories=['Car', 'Debt consolidation', 'Other', 'Vacation', 'Home', 'Business', 'Education', 'Medical'])

    
    df = pd.read_csv(file_path, index_col="id",
                    dtype={
                        'education_level': edu_dtype,
                        'grade_subgrade': grade_subgrade_dtype,
                        'gender': gender_dtype,
                        'marital_status': marital_status_dtype,
                        'employment_status': employment_status_dtype,
                        'loan_purpose': loan_purpose_dtype,
                    })

    if 'loan_paid_back' in df:
        loan_paid_status = df.loan_paid_back.map(lambda x: "Yes" if x >= 1 else "No")
        df["loan_paid_status"] = pd.Categorical(loan_paid_status, categories=['No', 'Yes'], ordered=True)
    
    # ordinal
    df["grade"] = df.grade_subgrade.str[0]
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', ]
    df['grade'] = pd.Categorical(df['grade'], categories=grade_order, ordered=True)
    
    return df


file_path =  "/kaggle/input/playground-series-s5e11/train.csv"

train_df = load_data(file_path)
train_df.sample(10)


print("Repayment Rate: {:.2%}".format(train_df.loan_paid_back.mean()))
train_df.loan_paid_status.value_counts()


mdf = train_df.sample(frac=.1)
# since the dataset is large, we will sample it for exploratory data analysis.


import pandas as pd
import numpy as np
from IPython.display import display
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Your categorical features
categorical_features = ['gender', 'marital_status', 'employment_status', 'loan_purpose',]
ordinal_features = ['education_level', 'grade']

results = []

df = mdf.rename(columns={'loan_paid_back': 'target'})
repayments = []
for cat_var in categorical_features + ordinal_features:
   
    # 1. CHI-SQUARE TEST
    contingency_table = pd.crosstab(df[cat_var], df['target'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    
    # Effect size: Cramer's V
    n = df.shape[0]
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    
    
    # Repayment rates by category
    repayment_rates = df.groupby(cat_var).agg(repayment_rate=('target', 'mean') , total=('target','count')).reset_index().rename(columns={cat_var: 'feature_value'})
    repayment_rates['feature_name'] = cat_var
    repayments.append(repayment_rates)
    
    # 2. MUTUAL INFORMATION (non-parametric)
    le = LabelEncoder()
    X_cat = le.fit_transform(df[cat_var].astype(str))
    mi_score = mutual_info_classif(X_cat.reshape(-1, 1), df['target'], random_state=42)[0]

    results.append({
        'feature': cat_var,
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'significant': p_value < 0.05,
        'mutual_info': mi_score,
    })


# Summary table
results_df = pd.DataFrame(results).sort_values('cramers_v', ascending=False)
print("\n" + "="*80)
print("ğŸ“Š COMPLETE CATEGORICAL FEATURE IMPORTANCE RANKING")
print("="*80)
results_df.round(4)[['feature', 'chi2', 'p_value', 'cramers_v', 'mutual_info', 'significant']]



LOAN_PAID_STATUS = "loan_paid_status"
numerical_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'interest_rate']


repayments_df = pd.concat(repayments)
repayments_df["feature"] = repayments_df["feature_name"] + " = " + repayments_df['feature_value']
repayments_df.sort_values("repayment_rate")


#sns.scatterplot(repayments_df, x="total", y="repayment_rate", hue="feature_name")

import plotly.express as px

fig = px.scatter(repayments_df, x="total", y="repayment_rate", color="feature_name",
                 hover_data=['feature'])
fig.show()


for col in categorical_features:
    sns.kdeplot(mdf, x="annual_income", hue=col)
    plt.tight_layout()
    plt.show()


sns.scatterplot(mdf, x="loan_amount", y="interest_rate", hue="loan_paid_status", alpha=0.5)


sns.scatterplot(mdf, x="loan_amount", y="interest_rate", hue="grade", alpha=0.5)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd

# Create transformations
df_transformed = df[['annual_income']].copy()

# 1. Original
df_transformed['original'] = df['annual_income']

# 2. Standard Scaler
scaler = StandardScaler()
df_transformed['std_scaler'] = scaler.fit_transform(df[['annual_income']]).flatten()

# 3. Log(1+x)
df_transformed['log'] = np.log1p(df['annual_income'])

# 4. Log + StandardScaler
log_scaler = StandardScaler()
df_transformed['log_std_scaler'] = log_scaler.fit_transform(np.log1p(df[['annual_income']])).flatten()

# 5. Reflect + Log (for left skew)
max_income = df['annual_income'].max()
df_transformed['reflect_log'] = np.log1p(max_income - df['annual_income'] + 1)

# 6. Yeo-Johnson (auto-handles direction)
pt = PowerTransformer(method='yeo-johnson')
df_transformed['yeo_johnson'] = pt.fit_transform(df[['annual_income']]).flatten()

# Melt for easy plotting
df_melted = df_transformed.melt(id_vars=None, var_name='transformation', value_name='value')

# Plot distributions (adjusted for 6 transformations)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

transformations = ['original', 'std_scaler', 'log', 'log_std_scaler', 'reflect_log', 'yeo_johnson']
titles = ['Original', 'Std Scaler', 'Log(1+x)', 'Log+StdScaler', 'Reflect+Log', 'Yeo-Johnson']

for i, (trans, title) in enumerate(zip(transformations, titles)):
    row, col = divmod(i, 3)
    
    # Histogram + KDE
    sns.histplot(data=df_melted[df_melted['transformation']==trans], 
                x='value', kde=True, ax=axes[row, col], bins=100)
    axes[row, col].set_title(f'{title}\nSkewness: {df_transformed[trans].skew():+.3f}')
    axes[row, col].axvline(df_transformed[trans].mean(), color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



# Compare all 6 transformations
summary_stats = df_transformed.agg(['mean', 'std', 'skew', 'kurtosis']).round(3)
summary_stats.loc['range'] = df_transformed.max() - df_transformed.min()
print("Transformation Comparison (6 Options):")
summary_stats.T.loc[transformations]


import pandas as pd
import numpy as np
from itertools import combinations
from functools import lru_cache
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

lvl2_features = categorical_features + ["education_level"]

all_pairs = list(combinations(lvl2_features, 2))

@lru_cache(maxsize=32)
def feature_test_cached(feature):
    
    contingency = pd.crosstab(df[feature], df['target'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
    
    le = LabelEncoder()
    X_cat = le.fit_transform(df[feature].astype(str))
    mi_score = mutual_info_classif(X_cat.reshape(-1, 1), df['target'], random_state=42)[0]
    
    return {'chi2': chi2, 'p_value': p, 'cramers_v': cramers_v, 'mi': mi_score}

print("ğŸ”¬ OPTIMIZED PAIRWISE INTERACTION ANALYSIS (MEMOIZED)")
print("="*100)
print(f"Testing {len(all_pairs)} pairs...")

results = []

for i, (feat1, feat2) in enumerate(all_pairs, 1):
    
    f1_stats = feature_test_cached(feat1)
    f2_stats = feature_test_cached(feat2)
    
    interaction_col = f'{feat1}_{feat2}'
    df[interaction_col] = df[feat1].astype(str) + '_' + df[feat2].astype(str)
    inter_stats = feature_test_cached(interaction_col)
    
    # Range strength
    inter_stats_df = df.groupby(feat1)['target'].mean()
    range_strength = (1 - inter_stats_df).max() - (1 - inter_stats_df).min()
    
    results.append({
        'pair': f'{feat1} Ã— {feat2}',
        'f1_cramers_v': f1_stats['cramers_v'],
        'f2_cramers_v': f2_stats['cramers_v'],
        'inter_cramers_v': inter_stats['cramers_v'],
        'inter_mi': inter_stats['mi'],
        'range_strength': range_strength,
        'significant': inter_stats['p_value'] < 0.001,
        'recommended': (inter_stats['cramers_v'] > 0.1) or (range_strength > 0.15)
    })
    
    # Clean up
    del df[interaction_col]


# RESULTS (same beautiful output)
results_df = pd.DataFrame(results).sort_values('inter_cramers_v', ascending=False)

print("\nğŸ�† TOP 10 INTERACTIONS")
display(results_df.head(10)[['pair', 'f1_cramers_v', 'f2_cramers_v', 'inter_cramers_v', 
                            'range_strength', 'recommended']].round(4)
       .style
       .background_gradient(subset=['inter_cramers_v', 'range_strength'], cmap='YlOrRd'))

recommended = results_df[results_df['recommended']]['pair'].head(3).tolist()
print(f"\nğŸš€ TOP 3 RECOMMENDED: {recommended}")
print(f"ğŸ“ˆ Expected AUC gain: +{len(recommended)*0.015:.3f}")




numerical_df = train_df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

train_df.drop(columns='loan_paid_status', inplace=True)



g = sns.PairGrid(mdf.drop(columns="loan_paid_back"), hue=LOAN_PAID_STATUS, diag_sharey=False, corner=True)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.scatterplot, alpha=.3)
g.add_legend()


X = train_df.drop(columns=['loan_paid_back', 'loan_paid_status'], errors='ignore', axis=1)
y = train_df['loan_paid_back']

numerical_features  , categorical_features , ordinal_features


# Experiment 1 - Simple Logistic Regression without any feature engineering
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('yeo', PowerTransformer(method='yeo-johnson', standardize=True), ["annual_income"]),
        ('num', StandardScaler(), ["debt_to_income_ratio", "credit_score", "interest_rate"]),
        #('num', StandardScaler(), ["debt_to_income_ratio", "credit_score", "interest_rate", "annual_income"]),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
        ('ord', OrdinalEncoder(categories=[edu_order, grade_subgrade_order,]), ['education_level', 'grade_subgrade']),
    ])


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create the preprocessing and modeling pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='saga', random_state=42, class_weight='balanced'))
])



from sklearn.model_selection import cross_val_score

score = cross_val_score(model_pipeline, X, y, scoring='roc_auc', cv=3, verbose=1,).mean()
print(f"Pipeline AUC: {score:.6f}")



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model pipeline
model_pipeline.fit(X_train, y_train)


# Make predictions on the test data
y_pred = model_pipeline.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


tn, fp, fn, tp = cm.ravel().tolist()
print("Total Misclassifications:", fp + fn)
print(cm)


from sklearn.metrics import roc_curve, roc_auc_score

# Get predicted probabilities for the positive class (1)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Score: ",auc_score)
# 0.9101587836546416
# 0.9100344616546963


# Plot the ROC curve
from sklearn.metrics import RocCurveDisplay
# model_pipeline_disp = RocCurveDisplay.from_estimator(model_pipeline, X_test, y_test)
model_pipeline_disp = RocCurveDisplay.from_predictions(y_test, y_pred_proba)


categorical_features


# Get feature names after preprocessing
# Numerical features are straightforward after StandardScaler
processed_numerical_features = numerical_features

# Get feature names for one-hot encoded categorical features
onehot_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
processed_categorical_features = onehot_encoder.get_feature_names_out(categorical_features)

# Get ordinal encoded grade_subgrade feature name
ordinal_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['ord']
processed_ordinal_features = ordinal_encoder.get_feature_names_out(["education_level", "grade_subgrade"])


# Combine all processed feature names
all_processed_features = list(processed_numerical_features) + list(processed_categorical_features) + list(processed_ordinal_features)




# Get the coefficients from the logistic regression model
coefficients = model_pipeline.named_steps['classifier'].coef_[0]

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': all_processed_features,
    'Coefficient': coefficients
})

# Sort by absolute coefficient value for importance
feature_importance_df['Absolute_Coefficient'] = abs(feature_importance_df['Coefficient'])
feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

print("Top 20 Feature Importances (Coefficients):")
feature_importance_df.head(20)


model_pipeline.fit(X, y)

test_df = load_data("/kaggle/input/playground-series-s5e11/test.csv")

if 'loan_paid_back' in test_df.columns:
    test_df.drop('loan_paid_back', axis=1)

test_df['loan_paid_back'] = model_pipeline.predict_proba(test_df)[:, 1]


test_df[['loan_paid_back']].to_csv("output_logit.csv")


from lightgbm import LGBMClassifier

xgb_pipe = Pipeline([
    ('classifier', LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=32,
        min_child_samples=100,
        categorical_features=categorical_features + ordinal_features,
        random_state=42,
        verbose=-1
    ))
])



xgb_auc = cross_val_score(xgb_pipe, X, y, scoring='roc_auc', cv=3).mean()
print(f"ğŸŸ¢ XGBoost (Raw Features) AUC: {xgb_auc:.4f}")



xgb_pipe.fit(X, y)

X_test = test_df.drop(columns=["loan_paid_back"],)
test_df['loan_paid_back'] = xgb_pipe.predict_proba(X_test)[:, 1]
test_df[['loan_paid_back']].to_csv("output_xgboost.csv")




