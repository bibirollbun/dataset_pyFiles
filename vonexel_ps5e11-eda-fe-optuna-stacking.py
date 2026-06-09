import os
import time
import joblib
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde, probplot
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu


import shap
import optuna
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score



pd.reset_option('display.max_columns')
warnings.filterwarnings('ignore')
%matplotlib inline


# setting seed
SEED = 64911002

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(SEED)


def nan_check(df):
    total_entries = df.shape[0] * df.shape[1]
    missing_entries_max = df.isnull().sum().sum()
    missing_entries_max_percentage = (missing_entries_max / total_entries) * 100
    print(f"Total entries in the dataset: {total_entries}")
    print(f"Maximum missing values in the dataset: {missing_entries_max}")
    print(f"Percentage of maximum missing values in the dataset: {missing_entries_max_percentage:.2f}%")
    return df.isna().sum()


def feature_value_counts(df, columns_to_include=None):
    """
    Outputs the Distribution of values for each attribute in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    columns_to_include (list, optional): List of column names to process. 
                                        If None, all columns will be processed.
    """
    # Ğ�Ğ¿Ñ€ĞµĞ´ĞµĞ»Ñ�ĞµĞ¼ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº Ğ´Ğ»Ñ� Ğ¾Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ¸
    columns = columns_to_include if columns_to_include is not None else df.columns
    
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue
            
        print(f"Distribution of '{column}' feature:")
        print(df[column].value_counts())
        print("\n")


def feature_distribution(df, columns_to_visualize=None, exclude_from_visualization=None):
    """
    Visualizes feature distributions with division into numerical and categorical ones
    """
    if columns_to_visualize is None:
        columns_to_visualize = df.columns.tolist()
    if exclude_from_visualization:
        columns_to_visualize = [col for col in columns_to_visualize if col not in exclude_from_visualization]
    num_cols = df[columns_to_visualize].select_dtypes(include=np.number).columns.tolist()
    cat_cols = df[columns_to_visualize].select_dtypes(exclude=np.number).columns.tolist()
    
    # numerical
    if num_cols:
        fig_num, axes_num = plt.subplots(
            len(num_cols), 3, 
            figsize=(24, 6 * len(num_cols)),
            squeeze=False
        )
        
        for i, col in enumerate(num_cols):
            ax_hist = axes_num[i, 0]
            ax_box = axes_num[i, 1]
            ax_stat = axes_num[i, 2]
            sns.histplot(df[col], kde=True, ax=ax_hist, bins=30, color='steelblue')
            mean_val = df[col].mean()
            std_val = df[col].std()
            ax_hist.axvline(mean_val, color='darkorange', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax_hist.axvline(mean_val - std_val, color='red', linestyle=':', label='Â±1 Std')
            ax_hist.axvline(mean_val + std_val, color='red', linestyle=':')
            ax_hist.legend()
            
            # box-plot with outliers detection
            sns.boxplot(y=df[col], ax=ax_box, color='steelblue')
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            
            if df[col].min() < lower:
                ax_box.add_patch(Rectangle(
                    (-0.4, df[col].min()), 0.8, lower - df[col].min(),
                    edgecolor='red', linestyle='--', fill=False))
            if df[col].max() > upper:
                ax_box.add_patch(Rectangle(
                    (-0.4, upper), 0.8, df[col].max() - upper,
                    edgecolor='red', linestyle='--', fill=False))
            
            # stats
            stats = [
                f"Mean: {mean_val:.2f}",
                f"Std: {std_val:.2f}",
                f"Min: {df[col].min():.2f}",
                f"Max: {df[col].max():.2f}",
                f"Skew: {df[col].skew():.2f}",
                f"Kurtosis: {df[col].kurtosis():.2f}"
            ]
            ax_stat.axis('off')
            ax_stat.text(0.1, 0.5, '\n'.join(stats), fontsize=12, va='center')
        
        plt.tight_layout()
        plt.show()
    
    # categorical
    if cat_cols:
        fig_cat, axes_cat = plt.subplots(
            len(cat_cols), 2, 
            figsize=(20, 5 * len(cat_cols)),
            squeeze=False
        )
        
        for i, col in enumerate(cat_cols):
            ax_count = axes_cat[i, 0]
            ax_top = axes_cat[i, 1]
            
            # plot n-categories
            top_categories = df[col].value_counts().nlargest(20).index
            df_plot = df[col].apply(lambda x: x if x in top_categories else 'Other')
            sns.countplot(x=df_plot, ax=ax_count, palette='coolwarm')
            ax_count.tick_params(axis='x', rotation=90)
            
            # plot 5-categories
            top5 = df[col].value_counts().nlargest(5)            
            sns.barplot(x=top5.values, y=top5.index, ax=ax_top, palette='coolwarm', orient='h')
        plt.tight_layout()
        plt.show();


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


df.head()


test.head()


df.info()


test.info()


print(f"Train dataset has {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Test dataset has {test.shape[0]} rows and {test.shape[1]} columns")


 nan_check(df)


 nan_check(test)


unique = df.select_dtypes(exclude=['object']).nunique()
unique = unique[unique == 1]
print(f"Columns with only one unique value: {unique}")


df.describe(include = 'all')


test.describe(include = 'all')


numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = '.2f', linewidths = .5, cmap='coolwarm', figsize = (16, 8))
plt.show();


loan_counts = df['loan_paid_back'].value_counts()

labels = ['Paid (1)', 'Defaulted (0)']
sizes = loan_counts.values
percentages = [f'{(count/sum(sizes))*100:.1f}%' for count in sizes]
plt.figure(figsize=(16, 8))
colors = sns.color_palette('coolwarm', n_colors=2)
colors = colors[::-1]

# pie chart
wedges, texts, autotexts = plt.pie(sizes, 
                                  labels=labels, 
                                  autopct='%1.1f%%',
                                  colors=colors,
                                  startangle=90,
                                  explode=(0.05, 0.05),  
                                  textprops={'fontsize': 14, 'fontweight': 'bold'},
                                  wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(16)
    
plt.title('Loan Repayment Status Distribution', 
          fontsize=24, 
          fontweight='bold', 
          pad=20,
          color='#2c3e50')

legend_labels = [
    f'Paid Loans: {sizes[0]:,} ({percentages[0]})',
    f'Defaulted Loans: {sizes[1]:,} ({percentages[1]})'
]

plt.legend(wedges, legend_labels,
          title="Loan Status Details",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=14,
          title_fontsize=16,
          frameon=True,
          framealpha=0.9,
          edgecolor='#2c3e50')

total_loans = sum(sizes)
plt.text(0, 0, f'Total Loans\n{total_loans:,}', 
         ha='center', 
         va='center', 
         fontsize=18, 
         fontweight='bold',
         color='#2c3e50',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#2c3e50', boxstyle='round,pad=1'))

plt.figtext(0.5, 0.02, 
           f'Insight: With a {percentages[0]} repayment rate, the data shows healthy performance\n'
           f'but the {percentages[1]} default rate represents significant recoverable risk exposure',
           ha='center', 
           fontsize=14, 
           fontstyle='italic',
           bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='orange', boxstyle='round,pad=0.5'))
plt.tight_layout(rect=[0, 0.05, 0.85, 1])


numeric_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
feature_value_counts(df, columns_to_include = numeric_features)


feature_distribution(df, columns_to_visualize = numeric_features)


categorical_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
feature_value_counts(df, columns_to_include = categorical_features)


feature_distribution(df, columns_to_visualize = categorical_features)


fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(numeric_features):
    sns.boxplot(x='loan_paid_back', y=feature, data=df, ax=axes[i], palette='coolwarm')
    axes[i].set_title(f'Distribution of {feature} by target', fontweight='bold')
    group_0 = df[df['loan_paid_back'] == 0][feature]
    group_1 = df[df['loan_paid_back'] == 1][feature]
    t_stat, p_value = stats.ttest_ind(group_0, group_1, nan_policy='omit')
    axes[i].text(0.5, 0.95, f'p-value: {p_value:.4f}', transform=axes[i].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
fig.delaxes(axes[5])
plt.tight_layout()
plt.show();


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()


for i, feature in enumerate(categorical_features):
    category_performance = df.groupby(feature)['loan_paid_back'].mean().sort_values(ascending=False)
    if feature == 'grade_subgrade':
        bars = axes[i].barh(category_performance.index, category_performance.values, 
                           color=plt.cm.coolwarm(category_performance.values))
        axes[i].set_title(f'Loan Repayment Percentage by {feature}', fontweight='bold', fontsize=16)
        axes[i].set_xlabel('Repayment percentage', fontsize=14)
        axes[i].set_ylabel(feature, fontsize=14)
        for j, bar in enumerate(bars):
            width = bar.get_width()
            x_pos = width + 0.01 if width < 0.9 else width - 0.03
            ha = 'left' if width < 0.9 else 'right'
            axes[i].annotate(f'{width:.3f}', 
                            (x_pos, bar.get_y() + bar.get_height()/2),
                            ha=ha, va='center', fontweight='bold', fontsize=10)
        axes[i].set_xlim(0, max(category_performance.values) * 1.15)
        
    else:
        barplot = sns.barplot(x=category_performance.index, y=category_performance.values, 
                            ax=axes[i], palette='coolwarm')
        axes[i].set_title(f'Loan Repayment Percentage by {feature}', fontweight='bold', fontsize=16)
        axes[i].set_ylabel('Repayment percentage', fontsize=14)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=25, labelsize=12)
        for p in barplot.patches:
            height = p.get_height()
            y_pos = height + 0.01 if height < 0.95 else height - 0.04
            va = 'bottom' if height < 0.95 else 'top'
            axes[i].annotate(f'{height:.3f}', 
                           (p.get_x() + p.get_width() / 2., y_pos),
                           ha='center', va=va, fontweight='bold', fontsize=11,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
if len(categorical_features) < 6:
    for j in range(len(categorical_features), 6):
        fig.delaxes(axes[j])
plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=3.0)
plt.subplots_adjust(top=0.92) 
plt.show();


cramer_matrix = pd.DataFrame(index=categorical_features, columns=categorical_features)

for i, feat1 in enumerate(categorical_features):
    for j, feat2 in enumerate(categorical_features):
        if i <= j:  # Compute only upper triangle to avoid redundant calculations
            contingency_table = pd.crosstab(df[feat1], df[feat2])
            chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            
            cramer_matrix.loc[feat1, feat2] = cramer_v
            cramer_matrix.loc[feat2, feat1] = cramer_v

# Convert to float type and handle diagonal
cramer_matrix = cramer_matrix.astype(float)
np.fill_diagonal(cramer_matrix.values, 1.0)

# Plot heatmap
plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(cramer_matrix, dtype=bool))
sns.heatmap(cramer_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            mask=mask, center=0.3, vmin=0, vmax=1,
            cbar_kws={'label': "Cramer's V (Association Strength)"},
            linewidths=0.5, annot_kws={'size': 12})
plt.title("Association Strength Between Categorical Features (Cramer's V)", 
          fontweight='bold', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show();


plt.figure(figsize=(36, 12))

unique_marital = df['marital_status'].unique()
unique_education = df['education_level'].unique()
total_combinations = len(unique_marital) * len(unique_education)

props = {}
for i, marital in enumerate(unique_marital):
    for j, education in enumerate(unique_education):
        norm_val = (i * len(unique_education) + j) / total_combinations
        color = plt.cm.coolwarm(1 - norm_val) 
        props[(marital, education)] = {'color': color}


mosaic_data = df[['marital_status', 'education_level']]
ax = mosaic(mosaic_data, ['marital_status', 'education_level'], 
            labelizer=lambda x: '',  
            properties=props,
            gap=0.02,
            ax=plt.gca())


plt.title('Proportional Distribution: Marital Status vs Education Level', 
          fontweight='bold', fontsize=16, pad=20)

plt.xlabel('Marital Status', fontsize=14)
plt.ylabel('Education Level', fontsize=14)


from matplotlib.patches import Patch
legend_elements = []
for i, marital in enumerate(unique_marital):
    for j, education in enumerate(unique_education):
        norm_val = (i * len(unique_education) + j) / total_combinations
        color = plt.cm.coolwarm(1 - norm_val)
        if ((df['marital_status'] == marital) & (df['education_level'] == education)).any():
            legend_elements.append(Patch(facecolor=color, 
                                       edgecolor='black',
                                       label=f'{marital}, {education}'))


if len(legend_elements) > 0:
    plt.legend(handles=legend_elements, 
              loc='upper right', 
              bbox_to_anchor=(1.18, 1.0),
              title='Category Combinations',
              fontsize=9,
              framealpha=0.9)

plt.tight_layout(rect=[0, 0.03, 0.85, 1])  
plt.show();


repaid_dti = df[df['loan_paid_back'] == 1]['debt_to_income_ratio']
defaulted_dti = df[df['loan_paid_back'] == 0]['debt_to_income_ratio']
t_stat, p_value_ttest = ttest_ind(repaid_dti, defaulted_dti, equal_var=False)
print(f"Welch's t-test for DTI: t={t_stat:.2f}, p={p_value_ttest:.6f}")


u_stat, p_value_mwu = mannwhitneyu(repaid_dti, defaulted_dti, alternative='two-sided')
print(f"Mann-Whitney U test for DTI: U={u_stat:.2f}, p={p_value_mwu:.6f}")


X_causal = df[['credit_score', 'debt_to_income_ratio']].copy()
X_causal = sm.add_constant(X_causal)
y_causal = df['interest_rate']
model_causal = sm.OLS(y_causal, X_causal).fit()
df['interest_rate_residual'] = model_causal.resid

residual_corr = df['interest_rate_residual'].corr(df['loan_paid_back'])
print(f"Correlation between interest rate residuals and repayment: {residual_corr:.4f}")


def chi_square_test(feature):
    contingency_table = pd.crosstab(df[feature], df['loan_paid_back'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, contingency_table

# Employment status chi-square test
chi2_emp, p_emp, emp_table = chi_square_test('employment_status')
print(f"Employment Status Chi-square: Ï‡Â²={chi2_emp:.1f}, p={p_emp:.6f}")

# Grade_subgrade chi-square test 
chi2_grade, p_grade, grade_table = chi_square_test('grade_subgrade')
print(f"Grade_subgrade Chi-square: Ï‡Â²={chi2_grade:.1f}, p={p_grade:.6f}")

# Gender chi-square test
chi2_gender, p_gender, gender_table = chi_square_test('gender')
print(f"Gender Chi-square: Ï‡Â²={chi2_gender:.2f}, p={p_gender:.4f}")


plt.figure(figsize=(24, 12))
plt.subplot(1, 2, 1)


df['dti_binned'] = pd.cut(df['debt_to_income_ratio'], bins=20)
dti_default_rate = df.groupby('dti_binned')['loan_paid_back'].apply(lambda x: (x == 0).mean()).reset_index()
dti_default_rate.columns = ['dti_bin', 'default_rate']


plt.plot([interval.mid for interval in dti_default_rate['dti_bin']], 
         dti_default_rate['default_rate'], 'b-', linewidth=2.5)
plt.axvline(x=0.15, color='r', linestyle='--', alpha=0.7, label='Threshold at 0.15')
plt.title('Non-linear Relationship: DTI vs Default Probability', fontsize=14)
plt.xlabel('Debt-to-Income Ratio (midpoint of bins)', fontsize=12)
plt.ylabel('Default Rate', fontsize=12)
plt.grid(alpha=0.3)
plt.legend(loc='lower right')


plt.subplot(1, 2, 2)
X = df[['debt_to_income_ratio']].values
y = df['loan_paid_back'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.plot(fpr, tpr, 'o-', color='darkorange', linewidth=2.5, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve for DTI as Single Predictor', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show();


X_train = df.drop(['id', 'loan_paid_back'], axis=1)
y_train = df['loan_paid_back']
X_test = test.drop('id', axis=1)
X_test_ids = test['id'].copy()


# identify strongly rightâ€‘skewed numeric columns
numeric_cols = ['annual_income', 'loan_amount', 'debt_to_income_ratio',
                'interest_rate', 'credit_score']
skewness = X_train[numeric_cols].skew()
skewed_features = skewness[skewness.abs() > 0.75].index.tolist()


# log (Yeoâ€‘Johnson) transformations for skewed numerics
pt = PowerTransformer(method='yeo-johnson', standardize=False)
X_train[skewed_features] = pt.fit_transform(X_train[skewed_features])
X_test[skewed_features] = pt.transform(X_test[skewed_features])


# winsorize at the 0.5â€¯% / 99.5â€¯% percentiles 
def winsorize_series(s, lower=0.005, upper=0.995):
    q_low, q_high = s.quantile([lower, upper])
    return s.clip(lower=q_low, upper=q_high)
winsor_cols = ['annual_income', 'loan_amount']
for col in winsor_cols:
    X_train[col] = winsorize_series(X_train[col])
    X_test[col] = winsorize_series(X_test[col])


# loan_to_income_ratio
X_train['loan_to_income_ratio'] = X_train['loan_amount'] / X_train['annual_income']
X_test['loan_to_income_ratio'] = X_test['loan_amount'] / X_test['annual_income']


# payment_to_income
X_train['payment_to_income'] = X_train['loan_amount'] * (1 + X_train['interest_rate']/100) / X_train['annual_income']
X_test['payment_to_income'] = X_test['loan_amount'] * (1 + X_test['interest_rate']/100) / X_test['annual_income']


# interest_burden
X_train['interest_burden'] = X_train['interest_rate'] * X_train['loan_to_income_ratio']
X_test['interest_burden'] = X_test['interest_rate'] * X_test['loan_to_income_ratio']


# high dti
X_train['high_dti'] = (X_train['debt_to_income_ratio'] > 0.15).astype(int)
X_test['high_dti'] = (X_test['debt_to_income_ratio'] > 0.15).astype(int)


# extra high dti
X_train['very_high_dti'] = (X_train['debt_to_income_ratio'] > 0.25).astype(int)
X_test['very_high_dti'] = (X_test['debt_to_income_ratio'] > 0.25).astype(int)


# low_interest_high_loan
X_train['low_interest_high_loan'] = ((X_train['interest_rate'] < 7) & (X_train['loan_amount'] > 15000)).astype(int)
X_test['low_interest_high_loan'] = ((X_test['interest_rate'] < 7) & (X_test['loan_amount'] > 15000)).astype(int)


if 'loan_grade' in X_train.columns:
    grade_map_ord = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    X_train['loan_grade_ord'] = X_train['loan_grade'].map(grade_map_ord)
    X_test['loan_grade_ord'] = X_test['loan_grade'].map(grade_map_ord)


# polynomial features
poly_cols = ['debt_to_income_ratio', 'interest_rate', 'credit_score']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_features = poly.fit_transform(X_train[poly_cols])
poly_features_test = poly.transform(X_test[poly_cols])
poly_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
X_train = pd.concat([X_train.reset_index(drop=True), pd.DataFrame(poly_features, columns=poly_names)], axis=1)
X_test = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(poly_features_test, columns=poly_names)], axis=1)


# interest_to_credit_ratio (captures pricing VS risk)
X_train['interest_credit_ratio'] = X_train['interest_rate'] / X_train['credit_score']
X_test['interest_credit_ratio'] = X_test['interest_rate'] / X_test['credit_score']


# income_to_dti (inverse of dti, easier to interpret)
X_train['income_to_dti'] = X_train['annual_income'] / X_train['debt_to_income_ratio']
X_test['income_to_dti'] = X_test['annual_income'] / X_test['debt_to_income_ratio']


# debt Service Ratio (DSR): close to debt_to_income_ratio but converts annual financial metrics to monthly equivalents
X_train['dsr'] = (X_train['loan_amount'] * (X_train['interest_rate'] / 100) / 12) / (X_train['annual_income'] / 12)
X_test['dsr'] = (X_test['loan_amount'] * (X_test['interest_rate'] / 100) / 12) / (X_test['annual_income'] / 12)


# monthly income
X_train['monthly_income'] = X_train['annual_income'] / 12
X_test['monthly_income'] = X_test['annual_income'] / 12


# credit utilization 
X_train['credit_utilization_proxy'] = (
    X_train['debt_to_income_ratio'] * 
    (1 - X_train['credit_score'] / 850) * 
    np.log1p(X_train['loan_amount'])
)

X_test['credit_utilization_proxy'] = (
    X_test['debt_to_income_ratio'] * 
    (1 - X_test['credit_score'] / 850) * 
    np.log1p(X_test['loan_amount'])
)

# Ğ�Ğ¾Ñ€Ğ¼Ğ°Ğ»Ğ¸Ğ·ÑƒĞµĞ¼
scaler_cu = StandardScaler()
X_train['credit_utilization_proxy'] = scaler_cu.fit_transform(X_train[['credit_utilization_proxy']])
X_test['credit_utilization_proxy'] = scaler_cu.transform(X_test[['credit_utilization_proxy']])

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
dti_credit_cols = ['debt_to_income_ratio', 'credit_score']
dti_credit = X_train[dti_credit_cols].values
dti_credit_test = X_test[dti_credit_cols].values
poly_features = poly.fit_transform(dti_credit)
poly_features_test = poly.transform(dti_credit_test)
feature_names = poly.get_feature_names_out(dti_credit_cols)
for i, name in enumerate(feature_names):
    if name not in dti_credit_cols:
        clean_name = f'poly_{name.replace(" ", "_")}'
        X_train[clean_name] = poly_features[:, i]
        X_test[clean_name] = poly_features_test[:, i]


# squared terms
X_train['dti_sq'] = X_train['debt_to_income_ratio'] ** 2
X_test['dti_sq'] = X_test['debt_to_income_ratio'] ** 2

X_train['credit_sq'] = X_train['credit_score'] ** 2
X_test['credit_sq'] = X_test['credit_score'] ** 2


# dti bins
X_train['dti_binned'] = pd.cut(
    X_train['debt_to_income_ratio'],
    bins=[0, 0.1, 0.15, 0.2, 1.0],
    labels=['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk'],
    include_lowest=True
)
X_test['dti_binned'] = pd.cut(
    X_test['debt_to_income_ratio'],
    bins=[0, 0.1, 0.15, 0.2, 1.0],
    labels=['low_risk', 'medium_low_risk', 'medium_high_risk', 'high_risk'],
    include_lowest=True
)
# the subâ€‘grade is inherently ordered (A1-F5); 
# map to 1â€‘30
grade_map = {g: i for i, g in enumerate(['A1','A2','A3','A4','A5',
                                         'B1','B2','B3','B4','B5',
                                         'C1','C2','C3','C4','C5',
                                         'D1','D2','D3','D4','D5',
                                         'E1','E2','E3','E4','E5',
                                         'F1','F2','F3','F4','F5'], start=1)}
X_train['grade_ordinal'] = X_train['grade_subgrade'].map(grade_map)
X_test['grade_ordinal'] = X_test['grade_subgrade'].map(grade_map)


# employment status
for cat in ['Retired', 'Unemployed', 'Student']:
    col = f"is_{cat.lower()}"
    X_train[col] = (X_train['employment_status'] == cat).astype(int)
    X_test[col] = (X_test['employment_status'] == cat).astype(int)


# loan purpose: especially â€œdebt consolidationâ€� dominates
X_train['is_debt_consolidation'] = (X_train['loan_purpose'] == 'Debt consolidation').astype(int)
X_test['is_debt_consolidation'] = (X_test['loan_purpose'] == 'Debt consolidation').astype(int)


# drop transformed features
features_to_drop = ['grade_subgrade', 'employment_status', 'loan_purpose']
for feature in features_to_drop:
    if feature in X_train.columns:
        X_train.drop(feature, axis=1, inplace=True)
    if feature in X_test.columns:
        X_test.drop(feature, axis=1, inplace=True)


# ensuring consistent column order
all_features = sorted(list(set(X_train.columns.tolist() + X_test.columns.tolist())))
X_train = X_train.reindex(columns=all_features, fill_value=0)
X_test = X_test.reindex(columns=all_features, fill_value=0)


# categorical features
categorical_features = ['gender',
                        'marital_status',
                        'education_level',
                        'dti_binned']


# numerical features
numeric_features = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_features = [c for c in numeric_features if c not in categorical_features]


# scale numerical features
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])
X_train[categorical_features] = X_train[categorical_features].astype('category')
X_test[categorical_features] = X_test[categorical_features].astype('category')


X_train.head()


# validation split
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)


# uncomment for hyperparameter tuning
#def objective_catboost(trial):
#    params = {
#        'iterations': trial.suggest_int('iterations', 1600, 1690),
#        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3, log=True),
#        'depth': trial.suggest_int('depth', 4, 5),
#        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 30, 60, log=True),
#        'border_count': trial.suggest_int('border_count', 230, 240),
#        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.004, 0.07),
#        'random_strength': trial.suggest_float('random_strength', 0.48, 0.78),
#        'scale_pos_weight': 4.0,
#        'eval_metric': 'F1',
#        'loss_function': 'Logloss',
#        'task_type': 'GPU',
#        'devices': '0-1',
#        'random_state': SEED,
#        'verbose': False,
#        'early_stopping_rounds': 50,
#        'cat_features': categorical_features
#    }
    
#    model = CatBoostClassifier(**params)
#    model.fit(
#        X_train_final, y_train_final,
#        eval_set=(X_val, y_val),
#        use_best_model=True,
#        verbose=False
#    )
    
#    y_pred_proba = model.predict_proba(X_val)[:, 1]
#    auc_score = roc_auc_score(y_val, y_pred_proba)
#    return auc_score

#study_catboost = optuna.create_study(direction='maximize', study_name='catboost_optimization')
#study_catboost.optimize(objective_catboost, n_trials=500)


#print(f"Best AUC score: {study_catboost.best_value:.5f}")
#print("\nBest hyperparameters:")
#for param_name, param_value in study_catboost.best_params.items():
#    print(f"  {param_name}: {param_value}")


# best params
#catboost_best_params = study_catboost.best_params
#catboost_best_params.update({
#    'eval_metric': 'F1',
#    'loss_function': 'Logloss',
#    'task_type': 'GPU',
#    'devices': '0-1',
#    'random_state': SEED,
#    'verbose': False,
#    'early_stopping_rounds': 50,
#    'scale_pos_weight': 4.0,
#    'cat_features': categorical_features
#})

catboost_best_params = {
    'iterations': 1603,          
    'learning_rate': 0.2560851269636987,      
    'depth': 4,                  
    'l2_leaf_reg': 54.96505861247231,
    'border_count': 240, 
    'bagging_temperature': 0.036850876921115924,
    'random_strength': 0.6490047103570082,
    'scale_pos_weight': 4.0,
    'eval_metric': 'F1',
    'loss_function': 'Logloss',
    'task_type': 'GPU',
    'devices': '0-1',
    'random_state': SEED,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'cat_features': categorical_features
}


# xgb params
xgb_best_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 6,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'n_estimators': 2000,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'enable_categorical': True,
    'random_state': SEED
}


# lgb params
lgb_best_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'max_depth': -1,
    'min_data_in_leaf': 50,
    'lambda_l1': 5,
    'lambda_l2': 5,
    'categorical_feature': 'auto',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'device': 'gpu',
    'random_state': SEED
}


k = 9
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


catboost_oof_preds = np.zeros(len(X_train))
xgb_oof_preds = np.zeros(len(X_train))
lgb_oof_preds = np.zeros(len(X_train))


catboost_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))
lgb_test_preds = np.zeros(len(X_test))


for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n Fold {fold+1}/{k}")
    fold_start_time = time.time()
    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_valid_fold = X_train.iloc[valid_idx]
    y_valid_fold = y_train.iloc[valid_idx]
    
    # catboost 
    cb_model = CatBoostClassifier(**catboost_best_params)
    cb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_valid_fold, y_valid_fold),
        use_best_model=True,
        verbose=False
    )
    catboost_oof_preds[valid_idx] = cb_model.predict_proba(X_valid_fold)[:, 1]
    catboost_test_preds += cb_model.predict_proba(X_test)[:, 1] / k
    
    # xgb 
    xgb_model = xgb.XGBClassifier(**xgb_best_params)
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        verbose=False
    )
    xgb_oof_preds[valid_idx] = xgb_model.predict_proba(X_valid_fold)[:, 1]
    xgb_test_preds += xgb_model.predict_proba(X_test)[:, 1] / k
    
    # lgb
    lgb_model = lgb.LGBMClassifier(**lgb_best_params)
    lgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)]
    )
    lgb_oof_preds[valid_idx] = lgb_model.predict_proba(X_valid_fold)[:, 1]
    lgb_test_preds += lgb_model.predict_proba(X_test)[:, 1] / k
    
    fold_time = time.time() - fold_start_time
    print(f" Fold {fold+1} completed in {fold_time:.2f} seconds")

# calculating metrics for basic models
catboost_auc = roc_auc_score(y_train, catboost_oof_preds)
xgb_auc = roc_auc_score(y_train, xgb_oof_preds)
lgb_auc = roc_auc_score(y_train, lgb_oof_preds)

print(f"CatBoost OOF AUC: {catboost_auc:.5f}")
print(f"XGBoost OOF AUC: {xgb_auc:.5f}")
print(f"LightGBM OOF AUC: {lgb_auc:.5f}")


stack_train = np.column_stack((catboost_oof_preds, xgb_oof_preds, lgb_oof_preds))
stack_test = np.column_stack((catboost_test_preds, xgb_test_preds, lgb_test_preds))

# LogisticRegression
meta_model = LogisticRegression(penalty='l2', C=0.1, max_iter=5000, solver='lbfgs', random_state=SEED)
meta_model.fit(stack_train, y_train)
final_test_preds = meta_model.predict_proba(stack_test)[:, 1]


submission_df = pd.DataFrame({
    'id': X_test_ids,
    'loan_paid_back': final_test_preds
})

submission_filename = f'submission.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"\nSubmission saved to: {submission_filename}")


submission_df.head()


explainer = shap.TreeExplainer(cb_model)
shap_values = explainer.shap_values(X_val)

plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X_val, max_display = 20, show = False)
plt.title('SHAP Values - CatBoost Model', fontsize=16, fontweight='bold')
plt.tight_layout();


meta_importance = pd.DataFrame({
    'Model': ['CatBoost', 'XGBoost', 'LightGBM'],
    'Importance': np.abs(meta_model.coef_[0])
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Model', data=meta_importance, palette='viridis')
plt.title('Meta-Model Feature Importance (Base Model Weights)', fontsize=14, fontweight='bold')
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.ylabel('Base Model', fontsize=12)
plt.tight_layout();

