import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as mnso
import warnings
import math

from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv',index_col = 'id')
original = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv',delimiter = ';')
train.head()


cat_features = []
for i in train.iloc[:, :-1].columns:
    if train[i].nunique() <= 20:
        cat_features.append(i)
num_features = [i for i in train.iloc[:, :-1].columns if i not in cat_features]
cat_features = cat_features + ['y']
num_features = num_features + ['y']

print('Categorical Features: ',cat_features)
print('Numercial Features: ',num_features)


corr_pearson = train[num_features].corr(method='pearson')
corr_spearman = train[num_features].corr(method='spearman')

# Mask for lower triangle (to avoid duplicate values)
mask_corr = np.tril(np.ones_like(corr_pearson, dtype=bool))

# ========================
# Cramér’s V
# ========================
def cramers_v(x, y):
    crosstab = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(crosstab)
    n = crosstab.sum().sum()
    return np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))

# Build matrix for categorical features
cramers_v_matrix = pd.DataFrame(index=cat_features, columns=cat_features)

for var1 in cat_features:
    for var2 in cat_features:
        cramers_v_matrix.loc[var1, var2] = cramers_v(train[var1], train[var2])

cramers_v_matrix = cramers_v_matrix.astype(float)

# Mask for upper triangle
mask_cramer = np.triu(np.ones_like(cramers_v_matrix, dtype=bool))

train_encoded = pd.get_dummies(train, drop_first=False)

X = train_encoded.drop(columns='y')
y = train_encoded['y']

mi = mutual_info_classif(X, y, discrete_features='auto')

mi_df = pd.DataFrame({
    'feature': X.columns,
    'MI': mi
}).sort_values(by='MI', ascending=False)

n_feats = len(cat_features) - 1  # exclude target
n_cols = 3
n_rows = 1 + math.ceil(n_feats / n_cols)  # 1 row for correlation + cramer, rest for MI

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
axes = axes.flatten()

sns.heatmap(corr_pearson, mask=~mask_corr, annot=True, cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Numerical Feature Correlation (Pearson)")

sns.heatmap(corr_spearman, mask=~mask_corr, annot=True, cmap="YlGnBu", ax=axes[1])
axes[1].set_title("Numerical Feature Correlation (Spearman)")

sns.heatmap(cramers_v_matrix, mask=mask_cramer, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[2])
axes[2].set_title("Cramér's V - Categorical Feature Association")

def get_original_feature(col_name):
    for f in cat_features[:-1]:  # exclude target
        if col_name.startswith(f + "_"):
            return f
    return col_name

mi_df['original_feature'] = mi_df['feature'].apply(get_original_feature)

# Find global max MI for consistent x-axis
max_mi = mi_df['MI'].max()

for ax, f in zip(axes[3:], cat_features[:-1]):
    sub = mi_df[mi_df['original_feature'] == f].sort_values('MI', ascending=False)
    sns.barplot(x='MI', y='feature', data=sub, ax=ax, palette='viridis')
    ax.set_title(f"Mutual Information: {f} vs target (y)")
    ax.set_xlabel("MI score")
    ax.set_ylabel("Encoded categories")
    ax.set_xlim(0, max_mi * 1.05)

# Remove unused axes (if any)
for ax in axes[3 + n_feats:]:
    ax.remove()

plt.tight_layout()
plt.show()


def plot_categorical_comparison(df=None, x=None, y=None, x_order=None, y_order=None):
    """
    Creates two plots side by side:
    (1) Stacked bar chart (normalized proportions of y within each x)
    (2) Countplot (raw counts of y within each x)
    Colors for y are consistent across both plots.

    Parameters:
    - df : DataFrame (default=train if not provided)
    - x  : categorical column (group by)
    - y  : categorical column (distribution within x)
    - x_order : list of categories for x (optional, e.g. months in order)
    - y_order : list of categories for y (optional, keeps colors consistent)
    """

    # Default df
    if df is None:
        df = train

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Determine order
    order = x_order if x_order is not None else sorted(df[x].value_counts().index.tolist())
    if y_order is None:
        y_order = df[y].value_counts().index.tolist()

    # ========================
    # Define consistent colors
    # ========================
    palette = dict(zip(y_order, sns.color_palette("tab10", len(y_order))))

    # ========================
    # (1) Stacked bar chart (normalized proportions)
    # ========================
    prop_df = df.groupby(x)[y].value_counts(normalize=True).unstack(fill_value=0)
    prop_df = prop_df.reindex(order)  # apply x order
    prop_df = prop_df.reindex(columns=y_order, fill_value=0)  # apply y order

    prop_df.plot(kind='bar', stacked=True, ax=axes[0], color=[palette[col] for col in prop_df.columns])

    axes[0].set_title(f"Distribution of {y.upper()} within each {x.upper()} (Proportion)")
    axes[0].set_xlabel(x)
    axes[0].set_ylabel("Proportion")
    axes[0].legend(title=y, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=45)

    # Add annotations
    for p in axes[0].patches:
        width, height = p.get_width(), p.get_height()
        x0, y0 = p.get_xy()
        if height > 0:
            axes[0].text(
                x0 + width/2, y0 + height/2,
                f"{height*100:.1f}%",
                ha="center", va="center",
                fontsize=9, color="black", weight="bold"
            )

    # ========================
    # (2) Countplot (raw counts)
    # ========================
    sns.countplot(x=x, hue=y, data=df, ax=axes[1],
                  order=order, hue_order=y_order, palette=palette)

    axes[1].set_title(f"Count of {y.upper()} within each {x.upper()} (Raw counts)")
    axes[1].set_xlabel(x)
    axes[1].set_ylabel("Count")
    axes[1].legend(title=y, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


plot_categorical_comparison(x = 'poutcome', y = 'y')


train["contact_status"] = np.where(train["pdays"] == -1, "New_customer", "Old_customer")
test["contact_status"] = np.where(test["pdays"] == -1, "New_customer", "Old_customer")
plot_categorical_comparison(x = 'contact_status', y = 'y')
plot_categorical_comparison(x="contact_status",y="poutcome")
def calculate_value_counts(group_columns):
    result = train.groupby(group_columns)['y'].value_counts().to_frame('value_counts')
    result['value_counts_normalized'] = train.groupby(group_columns)['y'].value_counts(normalize=True).mul(100).to_frame('value_counts_normalized')
    result = result.sort_values(group_columns)
    return result
calculate_value_counts(['contact_status', 'poutcome'])


month_order = ["jan","feb","mar","apr","may","jun",
               "jul","aug","sep","oct","nov","dec"]

plot_categorical_comparison(x="month", y="y", x_order=month_order)


def generate_palette(categories, cmap='tab10'):
    """
    Generate a color palette dictionary automatically
    for given categories using seaborn color palettes.
    """
    palette = sns.color_palette(cmap, n_colors=len(categories))
    return {cat: color for cat, color in zip(categories, palette)}

def plot_categorical_comparison_y(df, x, y, axes, month_order=None, cmap_x='tab10', cmap_y='Set2'):
    """
    Reusable function to plot stacked bar (A), countplot (B), and lineplot (C).
    Automatically generates orders and palettes for x and y.
    - If x == "month" and month_order is provided, uses month_order.
    """
    # Auto-generate orders
    if month_order is not None:
        x_order = month_order
    else:
        x_order = sorted(df[x].unique().tolist())
    
    y_order = df[y].unique().tolist()

    # Auto-generate palettes
    palette_x = generate_palette(x_order, cmap=cmap_x)
    palette_y = generate_palette(y_order, cmap=cmap_y)

    # --- Stacked bar chart (A) ---
    prop_df = df.groupby(x)[y].value_counts(normalize=True).unstack(fill_value=0)
    prop_df = prop_df.reindex(x_order)
    prop_df = prop_df.reindex(columns=y_order, fill_value=0)

    prop_df.plot(kind='bar', stacked=True, ax=axes['A'],
                 color=[palette_y[col] for col in prop_df.columns])
    
    axes['A'].set_title(f"Distribution of {y.upper()} within each {x.upper()} (Proportion)")
    axes['A'].set_xlabel(x)
    axes['A'].set_ylabel("Proportion")
    axes['A'].tick_params(axis='x', rotation=45)

    # Add annotations
    for p in axes['A'].patches:
        width, height = p.get_width(), p.get_height()
        x0, y0 = p.get_xy()
        if height > 0:
            axes['A'].text(
                x0 + width/2, y0 + height/2,
                f"{height*100:.1f}%",
                ha="center", va="center",
                fontsize=9, color="black", weight="bold")

    # --- Countplot (B) ---
    sns.countplot(x=x, hue=y, data=df, ax=axes['B'],
                  order=x_order, hue_order=y_order, palette=palette_y)
    axes['B'].set_title(f"Count of {y.upper()} within each {x.upper()} (Raw counts)")
    axes['B'].set_xlabel(x)
    axes['B'].set_ylabel("Count")
    axes['B'].tick_params(axis='x', rotation=45)

    # --- Lineplot (C) ---
    df_c = df.groupby([x, y])['y'].value_counts(normalize=True).to_frame('prop').reset_index()
    df_c = df_c[df_c['y'] == 1]
    df_c[x] = pd.Categorical(df_c[x], categories=x_order, ordered=True)
    df_c = df_c.sort_values(x)

    sns.lineplot(
        data=df_c,
        x=x,
        y='prop',
        hue=y,
        
        hue_order=y_order,
        style=y,
        markers=True,
        dashes=True,
        ax=axes['C'],
        palette=palette_y)

    axes['C'].set_title(f"Proportion of Target by {x.upper()} - {y.upper()} (When customer subscribe to a bank term deposit)")
    axes['C'].set_xlabel(x)
    axes['C'].set_ylabel("Proportion")
    axes['C'].set_ylim(0, 1)
    axes['C'].tick_params(axis='x', rotation=45)

    # Reference line
    axes['C'].axhline(0.5, color='red', linestyle='--', linewidth=1)
    axes['C'].text(
        x=0.02, y=0.52, s="50% threshold", 
        color='red', fontsize=10, weight='bold',
        ha='left', va='bottom', transform=axes['C'].transAxes)

# =====================
# Example usage
# =====================
mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(
    df=train,
    x="month",
    y="housing",
    axes=axd,
    month_order=month_order  # ✅ will apply your predefined order
)

plt.tight_layout()
plt.show()


mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(
    df=train,
    x="month",
    y="loan",
    axes=axd,
    month_order=month_order  # ✅ will apply your predefined order
)

plt.tight_layout()
plt.show()


def add_loan_status(df):
    df = df.copy()  # avoid modifying original dataframe
    df['loan_status'] = np.select(
        [(df['housing'] == 'yes') & (df['loan'] == 'yes'),
        (df['housing'] == 'yes') & (df['loan'] == 'no'),
        (df['housing'] == 'no') & (df['loan'] == 'yes'),
        (df['housing'] == 'no') & (df['loan'] == 'no')],
        ['Have both housing / personal loan',
        'Housing Loan',
        'Personal Loan',
        'No loan'])
    return df
train = add_loan_status(train)
test = add_loan_status(test)


mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(
    df=train,
    x="month",
    y="loan_status",
    axes=axd,
    month_order=month_order  # ✅ will apply your predefined order
)

plt.tight_layout()
plt.show()


mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(
    df=train,
    x="month",
    y="contact",
    axes=axd,
    month_order=month_order  # ✅ will apply your predefined order
)

plt.tight_layout()
plt.show()


mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(
    df=train,
    x="month",
    y="contact_status",
    axes=axd,
    month_order=month_order  # ✅ will apply your predefined order
)

plt.tight_layout()
plt.show()


def add_days_category(df):    
    # Define bins and labels
    bins = [-1, 7, 30, 90, 365, 730, np.inf]
    labels = ["1-week", "1-month", "1-quarter", "1-year", "2-year", "More than 2 years"]

    # Mask for Old_customer
    mask = df["contact_status"] == "Old_customer"

    # Use pd.cut but cast to string so we avoid Categorical conflicts
    df.loc[mask, "days_category"] = pd.cut(
        df.loc[mask, "pdays"], bins=bins, labels=labels
    ).astype(str)

    # Replace "nan" (string) with "0"
    # df["days_category"] = df["days_category"].replace("nan", "0")

    # Define final ordered categories, including "New"
    categories = ["New"] + labels
    df["days_category"] = pd.Categorical(df["days_category"], categories=categories, ordered=True)

    # Example: mark completely new customers as "New"
    new_mask = df["contact_status"] == "New_customer"
    df.loc[new_mask, "days_category"] = "New"

    return df

# Apply
train = add_days_category(train)
test = add_days_category(test)
mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(
    df=train[train["contact_status"] == "Old_customer"],
    y="poutcome",month_order = ['1-week','1-month','1-quarter','1-year','2-year','More than 2 years'],
    x="days_category",
    axes=axd)

plt.tight_layout()
plt.show()
df = calculate_value_counts(['days_category', 'poutcome'])
df.loc[df.index.get_level_values(0).isin(['1-month','1-quarter', '1-year', '2-year'])]


train["Contact_current_campaign"] = train["campaign"].apply(
    lambda x: "More than 4" if x > 3 else str(x))

test["Contact_current_campaign"] = test["campaign"].apply(
    lambda x: "More than 4" if x > 3 else str(x))

train["Contact_previous_campaign"] = train["previous"].apply(
    lambda x: "More than 4" if x > 3 else str(x))

test["Contact_previous_campaign"] = test["previous"].apply(
    lambda x: "More than 4" if x > 3 else str(x))


def add_age_bins(df, col="age", new_col="age_bin"):
    bins = [18, 30, 40, 50, 60, df[col].max()+1]  # +1 so 95 fits
    labels = ["18-30","30-40", "40-50", "50-60", "60+"]

    df[new_col] = pd.cut(df[col], bins=bins, labels=labels, right=False)
    return df
train = add_age_bins(train)
test = add_age_bins(test)

plot_categorical_comparison(x='age_bin',y='y')
mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(df=train,
    y="age_bin",
    x="loan_status",
    axes=axd)


train['balance_type'] = train['balance'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
test['balance_type'] = test['balance'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(df=train,
    y="balance_type",
    x="age_bin",
    axes=axd)


mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(df=train,
    y="balance_type",
    x="job",
    axes=axd)


mosaic = """
AB
CC
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(df=train,
    y="age_bin",
    x="job",
    axes=axd)


def compare_campaign(row):
    if row['Contact_current_campaign'] < row['Contact_previous_campaign']:
        return 'Less'
    elif row['Contact_current_campaign'] == row['Contact_previous_campaign']:
        return 'Equal'
    else:
        return 'More'
train['Compare_contact'] = train.apply(compare_campaign, axis=1)
test['Compare_contact'] = test.apply(compare_campaign, axis=1)


fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(df=train,
    y="Compare_contact",
    x="contact_status",axes=axd)


fig, axd = plt.subplot_mosaic(mosaic, figsize=(14, 12))

plot_categorical_comparison_y(df=train,
    y="Compare_contact",
    x="month",
    axes=axd,  month_order=month_order)


col_removes = ['campaign','previous','day','Contact_current_campaign','Contact_previous_campaign','age_bin','pdays']
# Remove from train and test
train = train.drop(columns=col_removes, errors="ignore")
test = test.drop(columns=col_removes, errors="ignore")
cols = list(train.columns)

# Move 'y' to the end
if 'y' in cols:
    cols.remove('y')
    cols.append('y')

# Reorder DataFrame
train = train[cols]


unorder_features = ['job','marital','contact','poutcome','loan_status']
order_features = ['education','Compare_contact']
features_map = {'yes':1,'no':0}     ### default, housing, loan
customer_map = {'New_customer':1, 'Old_customer':0}
balance_map = {'Positive':1, 'Negative':0}
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}


train['education'].unique()


for i in train.columns:
    if train[i].dtypes == 'O':
        print(f'Column {i} have {train[i].nunique()} values and they are {train[i].unique()}')


sns.countplot(x='y',data=train)
train['y'].value_counts(normalize=True)

