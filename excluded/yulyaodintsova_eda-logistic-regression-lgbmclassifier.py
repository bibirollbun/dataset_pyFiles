import pandas as pd
import numpy as np
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from scipy.stats import spearmanr
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split

pd.options.display.max_columns
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')


df_train=pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


from IPython.display import HTML

HTML('''
<h1 style="text-align:center; color:#2E86C1;">ğŸ“Š Exploratory Data Analysis (EDA)</h1>
<h3 style="text-align:center; color:#7D3C98;">Bank Marketing Campaign Dataset</h3>
''')


custom_colors = [
    "#8B2C00", "#B33B00", "#D14900", "#E06622", "#E88B4A", "#F0AD72", "#F8CE9E", 
    "#FAE5D3", "#FEF8F4", "#F4FBF6", "#D4F0E1", "#A9E3C2", 
    "#8CD7AD", "#66C796", "#3BB981", "#1FAE6C", "#159A5B", "#0E8749", "#0E6B3C", "#0B4F2E"
]

palette = sns.color_palette(custom_colors, n_colors=len(custom_colors))


sns.palplot(palette)
plt.title("Palette (Deep Orange â†’ White â†’ Deep Green)", fontsize=14)
plt.show()


df_train_info = pd.DataFrame({
    "DataType": df_train.dtypes,
    "MissingValues": df_train.isnull().sum(),
    "UniqueValues": df_train.nunique()
}).sort_values(by="MissingValues", ascending=False)

df_train_info['MissingValuesRatio'] = round(df_train_info['MissingValues'] / df_train.shape[0] ,2)

df_train_info


df_check = df_train[['pdays', 'previous']].copy()
df_check['has_previous_contact'] = (df_check['pdays'] != -1).astype(int)
no_prev_contact_ratio = (df_check[df_check['pdays'] == -1]['previous'] == 0).mean()
no_prev_contact_ratio 


discrete_features = ["age", "balance","duration","campaign","pdays","previous"]
categorical_features = ["job", "marital", "contact","poutcome"]
ordinal_features = ["education"]
binary_features = ["default", "housing", "loan"]
date_features = ["day", "month"]


feature_job = "job"
categorical_features_no_job = [col for col in categorical_features if col != feature_job]


fig = plt.figure(figsize=(16, 6))
feature_job = "job"

order = df_train.groupby(feature_job)['y'].mean().sort_values(ascending=False).index


sns.barplot(
    data=df_train,
    x=feature_job,
    y='y',
    order=order,
    color=custom_colors[18],
    errorbar=None
)

plt.title(f"Conversion Rate by {feature_job}", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


green_cmap = sns.color_palette(palette[10:], as_cmap=True)

pairs = list(combinations(categorical_features, 2))

fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 5 * len(pairs)))

for ax, (feat_A, feat_B) in zip(axes, pairs):
    dist = (
        df_train
        .groupby(feat_A)[feat_B]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )

    sns.heatmap(
        dist,
        cmap=green_cmap, 
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title(f"Distribution of '{feat_B}' within {feat_A}", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()


def plot_cat_num(df, cat_features, num_features, color):

    sns.set_theme(style="whitegrid", font_scale=1.1)
    

    for cat in cat_features: 
        for num in num_features:
            plt.figure(figsize=(12, 6))
            
            sns.boxplot(
                data=df, x=cat, y=num, 
                color=color
            )
            
            plt.title(f"{num.capitalize()} distribution by {cat}", fontsize=14, weight="bold")
            plt.xticks(rotation=30, ha="right", fontsize=11)
            plt.yticks(fontsize=11) 
            plt.xlabel(cat.capitalize(), fontsize=12)
            plt.ylabel(num.capitalize(), fontsize=12)
            plt.tight_layout()
            plt.show()


plot_cat_num(df_train, categorical_features, discrete_features, color=palette[5])


def plot_credit_distribution(df, features, palette, colors_idx=(5, 16)):
    colors = [palette[colors_idx[0]], palette[colors_idx[1]]]

    sns.set_theme(style="white", font_scale=1.15)
    fig, axes = plt.subplots(len(features), 1, figsize=(14, 4 * len(features)))
    if len(features) == 1:
        axes = [axes]

    for ax, col in zip(axes, features):
        data = df.groupby([col, "y"]).size().unstack(fill_value=0)
        data_pct = data.div(data.sum(axis=1), axis=0) * 100

        data_pct.plot(
            kind="bar", stacked=True, color=colors,
            ax=ax, width=0.85, edgecolor="none"
        )

        for cont in ax.containers:
            labels = [f"{bar.get_height():.1f}%" if bar.get_height() >= 6 else "" for bar in cont]
            ax.bar_label(cont, labels=labels, label_type="center", fontsize=9)

        ax.set_title(f"Credit decision distribution by '{col}'", fontsize=13, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Share (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=30, labelsize=10)
        ax.grid(False)
        ax.legend(
            ["Rejected", "Accepted"], title="Credit decision",
            ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False
        )

    plt.tight_layout()
    plt.show()


plot_credit_distribution(df_train, categorical_features, palette)


fig = plt.figure(figsize=(16, 6))
feature_education = "education"

order = df_train.groupby(feature_education)['y'].mean().sort_values(ascending=False).index


sns.barplot(
    data=df_train,
    x=feature_education,
    y='y',
    order=order,
    color=custom_colors[18],
    errorbar=None
)

plt.title(f"Conversion Rate by {feature_education}", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plot_cat_num(
    df=df_train,
    cat_features=["education"],
    num_features=["age", "balance"],
    color=palette[5]
)


plot_credit_distribution(df_train, binary_features, palette)


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 10))  # Ğ Ğ°Ğ·Ğ¼ĞµÑ€ Ğ¿Ğ¾Ğ»Ğ¾Ñ‚Ğ½Ğ°

for ax, column in zip(axes.flat, discrete_features):
    sns.boxplot(y=df_train[column], ax=ax, color=palette[16])
    
    Q1 = df_train[column].quantile(0.25)
    Q3 = df_train[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_train[(df_train[column] < lower_bound) | (df_train[column] > upper_bound)][column]
    outlier_share = (len(outliers) / len(df_train[column])) * 100  

    ax.set_title(f"Boxplot {column}\nOutliers: {outlier_share:.2f}%")

plt.tight_layout()
plt.show()


sns.set_theme(style="whitegrid", font_scale=1.2)

fig, axes = plt.subplots(len(discrete_features), 2, figsize=(16, 5 * len(discrete_features)))

for i, col in enumerate(discrete_features):
    sns.histplot(
        data=df_train, x=col, hue="y",
        bins=30, kde=False, stat="density", common_norm=False,
        palette={0: "#B33B00", 1: "#159A5B"}, ax=axes[i,0]
    )
    axes[i,0].set_title(f"Histogram of {col} by target", fontsize=14, weight="bold")
    axes[i,0].set_xlabel(col)
    axes[i,0].set_ylabel("Density")

    sns.kdeplot(
        data=df_train, x=col, hue="y",
        fill=True, common_norm=False,
        palette={0: "#B33B00", 1: "#159A5B"}, alpha=0.4, ax=axes[i,1]
    )
    axes[i,1].set_title(f"KDE of {col} by target", fontsize=14, weight="bold")
    axes[i,1].set_xlabel(col)
    axes[i,1].set_ylabel("Density")

plt.tight_layout()
plt.show()



values_0 = df_train[df_train["y"] == 0]['balance']
values_1 = df_train[df_train["y"] == 1]['balance']

ks_stat, p_value = ks_2samp(values_0, values_1)
print(f"{col}: KS = {ks_stat:.4f}, p-value = {p_value:.4e}")


plt.figure(figsize=(12,6)) 

sns.barplot(
    data=df_train,
    x="month",
    y="y",
    color=custom_colors[18], 
    errorbar=None
)

plt.title("Conversion Rate by Month", fontsize=16, weight="bold")
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Conversion Rate", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6)) 

sns.countplot(data=df_train, x="month",  color=custom_colors[18])
plt.title("Distribution of clients by month", fontsize=14, weight="bold")
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6)) 

sns.barplot(
    data=df_train,
    x="day",
    y="y",
    color=custom_colors[3], 
    errorbar=None
)

plt.title("Conversion Rate by day", fontsize=16, weight="bold")
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Day of month", fontsize=12)
plt.ylabel("Conversion Rate", fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6)) 

sns.countplot(data=df_train, x="day",  color=custom_colors[3])
plt.title("Distribution of clients by day", fontsize=14, weight="bold")
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.xlabel("Day", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.tight_layout()
plt.show()


corr, p_value = spearmanr(df_train["age"], df_train["balance"])

print(f"Spearman correlation (age vs balance): {corr:.4f}, p-value: {p_value:.4e}")


education_order = {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": 0}
df_train["education_num"] = df_train["education"].map(education_order)


corr_age, pval_age = spearmanr(df_train["age"], df_train["education_num"])
print(f"Spearman correlation (age vs education): {corr_age:.4f}, p-value = {pval_age:.4e}")


corr_bal, pval_bal = spearmanr(df_train["balance"], df_train["education_num"])
print(f"Spearman correlation (balance vs education): {corr_bal:.4f}, p-value = {pval_bal:.4e}")


from scipy.stats import chi2_contingency


def phi_coefficient(x, y):
    table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    return np.sqrt(chi2 / n)

for i in range(len(binary_features)):
    for j in range(i+1, len(binary_features)):
        col1, col2 = binary_features[i], binary_features[j]
        phi = phi_coefficient(df_train[col1], df_train[col2])
        print(f"{col1} vs {col2}: Phi = {phi:.4f}")


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum().sum()
    phi2 = chi2 / n
    r, k = table.shape
    return np.sqrt(phi2 / min(k-1, r-1))


cv = cramers_v(df_train["housing"], df_train["job"])
print(f"housing vs job: Cramer's V = {cv:.4f}")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ohe_cols = ["job", "marital", "contact", "month"]
    df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=False,dtype=int )

    if "poutcome" in df.columns:
        df["poutcome"] = (df["poutcome"] == "success").astype(int)

    binary_map = {"no": 0, "yes": 1}
    for col in ["default", "housing", "loan"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    for col in ["balance", "pdays"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 0 if x < 0 else x)
            df[col] = pd.qcut(df[col], 4, labels=False, duplicates="drop")

    drop_cols = ["day", "education","education_num"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df



df_train_prep = preprocess_df(df_train)
df_test_prep  = preprocess_df(df_test)

df_train_prep.head()


df_train_prep.shape


df_test_prep.shape


X = df_train_prep.drop(columns=["y", "id"], errors="ignore")
y = df_train_prep["y"].astype(int)

X_test = df_test_prep.drop(columns=["id"], errors="ignore")
X_test = X_test.reindex(columns=X.columns, fill_value=0)


pos_rate = y.mean()
scale_pos_weight = (1 - pos_rate) / pos_rate


X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import lightgbm as lgb
from sklearn.metrics import roc_auc_score


pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),  
    ("logreg", LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1
    ))
])

pipe.fit(X_tr, y_tr)

val_pred = pipe.predict_proba(X_val)[:, 1]
print("LogReg Validation ROC-AUC:", roc_auc_score(y_val, val_pred))

pipe.fit(X, y)
test_pred = pipe.predict_proba(X_test)[:, 1]


logreg = pipe.named_steps["logreg"]
coefs = pd.Series(logreg.coef_.ravel(), index=X.columns)

top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).head(20).index)

colors = top_coefs.apply(lambda x: "green" if x > 0 else "red")

plt.figure(figsize=(10, 7))
sns.barplot(
    x=top_coefs.values,
    y=top_coefs.index,
    palette=colors
)
plt.title("Top-20 important features (Logistic Regression)", fontsize=14, weight="bold")
plt.xlabel("Coefficient value", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.axvline(0, color="black", linestyle="--")
plt.tight_layout()
plt.show()


pos_rate = y.mean()
scale_pos_weight = (1 - pos_rate) / pos_rate


model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    learning_rate=0.03,        
    n_estimators=5000,         
    num_leaves=64,             
    min_child_samples=40,      
    subsample=0.8,             
    colsample_bytree=0.8,     
    reg_alpha=0.1,           
    reg_lambda=0.5,           
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

val_pred = model.predict_proba(X_val)[:, 1]
print("Best iteration:", model.best_iteration_)
print("Validation ROC-AUC:", roc_auc_score(y_val, val_pred))

best_n = model.best_iteration_ or model.get_params()["n_estimators"]

final_model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    learning_rate=0.02,
    n_estimators=best_n,
    num_leaves=64,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
final_model.fit(X, y)


test_pred = final_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({"id": df_test["id"], "y": test_pred})
submission.to_csv("submission_2.csv", index=False)
print(submission.head())


fi = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)

