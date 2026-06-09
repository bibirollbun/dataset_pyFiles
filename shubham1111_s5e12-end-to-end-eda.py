import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import seaborn as sns
sns.set_theme(style="whitegrid")
PALETTE = "Set2"

import warnings 

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


df_train.head()


df_test.head()


print('Train Shape:', df_train.shape)
print('Test Shape:', df_test.shape)


numeric_cols = [
    'age','alcohol_consumption_per_week','physical_activity_minutes_per_week',
    'diet_score','sleep_hours_per_day','screen_time_hours_per_day','bmi',
    'waist_to_hip_ratio','systolic_bp','diastolic_bp','heart_rate',
    'cholesterol_total','hdl_cholesterol','ldl_cholesterol','triglycerides'
]

cat_cols = [
    'gender','ethnicity','education_level','income_level','smoking_status',
    'employment_status','family_history_diabetes','hypertension_history',
    'cardiovascular_history','diagnosed_diabetes'
]

TARGET = 'diagnosed_diabetes'


def plot_histograms(df, cols, rows=4, cols_per_row=4):
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(25, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col], bins=30, kde=False, ax=ax, color=sns.color_palette(PALETTE)[0])
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle("Histograms Plot for numeric values", fontsize=20)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

plot_histograms(df_train, numeric_cols)


def plot_boxplots(df, cols, rows=4, cols_per_row=4):
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.boxplot(x=df[col], ax=ax, color=sns.color_palette(PALETTE)[1])
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle("boxplots numeric values", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

plot_boxplots(df_train, numeric_cols)


def kde_plot(df, cols, rows=4, cols_per_row=4):
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.kdeplot(df[col], ax=ax, fill=True, bw_method="scott")
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle("Combined KDEs (numeric)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

kde_plot(df_train, numeric_cols)


def violin_plot(df, cols, rows=4, cols_per_row=4):
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.violinplot(x=df[col], ax=ax, inner="quartile", color=sns.color_palette(PALETTE)[2])
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle("Combined violins (numeric)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

violin_plot(df_train, numeric_cols)


def plot_corr_heatmap(df, cols):
    corr = df[cols].corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, linewidths=0.3)
    plt.title("Correlation heatmap (numeric)", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

plot_corr_heatmap(df_train, numeric_cols)


def plot_pairplot_subset(df, cols, n=9):
    subset = cols[:n]
    sns.pairplot(df[subset].sample(min(len(df), 2000)), corner=True, plot_kws={"s": 10, "alpha": 0.6})
    plt.suptitle("Pairplot (subset)", fontsize=14)
    plt.show()

plot_pairplot_subset(df_train, numeric_cols, n=9)


def plot_cat_counts(df, cols, rows=4, cols_per_row=4):
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        order = df[col].value_counts().index
        sns.countplot(y=col, data=df, order=order, ax=ax, palette=PALETTE)
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle("Categorical counts", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

plot_cat_counts(df_train, cat_cols)


def plot_cat_vs_target_stacked(df, cat_cols, target, top_n=6):
    selected = [c for c in cat_cols if c != target][:top_n]
    fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 5))
    if len(selected) == 1:
        axes = [axes]
    for ax, c in zip(axes, selected):
        ct = pd.crosstab(df[c], df[target], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=ax, colormap="Accent")
        ax.set_title(c)
        ax.legend(title=str(target), bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

plot_cat_vs_target_stacked(df_train, cat_cols, TARGET)


def plot_numeric_by_target(df, numeric_cols, target, rows=4, cols_per_row=4):
    tvals = sorted(df[target].unique())
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x=target, y=col, data=df, ax=ax, palette=PALETTE)
        ax.set_title(f"{col} by {target}", fontsize=10)
    for ax in axes[len(numeric_cols):]:
        ax.axis('off')
    fig.suptitle("Numeric features split by target (boxplots)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

plot_numeric_by_target(df_train, numeric_cols, TARGET)


def plot_kdes_by_target(df, cols, target, subset=6):
    cols = cols[:subset]
    tvals = sorted(df[target].unique())
    fig, axes = plt.subplots(2, (subset+1)//2, figsize=(6*((subset+1)//2), 8))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        for t in tvals:
            sns.kdeplot(df.loc[df[target]==t, col], fill=False, ax=ax, label=str(t))
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis('off')
    plt.legend(title=str(target))
    plt.tight_layout()
    plt.show()

plot_kdes_by_target(df_train, numeric_cols, TARGET)


def plot_joint_age_bmi(df):
    if {'age','bmi'}.issubset(df.columns):
        sns.jointplot(x='age', y='bmi', data=df, kind='hex', height=8)
        plt.suptitle("Hexbin joint: age vs bmi", y=1.02)
        plt.show()

plot_joint_age_bmi(df_train)


def plot_regression(df, x, y):
    if {x,y}.issubset(df.columns):
        plt.figure(figsize=(8,6))
        sns.regplot(x=x, y=y, data=df, scatter_kws={"s":10, "alpha":0.5}, line_kws={"color":"red"})
        plt.title(f"{y} vs {x} (regression fit)")
        plt.show()

plot_regression(df_train, 'systolic_bp', 'diastolic_bp')



def plot_strip_outliers(df, cols, rows=4, cols_per_row=4):
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, 4*rows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        sns.stripplot(x=df[col], ax=ax, size=2, alpha=0.6, color=sns.color_palette(PALETTE)[3])
        ax.set_title(col, fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(cols):]:
        ax.axis('off')
    fig.suptitle("Strip charts (quick outlier view)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

plot_strip_outliers(df_train, numeric_cols)































