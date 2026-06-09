!pip install --quiet seaborn --upgrade
!pip install --quiet matplotlib --upgrade


import itertools
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import seaborn as sns


PATH_TRAIN = Path('/kaggle/input/playground-series-s5e6/train.csv')
PATH_TEST = Path('/kaggle/input/playground-series-s5e6/test.csv')


df_train = pd.read_csv(PATH_TRAIN).set_index("id")
df_test = pd.read_csv(PATH_TEST).set_index("id")


display(df_train)
display(df_test)


def display_summary(df):
    df_desc = pd.DataFrame(df.describe(include="all").transpose())
    df_summary = pd.DataFrame({
        'dtype': df.dtypes,
        '#missing': df.isnull().sum().values,
        '%missing': df.isnull().sum().values / len(df),  # 0.20 -> 20%
        '#duplicates': df.duplicated().sum(),
        'nunique': df.nunique().values,
        'min': df_desc['min'].values,
        'max': df_desc['max'].values,
        'avg': df_desc['mean'].values,
        'median': df_desc['50%'].values,
        'std dev': df_desc['std'].values,
        'all integer': (df == np.round(df)).all()
    })
    numerical_features = df.select_dtypes('number').columns
    conv_num_cols = ['min', 'max', 'avg', 'median', 'std dev']
    df_summary_numerical = df_summary[df_summary.index.isin(numerical_features)].astype({c: 'float64' for c in conv_num_cols})
    df_summary_categorical = df_summary[~df_summary.index.isin(numerical_features)]
    
    display(df_summary_numerical.style.background_gradient())
    display(df_summary_categorical.drop(columns=conv_num_cols+['all integer']).style.background_gradient())


display_summary(df_train)


display_summary(df_test)


for col in df_test.columns:
    only_train = set(df_train[col]).difference(df_test[col])
    only_test = set(df_test[col]).difference(df_train[col])
    print(f'{col} (nunique: {df_train[col].nunique()}/{df_test[col].nunique()})')
    if not only_train and not only_test:
        print('Same unique values')
    else:
        print(f'{only_train=}')
        print(f'{only_test=}')
    print()


def draw_correlation_heatmap(df_features: pd.DataFrame, 
                             absolute=False, 
                             annot=False,  # display correlation values in cells
                             figsize=(7,7),
                             title=None,
                            ) -> None:
    if not title:
        title = 'Pearson Correlation' + (' (absolute)' if absolute else '')
    df_corr = df_features.corr(method='pearson')
    if absolute:
        df_corr = df_corr.abs()
    triangle_mask = np.triu(np.ones_like(df_corr))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_corr, 
                mask=triangle_mask, 
                cmap=sns.color_palette("light:b", as_cmap=True) if absolute else 'RdYlBu_r', 
                cbar=True, 
                linewidth=1, 
                annot=annot, #fmt='.2f',
                ax=ax)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


draw_correlation_heatmap(df_features=df_train.select_dtypes('number'),
                         annot=True,
                        )    


def plot_numerical_feature_distributions_with_box_and_histograms(
    df_features: pd.DataFrame,
    df_features_2: pd.DataFrame,
    labels: tuple[str, str] = ('Train', 'Test'),    
    features: list[str] = None, 
    palette="Set2",
    ):
    features = features or df_features.select_dtypes(include='number').columns
    df_viz = pd.concat([df_features.assign(source=labels[0]), df_features_2.assign(source=labels[1])])
    colors = sns.color_palette(palette, 2).as_hex()
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(len(features), 3, figsize=(18, 4*len(features)))
    for i, feature_name in enumerate(features):
        is_discrete = is_integer_dtype(df_features[feature_name].dtype)
        sns.boxplot(data=df_viz,
                    x=feature_name, hue="source", palette=colors, ax = axes[i][0])
        plt.xlabel(feature_name)
        axes[i][0].set_title(f"Box Plot for {feature_name}")
        sns.histplot(data=df_features, x=feature_name, color=colors[0], kde=True, bins=30, label=labels[0], alpha=0.6, stat='density', discrete=is_discrete, ax = axes[i][1])
        if feature_name in df_features_2:  # target might be missing
            sns.histplot(data=df_features_2, x=feature_name, color=colors[1], kde=True, bins=30, label=labels[1], alpha=0.6, stat='density', discrete=is_discrete, ax = axes[i][1])
        axes[i][1].set_xlabel(feature_name)
        axes[i][1].set_ylabel("Density")
        axes[i][1].set_title(f"Histogram for {feature_name} (Density)")
        if feature_name in df_features_2:
            sns.histplot(data=df_features, x=feature_name, color=colors[0], kde=True, bins=30, label=labels[0], alpha=0.6, stat='frequency', discrete=is_discrete, ax = axes[i][2])
            sns.histplot(df_features_2, x=feature_name, color=colors[1], kde=True, bins=30, label=labels[1], alpha=0.6, stat='frequency', discrete=is_discrete, ax = axes[i][2])
            axes[i][2].set_ylabel("Frequency")
            axes[i][2].set_title(f"Histogram for {feature_name} (Frequency)")
            axes[i][2].legend(title="Dataset")
        else:
            # if we have only one distribution (e.g. train, not test), plot a cumulative percentage curve for first dataset
            sns.ecdfplot(df_features, x=feature_name, ax = axes[i][2], stat="percent")
            axes[i][2].set_title(f"Cumulative Percentage for {feature_name}")
            
        axes[i][2].set_xlabel(feature_name)
    plt.tight_layout()
    plt.show()


plot_numerical_feature_distributions_with_box_and_histograms(df_train, df_test)


pd.concat([
    (df_train['Potassium'].value_counts() / len(df_train)).sort_index().cumsum().rename('train'),
    (df_test['Potassium'].value_counts() / len(df_test)).sort_index().cumsum().rename('test')
    ], axis=1)


pd.concat([
    (df_train['Temparature'].value_counts() / len(df_train)).sort_index().cumsum().rename('train'),
    (df_test['Temparature'].value_counts() / len(df_test)).sort_index().cumsum().rename('test')
    ], axis=1)


def plot_pie_for_train_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    feature: str,
    size_multiplier = None,
):
    size_multiplier = size_multiplier or len(np.unique(np.concatenate((df_train[feature], df_test[feature])))) / 8
    fig, axes = plt.subplots(1, 2, figsize=(8 * size_multiplier, 4 * size_multiplier))
    axes[0].pie(df_train[feature].value_counts(), labels=df_train[feature].value_counts().index, autopct='%1.1f%%')
    axes[0].set_title('train')
    if df_test is not None:
        axes[1].pie(df_test[feature].value_counts(), labels=df_test[feature].value_counts().index, autopct='%1.1f%%')
        axes[1].set_title('test')
    else:
        axes[1].remove()
    plt.suptitle(f"'{feature}'")
    plt.show()


plot_pie_for_train_test(df_train, df_test, 'Soil Type', size_multiplier=1.5)


plot_pie_for_train_test(df_train, df_test, 'Crop Type', size_multiplier=1.5)


pair_plot = sns.pairplot(
    df_train.sample(n=4000, random_state=42),
    # hue=,
    # palette='Set2',
    diag_kind='kde',
    plot_kws={'alpha': 0.5, 's': 15},
)
plt.suptitle('Numerical Features', y=1.02, fontsize=16)
plt.show()


for feature in df_train.select_dtypes(include='number').columns:
    g = sns.FacetGrid(df_train, col='Fertilizer Name', sharex=False, sharey=False)
    g.map(sns.histplot, feature, kde=False, common_bins=False, bins=df_train[feature].nunique())  # all integer
    g.fig.suptitle(f'Distribution of {feature} by Fertilizer Name', y=1.05)
    plt.show()


def plot_categorical_feature_distribution_vs_categorical_target(df: pd.DataFrame, target: str, categorical_feature: str, palette="Set2"):
    ax = sns.histplot(
        df_train,
        x=target, 
        hue=categorical_feature, 
        multiple="fill", 
        palette=palette,
        )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


plot_categorical_feature_distribution_vs_categorical_target(df_train, "Fertilizer Name", "Soil Type")


plot_categorical_feature_distribution_vs_categorical_target(df_train, "Fertilizer Name", "Crop Type")


plot_pie_for_train_test(df_train, None, 'Fertilizer Name', size_multiplier=1.5)




