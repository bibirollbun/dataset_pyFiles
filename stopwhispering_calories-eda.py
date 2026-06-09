!pip install --quiet seaborn --upgrade


import itertools
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import seaborn as sns


PATH_TRAIN = Path('/kaggle/input/playground-series-s5e5/train.csv')
PATH_TEST = Path('/kaggle/input/playground-series-s5e5/test.csv')
PATH_ORIGINAL = Path('/kaggle/input/calories-burnt-prediction/calories.csv')  # not used here, yet TODO


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


arr_unique_not_int = df_train['Heart_Rate'].unique()[df_train['Heart_Rate'].unique() != df_train['Heart_Rate'].unique().round()]
print(f'{arr_unique_not_int=}')

df_train[df_train['Heart_Rate'].isin(arr_unique_not_int)]


arr_unique_not_int = df_train['Height'].unique()[df_train['Height'].unique() != df_train['Height'].unique().round()]
print(f'{arr_unique_not_int=}')

df_train[df_train['Height'].isin(arr_unique_not_int)]


df_train['Heart_Rate'] = df_train['Heart_Rate'].round().astype(np.float64)  # same as test
df_train['Height'] = df_train['Height'].round().astype(np.float64)  # same as test


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


plot_numerical_feature_distributions_with_box_and_histograms(df_train, df_test, features=['Age', 'Height', 'Weight'])


plot_numerical_feature_distributions_with_box_and_histograms(df_train, df_test, features=['Duration', 'Heart_Rate', 'Body_Temp', 'Calories'])


print(f'{"Feature" :<15} {"Outliers" :<8} {"Below" :<8} {"Above" :<8}')
for col in df_train.select_dtypes(np.number).columns:
    Q1 = df_train[col].quantile(0.25)
    Q3 = df_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_below = (df_train[col] < lower_bound).sum()
    n_above = (df_train[col] > upper_bound).sum()
    print(f'{col :<15} {n_below+n_above :<8} {n_below :<8} {n_above :<8}')


def plot_ranges_and_outliers(
    df_features: pd.DataFrame, 
    ncols=3,
    features=None,  # default: all numerical features
    palette=None, #"Set2",
    hue: str|None=None,  # optional feature to split by vertically
) -> None:
    
    features = features or df_features.select_dtypes(include='number').columns
    nrows = (math.ceil(df_features.shape[1]/ncols))
    
    fig = plt.figure(figsize=[15, nrows*1.5+2])
    plt.suptitle('Ranges & Outliers', fontsize=18, fontweight='bold')
    fig.subplots_adjust(top=0.92)
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    for i, feature in enumerate(features):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax = sns.violinplot(
            data=df_features, 
            x=feature, 
            hue=hue,
            inner="quart",  # "box",
            palette=palette,
            gap=.1)
        ax.legend_.remove()

        ax.set_title(f'{feature}')
        ax.set_xlabel('')
        ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),  # centered below the subplots
        ncol=2, frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    
plot_ranges_and_outliers(df_features=df_train, hue='Sex')


fig, axes = plt.subplots(1, 2, figsize=(8,4))
axes[0].pie(df_train['Sex'].value_counts(), labels=df_train['Sex'].value_counts().index, autopct='%1.1f%%')
axes[0].set_title('train')
axes[1].pie(df_test['Sex'].value_counts(), labels=df_test['Sex'].value_counts().index, autopct='%1.1f%%')
axes[1].set_title('test')
plt.suptitle("'Sex'")
plt.show()


pair_plot = sns.pairplot(
    df_train.sample(n=4000, random_state=42),
    hue='Sex',
    palette='Set2',
    diag_kind='kde',
    plot_kws={'alpha': 0.5, 's': 15},
)
plt.suptitle('Numerical Features (incl. Target)', y=1.02, fontsize=16)
plt.show()


for feature in df_train.drop("Calories", axis=1).select_dtypes('number').columns:
    p = sns.jointplot(data=df_train, x=feature, y="Calories", kind="hist")
    p.fig.suptitle(f"{feature} / Calories")
    p.fig.tight_layout()
    
plt.show()


# let's use line plots to show the tendencies with confidence interval; and distinguish by 'Sex'
for feature in df_train.drop("Calories", axis=1).select_dtypes('number').columns:
    p = sns.relplot(
        data=df_train, x=feature, y="Calories",
        kind="line",
        hue="Sex",
    )
    p.fig.suptitle(f"{feature} / Calories")
    p.fig.tight_layout()
    
plt.show()


def plot_mean_target_by_binned_numerical_feature(
    df_features: pd.DataFrame,
    ser_targets: pd.DataFrame,
    features=None,  # default: all numerical features
    palette="Set2"
):
    colors = sns.color_palette(palette, 2).as_hex() 
    df = pd.concat([df_features, ser_targets], axis=1).copy()
    features = features or df_features.select_dtypes(include='number').columns

    unique_values = ser_targets.unique()
    if len(unique_values) == 2:  # with binary targets, we plot separately for both distributions, otherwise (multi-categories), we don't
        label_0 = f'{ser_targets.name} = {unique_values[0]}'
        label_1 = f'{ser_targets.name} = {unique_values[1]}'
        df_0 = df[df[ser_targets.name] == unique_values[0]]
        df_1 = df[df[ser_targets.name] == unique_values[1]]
    else:
        pass
    
    fig, axes = plt.subplots(len(features), 2, figsize=(12, 4*len(features)))
    for i, col in enumerate(features):
    
        # plot numerical feature distribution
        if len(unique_values) == 2:
            sns.histplot(df_0[col], label=label_0, kde=True, stat="density", alpha=.4, element="step", ax=axes[i][0], color=colors[0])
            sns.histplot(df_1[col], label=label_1, kde=True, stat="density", alpha=.4, element="step", ax=axes[i][0], color=colors[1])
        else:
            sns.histplot(df[col], label=ser_targets.name, kde=True, stat="density", alpha=.4, element="step", ax=axes[i][0], color=colors[0])
        axes[i][0].grid(False)
        axes[i][0].legend()
        axes[i][0].set_title(f"{col}")    
    
        # plot target relationship with binned numerical feature (train only)
        df['bucket'], bin_edges = pd.cut(df[col], bins=10, retbins=True, labels=False)
        bucket_means = df.groupby('bucket')[ser_targets.name].mean()
        if df["bucket"].nunique() < 10:
            # we need to add dummy bucket_means for buckets with no data
            for i_bucket in range(10):
                if i_bucket not in bucket_means.index:
                    bucket_means = pd.concat([bucket_means, pd.Series([0], index=[i_bucket])])
            bucket_means = bucket_means.sort_index()
         
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2   
        axes[i][1].plot(bin_midpoints, bucket_means, marker='o', linestyle='-', color=colors[0])
        axes[i][1].set_xlabel(f'{col} (Binned)')
        axes[i][1].set_ylabel(f'Mean {ser_targets.name}')
        axes[i][1].set_title(f'Mean {ser_targets.name} per {col}')
        axes[i][1].set_xticks(bin_midpoints, labels=[round(r, 1) for r in bin_midpoints], rotation=45)
    
    fig.tight_layout() 
    plt.show()


plot_mean_target_by_binned_numerical_feature(
    df_features=df_train.drop(columns='Calories'),
    ser_targets=df_train['Calories'],
    # features=
    palette="tab10",
)


def plot_categorical_vs_target(df: pd.DataFrame, target: str, categorical_feature: str, palette="Set2"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    sns.histplot(df, x=target, hue=categorical_feature, multiple="stack", palette=palette, ax=axes[0])
    sns.violinplot(data=df, x=target, y=categorical_feature, bw_adjust=.3, cut=0, ax=axes[1])
    plt.show()
    
    sns.displot(df, x=target, col=categorical_feature, stat='density')
    plt.show()


plot_categorical_vs_target(df_train, target="Calories", categorical_feature="Sex")



    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    sns.kdeplot(x=df_train['Calories'], 
                ax=axes[0],
                color="#0398fc",
                fill=True, 
                alpha=0.5, 
                linewidth=0.5, )

    sns.kdeplot(np.log1p(df_train['Calories'].rename(f'Calories (np.log1p)')), 
                ax=axes[1], 
                color='#07a61c',
                fill=True, 
                alpha=0.5, 
                linewidth=0.5, )

    plt.show()




