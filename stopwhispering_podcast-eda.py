!pip install --quiet seaborn --upgrade


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
import seaborn as sns


# hide warning when displaying DataFrames ("RuntimeWarning: invalid value encountered in greater has_large_value")
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)


PATH_TRAIN = Path('/kaggle/input/playground-series-s5e4/train.csv')
PATH_TEST = Path('/kaggle/input/playground-series-s5e4/test.csv')
PATH_ORIGINAL = Path('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')


df_train = pd.read_csv(PATH_TRAIN).set_index("id")
df_test = pd.read_csv(PATH_TEST).set_index("id")


display(df_train.head())
display(df_test.head())


def display_summary(df):
    df_desc = pd.DataFrame(df.describe(include="all").transpose())
    df_summary = pd.DataFrame({
        'dtype': df.dtypes,
        '#missing': df.isnull().sum().values,
        '%missing': df.isnull().sum().values / len(df),  # 0.20 -> 20%
        '#duplicates': df.duplicated().sum(),
        '#unique': df.nunique().values,
        'min': df_desc['min'].values,
        'max': df_desc['max'].values,
        'avg': df_desc['mean'].values,
        'std dev': df_desc['std'].values,
        'all integer': (df == np.round(df)).all()
    })
    numerical_features = df.select_dtypes('number').columns
    conv_num_cols = ['min', 'max', 'avg', 'std dev']
    df_summary_numerical = df_summary[df_summary.index.isin(numerical_features)].astype({c: 'float64' for c in conv_num_cols})
    df_summary_categorical = df_summary[~df_summary.index.isin(numerical_features)]
    
    display(df_summary_numerical.style.background_gradient())
    display(df_summary_categorical.drop(columns=conv_num_cols+['all integer']).style.background_gradient())
    
display_summary(df_train)


display_summary(df_test)


sns.boxplot(data=pd.concat([df_train.assign(source='train'), df_test.assign(source='test')]),
            x='Episode_Length_minutes', 
            hue='source', 
            palette="Set2")
plt.show()


print(f"{df_train['Episode_Length_minutes'].mean()=}")
print(f"{df_test['Episode_Length_minutes'].mean()=}")
print(f"{df_train['Episode_Length_minutes'].median()=}")
print(f"{df_test['Episode_Length_minutes'].median()=}")
print(f"{df_train['Episode_Length_minutes'].max()=}")
print(f"{df_test['Episode_Length_minutes'].max()=}")
print(f"{df_train['Episode_Length_minutes'].max()=}")
print(f"{df_test['Episode_Length_minutes'].max()=}")

print(f"{(df_test['Episode_Length_minutes'] > 325).sum()=}")


df_test[df_test['Episode_Length_minutes'] > 325]


df_test.loc[df_test['Episode_Length_minutes'] > 325, 'Episode_Length_minutes'] = 325


sns.boxplot(data=pd.concat([df_train.assign(source='train'), df_test.assign(source='test')]),
            x='Episode_Length_minutes', 
            hue='source', 
            palette="Set2")
plt.show()


display(df_train['Number_of_Ads'].value_counts(dropna=False))


display(df_test['Number_of_Ads'].value_counts(dropna=False))


df_train.loc[df_train['Number_of_Ads'] > 3, 'Number_of_Ads'] = 3
df_test.loc[df_test['Number_of_Ads'] > 3, 'Number_of_Ads'] = 3

df_train['Number_of_Ads'] = df_train['Number_of_Ads'].fillna(3)

df_train['Number_of_Ads'] = df_train['Number_of_Ads'].astype(int)
df_test['Number_of_Ads'] = df_test['Number_of_Ads'].astype(int)

display(df_train['Number_of_Ads'].value_counts(dropna=False))
display(df_test['Number_of_Ads'].value_counts(dropna=False))


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


df_train_only_missing_eplen = df_train.loc[df_train['Episode_Length_minutes'].isna()]
print(f"{len(df_train_only_missing_eplen)=:_}")

df_train_without_missing_eplen = df_train.loc[df_train['Episode_Length_minutes'].notna()]
print(f"{len(df_train_without_missing_eplen)=:_}")


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.histplot(data=df_train_only_missing_eplen, 
             x='Listening_Time_minutes', kde=True, bins=30, alpha=0.6, stat='density', color='#a87732', ax=axes[0])
sns.histplot(data=df_train_without_missing_eplen, 
             x='Listening_Time_minutes', kde=True, bins=30, alpha=0.6, stat='density', color='#295587', ax=axes[1])
sns.histplot(data=df_train_only_missing_eplen, 
             x='Listening_Time_minutes', kde=True, bins=30, alpha=0.6, stat='density', color='#a87732', ax=axes[2], label='only nan')
sns.histplot(data=df_train_without_missing_eplen, 
             x='Listening_Time_minutes', kde=True, bins=30, alpha=0.6, stat='density', color='#295587', ax=axes[2], label='no nan')
axes[2].legend(title="Episode_Length_minutes")

plt.suptitle("Target Distribution by NaN Status of 'Episode_Length_minutes'")
plt.show()


def plot_categorical_feature_distribution(
    ser: pd.Series, 
    palette: str="Set2",
    explode_value=None,
) -> None:
    nunique = ser.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(15 + nunique*0.01, 5 + nunique*0.1))
    axes = axes.flatten()
    value_counts = ser.value_counts(ascending=True)
    labels = value_counts.index.tolist()
    colors = sns.color_palette(palette, len(labels)).as_hex()  # we borrow colors from a seaborn color palette
    # Donut Chart
    explodes=None if explode_value is None else [0.1 if i == explode_value else 0 for i in value_counts.index]
    axes[0].pie(
        value_counts, 
        autopct='%1.1f%%', 
        textprops={'size': 8, 'color': 'black'}, 
        colors=colors,
        wedgeprops=dict(width=0.4),  # donut wedge width
        startangle=80, 
        pctdistance=0.85,  # have percentage displayed within wedge
        explode=explodes,
        labels=labels,
    )
    # Count Plot 
    for i, v in enumerate(value_counts):
        axes[1].barh(y=i, width=v, color='none', edgecolor=colors[i], hatch='////')
        axes[1].text(x=v + 1, y=i, s=str(v), color='black', fontsize=10, va='center')
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels)
    sns.despine(left=True, bottom=True)  # remove default spines (borders) from plot
    axes[1].set_xticks([])
    fig.suptitle(f'{ser.name} Distribution', fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

plot_categorical_feature_distribution(ser=df_train['Podcast_Name'])


plot_categorical_feature_distribution(ser=df_train['Episode_Title'])


plot_categorical_feature_distribution(ser=df_train['Genre'])


plot_categorical_feature_distribution(ser=df_train['Publication_Day'])


plot_categorical_feature_distribution(ser=df_train['Publication_Time'])


plot_categorical_feature_distribution(ser=df_train['Episode_Sentiment'])


for feature in df_train.drop("Listening_Time_minutes", axis=1).select_dtypes('number').columns:
    p = sns.jointplot(data=df_train, x=feature, y="Listening_Time_minutes", kind="hist")
    p.fig.suptitle(f"{feature} / Listening_Time_minutes")
    p.fig.tight_layout()
    
plt.show()


# we need to get rid of that one outlier
df_viz = df_train.copy()
df_viz.loc[df_viz['Episode_Length_minutes'] > 150, 'Episode_Length_minutes'] = 64.5


g = sns.regplot(df_viz, x='Episode_Length_minutes', y='Listening_Time_minutes', 
            marker="o",  # "o" -> circle
            scatter_kws={'s': 0.1},  # make them smaller
            line_kws={"color": "red"},
           )
g.figure.set_size_inches(18.5, 10.5)


def plot_numerical_vs_target_with_for_multiple_categorical_trendlines(
    df: pd.DataFrame,
    target_column: pd.Series,
    numerical_feature: str,
    categorical_features: list[str],
    palette="deep",  #"tab10"
):
    for categorical_feature in categorical_features:
        sns.lmplot(
            df, x=numerical_feature, y=target_column, 
            scatter_kws={'s': 0.1},
            line_kws={'linewidth': 4},
            hue=categorical_feature,
            height=7,
            aspect=2,
            legend=False,  # dots are barely visible with default legend
            palette=palette,
            )
        plt.legend(fontsize=8, markerscale=40, title=categorical_feature)  # make markers much larger in the legend
        plt.title(f'{target_column} by {numerical_feature} and {categorical_feature}')
        plt.show()


plot_numerical_vs_target_with_for_multiple_categorical_trendlines(
    df=df_viz,
    target_column='Listening_Time_minutes',
    numerical_feature='Episode_Length_minutes',
    categorical_features=['Genre', 'Publication_Day', 'Publication_Time', 'Number_of_Ads', 'Episode_Sentiment']
)


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
    df_features=df_train.drop(columns='Listening_Time_minutes'),
    ser_targets=df_train['Listening_Time_minutes'],
    # features=['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads'],
    palette="tab10",
)


df_train_alt = df_train.copy()
df_train_alt['Listening_Time_Quota'] = df_train_alt['Listening_Time_minutes'] / df_train_alt['Episode_Length_minutes']

print(f'{df_train_alt.shape=}')


print(f"{(df_train_alt['Listening_Time_Quota'] > 1).sum()=:_}")

df_train_alt.drop((df_train_alt.loc[df_train_alt['Listening_Time_Quota'] > 1]).index, inplace=True)
print(f"{(df_train_alt['Listening_Time_Quota'] > 1).sum()=:_}")

print(f'{df_train_alt.shape=}')


for feature in df_train.select_dtypes('number').columns:
    p = sns.jointplot(data=df_train_alt, x=feature, y="Listening_Time_Quota", kind="hist")
    p.fig.suptitle(f"{feature} / Listening_Time_Quota")
    p.fig.tight_layout()
    
plt.show()


print(f"{df_train_alt.loc[df_train_alt['Episode_Length_minutes'] <= 20, 'Listening_Time_Quota'].mean()=}")
print(f"{df_train_alt.loc[df_train_alt['Episode_Length_minutes'] <= 20, 'Listening_Time_minutes'].mean()=}")


print(f"{df_train_alt.loc[df_train_alt['Episode_Length_minutes'] >= 100, 'Listening_Time_Quota'].mean()=}")
print(f"{df_train_alt.loc[df_train_alt['Episode_Length_minutes'] >= 100, 'Listening_Time_minutes'].mean()=}")


(df_train_alt.loc[df_train_alt['Episode_Length_minutes'] >= 100, 'Listening_Time_Quota'] < 0.4).sum() / len(df_train_alt.loc[df_train_alt['Episode_Length_minutes'] >= 100])


(df_train.loc[df_train['Episode_Length_minutes'] >= 100, 'Listening_Time_minutes'] < 43).sum() / len(df_train.loc[df_train['Episode_Length_minutes'] >= 100])


def plot_categorical_vs_target(df: pd.DataFrame, target: str, categorical_feature: str, palette="Set2"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    sns.histplot(df, x=target, hue=categorical_feature, multiple="stack", palette=palette, ax=axes[0])
    sns.violinplot(data=df, x=target, y=categorical_feature, bw_adjust=.3, cut=0, ax=axes[1])
    plt.show()
    
    sns.displot(df, x=target, col=categorical_feature, stat='density')
    plt.show()

plot_categorical_vs_target(df_train.assign(Number_of_Ads_as_cat=df_train['Number_of_Ads'].astype(str)), target="Listening_Time_minutes", categorical_feature="Number_of_Ads_as_cat")


plot_categorical_vs_target(df_train, target="Listening_Time_minutes", categorical_feature="Publication_Time")


plot_categorical_vs_target(df_train, target="Listening_Time_minutes", categorical_feature="Episode_Sentiment")


plot_categorical_vs_target(df_train, target="Listening_Time_minutes", categorical_feature="Publication_Day")


plot_categorical_vs_target(df_train, target="Listening_Time_minutes", categorical_feature="Genre")


sns.catplot(data=df_train, x="Listening_Time_minutes", y="Podcast_Name", kind='violin', bw_adjust=.3, cut=0, height=20)
plt.show()


with pd.option_context("display.max_rows", 200):
    display(pd.concat([
        df_train.groupby('Episode_Title')['Listening_Time_minutes'].mean().rename('mean'),
        df_train.groupby('Episode_Title')['Listening_Time_minutes'].median().rename('median'),
        df_train.groupby('Episode_Title')['Listening_Time_minutes'].std().rename('std'),
        df_train.groupby('Episode_Title')['Listening_Time_minutes'].min().rename('min'),
        df_train.groupby('Episode_Title')['Listening_Time_minutes'].max().rename('max'),
    ], axis=1).sort_index())


df_train


weekday_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
df_pivot = df_train.pivot_table(index='Publication_Day', columns='Publication_Time', values='Listening_Time_minutes', aggfunc='mean').sort_index(key=lambda x: x.map(weekday_map))
df_pivot = df_pivot[['Morning', 'Afternoon', 'Evening', 'Night']]
sns.heatmap(df_pivot, annot=True, fmt=".1f", cmap="crest")
plt.show()


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


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


%%time
model = lgb.LGBMRegressor(**{"random_state": 42, "verbose": -1})
kf = KFold(n_splits=5, shuffle=True, random_state=42)

X = df_train.drop(columns=['Listening_Time_minutes'])
cat_cols = X.select_dtypes('object').columns
X[cat_cols] = X[cat_cols].astype('category')
y = df_train['Listening_Time_minutes']

y_pred_oof = pd.Series(index=X.index, name='pred')
for idx_train, idx_val in kf.split(X):
    X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
    y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]
    model.fit(X_train, y_train)
    y_pred_oof.loc[idx_val] = model.predict(X_val)

rmse = mean_squared_error(df_train['Listening_Time_minutes'], y_pred_oof, squared=False)
print(f'{rmse=}')


df_viz = pd.concat([df_train['Listening_Time_minutes'], 
                    y_pred_oof, 
                    (y_pred_oof - df_train['Listening_Time_minutes']).rename('res'), 
                    df_train['Episode_Length_minutes'].isna().rename('Episode_Length_minutes_isna')], 
                   axis=1)


sns.relplot(df_viz, x='Listening_Time_minutes', y='pred', height=10, hue='Episode_Length_minutes_isna')
plt.show()


sns.relplot(df_viz.iloc[:10000], x='Listening_Time_minutes', y='pred', height=10, hue='res', palette=sns.color_palette("icefire", as_cmap=True))
plt.show()


idx_isna = df_train['Episode_Length_minutes'].isna()

rmse_missing_Episode_Length_minutes = mean_squared_error(df_train.loc[idx_isna, 'Listening_Time_minutes'], y_pred_oof.loc[idx_isna], squared=False)
print(f'{rmse_missing_Episode_Length_minutes=}')

rmse_has_Episode_Length_minutes = mean_squared_error(df_train.loc[~idx_isna, 'Listening_Time_minutes'], y_pred_oof.loc[~idx_isna], squared=False)
print(f'{rmse_has_Episode_Length_minutes=}')

