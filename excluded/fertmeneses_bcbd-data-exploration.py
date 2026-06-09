import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================
# Explore leaderboard

def explore_LB(
    LB_file,target_col,cum_plot=True,
    bins=100,x_lim=None,figsize=(10,5)
    ):
    """
    Study the Leaderboard by making a histogram and (optionally) a cumulative plot
    """
    # Get data:
    LB_data = pd.read_csv(LB_file)
    # Make figure:
    fig, ax1 = plt.subplots(figsize=figsize)
    # Histogram:
    counts, bins, patches = ax1.hist(
        LB_data[target_col],bins=bins,edgecolor="navy",
        alpha=0.8,label='Histogram')
    ax1.set_title(f'Leaderboard {target_col} histogram')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel("Frequency")
    # Prepare handles for legend
    handles, labels = ax1.get_legend_handles_labels()
    # Cumulative (optional):
    if cum_plot:
        ax2 = ax1.twinx()
        cum_counts = counts.cumsum()
        cum_counts = cum_counts / cum_counts[-1]  # normalize to 1
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        line, = ax2.plot(
            bin_centers, cum_counts, color="orange", 
            linewidth=2, label="Cumulative")
        ax2.set_ylabel("Cumulative proportion")
        # Add cumulative to legend
        handles.append(line)
        labels.append("Cumulative")
    # Unified legend
    ax1.legend(handles, labels, loc="upper left")
    # Set x limits if required:
    if x_lim is not None:
        plt.xlim(x_lim)


LB_file = '/kaggle/input/leaderboard/playground-series-s5e8-publicleaderboard-2025-08-25.csv'
target_col = 'Score'
explore_LB(LB_file,target_col,bins=300,x_lim=[0.7,1])


def load_dataset(
    dataset_file,id_col=None,
    ):
    """
    Load a dataset and identify the id column, if any.
    The output is the dataset. If {id_col} is provided, then the output dataset
    and id column are provided as separate variables.
    """
    # Load original datasets:
    df = pd.read_csv(dataset_file)
    # Print some data examples:
    display(df.head())
    # Separate id column if required:
    if id_col is not None:
        ids = df[id_col]
        df = df.drop(columns=[id_col])
    print('_'*40+'\n\nInformation about dataset:\n')
    df.info()
    return df, ids if id_col is not None else df


# Get datasets:
train_file = '/kaggle/input/playground-series-s5e8/train.csv'
test_file = '/kaggle/input/playground-series-s5e8/test.csv'
id_col = 'id'
train_df, train_id = load_dataset(train_file,id_col=id_col)
test_df, test_id = load_dataset(test_file,id_col=id_col)


def plot_target_dist(
    df, target_col, pie_threshold=10, reg_bins=50
    ):
    """
    Plot the target distribution in a dataset.
    --- Inputs ---
    {df} [pandas.DataFrame]: Input dataframe.
    {target_col} [str]: Name of the target column.
    {pie_threshold} [int]: For a classification task, it defines the maximum number
    of classes for which the output will be a pie chart, else it will be a bar plot.
    {reg_bins} [int]: For regression, define the number of bins for the histogram.
    
    --- Output ---
    - If the target column is numeric and has more than {pie_threshold} unique values,
    then plot a histogram.
    - If the target column is categorical (or numeric with few unique values):
        - Pie chart if classes <= {pie_threshold}.
        - Horizontal bar plot if classes > {pie_threshold}.
    """
    # Identify target column:
    target = df[target_col]
    nunique = target.nunique()
    target_type = 'Numeric' if pd.api.types.is_numeric_dtype(target) else 'Object'
    
    # If target is numeric and have few unique values, plot histogram:
    if pd.api.types.is_numeric_dtype(target) and nunique > pie_threshold:
        plt.figure(figsize=(8, 5))
        plt.hist(target, bins=reg_bins, edgecolor="navy", alpha=0.7)
        plt.title(f"Histogram of {target_col} (Regression)")
        plt.xlabel(target_col)
        plt.ylabel("Frequency")
        plt.show()
    # Else, plot either a pie chart of horizontal bar plot:
    else:
        value_counts = target.value_counts().sort_values(ascending=False)
        if nunique <= pie_threshold:
            # Pie chart
            plt.figure(figsize=(6, 6))
            wedges, texts, autotexts = plt.pie(
                value_counts,labels=None,autopct="%1.1f%%",startangle=90)
            plt.title(f"Class distribution of {target_col} ({target_type})")
            plt.legend(wedges, value_counts.index,
                title="Classes",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
            plt.show()
        else:
            # Horizontal bar plot
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind="barh", color="steelblue", edgecolor="black")
            plt.title(f"Class distribution of {target_col} ({target_type})")
            plt.xlabel("Frequency")
            plt.ylabel("Classes")
            plt.gca().invert_yaxis()
            plt.show()


plot_target_dist(train_df,'y')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def _aligned_value_counts(
    s_train, s_test
    ):
    """
    Return aligned value_counts for train/test with a common index order.
    """
    # Get all categories in the combined datasets:
    cats = sorted(
        set(s_train.dropna().unique()).union(set(s_test.dropna().unique())))
    # Count unique values for each class:
    vc_train = s_train.value_counts().reindex(cats, fill_value=0)
    vc_test  = s_test.value_counts().reindex(cats, fill_value=0)
    return vc_train, vc_test

def _plot_categorical_donut(
    train_series, test_series, feature, ax, legend_loc=(1.02, 0.5)
    ):
    """
    Donut chart: outer ring=train, inner ring=test.
    """
    # Get all value counts and prepare labels:
    vc_train, vc_test = _aligned_value_counts(train_series, test_series)
    labels = vc_train.index.astype(str)
    # Colors
    colors = sns.color_palette('tab20', n_colors=max(3, len(labels)))
    # Outer ring (train)
    wedges_train, *_ = ax.pie(
        vc_train.values, radius=1.0, startangle=90, labels=None,
        wedgeprops=dict(width=0.25, edgecolor='white'), colors=colors
    )
    # Inner ring (test)
    wedges_test, *_ = ax.pie(
        vc_test.values, radius=0.7, startangle=90, labels=None,
        wedgeprops=dict(width=0.25, edgecolor='white'), colors=colors
    )
    # Hole
    ax.add_artist(plt.Circle((0, 0), 0.4, color='white', linewidth=0))
    ax.set_title(f'{feature}: Distribution (Train outer, Test inner)')
    # Legend with class labels
    ax.legend(
        wedges_train, labels, title="Classes",
        loc='center left', bbox_to_anchor=legend_loc
    )

def _plot_numeric_hist(
    train_series, test_series, feature, ax, reg_bins=50
    ):
    """
    Histogram plot: overlay train/test hist with shared bins.
    """
    # Drop NaN values and check if feature is empty:
    tr = train_series.dropna().astype(float)
    te = test_series.dropna().astype(float)
    if tr.empty and te.empty:
        ax.text(0.5, 0.5, "No numeric data", ha='center', va='center')
        ax.set_axis_off()
        return
    # Define boundaries:
    vmin = np.nanmin([tr.min() if not tr.empty else np.nan,
                      te.min() if not te.empty else np.nan])
    vmax = np.nanmax([tr.max() if not tr.empty else np.nan,
                      te.max() if not te.empty else np.nan])
    # Make bar plot if there was a problem with the boundaries:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Fallback: simple count bar if constant
        ax.bar(['Train','Test'], [tr.size, te.size], edgecolor='navy', alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title(f'{feature}: Constant/degenerate')
    # Make histogram:
    else:
        bins = np.linspace(vmin, vmax, reg_bins)
        tr.plot(kind='hist', bins=bins, density=True, alpha=0.6,
                edgecolor='navy', color='teal', ax=ax)
        te.plot(kind='hist', bins=bins, density=True, alpha=0.6,
                edgecolor='navy', color='orange', ax=ax, rwidth=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}: Distribution')
        ax.legend(['Train', 'Test'])
        ax.set_yticks([])
    # Additional configuration:
    for s in ["top","right","left", "bottom"]:
        ax.spines[s].set_visible(False)

def _is_binary(
    series
    ):
    """
    Determine whether the series is binary or not.
    """
    vals = pd.Series(series.dropna().unique())
    return len(vals) == 2

def _plot_kde_by_binary_ref(
    train_df, feature, ref_feature, ax
    ):
    """
    Stacked KDE by ref_feature (requires binary ref).
    """
    # Check if there are any incompatibilities:
    if ref_feature not in train_df.columns or not _is_binary(train_df[ref_feature]):
        ax.text(0.5, 0.5, f'No binary "{ref_feature}" to split', ha='center', va='center')
        ax.set_axis_off()
        return
    if not pd.api.types.is_numeric_dtype(train_df[feature]):
        ax.text(0.5, 0.5, "Non-numeric feature", ha='center', va='center')
        ax.set_axis_off()
        return
    # Make KDE plot:
    sns.kdeplot(
        x=feature, data=train_df.dropna(subset=[feature, ref_feature]),
        hue=ref_feature, multiple='stack', fill=True, cut=0,
        bw_method=0.15, lw=1.2, edgecolor='lightgray', ax=ax, palette='PuBu'
    )
    ax.set_title(f'{feature}: "{ref_feature}" density (Train)')
    ax.set_ylabel('Density')
    ax.set_yticks([])
    for s in ["top","right","left", "bottom"]:
        ax.spines[s].set_visible(False)

def _plot_swarm_like(
    train_df, feature, ref_feature, ax, seed=42
    ):
    """
    Swarm-like scatter per category showing rate of ref_feature==1.
    Requires categorical-like feature and binary ref.
    """
    # Check if there are any incompatibilities:
    if ref_feature not in train_df.columns or not _is_binary(train_df[ref_feature]):
        ax.text(0.5, 0.5, f'No binary "{ref_feature}" to split', ha='center', va='center')
        ax.set_axis_off()
        return
    # Set category order and check if data is empty:
    cats = sorted(x for x in train_df[feature].dropna().unique())
    if len(cats) == 0:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.set_axis_off()
        return
    # Prepare plot parameters:
    rng = np.random.default_rng(seed)
    palette = sns.color_palette('tab20', n_colors=max(3, len(cats)))
    # Make swarm plot:
    for i, cat in enumerate(cats):
        # Determine reference boundary for the plots:
        mask = train_df[feature] == cat
        sub = train_df.loc[mask, ref_feature].dropna().astype(int)
        if sub.empty:
            continue
        rate = sub.mean()
        n = sub.shape[0]
        n_pos = int(round(n * rate))
        n_neg = n - n_pos

        # Random vertical scatters in [0, rate] and [rate, 1]
        y_pos = rng.uniform(0, rate if rate > 0 else 1e-6, n_pos)
        y_neg = rng.uniform(rate, 1, n_neg)
        x_pos = i + rng.uniform(-0.28, 0.28, n_pos)
        x_neg = i + rng.uniform(-0.28, 0.28, n_neg)

        # Swarm plot:
        col = palette[i]
        ax.scatter(x_neg, y_neg, s=10, color=col, alpha=0.12, edgecolor=(0,0,0,0.15))
        ax.scatter(x_pos, y_pos, s=10, color=col, alpha=0.45, edgecolor=(0,0,0,0.15))
        ax.plot([i-0.3, i+0.3], [rate, rate], ls='--', color='k', lw=1)
    # Additional configuration:
    ax.set_xlim(-0.6, len(cats)-0.4)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([str(c)[0] for c in cats], rotation=0, ha='center')
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.set_title(f'{feature}: "{ref_feature}" rate (Train)')
    for s in ["top","right","left", "bottom"]:
        ax.spines[s].set_visible(False)

def _explain_stats(
    df_train, feature, ref_feature
    ):
    """
    Tabular summary when low cardinality.
    """
    # Get unique values information:
    unique_vals = sorted(v for v in df_train[feature].dropna().unique())
    df_expl = pd.DataFrame({
        feature: unique_vals,
        '#Rows': [int((df_train[feature] == v).sum()) for v in unique_vals],
    })
    # Calculate classes distribution related to the reference feature:
    if ref_feature in df_train.columns and _is_binary(df_train[ref_feature]):
        rates = []
        for v in unique_vals:
            sub = df_train.loc[df_train[feature] == v, ref_feature]
            rates.append(100.0 * (sub == 1).mean())
        df_expl[f'{ref_feature}_Rate[%]'] = np.round(rates, 2)
    print(df_expl)

def explore_features(
    train_df,test_df,ref_feature,
    features=None,max_unique_val=15,reg_bins=50
    ):
    """
    Explore features with appropriate plots decided internally.
    - Numeric: histogram (train vs test) + KDE by ref_feature (if binary).
    - Object/Categorical: donut pie (train outer, test inner) + swarm-like rate plot.
      If unique values in train > max_unique_val, skip and print the count.
    Also prints a small table when unique values (train) <= max_unique_val.
    --- Inputs ---
    {train_df} [pandas Dataframe]: training dataset.
    {test_df} [pandas Dataframe]: test dataset.
    {ref_feature} [string]: Reference (target) feature.
    {features} [list]: Features to be analyzed, must be present in both datasets.
    {max_unique_val} [int]: Threshold for maximum number of unique values, else the
    feature analysis will be skipped.
    {reg_bins} [int]: Number of bins for the histogram plots.
    """
    # Default feature list: columns present in both train and test, excluding ref_feature
    if features is None:
        features = [c for c in test_df.columns if c in train_df.columns and c != ref_feature]
    # Analyze each feature:
    for feature in features:
        s_train = train_df[feature]
        s_test  = test_df[feature]
        # Numerical feature:
        if pd.api.types.is_numeric_dtype(s_train):
            print('-' * 25, feature, '-' * 25)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
            _plot_numeric_hist(s_train, s_test, feature, ax1, reg_bins=reg_bins)
            _plot_kde_by_binary_ref(train_df, feature, ref_feature, ax2)
            fig.tight_layout()
            plt.show()
            # Table only if numeric but very low cardinality:
            nunique = s_train.dropna().nunique()
            if nunique <= max_unique_val:
                _explain_stats(train_df, feature, ref_feature=ref_feature)
            else:
                print('(Not suitable for a table)')
        # Object feature:
        else:
            # Count number of unique values:
            nunique = s_train.dropna().nunique()
            if nunique > max_unique_val:
                print(f'Skipping "{feature}": {nunique} unique values (> {max_unique_val}).')
                continue
            # Make plot:
            print('-' * 10, f'{feature} | Outer: train, Inner: test', '-' * 10)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
            _plot_categorical_donut(s_train, s_test, feature, ax1)
            _plot_swarm_like(train_df, feature, ref_feature, ax2)
            fig.tight_layout()
            plt.show()
            # Table only if low cardinality:
            if nunique <= max_unique_val:
                _explain_stats(train_df, feature, ref_feature=ref_feature)
            else:
                print('(Not suitable for a table)')


explore_features(train_df,test_df,'y',reg_bins=25)


import matplotlib.patches as patches

def encode_and_correlation(
    df,target_col,max_cats=10,
    c_thres=0.1,plot_corr=True
    ):
    """
    Encodes non-numeric features in a dataset and plots the correlation matrix.
    --- Inputs ---
    {df} [dataframe]: Dataframe to be encoded.
    {target_col} [string]: Target column, it won't be encoded.
    {max_cats} [int]: Maximum number of categories for a non-numeric
    feature to be encoded. If the feature has more categories, it is ignored.
    {c_thres} [float]: Threshold correlation value, only exceeding values
    will be painted in the correlation matrix.
    {plot_corr} [boolean]: If True, plots the correlation matrix. If False, 
    doesn't plot it, but still returns the encoded dataframe.
    --- Outputs ---
    {df_enc} [dataframe]: Encoded dataframe.
    """
    # ===== ENCODING ===== 
    # One-Hot encode non-numerical features with few unique values:
    feat_enc = [feat for feat in df.columns if 
                df[feat].dtype == 'object' and # Numeric or...
                df[feat].nunique()<=max_cats] # ...Few unique values

    # Encode features:
    if len(feat_enc)>1:
        data_enc = pd.get_dummies(df[feat_enc])
        # Prepare dataframes having only numeric/boolean features:
        df_enc = pd.concat([df, data_enc], axis=1)
    else:
        df_enc = df.copy()
    # Sort columns alphabetically, but leave 'y' at the beggining in training dataset:
    df_enc = df_enc.reindex(sorted(df_enc.columns), axis=1)
    if target_col in df_enc:
        df_enc = df_enc[
        [target_col] + [col for col in df_enc.columns if col != target_col]
        ]
    else:
        df_enc = df_enc[[col for col in df_enc.columns]]
    # Drop source-encoded features: 
    df_enc = df_enc.drop(feat_enc,axis=1)
    # Drop not suitable features:
    feat_not_enc = [feat for feat in df.columns if
                    df[feat].dtype == 'object' and # Numeric or...
                    df[feat].nunique()>10] # ...Few unique values
    df_enc = df_enc.drop(feat_not_enc,axis=1)
    
    # ===== CORRELATION MATRIX ===== 
    if plot_corr:
        # Prepare data:
        corr = df_enc.corr() # Obtain correlations
        trimask = np.triu(np.ones_like(corr, dtype=bool)) # Mask upper triangle in correlations    
        # Plot correlations in training dataset:
        fig, ax = plt.subplots(figsize=(20, 10))
        sns.heatmap(corr, ax=ax, square=True,cmap='coolwarm',
                    vmin=-1, vmax=1, linecolor='w',lw=0.5, 
                    mask=trimask | (np.abs(corr) <= c_thres))
        # Separate the target correlations:
        ax.plot([1,1],[1,len(df_enc.columns)],color='k',lw=2)
        # Draw the correlations' triangle:
        ax.plot([1,len(df_enc.columns)],[1,len(df_enc.columns)],color='k',lw=0.5)
        ax.plot([1,len(df_enc.columns)],[len(df_enc.columns),len(df_enc.columns)],
                color='k',lw=1.5)
        # Identify blocks of similar features:
        main_name = [feat.split('_')[0] for feat in list(df_enc.columns)]
        index_counts = [(main_name.index(name),main_name.count(name))
                        for name in sorted(set(main_name))]
        for index, counts in index_counts:
            if counts>1: 
                triangle = [[index, index],
                            [index+counts, index+counts],
                            [index, index+counts]]
                ax.add_patch(patches.Polygon(triangle, edgecolor="k",facecolor='none',ls="--",lw=2))
        ax.set_title(f"Dataset Correlations", size=15)
        fig.tight_layout()
        plt.show()    
    
    return df_enc


train_df_enc = encode_and_correlation(train_df,'y')


# Make copies of original datasets:
train_df_fe = train_df.copy()
test_df_fe = test_df.copy()


def bin_feature(
    orig_feature,datasets,bins,drop_original=False
    ):
    """
    Bin a feature into intervals.
    """
    for dataset in datasets:
        if orig_feature in dataset and f'{orig_feature}_range' not in dataset:
            intervals = pd.cut(dataset[orig_feature].astype(int), bins, include_lowest=True)
            dataset[f'{orig_feature}_range'] = pd.factorize(intervals, sort=True)[0]+1
            if drop_original:
                dataset.drop(columns=orig_feature, inplace=True)
    return datasets


# Split in 5-years-bins and generate 'Age_Range' new feature:
bins = np.concatenate(([0],np.arange(20, 61, 5),[120]))
train_df_fe,test_df_fe = bin_feature('age',[train_df_fe,test_df_fe],bins,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['age_range'])


def group_feature(
    orig_feature,datasets,mapping_group,drop_original=False
    ):
    """
    Groups a feature according to a predefined map.
    """
    for df in datasets:
        if orig_feature in df and f'{orig_feature}_group' not in df:
            # Apply mapping
            df[f'{orig_feature}_group'] = df[orig_feature].map(mapping_group)
            if drop_original:
                df.drop(columns=orig_feature, inplace=True)
    return datasets


job_map = {
    # Below 10% y rate
    'blue-collar': 1, 'entrepreneur': 1, 'housemaid': 1,
    'services': 1,
    # Below 20% y rate
    'admin.': 2, 'management': 2, 'self-employed': 2, 
    'technician': 2, 'unemployed': 2, 'unknown': 2,
    # Below 30% y rate
    'retired': 3, 
    # Above 30% y rate
    'student': 4, 
}
train_df_fe,test_df_fe = group_feature(
    'job',[train_df_fe,test_df_fe],job_map,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['job_group'])


def make_boolean_feature(
    orig_feature,ref_value,datasets,drop_original=False
    ):
    """
    Make a feature of boolean type, by checking if the value of the feature
    is equal to the reference value.
    """
    for df in datasets:
        if orig_feature in df and f"{orig_feature}_is_{ref_value}" not in df:
            df[f"{orig_feature}_is_{ref_value}"] = (df[orig_feature] == ref_value).astype(int)
            if drop_original:
                df.drop(columns=orig_feature, inplace=True)
    return datasets


train_df_fe,test_df_fe = make_boolean_feature(
    'marital','single',[train_df_fe,test_df_fe],drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['marital_is_single'])


education_map = {
    'primary': 1,
    'secondary': 2,
    'tertiary': 3,
    'unknown': 2.5
}
train_df_fe,test_df_fe = group_feature(
    'education',[train_df_fe,test_df_fe],education_map,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['education_group'])


default_map = {
    'no': 1,
    'yes': 2
}
train_df_fe,test_df_fe = group_feature(
    'default',[train_df_fe,test_df_fe],default_map,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['default_group'])


def make_num_boundary_boolean(
    orig_feature,boundary_value,datasets,drop_original=False
    ):
    """
    Make a numeric feature boolean based on the a numerical boundary (smaller or greater).
    """
    # Set extension name:
    extension = '_is_positive' if boundary_value == 0 else '_above_boundary'
    for df in datasets:
        if orig_feature in df and f"{orig_feature}{extension}" not in df:
            df[f"{orig_feature}{extension}"] = (df[orig_feature] >= boundary_value).astype(int)
            if drop_original:
                df.drop(columns=orig_feature, inplace=True)
    return datasets


make_num_boundary_boolean(
    'balance',0,[train_df_fe,test_df_fe],drop_original=False)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['balance_is_positive'])


def make_log_feature(
    orig_feature,datasets,drop_original=False
    ):
    """
    Transform a numeric feature into its logarithmic version. All zero or negative values
    are assigned as -1.
    """
    for df in datasets:
        if orig_feature in df and f"{orig_feature}_log" not in df:
            df[f"{orig_feature}_log"] = -1.0  # initialize with -1
            df.loc[df[orig_feature] > 0, f"{orig_feature}_log"] = np.log(
                df.loc[df[orig_feature] > 0, orig_feature])
            if drop_original:
                df.drop(columns=orig_feature, inplace=True)
    return datasets


train_df_fe,test_df_fe = make_log_feature(
    'balance',[train_df_fe,test_df_fe],drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['balance_log'])


train_df_fe,test_df_fe = make_boolean_feature(
    'contact','unknown',[train_df_fe,test_df_fe],drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['contact_is_unknown'])


explore_features(train_df_fe,test_df_fe,'y',reg_bins=31,features=['day'],max_unique_val=40)


day_map = {
    # Below 10% y rate
    6:1,7:1,17:1,19:1,20:1,21:1,28:1,29:1,31:1,
    # Below 20% y rate
    2:2,4:2,5:2,8:2,9:2,11:2,12:2,13:2,14:2,15:2,16:2,18:2,22:2,23:2,24:2,25:2,26:2,27:2,
    # Below 30% y rate
    3:3,10:3,30:3,
    # Above 30% y rate
    1: 4, 
}
train_df_fe,test_df_fe = group_feature(
    'day',[train_df_fe,test_df_fe],day_map,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['day_group'])


month_map = {
    # Close to 10% y rate
    'jan':1, 'may':1, 'jun':1, 'jul':1, 'aug':1, 'nov':1,
    # Close to 20% y rate
    'feb':2, 'apr':2, 
    # Close to 50% y rate
    'mar':3, 'sep':3, 'oct':3, 'dec':3,
}
train_df_fe,test_df_fe = group_feature(
    'month',[train_df_fe,test_df_fe],month_map,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=12,features=['month_group'])


def limit_min_max_values(
    orig_feature,datasets,min_bound=None,max_bound=None,drop_original=False
    ):
    """
    Limits the value of the feature within the min and max boundaries, if provided.
    Any value outside the boundaries is reassigned as the closest boundary value.
    """
    # Sanity check for boundary conditions:
    if (min_bound is not None) and (max_bound is not None) and (min_bound >= max_bound):
        print('Error: min_bound must be lower than max_bound if both are provided.')
        return datasets
        
    for df in datasets:
        if orig_feature in df and f"{orig_feature}_bounded" not in df:
            # Clip with whichever bounds are provided (None is okay)
            df[f"{orig_feature}_bounded"] = df[orig_feature].clip(
                lower=min_bound, upper=max_bound)
            if drop_original:
                df.drop(columns=orig_feature, inplace=True)
    return datasets


train_df_fe,test_df_fe = limit_min_max_values(
    'duration',[train_df_fe,test_df_fe],max_bound=1000,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['duration_bounded'])


# Explore the y positive rate in small bins:
bins = np.arange(0, 76, 5)
train_df_fe,test_df_fe = bin_feature('campaign',[train_df_fe,test_df_fe],bins,drop_original=False)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['campaign_range'])
for df in [train_df_fe,test_df_fe]:
    df.drop(columns='campaign_range', inplace=True)


train_df_fe,test_df_fe = make_num_boundary_boolean(
    'campaign',5,[train_df_fe,test_df_fe],drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['campaign_above_boundary'])


# Explore the y positive rate in small bins:
bins = np.arange(0, 1000, 50)
train_df_fe,test_df_fe = bin_feature('pdays',[train_df_fe,test_df_fe],bins,drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['pdays_range'],max_unique_val=50)


for df in [train_df_fe,test_df_fe]:
    if 'previous' in df:
        df.drop(columns='previous', inplace=True)


train_df_fe,test_df_fe = make_boolean_feature(
    'poutcome','success',[train_df_fe,test_df_fe],drop_original=True)
explore_features(train_df_fe,test_df_fe,'y',reg_bins=25,features=['poutcome_is_success'])


# Select datasets:
train_df_sel = train_df_fe
test_df_sel = test_df_fe

# Determine column types:
col_numeric = [col for col in test_df_sel.columns if 
               test_df_sel[col].dtypes != 'object']
col_object = [col for col in test_df_sel.columns if col not in col_numeric]
for dataset in [train_df_sel,test_df_sel]:
    # Correct numeric columns:
    for col in col_numeric:
        dataset[col] = dataset[col].fillna(value=dataset[col].mean())
    # Correct non-numeric columns:
    for col in col_object:
        most_freq = dataset[col].value_counts(sort=True).index[0]
        dataset[col] = dataset[col].fillna(value=most_freq)
        
print('-'*10,'Numeric columns:','-'*10)
for col in col_numeric:
    print(f'{col}: {test_df_sel[col].dtypes}')
print('-'*10,'Non-numeric columns:','-'*10)
for col in col_object:
    print(f'{col}: {test_df_sel[col].dtypes}; {test_df_sel[col].nunique()} unique values')


# Start encoded datasets:
train_df_enc, test_df_enc = train_df_sel.copy(), test_df_sel.copy()
# OneHot-encode columns:
enc_cols = col_object
enc_train = pd.get_dummies(train_df_enc[enc_cols])
enc_test = pd.get_dummies(test_df_enc[enc_cols])
train_df_enc = pd.concat([train_df_enc, enc_train], axis=1)
test_df_enc = pd.concat([test_df_enc, enc_test], axis=1)
# Drop columns:
drop_cols = enc_cols
train_df_enc = train_df_enc.drop(drop_cols,axis=1)
test_df_enc = test_df_enc.drop(drop_cols,axis=1)


# train_df_enc


# Define models:
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

def train_submit_ML(model,train_df,test_df,test_id,plot_roc=True):
    """
    Trains a ML model, and saves the submission file.
    --- Inputs ---
    {model} [Machine Learning model]: ML model to be trained.
    {train_df} [Dataframe]: Training dataset, it must contain the
    columns "PassengerId" and "y".
    {test_df} [Dataframe]: Testing dataset, it must contain the
    column "PassengerId".
    --- Outputs ---
    {train_score} [Float]: Training score.
    """
    # Extract PassengerId and y columns, if applicable:
    y_train = train_df['y']
    x_train = train_df.drop(['y'],axis=1)
    x_test = test_df
    # Set seed if supported
    try:
        if 'random_state' in model.get_params():
            model.set_params(random_state=0)
    except Exception:
        pass

       # Ensure we can produce probabilities
    if not hasattr(model, "predict_proba"):
        if hasattr(model, "decision_function"):
            # Calibrate to get probabilities
            model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        else:
            raise ValueError("Model lacks predict_proba and decision_function; cannot produce probabilities.")

    # Train the model:
    model.fit(x_train,y_train)

    # Train probabilities and AUC
    y_train_pred = model.predict_proba(x_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_pred)

    # ROC curve (training)
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_train, y_train_pred)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC = {train_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Training ROC Curve')
        plt.legend()
        plt.tight_layout()
        
    # Get probabilities for the positive class
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    
   # Build and save submission
    subm = pd.DataFrame({
        "id": np.asarray(test_id),   # keep order aligned with test_df
        "y": y_pred_proba
    })
    subm.to_csv('submission.csv', index=False)
    print(f"Submission file saved to 'submission.csv' with {len(subm)} rows.")
    print(f"Training ROC AUC: {train_auc:.4f}")

    return y_pred_proba, train_auc


# Options:
# RandomForestClassifier(n_estimators=100, max_depth=5)
# LogisticRegression()
# GradientBoostingClassifier(n_estimators=100, max_depth=3)
# XGBClassifier(n_estimators=100, max_depth=3)
# KNeighborsClassifier(n_neighbors=5)
# LinearSVC(dual=False)

# Define and train simple model:
model = XGBClassifier(n_estimators=150, max_depth=3)
y_pred_proba, train_auc = train_submit_ML(model,train_df_enc,test_df_enc,test_id)




