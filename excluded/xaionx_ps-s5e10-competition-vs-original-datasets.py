# seaborn v0.13.2 and up required
# !pip install seaborn --upgrade


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path
import random
from tabulate import tabulate, SEPARATING_LINE
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
orig_color = sns.color_palette()[0] # color used for original dataset
comp_color = sns.color_palette()[1] # color used for competition dataset
palette = {"Original": orig_color, "Competition": comp_color}

def print_frequency_table(original, competition, curr_feature):
    od_f = original[curr_feature].value_counts(normalize=False).sort_index().to_dict()
    cd_f = competition[curr_feature].value_counts(normalize=False).sort_index().to_dict()
    od_fn = original[curr_feature].value_counts(normalize=True).sort_index().to_dict()
    cd_fn = competition[curr_feature].value_counts(normalize=True).sort_index().to_dict()
    values = sorted(list(set([*od_f.keys(), *cd_f.keys()])))
    table = [[value, f"{od_f[value]} (~{od_fn[value]:.2%})" if value in od_f else "0 (0.00%)", f"{cd_f[value]} (~{cd_fn[value]:.2%})" if value in cd_f else "0 (0.00%)"] for value in values]
    print(f"\nFrequencies & Proportions:")
    print(tabulate(table, headers=["Value", "Original", "Competition"], tablefmt="psql", disable_numparse=True))

def print_stats_table(original, competition, curr_feature):
    od_s = original[curr_feature].describe().to_dict()
    od_s["IQR"] = od_s["75%"] - od_s["25%"]
    cd_s = competition[curr_feature].describe().to_dict()
    cd_s["IQR"] = cd_s["75%"] - cd_s["25%"]
    stats_to_show = ["mean", "std", "min", "25%", "50%", "75%", "max", "IQR"]
    table = [[stat, round(od_s[stat], 3), round(cd_s[stat], 3)] for stat in stats_to_show]
    print(f"\n{curr_feature} Stats:")
    print(tabulate(table, headers=["", "Original", "Competition"], tablefmt="psql", disable_numparse=True))

def print_by_target_table(original, competition, curr_feature, target_col):
    od_v = original[curr_feature].unique()
    cd_v = competition[curr_feature].unique()
    values = sorted(list(set([*od_v, *cd_v])))
    table = []
    for value in values:
        if (len(table) > 0): table.append(["","","",""])
        od_s = original[original[curr_feature] == value][target_col].describe().to_dict()
        od_s["IQR"] = od_s["75%"] - od_s["25%"]
        cd_s = competition[competition[curr_feature] == value][target_col].describe().to_dict()
        cd_s["IQR"] = cd_s["75%"] - cd_s["25%"]
        stats_to_show = ["mean", "std", "min", "25%", "50%", "75%", "max", "IQR"]
        for (i, stat) in enumerate(stats_to_show): table.append([value if i == 0 else "", stat, round(od_s[stat], 4), round(cd_s[stat], 4)])
    print(f"\nTarget ({target_col}) by {curr_feature}:")
    print(tabulate(table, headers=["Value", "", "Original", "Competition"], tablefmt="psql", disable_numparse=True))

def print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col):
    def cliffs_delta_from_u(u_stat, n1, n2):
        return (2 * u_stat) / (n1 * n2) - 1

    def rank_biserial(u_stat, n1, n2):
        return 1 - (2 * u_stat) / (n1 * n2)

    def interpret_cliffs(delta):
        ad = abs(delta)
        if ad < 0.147:
            return "negligible"
        elif ad < 0.33:
            return "small"
        elif ad < 0.474:
            return "medium"
        else:
            return "large"
    
    results = []

    # --- Mann-Whitney + Effect Sizes per curr_feature ---
    print(f"\nPer {curr_feature} Comparison (Original vs Competition):")
    
    for value in sorted(combined[curr_feature].unique()):
        orig_vals = combined[(combined[curr_feature] == value) & (combined["source"] == "Original")][target_col].values
        comp_vals = combined[(combined[curr_feature] == value) & (combined["source"] == "Competition")][target_col].values
        if len(orig_vals) > 0 and len(comp_vals) > 0:
            u_stat, p = stats.mannwhitneyu(orig_vals, comp_vals, alternative="two-sided")
            p = np.format_float_scientific(p, 3)
            delta = round(cliffs_delta_from_u(u_stat, len(orig_vals), len(comp_vals)), 4)
            r_rb = rank_biserial(u_stat, len(orig_vals), len(comp_vals))
            interpretation = interpret_cliffs(delta)
        else:
            print(f"Skipping {curr_feature}='{value}' — missing in {'Competition' if len(comp_vals) == 0 else 'Original'} dataset.")
            u_stat = p = delta = r_rb = interpretation = "-"
        results.append([value, u_stat, p, delta, interpretation])

    # cols = [curr_feature, "U-stat", "p-value", "Cliffs_Delta", "Interpretation", "Rank-Biserial"]
    cols = [curr_feature, "U-stat", "p-value", "Cliffs_Delta", "Interpretation"]
    print(tabulate(results, headers=cols, tablefmt="psql", disable_numparse=True))

def print_kruskal_wallis_test(combined, curr_feature, target_col):
    # --- Kruskal-Wallis for Original ---
    groups_orig = [
        combined[(combined[curr_feature] == value) & (combined["source"] == "Original")][target_col]
        for value in combined[curr_feature].unique()
    ]
    groups_orig = [g for g in groups_orig if len(g) > 0]

    if len(groups_orig) > 1:
        stat_orig, p_orig = stats.kruskal(*groups_orig)
    else:
        stat_orig, p_orig = float('nan'), float('nan')

    # --- Kruskal-Wallis for Competition ---
    groups_comp = [
        combined[(combined[curr_feature] == value) & (combined["source"] == "Competition")][target_col]
        for value in combined[curr_feature].unique()
    ]
    groups_comp = [g for g in groups_comp if len(g) > 0]

    if len(groups_comp) > 1:
        stat_comp, p_comp = stats.kruskal(*groups_comp)
    else:
        stat_comp, p_comp = float('nan'), float('nan')

    print(f"\nKruskal-Wallis Test Across {curr_feature}:")
    kw_table = [
        ["Original", stat_orig, p_orig],
        ["Competition", stat_comp, p_comp]
    ]
    print(tabulate(kw_table, headers=["Dataset", "H-stat", "p-value"], tablefmt="psql", disable_numparse=True))

def js_divergence(p, q):
    p = np.array(p) / np.sum(p)
    q = np.array(q) / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5*stats.entropy(p, m, base=2) + 0.5*stats.entropy(q, m, base=2)

def print_js_divergence(original, competition, curr_feature):
    od_v = original[curr_feature].unique()
    cd_v = competition[curr_feature].unique()
    values = sorted(list(set([*od_v, *cd_v])))
    p = original[curr_feature].value_counts(normalize=True).reindex(values, fill_value=0)
    q = competition[curr_feature].value_counts(normalize=True).reindex(values, fill_value=0)
    print(f"\nJensen-Shannon Divergence = {js_divergence(p,q):.3f}")

def print_js_divergence_num(original, competition, curr_feature):
    # js divergence for binned numerical feature
    hist_range = (min(original[curr_feature].min(), competition[curr_feature].min()), 
                  max(original[curr_feature].max(), competition[curr_feature].max()))
    p_hist, bins = np.histogram(original[curr_feature], bins=50, range=hist_range)
    q_hist, _ = np.histogram(competition[curr_feature], bins=50, range=hist_range)
    p = p_hist / p_hist.sum()
    q = q_hist / q_hist.sum()
    print(f"\nJensen-Shannon Divergence = {js_divergence(p,q):.3f}")

def print_wasserstein_distance(original, competition, curr_feature):
    w_dist = stats.wasserstein_distance(original[curr_feature], competition[curr_feature])
    print(f"Wasserstein distance = {w_dist:.4f}")

def print_correlations_with_target(original, competition, curr_feature, target_col):
    pearson1, _ = stats.pearsonr(original[curr_feature], original[target_col])
    spearman1, _ = stats.spearmanr(original[curr_feature], original[target_col])
    pearson2, _ = stats.pearsonr(competition[curr_feature], competition[target_col])
    spearman2, _ = stats.spearmanr(competition[curr_feature], competition[target_col])
    print("\n=== Correlations with target ===")
    print(f"Original: Pearson={pearson1:.3f}, Spearman={spearman1:.3f}")
    print(f"Competition: Pearson={pearson2:.3f}, Spearman={spearman2:.3f}")

def print_ks_test(original, competition, curr_feature):
    stat, p = stats.ks_2samp(original[curr_feature], competition[curr_feature])
    print(f"\nKolmogorov-Smirnov Test: stat={stat:.3f}, p={p:.3e}")

def print_target_stats(original, competition, target_col):
    cd = competition[target_col].describe().to_dict()
    cd["normal_stat"], cd["normal_p"] = stats.normaltest(competition[target_col])
    cd["interpretation"] = "Not Normal (p < 0.05)" if cd["normal_p"] < 0.05 else "-"

    od = original[target_col].describe().to_dict()
    od["normal_stat"], od["normal_p"] = stats.normaltest(original[target_col])
    od["interpretation"] = "Not Normal (p < 0.05)" if od["normal_p"] < 0.05 else "-"

    print(f"Target variable: {target_col} overview:\n")
    stats_to_show = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    table = [[stat, od[stat], cd[stat]] for stat in stats_to_show]
    print(tabulate(table, headers=["", "Original", "Competition"], floatfmt=".4f"))
    
    print("\nD’Agostino and Pearson’s Test For Normality")
    stats_to_show = ["normal_stat", "normal_p", "interpretation"]
    table = [[stat, od[stat], cd[stat]] for stat in stats_to_show]
    print(tabulate(table, headers=["", "Original", "Competition"], floatfmt=".4f"))

def plot_feature_distribution_cat(combined, curr_feature, target_col):
    # this function accepts true categorical variables as well as ordinal numeric features
    # Make the feature categorical
    combined_temp = combined.copy()
    combined_temp[curr_feature] = combined_temp[curr_feature].astype(str)
    
    # Try to sort numerically first, fall back to alphabetical
    unique_vals = combined_temp[curr_feature].unique()
    try:
        categories = sorted(unique_vals, key=float)
    except (ValueError, TypeError):
        categories = sorted(unique_vals)
    
    combined_temp[curr_feature] = pd.Categorical(
        combined_temp[curr_feature], 
        categories=categories,
        ordered=True
    )
    
    fig = plt.figure(figsize=(8, 5))
    
    ax = sns.histplot(
        data=combined_temp,
        x=curr_feature,
        hue="source",
        multiple="dodge",
        shrink=0.8,
        stat="percent",
        common_norm=False,
        discrete=True
    )
    plt.title(f"{curr_feature} Distribution (Proportions)")
    plt.xlabel(curr_feature)
    plt.ylabel("Proportion (%)")
    
    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.075), ncol=2, title="Dataset")
    
    plt.tight_layout()
    plt.show()

def plot_feature_vs_target_cat(combined, curr_feature, target_col):
    # this function accepts true categorical variables as well as ordinal numeric features
    # Make the feature categorical
    combined_temp = combined.copy()
    combined_temp[curr_feature] = combined_temp[curr_feature].astype(str)
    
    # Try to sort numerically first, fall back to alphabetical
    unique_vals = combined_temp[curr_feature].unique()
    try:
        categories = sorted(unique_vals, key=float)
    except (ValueError, TypeError):
        categories = sorted(unique_vals)
    
    combined_temp[curr_feature] = pd.Categorical(
        combined_temp[curr_feature], 
        categories=categories,
        ordered=True
    )

    plt.figure(figsize=(8, 5))

    # Plot target_col by curr_feature
    ax = sns.boxplot(
        data=combined_temp,
        x=curr_feature,
        y=target_col,
        hue="source",
        gap=0.1,
        order=combined_temp[curr_feature].cat.categories
    )
    plt.title(f"{target_col} by {curr_feature}")
    plt.xlabel(curr_feature)
    plt.ylabel(target_col)

    sns.move_legend(ax, "lower center", bbox_to_anchor=(0.5, 1.075), ncol=2, title="Dataset")
    
    plt.tight_layout()
    plt.show()

def plot_feature_distribution_num(combined, curr_feature):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.kdeplot(combined, x = curr_feature, hue="source", common_norm = False, fill=True, alpha=0.4, ax=axes[0], legend=False, palette=palette)
    sns.boxplot(combined, x = curr_feature, hue="source", gap=0.1, ax=axes[1], palette=palette)
    
    # Single combined legend
    axes[1].get_legend().remove()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(labels))
    fig.suptitle(f"{curr_feature} Distribution")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.show()

def plot_feature_vs_target_num(combined, curr_feature, target_col):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    sns.scatterplot(combined.query('source == "Original"'), x = curr_feature, y = target_col, ax=axes[0], legend=False, color=orig_color, alpha=0.2)
    sns.scatterplot(combined.query('source == "Competition"'), x = curr_feature, y = target_col, ax=axes[1], legend=False, color=comp_color, alpha=0.2)
    axes[0].set_title(f"{target_col} by {curr_feature} (Original)")
    axes[1].set_title(f"{target_col} by {curr_feature} (Competition)")
    combined_temp = combined.copy()
    combined_temp["bin"] = pd.qcut(combined_temp[curr_feature], q=10, duplicates="drop")
    sns.boxplot(x="bin", y=target_col, data=combined_temp, hue="source", palette=palette, gap=0.2, ax=axes[2])
    axes[2].set_title(f"{target_col} distribution by binned {curr_feature}")
    axes[2].set_xlabel(f"binned {curr_feature}")
    axes[2].tick_params(axis="x", rotation=60)

    # Single combined legend
    axes[2].get_legend().remove()
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, title="Dataset", loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(labels))
    fig.suptitle(f"{curr_feature} vs {target_col}")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.show()

def plot_target(original, competition, target_col):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns

    # --- Distribution plot ---
    sns.kdeplot(original[target_col], label='Original', fill=True, alpha=0.4,
                ax=axes[0], color=orig_color)
    sns.kdeplot(competition[target_col], label='Competition', fill=True, alpha=0.4,
                ax=axes[0], color=comp_color)
    axes[0].set_title(f"Distribution: {target_col}")
    axes[0].legend()

    # --- Q-Q plot for Original ---
    stats.probplot(original[target_col], dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(orig_color)  # sample points
    axes[1].get_lines()[1].set_color(orig_color)  # fit line
    axes[1].set_title("Q-Q Plot: Original")

    # --- Q-Q plot for Competition ---
    stats.probplot(competition[target_col], dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_color(comp_color)  # sample points
    axes[2].get_lines()[1].set_color(comp_color)  # fit line
    axes[2].set_title("Q-Q Plot: Competition")

    plt.tight_layout()
    plt.show()

def generate_synthetic_data(num_rows=10000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = {
        "road_type" : np.random.choice(["highway", "urban","rural"], num_rows),
        "num_lanes" : np.random.randint(1, 5, num_rows),
        "curvature" : np.round(np.random.uniform(0.0, 1.0, num_rows), 2),
        "speed_limit" : np.random.choice([25, 35, 45, 60, 70],num_rows),
        "lighting" : np.random.choice(["daylight", "night", "dim"], num_rows),
        "weather" : np.random.choice(["clear", "rainy", "foggy"], num_rows),
        "road_signs_present" : np.random.choice([True, False], num_rows),
        "public_road" : np.random.choice([True, False], num_rows),
        "time_of_day" : np.random.choice(["morning", "evening", "afternoon"], num_rows),
        "holiday" : np.random.choice([True, False], num_rows),
        "school_season" : np.random.choice([True, False], num_rows),
        "num_reported_accidents" : np.random.poisson(lam=1.5, size=num_rows)
    }

    # simulate risk score  influenced by  features + noise
    base_risk = (
        0.3 * data["curvature"] + 
        0.2 * (data["lighting"] == "night").astype(int) + 
        0.1 * (data["weather"] != "clear").astype(int) + 
        0.2 * (data["speed_limit"] >= 60).astype(int) + 
        0.1 * (np.array(data["num_reported_accidents"]) > 2).astype(int)
    )

    # add noise and clip to  [0,1]

    noise = np.random.normal(0, 0.05, num_rows)
    risk_score = np.clip(base_risk + noise, 0, 1)
    data["accident_risk"] = np.round(risk_score, 2)

    return pd.DataFrame(data)


# load the competition dataset (train)
competition = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv", index_col="id")

# create the original dataset with the same number of entries as the competition dataset (train)
original = generate_synthetic_data(num_rows=len(competition), seed=42)

# create a combined dataset with a label column (source)
original_tmp = original.copy()
original_tmp['source'] = "Original"
competition_tmp = competition.copy()
competition_tmp['source'] = "Competition"
combined = pd.concat([original_tmp, competition_tmp], ignore_index=True)


target_col = "accident_risk"

print_target_stats(original, competition, target_col)
print("\n")
plot_target(original, competition, target_col)


features = [x for x in competition.columns.tolist() if x != target_col]

table = []
for feature in features:
    dtype = competition[feature].dtype
    kind = dtype.kind
    table.append([feature, dtype, "Numeric Discrete" if kind in "iu" else ("Numeric Continuous" if kind == "f" else "Categorical")])

print("Features Overview:\n")
print(tabulate(table, headers=["Feature", "dtype", "Type"]))


curr_feature = "road_type"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "road_type"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "num_lanes"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "num_lanes"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "curvature"

plot_feature_distribution_num(combined, curr_feature)
print_stats_table(original, competition, curr_feature)
print_ks_test(original, competition, curr_feature)
print_js_divergence_num(original, competition, curr_feature)
print_wasserstein_distance(original, competition, curr_feature)


curr_feature = "curvature"

plot_feature_vs_target_num(combined, curr_feature, target_col)
print_correlations_with_target(original, competition, curr_feature, target_col)


curr_feature = "speed_limit"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "speed_limit"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "lighting"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "lighting"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "weather"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "weather"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "road_signs_present"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "road_signs_present"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "public_road"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "public_road"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "time_of_day"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "time_of_day"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "holiday"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "holiday"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "school_season"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "school_season"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)


curr_feature = "num_reported_accidents"

plot_feature_distribution_cat(combined, curr_feature, target_col)
print_frequency_table(original, competition, curr_feature)
print_js_divergence(original, competition, curr_feature)


curr_feature = "num_reported_accidents"

plot_feature_vs_target_cat(combined, curr_feature, target_col)
print_by_target_table(original, competition, curr_feature, target_col)
print_mann_whitney_u_test_and_effect_size(combined, curr_feature, target_col)
print_kruskal_wallis_test(combined, curr_feature, target_col)

