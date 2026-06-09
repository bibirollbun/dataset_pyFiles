# Basic imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Disable warning
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option("display.max_columns", 500)


!pip install /kaggle/input/pip-install-libaries/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-libaries/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-libaries/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-libaries/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/pip-install-libaries/lifelines-0.30.0-py3-none-any.whl

!pip install /kaggle/input/pip-install-libaries/ecos-2.0.14-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/pip-install-libaries/scikit_learn-1.5.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/pip-install-libaries/scikit_survival-0.23.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# Import data
df_train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
df_test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

df_train.head()


df_target = df_train[["efs", "efs_time"]].copy() # copy the target data in a seperate data frame

# Plot the target distribution
def plot_distribution(x, hue, color="husl", title=None, xlabel="EFS time (months)", ax=None):

    if ax is None:  # Create a standalone figure is no axes is provided 
        fig, ax = plt.subplots(figsize=(6, 4)) 

    # Get unique categories 
    unique_categories = sorted(hue.unique()) 
    palette = sns.color_palette(color, n_colors=len(unique_categories))  

    # Plot histogram with custom palette
    sns.histplot(ax=ax, x=x, hue=hue, edgecolor=None, alpha=0.7, palette=palette)

    ax.set_xlabel(xlabel, fontsize=10) 
    ax.set_ylabel("Count", fontsize=10)
    if title:
        ax.set_title(title, fontsize=12)
    

plot_distribution(df_target["efs_time"], df_target["efs"])
plt.show()


# Functions for target transformation

from lifelines import KaplanMeierFitter
from lifelines import  NelsonAalenFitter
from sklearn.preprocessing import quantile_transform # See https://www.kaggle.com/code/ambrosm/esp-eda-which-makes-sense


def create_target_COX(data):
    
    data_copy = data.copy()
    data_copy["efs_time2"] = data_copy.efs_time.copy()
    data_copy.loc[data_copy.efs==0,"efs_time2"] *= -1 # Negitated censored data

    return data_copy["efs_time2"]


def create_target_KMF(data):        

    kmf = KaplanMeierFitter()
    kmf.fit(durations=data["efs_time"], event_observed=data["efs"])

    target = kmf.survival_function_at_times(data["efs_time"]).values
    #target[data.efs==1] += 0.3

    return target


def create_target_nelson(data):
    
    naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
    naf.fit(durations=data["efs_time"], event_observed=data["efs"])
    target = naf.cumulative_hazard_at_times(data["efs_time"]).values * -1
    
    return target 


def create_quantile_transform(data):
    time = data["efs_time"]
    event = data["efs"]
    
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(- time[event == 1].values.reshape(-1, 1)).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - 0.3
    
    return transformed


# Create and visulalize target distributions

target_cox = create_target_COX(df_target) # Create target cox
target_kmf = create_target_KMF(df_target) # Create target kaplan-meier
target_nelson = create_target_nelson(df_target) # Create target nelson
target_quant = create_quantile_transform(df_target) # Create target quantile transform


fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey=True) 
axs = axs.flatten()

plot_distribution(target_cox, df_target["efs"], title="Cox", xlabel="Transformed target", ax=axs[0]) # Plot the distribution of the transformed target
plot_distribution(target_kmf, df_target["efs"], title="Kaplan-Meier", xlabel="Transformed target", ax=axs[1])
plot_distribution(target_nelson, df_target["efs"], title="Nelson-Aalen", xlabel="Transformed target", ax=axs[2])
plot_distribution(target_quant, df_target["efs"], title="Quantile transform", xlabel="Transformed target", ax=axs[3])

# Adjust layout for better spacing
fig.suptitle("Target transformations", fontsize=16, fontweight="bold")  # Set overall title
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show()


# Define plotting functions

def plot_barplot(x, data, title=None, ax=None, hue=None, normalize=False, legend_outside=False, tilt_xlabels=False, color="husl"):
    if ax is None:  # Create a standalone figure if no axes are provided 
        fig, ax = plt.subplots(figsize=(6, 4))
    data = data.copy()  # Avoid modifying the original DataFrame
    data[x] = pd.Categorical(data[x], categories=sorted(data[x].dropna().unique()), ordered=True)
    if hue:
        data[hue] = pd.Categorical(data[hue], categories=sorted(data[hue].dropna().unique()), ordered=True)
        palette = sns.color_palette(color, n_colors=data[hue].nunique())  # Get colors from palette
    else:
        palette = sns.color_palette(color, n_colors=data[x].nunique()) 
    if normalize and hue:
        # Compute value counts for each category and normalize within each hue group
        norm_data = data.groupby([x, hue]).size().reset_index(name="count")
        norm_data["percentage"] = norm_data.groupby(x)["count"].transform(lambda x: x / x.sum())
        sns.barplot(x=x, y="percentage", hue=hue, data=norm_data, ax=ax, palette=palette,  edgecolor="white")
        ax.set_ylabel("Proportion")
    else:
        sns.countplot(x=x, hue=hue, data=data, ax=ax, palette=palette, edgecolor="white")
    if title:
        ax.set_title(title)
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    if tilt_xlabels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


import textwrap
def plot_stacked_countplot(df, x, hue, title=None, color="husl", ax=None):
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Get unique categories 
    unique_categories = sorted(df[hue].dropna().unique()) 
    palette = sns.color_palette(color, n_colors=len(unique_categories))  
    
    sns.countplot(x=x, hue=hue,  data=df_train, ax=ax, dodge=False, palette=palette)

    if title:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_ylim(0, 5500)
    
    # Automatically wrap x-tick labels
    wrapped_labels = [textwrap.fill(label.get_text(), width=10) for label in ax.get_xticklabels()]
    ax.set_xticklabels(wrapped_labels, fontsize=10)
    
    plt.legend(fontsize=10)  


from sksurv.nonparametric import kaplan_meier_estimator
def plot_survival_curve(df, col, filter_col=None, filter_var=None, title=None, ax=None, color="husl", legend_outside=False):
    
    if ax is None:  # Create a standalone figure if no axes are provided 
        fig, ax = plt.subplots(figsize=(6, 4))

    if filter_col: # Subset dataset by filter_var 
         df_subset = df[df[filter_col] == filter_var]
    else:
        df_subset = df
        
    unique_values = sorted(df_subset[col].dropna().unique())  # Sort categories
    colors = sns.color_palette(color, n_colors=len(unique_values))  # Get colors from palette

    for var, color in zip(unique_values, colors):  # Assign colors dynamically
        mask = df_subset[col] == var

        if mask.sum() == 0:
            continue  # Skip if no samples for this category
            
        time, survival_prob, conf_int = kaplan_meier_estimator(
            df_subset["efs"][mask].astype("bool"),
            df_subset["efs_time"][mask],
            conf_type="log-log")

        ax.step(time, survival_prob, where="post", label=f"{var}", color=color)  
        ax.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post", color=color)  

    ax.set_ylim(0, 1)
    ax.set_ylabel("Estimated probability of survival")
    ax.set_xlabel("Time (months)")
    if title:
        ax.set_title(title)
    if legend_outside:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(loc="best")


# Copy of df_train
df_mod = df_train.copy()


fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
plot_stacked_countplot(df_train, "race_group", "ethnicity", ax=axes[0])
plot_survival_curve(df_train, "race_group", ax=axes[1])

fig.suptitle("Survival curves by race group and ethnicity", fontsize=16, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(12, 8)) 
axes = axes.flatten()

for i, race_group in enumerate(df_train["race_group"].unique()):
    plot_survival_curve(df_train,"ethnicity",
                        filter_col="race_group", filter_var=race_group,
                        title=race_group, ax=axes[i])

# Adjust layout to prevent overlap
fig.suptitle("Survival curves by race group and ethnicity", fontsize=16, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show() 


def categorize_age(age):
    if age <= 2:
        return "1: Baby"
    elif age < 12:
        return "2: Young Child"
    elif age < 19:
        return "3: Teenager"
    elif age < 35:
        return "4: Young Adult"
    elif age < 65:
        return "5: Middle Age"
    else:
        return "6: Elderly"

df_mod["age_category"] = df_train["age_at_hct"].apply(categorize_age)
df_mod["age_category_donor"] = df_train["donor_age"].apply(categorize_age)


fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 
axes=axes.flatten()

sns.histplot(data=df_mod, x="age_at_hct", hue="efs", ax=axes[0], edgecolor=None, binwidth=1, palette="husl")
plot_survival_curve(df_mod, "age_category", ax=axes[1], color="husl")

fig.suptitle("Survival curves by recipient age", fontsize=16, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show()


def simplify_karnoshsky(karnofsky_score):
    if karnofsky_score == -1: 
        return "Missing"
    elif karnofsky_score >= 80: 
        return "High (80-100)"
    else :
        return "Low (<80)"

def simplify_comorbidity(comorbidity_score):
    if comorbidity_score == -1: 
        return "Missing"
    elif comorbidity_score == 0: 
        return "0: Low"
    elif comorbidity_score <= 3:
        return "1-3: Medium"
    else:
        return "4-10: High"


df_mod["karnofsky_score_simple"] = df_train["karnofsky_score"].apply(simplify_karnoshsky)
df_mod["comorbidity_score_simple"] = df_train["comorbidity_score"].apply(simplify_comorbidity)

fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 
axes = axes.flatten()
plot_barplot("age_category", df_mod, hue="karnofsky_score_simple", 
             ax=axes[0], normalize=True, legend_outside=True, tilt_xlabels=True)
plot_barplot("age_category", df_mod, hue="comorbidity_score_simple", 
             ax=axes[1], normalize=True, legend_outside=True, tilt_xlabels=True)
fig.suptitle("Recipient age and Health Metrics", fontsize=16, fontweight="bold")
plt.tight_layout(pad=1)
plt.show()


def classify_disease(disease):
    acute = {"AML", "ALL", "Other acute leukemia"}
    chronic = {"CML", "MPN", "MDS", "Other leukemia"}
    lymphoma = {"NHL", "HD"}
    bone_marrow_failure = {"SAA"}
    plasma_cell_disorders = {"PCD"}
    solid_tumors = {"Solid tumor"}
    immune_autoimmune = {"IMD", "AI"}
    histiocytic_disorders = {"HIS"}

    if disease in acute:
        return "Acute Leukemia"
    elif disease in chronic:
        return "Chronic Leukemia & Related Disorders"
    elif disease in lymphoma:
        return "Lymphoma"
    elif disease in bone_marrow_failure:
        return "Bone Marrow Failure Syndromes"
    elif disease in plasma_cell_disorders:
        return "Plasma Cell Disorders"
    elif disease in solid_tumors:
        return "Solid Tumors"
    elif disease in immune_autoimmune:
        return "Immune & Autoimmune Disorders"
    elif disease in histiocytic_disorders:
         return "Histiocytic Disorders"
    else:
        return "Other"

df_mod["disease_class"] = df_train["prim_disease_hct"].apply(classify_disease)

fig, axes = plt.subplots(1, 1, figsize=(12, 4)) 
plot_barplot("age_category", df_mod, hue="disease_class", 
             ax=axes, normalize=True, legend_outside=True, color="husl")
fig.suptitle("Recipient age and Primary disease", fontsize=16, fontweight="bold")
plt.tight_layout(pad=1)
plt.show()


df_mod[["donor_sex", "recipient_sex"]] = df_mod["sex_match"].str.split("-", expand=True)
df_mod[["donor_sex", "recipient_sex"]] = df_mod[["donor_sex", "recipient_sex"]].fillna("Missing")

fig, axes = plt.subplots(2, 4, figsize=(12, 6)) 
axes=axes.flatten()

#sns.histplot(data=df_mod, x="donor_sex", hue="efs", ax=axes[0], edgecolor="white", binwidth=1, palette="husl")
plot_barplot("donor_sex", df_mod, ax=axes[0])
plot_survival_curve(df_mod, "donor_sex", ax=axes[1], color="husl")
plot_barplot("recipient_sex", df_mod, ax=axes[2])
plot_survival_curve(df_mod, "recipient_sex", ax=axes[3], color="husl")
plot_barplot("sex_match", df_mod, ax=axes[4], tilt_xlabels=True)
plot_survival_curve(df_mod, "sex_match", ax=axes[5], color="husl", legend_outside=True)
for i in range(6, len(axes)):  # Remove empty subplots
    fig.delaxes(axes[i])
    
fig.suptitle("Survival curves by donor and recipient sex", fontsize=16, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show()


def impute_hla_columns(df, fill_with_mean=False):
    # For each locus (A, B, C, DRB1, DQB1)
    loci = ['a', 'b', 'c', 'drb1', 'dqb1']
    
    for locus in loci:
        high_col = f'hla_match_{locus}_high'
        low_col = f'hla_match_{locus}_low'
        
        # Impute missing low resolution with high resolution and vice versa
        df[low_col].fillna(df[high_col], inplace=True)
        df[high_col].fillna(df[low_col], inplace=True)

        if fill_with_mean:
            # Check where both high and low are missing
            missing_rows = df[high_col].isna() & df[low_col].isna()
            
            # For rows where both are missing, impute with the rounded mean of the available high and low columns in that row
            for idx in df[missing_rows].index:
                # Get all available values in the current row from high and low resolution columns
                row_values = df.loc[idx, [f'hla_match_{locus}_high' for locus in loci] + [f'hla_match_{locus}_low' for locus in loci]].dropna()
                
                if len(row_values) > 0:
                    # Calculate the row-wise mean and round it
                    mean_value = row_values.mean().round()
                    
                    # Assign the rounded mean value to the missing high and low resolution columns
                    df.loc[idx, high_col] = mean_value
                    df.loc[idx, low_col] = mean_value
    
    return df


def impute_sum_columns(df):
    
    # Impute missing values in sum columns based on sum of individual high/low res features
    df['hla_low_res_6'].fillna(
        df['hla_match_a_low'] + df['hla_match_b_low'] + df['hla_match_drb1_low'], 
        inplace=True)

    df['hla_nmdp_6'].fillna(
        df['hla_match_a_low'] + df['hla_match_b_low'] + df['hla_match_drb1_high'], 
        inplace=True)
    
    df['hla_low_res_8'].fillna(
        df['hla_match_a_low'] + df['hla_match_b_low'] + df['hla_match_c_low'] + df['hla_match_drb1_low'], 
        inplace=True)
    
    df['hla_low_res_10'].fillna(
        df['hla_match_a_low'] + df['hla_match_b_low'] + df['hla_match_c_low'] + df['hla_match_drb1_low'] + df['hla_match_dqb1_low'], 
        inplace=True)

    df['hla_high_res_6'].fillna(
        df['hla_match_a_high'] + df['hla_match_b_high'] + df['hla_match_drb1_high'], 
        inplace=True)
    
    df['hla_high_res_8'].fillna(
        df['hla_match_a_high'] + df['hla_match_b_high'] + df['hla_match_c_high'] + df['hla_match_drb1_high'], 
        inplace=True)
    
    df['hla_high_res_10'].fillna(
        df['hla_match_a_high'] + df['hla_match_b_high'] + df['hla_match_c_high'] + df['hla_match_drb1_high'] + df['hla_match_dqb1_high'], 
        inplace=True)
    
    return df


# Fill in missing values
df_train_filled = impute_hla_columns(df_train)
df_train_filled = impute_sum_columns(df_train)


cols_hla = ['hla_match_a_low', 'hla_match_a_high',
           'hla_match_b_low', 'hla_match_b_high',
           'hla_match_c_low', 'hla_match_c_high',
           'hla_match_drb1_low', 'hla_match_drb1_high',
           'hla_match_dqb1_low', 'hla_match_dqb1_high']

fig, axes = plt.subplots(3, 4, figsize=(12, 10)) 
axes = axes.flatten()

for i, col in enumerate(cols_hla):
    plot_survival_curve(df_train_filled, col, title=col, ax=axes[i])
for i in range(len(cols_hla), len(axes)):  # Remove empty subplots
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
fig.suptitle("Survival curves by hla scores per allel", fontsize=16, fontweight="bold")  # Set overall title
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show() 


cols_hla_sum = ['hla_low_res_6', 'hla_high_res_6',
               'hla_low_res_8', 'hla_high_res_8', 
               'hla_low_res_10', 'hla_high_res_10',
               'hla_nmdp_6']

fig, axes = plt.subplots(2, 4, figsize=(12, 6)) 
axes = axes.flatten()

for i, col in enumerate(cols_hla_sum):
    plot_survival_curve(df_train_filled, col, title=col, ax=axes[i])
for i in range(len(cols_hla_sum), len(axes)):  # Remove empty subplots
    fig.delaxes(axes[i])
    
# Adjust layout to prevent overlap
fig.suptitle("Survival curves by summed hla scores", fontsize=16, fontweight="bold")  # Set overall title
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show() 


plot_barplot("hla_nmdp_6", df_mod, hue="donor_related") 
plt.xlabel("HLA score: nmdp_6")
plt.ylabel("Counts")
plt.show()


fig, axes = plt.subplots(3, 1, figsize=(12, 12)) 

# First barplot
plot_barplot("hla_nmdp_6", df_mod, hue="conditioning_intensity", normalize=True, ax=axes[0], legend_outside=True)
axes[0].set_xlabel("HLA score: nmdp_6")  
axes[0].set_ylabel("Proportion by HLA score (%)")  
axes[0].set_title("Score Distribution by Conditioning Intensity")

# Second barplot
plot_barplot("hla_nmdp_6", df_mod, hue="tbi_status", normalize=True, ax=axes[1], legend_outside=True)
axes[1].set_xlabel("HLA score: nmdp_6")  
axes[1].set_ylabel("Proportion by HLA score  (%)")  
axes[1].set_title("Score Distribution by TBI Status")


# Tird barplot
plot_barplot("hla_nmdp_6", df_mod, hue="gvhd_proph", normalize=True, ax=axes[2], legend_outside=True)
axes[2].set_xlabel("HLA score: nmdp_6")  
axes[2].set_ylabel("Proportion by HLA score  (%)")  
axes[2].set_title("Score Distribution by GVHD Prophylaxis")

plt.tight_layout()
plt.show()


def categorize_tbi(tbi_status):
    if pd.isna(tbi_status) or tbi_status == "No TBI": # "No TBI" is the mode
        return "No TBI"
    elif ">cGy" in tbi_status:
        return "High-Dose TBI"
    elif "Cy"in tbi_status:
        return "TBI + Cy"
    else:
        return "Low-Dose TBI"


df_mod["tbi_category"] = df_train["tbi_status"].apply(categorize_tbi)
df_mod["conditioning_intensity"] = df_mod["conditioning_intensity"].replace(["N/A, F(pre-TED) not submitted", "No drugs reported"], "TBD")
cat_cols = df_mod.select_dtypes(include=["object", "category"]).columns
num_cols = df_mod.select_dtypes(include=["number"]).columns
df_mod.loc[:, cat_cols] = df_mod[cat_cols].fillna("Missing")
df_mod.loc[:, num_cols] = df_mod[num_cols].fillna(-1)

fig, axes = plt.subplots(4, 2, figsize=(12, 12)) 
axes = axes.flatten()
plot_barplot("conditioning_intensity", df_mod, ax=axes[0], title="Conditioning")
plot_survival_curve(df_mod, "conditioning_intensity", ax=axes[1], title="Conditioning")
plot_barplot("tbi_category", df_mod, ax=axes[2], title="TBI category")
plot_survival_curve(df_mod, "tbi_category", ax=axes[3], title="TBI category")
plot_barplot("rituximab", df_mod, ax=axes[4], title="Rituximab")
plot_survival_curve(df_mod, "rituximab", ax=axes[5], title="Rituximab")
plot_barplot("melphalan_dose", df_mod, ax=axes[6], title="Melphalan")
plot_survival_curve(df_mod, "melphalan_dose", ax=axes[7], title="Melphalan")

fig.suptitle("Conditioning regimes", fontsize=16, fontweight="bold")
plt.tight_layout(pad=1)
plt.show()


def simplify_karnoshsky(karnofsky_score):
    if karnofsky_score == -1: 
        return "Missing"
    elif karnofsky_score >= 80: 
        return "High (80-100)"
    else :
        return "Low (<80)"

def simplify_comorbidity(comorbidity_score):
    if comorbidity_score == -1: 
        return "Missing"
    elif comorbidity_score == 0: 
        return "0: Low"
    elif comorbidity_score <= 3:
        return "1-3: Medium"
    else:
        return "4-10: High"


df_mod["karnofsky_score_simple"] = df_train["karnofsky_score"].apply(simplify_karnoshsky)
df_mod["comorbidity_score_simple"] = df_train["comorbidity_score"].apply(simplify_comorbidity)

fig, axes = plt.subplots(1, 2, figsize=(12, 3)) 
axes = axes.flatten()
plot_barplot("karnofsky_score_simple", df_mod, hue="conditioning_intensity", 
             ax=axes[0], normalize=True, legend_outside=True)
plot_barplot("comorbidity_score_simple", df_mod, hue="conditioning_intensity", 
             ax=axes[1], normalize=True, legend_outside=True)
fig.suptitle("Distribution of Conditioning Regimens by Patient Health Metrics", fontsize=16, fontweight="bold")
plt.tight_layout(pad=1)
plt.show()


import pandas as pd
import re

# Define a function to simplify the gvhd_proph column
def simplify_gvhd(row):
    # Remove everything inside parentheses or brackets
    row = re.sub(r"\(.*?\)", "", row)  # Removes text inside parentheses
    row = row.strip()  # Remove extra spaces
    
    # Extract main treatment
    if 'FK' in row:
        main_treatment = 'FK'
    elif 'CSA' in row:
        main_treatment = 'CSA'
    elif 'Cyclophosphamide' in row:
        main_treatment = 'Cyclophosphamide'
    elif 'CDselect' in row:
        main_treatment = 'CDselect'
    elif 'TDEPLETION' in row:
        main_treatment = 'TDEPLETION'
    elif 'No GvHD Prophylaxis' in row:
        main_treatment = 'None'
    elif 'Parent Q' in row:
        main_treatment = 'Parent Q'
    else:
        main_treatment = 'Other'

    # Identify presence of MTX, MMF, and "others"
    has_mtx = 'Yes' if 'MTX' in row else 'No'
    has_mmf = 'Yes' if 'MMF' in row else 'No'
    has_others = 'Yes' if 'other' in row.lower() and 'MTX' not in row and 'MMF' not in row else 'No'

    # Determine Treatment Category
    if has_mtx == 'No' and has_mmf == 'No' and has_others == 'No':
        treatment_category = 'Single Treatment'
    elif has_mtx == 'Yes' and has_mmf == 'No' and has_others == 'No':
        treatment_category = 'Plus MTX'
    elif has_mtx == 'No' and has_mmf == 'Yes' and has_others == 'No':
        treatment_category = 'Plus MMF'
    elif has_mtx == 'No' and has_mmf == 'No' and has_others == 'Yes':
        treatment_category = 'Plus Others'
    else:
         treatment_category = 'Other combination'
    
    return pd.Series([main_treatment, treatment_category])

# Apply function to create new columns
df_mod[["Main_GVHD", "GVHD_Category"]] = df_train["gvhd_proph"].copy().fillna("Other").apply(simplify_gvhd)


fig, axes = plt.subplots(2, 2, figsize=(12, 8)) 
axes = axes.flatten()

plot_barplot("Main_GVHD", df_mod, ax=axes[0], title="Main GVHD treatment", tilt_xlabels=True)
plot_survival_curve(df_mod, "Main_GVHD", ax=axes[1], title="Main GVHD treatment", legend_outside=True)
plot_barplot("GVHD_Category", df_mod, ax=axes[2], title="GVHD Treatment Category", tilt_xlabels=True)
plot_survival_curve(df_mod, "GVHD_Category", ax=axes[3], title="GVHD Treatment Category", legend_outside=True)

fig.suptitle("GVHD regimes", fontsize=16, fontweight="bold")
plt.tight_layout(pad=2)
plt.show()


fig, axes = plt.subplots(1, 1, figsize=(12, 6)) 
plot_barplot("prim_disease_hct", df_mod, hue="Main_GVHD", 
             ax=axes, normalize=True, legend_outside=True, tilt_xlabels=True)

fig.suptitle("Distribution of GVHD Regimens by Disease", fontsize=16, fontweight="bold")
plt.tight_layout(pad=1)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 6)) 
axes = axes.flatten()

plot_barplot("in_vivo_tcd", df_mod, hue="Main_GVHD", ax=axes[0], title="In vivo T-cell Depleteion", tilt_xlabels=True, legend_outside=True)
for i, group in enumerate(df_mod["in_vivo_tcd"].unique()):
    plot_survival_curve(df_mod,"Main_GVHD",
                        filter_col="in_vivo_tcd", filter_var=group,
                        title=f"in vivo TCD: {group}", ax=axes[i+1], legend_outside=True)

# Adjust layout to prevent overlap
fig.suptitle("in vivo TCD and main GVHD treatment", fontsize=16, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show() 


df_mod["year_hct"] = df_mod["year_hct"].replace(2020, 2019)

fig, axes = plt.subplots(1, 1, figsize=(6, 4)) 
plot_survival_curve(df_mod,"year_hct", legend_outside=True, ax=axes) 
fig.suptitle("Survival improves with the years", fontsize=12, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show() 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_stacked_bar(df, time_col, cat_col, ax=None, palette="husl"):

    # Group by time and categorical column, then count occurrences
    pivot_df = df.groupby([time_col, cat_col]).size().unstack(fill_value=0)

    # Normalize to percentages
    pivot_df_percent = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100  

    # Get Seaborn colors
    colors = sns.color_palette(palette, n_colors=len(pivot_df_percent.columns))

    # If no axes are provided, create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot stacked bar chart
    pivot_df_percent.plot(kind="bar", stacked=True, color=colors, alpha=0.85, ax=ax)

    # Formatting
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Distribution of {cat_col} Over Years")
    ax.legend(title=cat_col, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
    ax.set_xticks(range(len(pivot_df_percent.index)))  # Ensure correct tick positions
    ax.set_xticklabels(pivot_df_percent.index, rotation=45)  # Rotate x-axis labels for readability
    ax.set_ylim(0, 100)  # Ensure y-axis represents percentage scale


fig, axes = plt.subplots(3, 1, figsize=(12, 12)) 
axes = axes.flatten()

plot_stacked_bar(df_mod, "year_hct", "conditioning_intensity", ax=axes[0])
plot_stacked_bar(df_mod, "year_hct", "Main_GVHD", ax=axes[1])
plot_stacked_bar(df_mod, "year_hct", "GVHD_Category", ax=axes[2])

fig.suptitle("Changes in conditioning and GVHD regimes over time", fontsize=12, fontweight="bold")  
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.95])  
plt.show() 


score_cols = ["karnofsky_score", "comorbidity_score"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 
axes = axes.flatten()

for i, col in enumerate(score_cols):
    plot_survival_curve(df_mod, col,
                        title=col, ax=axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 


disease_cols = ["diabetes", "obesity", "psych_disturb",
                "arrhythmia", "cardiac","vent_hist",
                "hepatic_mild", "hepatic_severe","renal_issue",
                "pulm_moderate", "pulm_severe", "prior_tumor",
                "peptic_ulcer",]

fig, axes = plt.subplots(5, 3, figsize=(12, 12)) 
axes = axes.flatten()

for i, col in enumerate(disease_cols):
    plot_survival_curve(df_mod, col, title=col, ax=axes[i])
for i in range(len(disease_cols), len(axes)):  # Remove empty subplots
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 


def count_diseases(df, cols, new_col_name="disease_count"):
    df_copy = df.copy()
    df_copy[new_col_name] = df_copy[cols].apply(lambda row: row.isin(["Yes"]).sum(), axis=1)
    return df_copy

disease_cols_1 = ["psych_disturb", "diabetes", "arrhythmia",
                  "pulm_severe", "obesity","cardiac",
                  "pulm_moderate", "prior_tumor", "vent_hist"]

disease_cols_2 = ["renal_issue", "hepatic_severe", "peptic_ulcer",
                  "rheum_issue", "hepatic_mild"]

# Add disease count
df_mod = count_diseases(df_mod, disease_cols_1, "disease_count_negative")
df_mod = count_diseases(df_mod, disease_cols_2, "disease_count_positive")


fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 
axes = axes.flatten()

for i, col in enumerate(["disease_count_negative", "disease_count_positive"]):
    plot_survival_curve(df_mod, col,
                        title=col, ax=axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 


def simplify_dri_score(dri_value):
    # Mappings to simplify dri_scores
    mapping = {
        "Low": "1: Low",
        "Intermediate": "2: Intermediate",
        "High": "3: High",
        "Very high": "4: Very High",
        "Intermediate - TED AML case <missing cytogenetics": "2: Intermediate",
        "High - TED AML case <missing cytogenetics": "3: High",
        "TBD cytogenetics": "0: Unknown",
        "Missing disease status": "0: Unknown",
        "N/A - disease not classifiable": "0: Unknown",
        "N/A - pediatric": "0: Unknown",
        "N/A - non-malignant indication": "1: Low",  # Often considered low-risk
        np.nan: "0: Unknown"  # Handle NaNs
    }
    
    return mapping.get(dri_value, "0: Unknown")  # Default to "Unknown" if not listed


def assign_dri_risk(disease, cyto_score, mrd_hct=None):
    # Define risk groups based on disease
    low_risk = {"IEA", "HIS", "SAA"}  # Low-risk conditions
    intermediate_risk = {"AML", "ALL", "CML", "NHL", "HD", "IMD", "PCD"}  # Common transplant diseases
    high_risk = {"MDS", "MPN", "Other leukemia", "Other acute leukemia"}  # High-risk diseases
    very_high_risk = {"Solid tumor", "IIS", "IPA", "AI"}  # Non-hematologic or aggressive conditions

    if pd.isna(disease) or pd.isna(cyto_score) or cyto_score == "Unknown":
        return "0: Unknown"
    
    # Default risk based on disease
    if disease in low_risk:
        risk = 1  # Low
    elif disease in intermediate_risk:
        risk = 2  # Intermediate
    elif disease in high_risk:
        risk = 3  # High
    elif disease in very_high_risk:
        risk = 4  # Very High
    else:
        return "0: Unknown"

    # Adjust risk based on cytogenetics
    if cyto_score in {"Favorable", "Normal"}:
        risk = max(1, risk - 1)  # Reduce risk level by 1 (min is 1)
    elif cyto_score == "Poor":
        risk = min(4, risk + 1)  # Increase risk level by 1 (max is 4)

    # Adjust risk if MRD at HCT is negative (??? MRD is bad but this seems to work)
    if mrd_hct == "Negative":
        risk = min(4, risk + 1)  # Increase risk level by 1 (max is 4)

    return f"{risk}: {'Low' if risk == 1 else 'Intermediate' if risk == 2 else 'High' if risk == 3 else 'Very High'}"



# Apply to DataFrame
df_mod["dri_score_custom"] = df_train.apply(lambda row: assign_dri_risk(row["prim_disease_hct"], row["cyto_score_detail"], row["mrd_hct"]), axis=1)
df_mod["dri_simple"] = df_train["dri_score"].apply(simplify_dri_score)

fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 
axes = axes.flatten()

plot_survival_curve(df_mod, "dri_score_custom", title="Custom dri-score", ax=axes[0])
plot_survival_curve(df_mod, "dri_simple", title="Simplified dri-score", ax=axes[1])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 


df_mod["merged_product_type"] = df_mod["graft_type"] + "_" + df_mod["prod_type"]

cols = ["merged_product_type","graft_type", "prod_type"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4)) 
axes = axes.flatten()

for i, col in enumerate(cols):
    plot_survival_curve(df_mod, col,
                        title=col, ax=axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 


df_mod["cyto_score_"] = df_mod["cyto_score"] + "_"
df_mod["cyto_combined"] = np.where(df_mod["prim_disease_hct"].isin(["AML", "MDS"]),
                                   df_mod["cyto_score_detail"], 
                                   df_mod["cyto_score_"])
df_mod["cyto_combined"] = df_mod["cyto_combined"].replace(["TBD_", "Not tested", "Not tested_"], "TBD")

cols = ["cyto_combined", "cyto_score", "cyto_score_detail"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4)) 
axes = axes.flatten()

for i, col in enumerate(cols):
    plot_survival_curve(df_mod, col,
                        title=col, ax=axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 


def combine_tce_match(row):
    # Define severity ranking
    severity_order = [
        "Missing",
        "Bi-directional non-permissive",
        "GvH non-permissive",
        "HvG non-permissive",
        "Permissive mismatched",
        "Permissive",
        "Fully matched"]
    
    # Get the values from both columns
    tce_1 = row["tce_match"]
    tce_2 = row["tce_div_match"]
    
    # Deal with missing values
    if pd.isna(tce_1) or tce_1 == "Missing":
        return tce_2
    if pd.isna(tce_2) or tce_2 == "Missing":
        return tce_1
    
    # Return the least severe category based on the ranking
    return tce_1 if severity_order.index(tce_1) >  severity_order.index(tce_2) else tce_2

df_mod["tce_combined"] = df_mod.apply(combine_tce_match, axis=1)

cols = ["tce_combined", "tce_match", "tce_div_match"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4)) 
axes = axes.flatten()

plot_survival_curve(df_mod, "tce_combined", title="tce_combined", ax=axes[0])
plot_survival_curve(df_mod, "tce_match", title="tce_match", ax=axes[1])
plot_survival_curve(df_mod, "tce_div_match", title="tce_div_match", ax=axes[2])

# Adjust layout to prevent overlap
plt.tight_layout(pad=1)  
plt.show() 




