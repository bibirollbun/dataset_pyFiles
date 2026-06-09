import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# We load the data

train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col="id")
origin = pd.read_csv("/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv")


print("Train data shape:", train.shape)
print("Origin data shape:", origin.shape)


train.head()


origin.head()


train.describe()


train.describe(exclude=np.number)


train.info()


#di_origin = origin[train.columns]
#di_train = pd.concat([train, di_origin], ignore_index=True)


# Establishing the seaborn aesthetic

sns.set_style("darkgrid")

# We establish the color palette

palette = sns.set_palette("Greens_r")


# Function to analyze number distributions

def plot_number_analyzer(data, column, figsize, suptitle):

    print(
    "Variable: ", column,
    "\nFormat: ", data[column].dtype,
    "\nNumber of null values: ", data[column].isnull().sum(),
    "\nUnique values: ", data[column].nunique(),
    "\nVariable range:", data[column].min(), "to", data[column].max(), "\n\n"
    )

    # We graph the distribution
    
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    sns.histplot(
        data=data, 
        x=column, 
        palette=palette,
        edgecolor="k",
        ax=axes[0]
    )
    sns.boxplot(
        data=data, 
        x=column,
        palette=palette,
        ax=axes[1]
    )
    plt.suptitle(t=suptitle)
    plt.tight_layout()
    plt.show()


# Function for categorical variables

def plot_categories(data, figsize, sizes, labels, colors, explode, title):

    print(
    "\nNumber of null values: ", data.isnull().sum(),
    "\nUnique values: ", data.nunique(),
    "\nDistribution of values: \n", data.value_counts(), "\n\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        wedgeprops={"edgecolor" : "k"},
        explode=explode,
        textprops={'fontsize': 12}
    )
    axes[0].set_ylabel("")
    
    sns.histplot(
        data=data,
        color="green",
        edgecolor="k",
        ax=axes[1]
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    
    # Add labels with the exact value above each bar
    
    for container in axes[1].containers:
        axes[1].bar_label(container, fontsize=12)
    
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# We create a variable for the analysis

di_di = train["diagnosed_diabetes"].replace([0, 1], ["No", "Yes"]).astype("category")

# We print and graph the distribution

counts_t = di_di.value_counts()
labels_t = counts_t.index
sizes_t = counts_t.values
color_t = ["#409e70", "#7b79c7"]
explode_t = (0.0, 0.2)
title_t = "Target variable distribution of values"

plot_categories(di_di, (12, 4), sizes_t, labels_t, color_t, explode_t, title_t)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "age", 
    (12, 4), 
    "Distribution of values of the 'age' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "alcohol_consumption_per_week", 
    (12, 4), 
    "Distribution of values of the 'alcohol_consumption_per_week' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "physical_activity_minutes_per_week", 
    (12, 4), 
    "Distribution of values of the 'physical_activity_minutes_per_week' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "diet_score", 
    (12, 4), 
    "Distribution of values of the 'diet_score' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "sleep_hours_per_day", 
    (12, 4), 
    "Distribution of values of the 'sleep_hours_per_day' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "screen_time_hours_per_day", 
    (12, 4), 
    "Distribution of values of the 'screen_time_hours_per_day' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "bmi", 
    (12, 4), 
    "Distribution of values of the 'bmi' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "waist_to_hip_ratio", 
    (12, 4), 
    "Distribution of values of the 'waist_to_hip_ratio' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "systolic_bp", 
    (12, 4), 
    "Distribution of values of the 'systolic_bp' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "diastolic_bp", 
    (12, 4), 
    "Distribution of values of the 'diastolic_bp' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "heart_rate", 
    (12, 4), 
    "Distribution of values of the 'heart_rate' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "cholesterol_total", 
    (12, 4), 
    "Distribution of values of the 'cholesterol_total' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "hdl_cholesterol", 
    (12, 4), 
    "Distribution of values of the 'hdl_cholesterol' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "ldl_cholesterol", 
    (12, 4), 
    "Distribution of values of the 'ldl_cholesterol' variable"
)


# We print and graph the distribution

plot_number_analyzer(
    train, 
    "triglycerides", 
    (12, 4), 
    "Distribution of values of the 'triglycerides' variable"
)


# We print and graph the distribution

counts_g = train["gender"].value_counts()
labels_g = counts_g.index
sizes_g = counts_g.values
color_g = ["#409e70", "#7b79c7", "#d7d684"]
explode_g = (0.0, 0.1, 0.5)
title_g = "Gender variable distribution of values"

plot_categories(train["gender"], (14, 6), sizes_g, labels_g, color_g, explode_g, title_g)


# We print and graph the distribution

counts_e = train["ethnicity"].value_counts()
labels_e = counts_e.index
sizes_e = counts_e.values
color_e = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db", "#8d0c34"]
explode_e = (0.1, 0.2, 0.3, 0.2, 0.4)
title_e = "Ethnicity variable distribution of values"

plot_categories(train["ethnicity"], (14, 6), sizes_e, labels_e, color_e, explode_e, title_e)


# We graph the relationship

fig, axes = plt.subplots(figsize=(12, 4))

sns.histplot(
    data=train, 
    x="ethnicity",
    hue="gender", 
    multiple="dodge", 
    shrink=.8,
    edgecolor="k",
    palette="Paired",
    ax=axes
)

plt.title("Ethnicity distribution by Gender")
plt.tight_layout()
plt.show()


# We print and graph the distribution

counts_ed = train["education_level"].value_counts()
labels_ed = counts_ed.index
sizes_ed = counts_ed.values
color_ed = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db"]
explode_ed = (0.1, 0.2, 0.3, 0.5)
title_ed = "Education Level variable distribution of values"

plot_categories(train["education_level"], (14, 6), sizes_ed, labels_ed, color_ed, explode_ed, title_ed)


# We print and graph the distribution

counts_i = train["income_level"].value_counts()
labels_i = counts_i.index
sizes_i = counts_i.values
color_i = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db", "#8d0c34"]
explode_i = (0.1, 0.2, 0.3, 0.2, 0.4)
title_i = "Income Level variable distribution of values"

plot_categories(train["income_level"], (14, 6), sizes_i, labels_i, color_i, explode_i, title_i)


# We print and graph the distribution

counts_ss = train["smoking_status"].value_counts()
labels_ss = counts_ss.index
sizes_ss = counts_ss.values
color_ss = ["#409e70", "#7b79c7", "#d7d684"]
explode_ss = (0.0, 0.1, 0.2)
title_ss = "Smoking Status variable distribution of values"

plot_categories(train["smoking_status"], (14, 6), sizes_ss, labels_ss, color_ss, explode_ss, title_ss)


# We print and graph the distribution

counts_es = train["employment_status"].value_counts()
labels_es = counts_es.index
sizes_es = counts_es.values
color_es = ["#409e70", "#7b79c7", "#d7d684", "#cdf3db"]
explode_es = (0.1, 0.2, 0.3, 0.5)
title_es = "Employment Status variable distribution of values"

plot_categories(train["employment_status"], (14, 6), sizes_es, labels_es, color_es, explode_es, title_es)


# We create a variable for the analysis

fhd = train["family_history_diabetes"].replace([0, 1], ["No", "Yes"]).astype("category")

# We print and graph the distribution

counts_fhd = fhd.value_counts()
labels_fhd = counts_fhd.index
sizes_fhd = counts_fhd.values
color_fhd = ["#409e70", "#7b79c7"]
explode_fhd = (0.0, 0.2)
title_fhd = "Family history diabetes distribution of values"

plot_categories(fhd, (12, 4), sizes_fhd, labels_fhd, color_fhd, explode_fhd, title_fhd)


# We create a variable for the analysis

hh = train["hypertension_history"].replace([0, 1], ["No", "Yes"]).astype("category")

# We print and graph the distribution

counts_hh = hh.value_counts()
labels_hh = counts_hh.index
sizes_hh = counts_hh.values
color_hh = ["#409e70", "#7b79c7"]
explode_hh = (0.0, 0.2)
title_hh = "Hypertension History distribution of values"

plot_categories(hh, (12, 4), sizes_hh, labels_hh, color_hh, explode_hh, title_hh)


# We create a variable for the analysis

ch = train["cardiovascular_history"].replace([0, 1], ["No", "Yes"]).astype("category")

# We print and graph the distribution

counts_ch = ch.value_counts()
labels_ch = counts_ch.index
sizes_ch = counts_ch.values
color_ch = ["#409e70", "#7b79c7"]
explode_ch = (0.0, 0.5)
title_ch = "Cardiovascular History distribution of values"

plot_categories(ch, (12, 4), sizes_ch, labels_ch, color_ch, explode_ch, title_ch)

