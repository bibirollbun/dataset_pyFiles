import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

import warnings, os, gc, sys, math, json, random, itertools

from scipy import stats
from scipy.stats import ks_2samp


warnings.filterwarnings("ignore")
plt.style.use("seaborn-whitegrid")
sns.set_palette("crest")
pd.set_option("display.max_columns", 100)


train = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/train.csv')
test = pd.read_csv('/kaggle/input/california-homlessness-prediction-challenge/test.csv')


def quick_overview(df, name="train"):
    print(f"\n{name.upper()} SHAPE: {df.shape}")
    display(df.head())
    display(df.describe(include="all").T)

quick_overview(train, "train")
quick_overview(test , "test")

print(f"Duplicate rows (train): {train.duplicated().sum()}  |  (test): {test.duplicated().sum()}")


train.isnull().sum() 



def plot_kde(data, name, columns=None, figsize=(8, 4), fill=True, max_density=None):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    columns = data.select_dtypes(include='number').columns.tolist()
    plt.figure(figsize=figsize)
    for col in columns:
        sns.kdeplot(data[col], label=col, linewidth=2,clip=(0, None),linestyle="-.")
        
    if max_density is not None:
        plt.ylim(0, max_density)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

age_cols1 = ['AGE_U18_PCT','AGE_18_24_PCT','AGE_25_34_PCT','AGE_35_44_PCT']
age_cols2 = ['AGE_45_54_PCT','AGE_55_59_PCT','AGE_60_61_PCT','AGE_62_64_PCT']
age_cols3 = ['AGE_65_69_PCT','AGE_70_79_PCT','AGE_80_PLUS_PCT','AGE_25_PLUS_PCT']
plot_kde(train[age_cols1], "Age Distributions 0-44 in Training Set", max_density=70)
plot_kde(test[age_cols1], "Age Distributions 0-44 in Test Set", max_density=70)
plot_kde(train[age_cols2], "Age Distributions 45-64 in Training Set", max_density=70)
plot_kde(test[age_cols2], "Age Distributions 45-64 in Test Set", max_density=70)
plot_kde(train[age_cols3], "Age Distributions 65-max in Training Set", max_density=70)
plot_kde(test[age_cols3], "Age Distributions 65-max in Test Set", max_density=70)


race_cols1=['RACE_WHITE_NH_PCT',	'RACE_BLACK_NH_PCT', 'RACE_ASIAN_NH_PCT',
           'RACE_TWO_OR_MORE_NH_PCT', 'RACE_HISPANIC_ANY_PCT']
race_cols2=['RACE_NATIVE_NH_PCT', 'RACE_PACIFIC_NH_PCT']
plot_kde(train[race_cols1],"Race Distributions in Training Set without Native and Pacific", max_density=50)
plot_kde(test[race_cols1],"Race Distributions in Test Set without Native and Pacific", max_density=50)
plot_kde(train[race_cols2],"Native and Pacific Distributions in Training Set", max_density=280)
plot_kde(test[race_cols2],"Native and Pacific Distributions in Test Set", max_density=280)


vet_cols = ['VETERAN_POP_PCT', 'NONVETERAN_POP_PCT']
dis_cols = ['DISABILITY_POP_PCT', 'NODISABILITY_POP_PCT']
hh_cols = ['TOTAL_HOUSEHOLDS_PCT', 'FAMILY_HH_TOTAL', 'FAMILY_HH_CHILD_LT18_PCT']
nf_cols = ['NONFAMILY_SINGLE_MALE_PCT', 'NONFAMILY_SINGLE_FEMALE_PCT', 'MULTI_PERSON_NONFAMILY_HH_PCT']
nif_cols = ['INDIVIDUALS_NOT_IN_FAMILY_UNITS_PCT']

plot_kde(train[vet_cols], "Veteran and Non Veteran Distributions in Training Set")
plot_kde(test[vet_cols], "Veteran and Non Veteran Distributions in Test Set")
plot_kde(train[dis_cols], "Disability and Non Disability Distributions in Training Set")
plot_kde(test[dis_cols], "Disability and Non Disability Distributions in Test Set")
plot_kde(train[hh_cols], "Household Distributions in Training Set")
plot_kde(test[hh_cols], "Household Distributions in Test Set")
plot_kde(train[nf_cols], "Non Family Distributions in Training Set")
plot_kde(test[nf_cols], "Non Family Distributions in Test Set")
plot_kde(train[nif_cols], "Not In Family Units Distributions in Training Set")
plot_kde(test[nif_cols], "Not In Family Units Distributions in Test Set")


plot_kde(train['HOMELESS_RATE'], "Homeless Rate Distribution in Training Set")


outlier_summary = {}
for col in train.drop(columns=['ID']):
    z = np.abs(stats.zscore(train[col].dropna()))
    outlier_summary[col] = (z>3).sum()   # 3-σ rule

pd.Series(outlier_summary, name="#outliers (>3σ)").sort_values(ascending=False).to_frame().style.bar()


corr = train.drop(columns=['ID','FAMILY_MEMBERS_UNDER_18_PCT']).corr()
#Dropping Family Members Under 18 since it has the same values as Age Under 18
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,cbar_kws={"shrink": 0.8})

plt.xticks(rotation=90, ha='right', fontsize=8) 
#rotation = 45 makes it diagonal if its easier for you to read
plt.yticks(rotation=0, fontsize=8)
plt.title("Pearson Correlation – Numeric Features")
plt.show()


target_corr = train.drop(columns=['ID','FAMILY_MEMBERS_UNDER_18_PCT']).corr()["HOMELESS_RATE"].drop(
    "HOMELESS_RATE").sort_values()
display(target_corr.to_frame("corr_with_target").style.bar(vmin=-1,vmax=1))

