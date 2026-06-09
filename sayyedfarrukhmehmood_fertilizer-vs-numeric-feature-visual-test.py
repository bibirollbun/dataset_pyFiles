!pip install -qU seaborn

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings 
warnings.filterwarnings('ignore')

class CFG:
    train_path="/kaggle/input/playground-series-s5e6/train.csv"
    test_path="/kaggle/input/playground-series-s5e6/test.csv"

    target = "Fertilizer Name"


train=pd.read_csv(CFG.train_path,index_col='id')
test=pd.read_csv(CFG.test_path,index_col='id')


def target_vs_num_feature(feature):
    print(f"\nğŸ”� Analyzing Relationship Between Fertilizer and '{feature}'\n")
    
    # â–¶ï¸� Raw Count Distribution (Before Normalization)
    print("\nğŸ“Š Plot 1: Raw counts of each fertilizer across feature bins \n")
    sns.histplot(hue=CFG.target, x=feature, stat='count', data=train, multiple='dodge', bins=7)
    plt.title(f'Raw Count of Fertilizers Across {feature} (Before Normalization)')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # â–¶ï¸� Prepare normalized data
    counts = train.groupby(['Fertilizer Name', feature]).size().reset_index(name='raw_count')
    totals = train.groupby('Fertilizer Name').size().reset_index(name='fert_total')
    merged = counts.merge(totals, on='Fertilizer Name')
    merged['pct_of_fertilizer'] = merged['raw_count'] / merged['fert_total']

    # â–¶ï¸� Percentage Distribution (After Normalization)
    print("\nğŸ“Š Plot 2: Percentage distribution of each fertilizer \n")
    plt.figure(figsize=(13, 4))
    sns.barplot(
        data=merged,
        x=feature,
        y='pct_of_fertilizer',
        hue='Fertilizer Name',
        ci=None,
        edgecolor='black'
    )
    plt.ylabel('Proportion of that Fertilizer')
    plt.xlabel(feature)
    plt.title(f'Normalized Percentage Distribution of Each Fertilizer Across {feature}')
    plt.legend(title='Fertilizer', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0, merged['pct_of_fertilizer'].max() * 1.1)
    plt.tight_layout()
    plt.show()

    # â–¶ï¸� Line Plot for Trend 
    print("\nğŸ“ˆ Plot 3: Trend of fertilizer proportions across feature \n")
    sns.lineplot(
        data=merged,
        x=feature,
        y='pct_of_fertilizer',
        hue='Fertilizer Name',
        marker='o'
    )
    plt.ylabel('Proportion of that Fertilizer')
    plt.xlabel(feature)
    plt.title(f'Trend of Fertilizer Share Across {feature} (After Normalization)')
    plt.legend(title='Fertilizer', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylim(0, merged['pct_of_fertilizer'].max() * 1.1)
    plt.tight_layout()
    plt.show()

    # â–¶ï¸� Chi-square test for independence
    print("\nğŸ§ª Statistical Test: Chi-square Test of Independence")
    contingency = pd.crosstab(train[feature], train['Fertilizer Name'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nâœ… P-value for '{feature}': {p:.4f}")
    
    if p < 0.05:
        print(f"\nğŸ“Œ Insight: Statistically significant association between {feature} and fertilizer choice.")
    else:
        print(f"\nğŸ“Œ Insight: No significant association between {feature} and fertilizer choice.")



target_vs_num_feature("Temparature")


target_vs_num_feature("Humidity")


target_vs_num_feature("Moisture")


train.head(1)


target_vs_num_feature("Nitrogen")


target_vs_num_feature("Potassium")


target_vs_num_feature("Phosphorous")

