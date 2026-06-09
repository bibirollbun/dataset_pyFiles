import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# supress warnings
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


train.tail()


train.info()


train.isna().sum().sort_values(ascending=False)



test.isna().sum().sort_values(ascending=False)


print("Shape of train: ", train.shape)

print("Shape of test: ", test.shape)


train.describe().T


# determine all numerical columns

columns_numerical = [col for col in train.columns if pd.api.types.is_numeric_dtype(train[col])]


columns_numerical




num_vars = len(columns_numerical)




# Color palette (color-blind friendly)
palette = sns.color_palette("Set3", n_colors=num_vars)

train_smaller = train.sample(frac = 0.005)


# Calculate number of rows & columns

num_cols = 2  # Keep 2 columns
if num_vars // num_cols == num_vars / num_cols:
    num_rows = num_vars // num_cols
else:
    num_rows = num_vars // num_cols + 1




# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4.5))
axes = axes.flatten()



# Plotting each variable
for i, var in enumerate(columns_numerical):
    sns.histplot(
        data=train_smaller,
        x=var,
        kde=True,
        color=palette[i],
        bins=30,
        edgecolor="white",
        linewidth=1.3,
        ax=axes[i]
    )
    axes[i].set_title(f"{var}", fontsize=14, weight="bold")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")
    axes[i].tick_params(axis='x', labelrotation=15)

# Remove unused axes
#for j in range(i + 1, len(axes)):
#    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()


# Plotting each variable
for i, var in enumerate(columns_numerical):

    plt.subplot(1, 2, 1)
    sns.histplot(
        data=train_smaller[var],
        kde=True,
        color=palette[i],
        bins=30,
        edgecolor="white",
        linewidth=1.3
    )
    plt.title(f"{var}", fontsize=14, weight="bold")
    plt.xlabel("")
    plt.ylabel("")
    # plt.tick_params(axis='x', labelrotation=15)

    plt.subplot(1, 2, 2)
    sns.boxplot(
        x=train_smaller[var],
        color = palette[i],
    )
    plt.title(f"Box Plot of {var}", fontsize = 14)

    plt.tight_layout(h_pad=2.5)
    plt.show()
    





# Plotting each variable


for i, var in enumerate(columns_numerical[1:]):  # remove ID



    plt.subplot(1, 2, 1)
    sns.boxplot(
        y=train_smaller[var],
        color = palette[i]
    )
    plt.title(f"{var} - Boxplot")

    plt.subplot(1, 2, 2)
    sns.violinplot(
        data=train_smaller, 
        y=var, 
        color=palette[i]
    )
    plt.title(f'Violin Plot of {var}')
    plt.xlabel('')
    plt.ylabel(var)
    
    plt.title(f"{var} - Violinplot", fontsize=14)
    plt.xlabel("")
    plt.ylabel("")
    # plt.tick_params(axis='x', labelrotation=15)

    plt.tight_layout(h_pad=2.5)
    plt.show()





# age
sns.histplot(
        data=train_smaller,
        x='Age',
        kde=True,
        color=palette[0],
        bins=100,
        edgecolor="white",
        linewidth=1.3
    )
plt.title(f"Age", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("")
plt.tick_params(axis='x', labelrotation=15)

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()


# Height
sns.histplot(
        data=train_smaller,
        x='Height',
        kde=True,
        color=palette[1],
        bins=80,
        edgecolor="white",
        linewidth=1.3
    )
plt.title(f"Height", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("")
plt.tick_params(axis='x', labelrotation=15)

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()


# Weight
sns.histplot(
        data=train_smaller,
        x='Weight',
        kde=True,
        color=palette[2],
        bins=100,
        edgecolor="white",
        linewidth=1.3
    )
plt.title(f"Weight", fontsize=14, weight="bold")
plt.xlabel("")
plt.ylabel("")
plt.tick_params(axis='x', labelrotation=15)

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()


for i, var in enumerate(columns_numerical[1:-1]):  # remove ID and remove Calories


    sns.scatterplot(
        x=train_smaller[var],
        y = train_smaller['Calories'],
        alpha = 0.5,
        color = palette[i]
    )
    plt.title(f"Box Plot of {var}")




    plt.tight_layout(h_pad=2.5)
    plt.show()


# let's count how many of those we have
genre_counts = pd.DataFrame(train['Sex'].value_counts())

genre_counts = genre_counts.reset_index()
genre_counts.columns = ['Genre', 'counts']


sns.barplot(
    x = genre_counts['Genre'],
    y = genre_counts['counts'],
    palette = palette
    
)

plt.title(f"Distribution of Sex")
plt.xticks(rotation=45)


# target variable 

sns.boxplot(
    x = train_smaller["Sex"], 
    y = train_smaller["Calories"],
    palette = palette
)
plt.title("Categorical: Sex vs. Calories")
plt.xlabel("Sex")
plt.ylabel("Calories")
# plt.xticks(rotation=45)
plt.show()


sns.histplot(
    data = train_smaller,
    x = "Calories",
    hue = "Sex",
    palette = palette,
    kde = True
)
plt.title("Distribution of Calories by Sex (Histogram)")
plt.xlabel("Calories")
plt.ylabel("Frequency")
plt.show()


sns.pairplot(
    train_smaller,
    hue = 'Sex',
    palette = palette,
)


!pip install jupyter-summarytools


from summarytools import dfSummary
dfSummary(train_smaller)

