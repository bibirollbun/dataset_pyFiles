!pip install -q ipyplot


import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from io import BytesIO
#from ipyplot import show_images
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings("ignore")

sns.set(style = "whitegrid", context = "notebook")


path = "/kaggle/input/playground-series-s5e5/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

train.shape


test.shape


train.head()


test.head()


def quick_overview(df, name = "dataframe"):
    print(f"\n{name.upper()} - Basic Info")
    display(df.info())
    display(df.describe(include = "all").T)

quick_overview(train, "train")
quick_overview(test, "test")


msno.matrix(train, figsize = (10, 4))
plt.title("Missing Value Matrix - Train Dataset")
plt.show()


msno.matrix(test, figsize = (10, 4))
plt.title("Missing Value Matrix - Test Dataset")
plt.show()


msno.bar(train, figsize = (10, 4))
plt.title("Missing Count per Column - Train Dataset")
plt.show()


fig, axis = plt.subplots(1, 2, figsize = (14,4))
sns.histplot(train["Calories"], bins = 60, kde = True, ax = axis[0])
axis[0].set_title("Calories (Raw Scale)")

sns.histplot(np.log1p(train["Calories"]), bins = 60, kde = True, ax = axis[1], color = "tomato")
axis[1].set_title("Calories (log1p Scale")
plt.show()


print("Skewness :", train["Calories"].skew().round(3))
print("Kurtosis :", train["Calories"].kurt().round(3))


num_cols = train.select_dtypes(include = ["int64", "float64"]).columns.tolist()
num_cols.remove("Calories")

fig, axis = plt.subplots(len(num_cols) // 3 + 1, 3, figsize = (15,4*len(num_cols)//3))

for i, col in enumerate(num_cols):
    r, c = divmod(i, 3)
    sns.histplot(train[col], kde = True, ax = axis[r][c], color = "steelblue")
    axis[r][c].set_title(f"Train Dataset - {col}")

plt.tight_layout()
plt.show()


fig, axis = plt.subplots(len(num_cols) // 3 + 1, 3, figsize = (15,4*len(num_cols)//3))

for i, col in enumerate(num_cols):
    r, c = divmod(i, 3)
    sns.histplot(test[col], kde = True, ax = axis[r][c], color = "tomato")
    axis[r][c].set_title(f"Test Dataset - {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize = (4,3))
sns.countplot(y = train['Sex'], palette = "muted")
plt.title("Sex Distribution in Train Dataset")
plt.show()


plt.figure(figsize = (4,3))
sns.countplot(y = test['Sex'], palette = "muted")
plt.title("Sex Distribution in Test Dataset")
plt.show()


corr = train[num_cols + ["Calories"]].corr(method = "spearman")

plt.figure(figsize = (10,7))
sns.heatmap(corr, cmap = "RdBu_r", center = 0, annot = True, fmt = ".2f")
plt.title("Spearman Correlations")
plt.show()


for col in num_cols:
    plt.figure(figsize = (4,3))
    sns.scatterplot(x = train[col], y = train["Calories"], alpha = 0.2)
    sns.regplot(x = train[col], y = train["Calories"], scatter = False, color = "red")
    plt.title(f"{col} vs Calories")
    plt.show()


plt.figure(figsize = (4,3))
sns.violinplot(x = "Sex", y = "Calories", data = train, palette = "pastel", inner = "quartile")
plt.title("Sex va Calories")
plt.show()


pair_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
sns.pairplot(train[pair_cols], corner = True, diag_kind = "kde", hue = None)
plt.suptitle("Pairwise scatter", y = 1.02)
plt.show()


def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr   = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    return ((series < lower) | (series > upper)).sum()

print("\nOutlier counts (IQR rule):")

for col in num_cols + ["Calories"]:
    print(f"{col:<12} : {iqr_outliers(train[col])}")


X_vif = train[num_cols].assign(constant = 1)
vif_df = pd.DataFrame({
    "feature": num_cols,
    "VIF"    : [variance_inflation_factor(X_vif.values, i) for i in range(len(num_cols))]
})
display(vif_df.sort_values("VIF", ascending=False).style.background_gradient(cmap="Reds"))


scaled = (train[num_cols] - train[num_cols].mean()) / train[num_cols].std()
pca = PCA(n_components=2, random_state = 42).fit_transform(scaled)
plt.figure(figsize = (6, 5))
sns.scatterplot(x = pca[:,0], y = pca[:,1],
                hue = pd.qcut(train["Calories"], 5, labels = False), palette = "viridis",
                alpha = 0.3, s = 10)
plt.title("PCA – coloured by Calories quintile"); plt.legend(title = "Quintile", bbox_to_anchor = (1.05,1))
plt.show()


compare_cols = num_cols + ["Sex"]
train_tag = train.assign(dataset = "train")[compare_cols + ["dataset"]]
test_tag  = test.assign(dataset = "test")[compare_cols + ["dataset"]]
combo = pd.concat([train_tag, test_tag], axis = 0)

for col in compare_cols:
    if col is "Sex":
        ct = pd.crosstab(combo[col], combo["dataset"], normalize = "columns") * 100
        ct.plot.barh(figsize = (6, 4), stacked=False, title = f"{col} – train vs test %")
        plt.show()
    else:
        sns.kdeplot(data = combo, x = col, hue = "dataset", fill = True, common_norm = False, alpha = 0.4)
        plt.title(f"{col} – train vs test distribution"); plt.show()




