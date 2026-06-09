import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns


# 讀資料
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


train.shape


test.shape


sample.shape


train.head()


test.head()


def over_view(df,name = "dataframe"):
    print(f"\n{name.upper()} - Basic Info")
    display(df.info())
    display(df.describe(include = "all").T)
over_view(train,"train")
over_view(test,"test")


# 合併
combined = pd.concat([train, test], ignore_index=True)


combined


# 辨識重複項
combined.duplicated().sum()


# 辨識缺失項
combined.isnull().sum()


# 辨識缺失項(畫圖)
msno.matrix(train, figsize = (10, 4))
plt.title("Missing Value Matrix - Train Dataset")
plt.show()


msno.matrix(test, figsize = (10, 4))
plt.title("Missing Value Matrix - test Dataset")
plt.show()


msno.bar(train, figsize = (8, 4))
plt.title("Missing Value Matrix - train Dataset")
plt.show()


msno.bar(test, figsize = (8, 4))
plt.title("Missing Value Matrix - test Dataset")
plt.show()


fig, axis = plt.subplots(1, 2, figsize = (14,4))
sns.histplot(train["Calories"], bins = 60, kde = True, ax = axis[0])
axis[0].set_title("Calories (Raw Scale)")

sns.histplot(np.log1p(train["Calories"]), bins = 60, kde = True, ax = axis[1], color = "tomato")
axis[1].set_title("Calories (log1p Scale")
plt.show()



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


# Create a figure with two side-by-side subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# ----------------------------
# 比較1: 時長 vs 卡路里 
# ----------------------------
axs[0, 0].scatter(
    combined['Duration'],     # x-axis:時長
    combined['Calories'],     # y-axis:卡路里
    alpha=0.2           #點透明圖
)
sns.regplot(
    x=combined['Duration'],
    y=combined['Calories'],
    scatter=False,          # 不重複繪製散佈點
    color="red",            # 迴歸線顏色
    ax=axs[0, 0]            # 指定繪製在 axs[0,0] 上
)
axs[0, 0].set_xlabel('Duration')       # label x-axis
axs[0, 0].set_ylabel('Calories')       # label y-axis
axs[0, 0].set_title('Duration vs. Calories')
    
# ----------------------------
# 比較2: 心律 vs 卡路里
# ----------------------------
axs[0, 1].scatter(
    combined['Heart_Rate'],   # x-axis:心律
    combined['Calories'],     # y-axis:卡路里
    alpha=0.2          
)
sns.regplot(
    x=combined['Heart_Rate'],
    y=combined['Calories'],
    scatter=False,          # 不重複繪製散佈點
    color="red",            # 迴歸線顏色
    ax=axs[0, 1]            # 指定繪製在 axs[0,0] 上
)
axs[0, 1].set_xlabel('Heart Rate')   
axs[0, 1].set_ylabel('Calories')     
axs[0, 1].set_title('Heart Rate vs. Calories')

# ----------------------------
# 比較3:  高度 vs 卡路里
# ----------------------------
axs[1, 0].scatter(
    combined['Height'],     # x-axis:高度
    combined['Calories'],     # y-axis:卡路里
    alpha=0.2           #點透明圖
)
sns.regplot(
    x=combined['Height'],
    y=combined['Calories'],
    scatter=False,          # 不重複繪製散佈點
    color="red",            # 迴歸線顏色
    ax=axs[1, 0]            # 指定繪製在 axs[0,0] 上
)
axs[1, 0].set_xlabel('Height')       # label x-axis
axs[1, 0].set_ylabel('Calories')       # label y-axis
axs[1, 0].set_title('Height vs. Calories')

# ----------------------------
# 比較4: 體重 vs 卡路里
# ----------------------------
axs[1, 1].scatter(
    combined['Weight'],     # x-axis:體重
    combined['Calories'],     # y-axis:卡路里
    alpha=0.2           #點透明圖
)
sns.regplot(
    x=combined['Weight'],
    y=combined['Calories'],
    scatter=False,          # 不重複繪製散佈點
    color="red",            # 迴歸線顏色
    ax=axs[1, 1 ]            # 指定繪製在 axs[0,0] 上
)
axs[1, 1].set_xlabel('Weight')       # label x-axis
axs[1, 1].set_ylabel('Calories')       # label y-axis
axs[1, 1].set_title('Weight vs. Calories')

# Adjust layout so titles/labels don’t overlap, then display
plt.tight_layout()
plt.show()


num_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
selected_features = ['Age', 'Body_Temp']

for col_name in num_cols:
    if col_name in selected_features:
        plt.figure(figsize = (4,3))
        sns.scatterplot(x = train[col_name], y = train["Calories"], alpha = 0.2)
        sns.regplot(x = train[col_name], y = train["Calories"], scatter = False, color = "red")
        plt.title(f"{col_name} vs Calories")
        plt.show()


plt.figure(figsize = (4,3))
sns.violinplot(x = "Sex", y = "Calories", data = train, palette = "pastel", inner = "quartile")
plt.title("Sex va Calories")
plt.show()


plt.figure(figsize = (4,3))
sns.countplot(y = train['Sex'], palette = "muted")
plt.title("Sex Distribution in Train Dataset")
plt.show()


plt.figure(figsize = (4,3))
sns.countplot(y = test['Sex'], palette = "muted")
plt.title("Sex Distribution in Test Dataset")
plt.show()


plt.figure(figsize=(10, 5))
sns.histplot(
    data=train,      # 指定數據來源的 DataFrame
    x='Age',         # 指定要繪製分佈的欄位名 (X軸)
    bins='auto',     # 'auto' 會讓 seaborn 自動選擇合適的條柱數量，你也可以指定一個數字，例如 bins=20
    kde=True,        # 是否同時繪製核密度估計曲線 (Kernel Density Estimate)
    color='skyblue', # 設定條柱顏色 (可選)
    edgecolor='black'# 設定條柱邊緣顏色 (可選，讓條柱更清晰)
)


plt.figure(figsize=(10, 5))
sns.histplot(
    data=test,      # 指定數據來源的 DataFrame
    x='Age',         # 指定要繪製分佈的欄位名 (X軸)
    bins='auto',     # 'auto' 會讓 seaborn 自動選擇合適的條柱數量，你也可以指定一個數字，例如 bins=20
    kde=True,        # 是否同時繪製核密度估計曲線 (Kernel Density Estimate)
    color='skyblue', # 設定條柱顏色 (可選)
    edgecolor='black'# 設定條柱邊緣顏色 (可選，讓條柱更清晰)
)


plt.figure(figsize=(10, 6))
sns.heatmap(combined.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()


corr = train[num_cols + ["Calories"]].corr(method = "spearman")

plt.figure(figsize = (10,7))
sns.heatmap(corr, cmap = "RdBu_r", center = 0, annot = True, fmt = ".2f")
plt.title("Spearman Correlations")
plt.show()


pair_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
sns.pairplot(train[pair_cols], corner = True, diag_kind = "kde", hue = None)
plt.suptitle("Pairwise scatter", y = 1.02)
plt.show()

