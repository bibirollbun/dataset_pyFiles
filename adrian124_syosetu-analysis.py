import numpy as np
import pandas as pd 


!pip install japanize_matplotlib


import japanize_matplotlib


train = pd.read_csv("/kaggle/input/narou/train.csv")
train.info()


train["global_point"].describe()


from matplotlib import pyplot as plt

plt.hist(train['global_point'], log=True)


print(train["biggenre"].unique())
print(len(train["biggenre"].unique()))


train["biggenre"].value_counts().plot(kind='pie')


train["biggenre"].value_counts().plot(kind='bar', rot=0)


print(train["genre"].unique())
print(len(train["genre"].unique()))


train["genre"].value_counts().plot(kind='pie')


train["genre"].value_counts().plot(kind='bar')


print(train["istensei"].unique())
print(train["istenni"].unique())


isekai_novels = train.loc[(train['istensei'] == "Yes") | (train['istenni'] =="Yes")]
non_isekais = train.loc[train.index.difference(isekai_novels.index)]


print("Isekais:")
isekai_novels["global_point"].describe()


print("Non-isekais:")
non_isekais["global_point"].describe()


plt.hist([non_isekais["global_point"], isekai_novels["global_point"]], alpha=0.5, label=['Non-isekai novels','Isekai novels'], log=True)
plt.legend(loc='upper right')
plt.title("Point distribution of isekai and other novels")
plt.show()


plt.boxplot(x=[non_isekais["global_point"], isekai_novels["global_point"]], labels=['Non-isekai novels','Isekai novels'])
plt.title("Distribution of points for isekai and non-isekai novels")
plt.yscale("log")
plt.ylabel("Points")
plt.show()


plt.hist(non_isekais["global_point"], alpha=0.5, label='Non-isekai novels', log=True)
plt.legend(loc='upper right')
plt.title("Distribution of points for non-isekai novels")
plt.show()


plt.hist(isekai_novels["global_point"], alpha=0.5, label='Isekai novels', log=True)
plt.legend(loc='upper right')
plt.title("Distribution of points for isekai novels")
plt.show()


train["isbl"].value_counts()


train["isgl"].value_counts()


bl_novels = train.loc[(train['isbl'] == "Yes")]
gl_novels = train.loc[(train['isgl'] == "Yes")]


bl_novels["global_point"].describe()


gl_novels["global_point"].describe()


plt.hist([gl_novels["global_point"], bl_novels["global_point"]], alpha=0.5, label=['GL novels','BL novels'], log=True)
plt.legend(loc='upper right')
plt.title("Point distribution of BL and GL novels")
plt.show()


other_novels = train.loc[(train['isbl'] == "No")&(train['isgl'] == "No")]
plt.boxplot(x=[gl_novels["global_point"], bl_novels["global_point"],other_novels["global_point"]], labels=['GL novels','BL novels',"other novels"])
plt.title("Distribution of points for BL, GL and other novels")
plt.yscale("log")
plt.ylabel("Points")
plt.show()


r15_novels = train.loc[(train['isr15'] == "Yes")]
non_r15_novels = train.loc[train.index.difference(r15_novels.index)]


violent_novels = train.loc[(train['iszankoku'] == "Yes")]
non_violent_novels = train.loc[train.index.difference(violent_novels.index)]


r15_novels["global_point"].describe()


non_r15_novels["global_point"].describe()


violent_novels["global_point"].describe()


non_violent_novels["global_point"].describe()


train["sasie_cnt_per_story"].describe()


train["sasie_cnt_per_story"].plot(kind="hist",range=[1,184])


train["sasie_cnt_per_story"].plot(kind="hist",range=[1,5])


train["kaiwaritu"].plot(kind="hist")


train["kaiwaritu"].plot(kind="hist", range=[80,100])


train["length_per_story"].plot(kind="hist")


train["length_per_story"].plot(kind="hist", range=[0,25_000])


train["length_per_story"].plot(kind="hist", range=[0,5000])


train["time_per_story_submission"].plot(kind="hist")


train["time_per_story_submission"].plot(kind="hist", range=[0,125])


train["time_per_story_submission"].plot(kind="hist", range=[0,30])


import seaborn as sns

sns.pairplot(data=train, y_vars="global_point",x_vars=["kaiwaritu","length_per_story","sasie_cnt_per_story","time_per_story_submission"])


train["keyword"].head()


train.fillna({"keyword":"UNTAGGED_NOVEL"}, inplace=True)

def str_tags_to_set(tags):
    return set(tags.split(" "))

keywords_set = train["keyword"].apply(lambda x: str_tags_to_set(x))
keywords_set_list = keywords_set.tolist()
keywords_collection = set()


for i in keywords_set_list:
    keywords_collection.update(i)
keywords_dict = dict()
for i, el in enumerate(keywords_collection):
    keywords_dict[el] = i


len(keywords_dict)


from collections import Counter
keywords_array = train['keyword'].str.split()
keywords_array_flattened = [keyword for individual_list in keywords_array for keyword in individual_list]
keywords_counter = Counter(keywords_array_flattened)
most_popular_keywords = keywords_counter.most_common(100)

most_popular_keywords_df = pd.DataFrame(most_popular_keywords, columns=['keyword', 'frequency'])
print(most_popular_keywords_df)


most_popular_keywords_df[:10]


train["writer"].value_counts()


train["writer"].value_counts().plot(kind="hist", bins=[i-0.5 for i in range(8)])


train.sort_values('global_point', ascending=False)[:10]


title_lengths = train["title"].apply(len)

plt.hist(title_lengths)
plt.title("Novel title lengths")
plt.ylabel("Frequency")
plt.xlabel("Title length (characters)")


plt.hist(title_lengths, range=[70,110])
plt.title("Max novel title lengths")
plt.ylabel("Frequency")
plt.xlabel("Title length (characters)")


import re

def count_kanji(text):
    return len(re.findall(r'[\u4E00-\u9FFF]', text))/len(text)

def count_hiragana(text):
    return len(re.findall(r'[\u3040-\u309F]', text))/len(text)

def count_katakana(text):
    return len(re.findall(r'[\u30A0-\u30FF]', text))/len(text)

train["title_kanji_rate"] = train["title"].apply(count_kanji)
train["title_hiragana_rate"] = train["title"].apply(count_hiragana)
train["title_katakana_rate"] = train["title"].apply(count_katakana)


train[["title","title_kanji_rate","title_hiragana_rate","title_katakana_rate"]].head()


train["title_kanji_rate"].plot(kind="hist")


train["title_hiragana_rate"].plot(kind="hist")


train["title_katakana_rate"].plot(kind="hist")


train["title_length"] = train["title"].apply(len)


sns.pairplot(train[["title_length","title_kanji_rate","title_hiragana_rate","title_katakana_rate"]],corner=True)


train["log_global_point"] = np.log1p(train["global_point"])


def string_binary_to_boolean(x):
    if x=="Yes":
        return True
    return False

binary_cols = ["isr15","isbl","isgl","iszankoku","istensei","istenni"]

for col in binary_cols:
    train[col] = train[col].apply(string_binary_to_boolean)


train["genre"] = train["genre"].astype("category")


numerical_cols = ["length_per_story","sasie_cnt_per_story","time_per_story_submission","kaiwaritu","title_length","title_kanji_rate","title_hiragana_rate","title_katakana_rate"]
for col in numerical_cols:
    train[col] = train[col].astype(np.float64)


from sklearn.model_selection import train_test_split

train, valid = train_test_split(train, test_size=0.1, random_state=42)


x_cols = ["genre","isr15","isbl","isgl","iszankoku","istensei","istenni","kaiwaritu","length_per_story","sasie_cnt_per_story","time_per_story_submission","title_length","title_kanji_rate","title_hiragana_rate","title_katakana_rate"]
y_col = ["log_global_point"]


train_x = train[x_cols]
valid_x = valid[x_cols]

train_y = train[y_col]
valid_y = valid[y_col]


import xgboost as xgb
custom_parameters = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    "early_stopping_rounds": 10,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "enable_categorical": True,
    "seed": 42
    
}
model = xgb.XGBRegressor(**custom_parameters)


model.fit(train_x, train_y, 
            eval_set=[(valid_x, valid_y)]) 


1.89**2


!pip install shap==0.46.0


import shap
explainer = shap.Explainer(model)
shap_values = explainer(train_x)


shap.summary_plot(shap_values)


shap.plots.scatter(shap_values[:, "length_per_story"])


shap.plots.scatter(shap_values[:, "time_per_story_submission"])


shap.plots.scatter(shap_values[:, "kaiwaritu"])


shap.plots.scatter(shap_values[:, "title_length"])


shap.plots.scatter(shap_values[:, "title_kanji_rate"])


shap.plots.scatter(shap_values[:, "title_hiragana_rate"])


shap.plots.scatter(shap_values[:, "title_katakana_rate"])


train_info = train[["title","log_global_point"]]

def explain_prediction(novel_id : int):
    novel_iloc = train.index.get_loc(novel_id)
    print(novel_iloc)
    print(f"Novel title: {train.loc[novel_id]['title']}")
    print(f"True score (log-transformed): {train.loc[novel_id]['log_global_point']}")
    shap.plots.waterfall(shap_values[novel_iloc])


train[train['title'].str.contains("転生したらスライムだった件|蜘蛛ですが、なにか？|死神を食べた少女")] # Looking for some specific, more popular novels


explain_prediction(115)


explain_prediction(25)


explain_prediction(13635)


explain_prediction(8695)


explain_prediction(42860)


shap_values[:, "genre"]


genre_vals = shap_values[:, "genre"].values


genre_ids = shap_values[:, "genre"].data


genre_df = pd.DataFrame({"shap_value":genre_vals, "genre": genre_ids})


genre_df.groupby("genre").mean().sort_values(by="shap_value")


genre_df.groupby("genre").mean().sort_values(by="shap_value").plot(kind="bar") # Mean


genre_df.groupby("genre").median().sort_values(by="shap_value").plot(kind="bar") # Median




