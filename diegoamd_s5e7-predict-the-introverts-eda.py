import kagglehub

# Download latest version
path = kagglehub.dataset_download("rakeshkapilavai/extrovert-vs-introvert-behavior-data")

print("Path to dataset files:", path)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


original_full = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")
original_miss = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")
train_set = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_set = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


print(original_full.info())


print(train_set.info())


print(test_set.info())


def print_num_descriptions(feat_name):
    print("Original Full")
    print(original_full[feat_name].describe())
    print("-" * 30)
    print("Train Set")
    print(train_set[feat_name].describe())
    print("-" * 30)
    print("Test Set")
    print(test_set[feat_name].describe())


def num_boxplots(feat_name):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sns.boxplot(y = feat_name, data = original_full, ax=ax[0])
    ax[0].set_title("Original Full")
    sns.boxplot(y = feat_name, data = train_set, ax=ax[1])
    ax[1].set_title("Train Set")
    sns.boxplot(y = feat_name, data = test_set, ax=ax[2])
    ax[2].set_title("Test Set")
    plt.tight_layout()
    plt.show()


def num_boxplots_vs_target(feat_name, target):
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sns.boxplot(y = feat_name, x = target, data = original_full, ax=ax[0])
    ax[0].set_title("Original Full")
    sns.boxplot(y = feat_name, x = target, data = train_set, ax=ax[1])
    ax[1].set_title("Train Set")
    plt.tight_layout()
    plt.show()


def print_cat_descriptions(feat_name):
    print("Original Full")
    print(original_full[feat_name].value_counts(normalize = True))
    print("-" * 30)
    print("Train Set")
    print(train_set[feat_name].value_counts(normalize = True))
    print("-" * 30)
    print("Test Set")
    print(test_set[feat_name].value_counts(normalize = True))


def cat_countplots(feat_name):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    sns.countplot(x = feat_name, data = original_full, ax = ax[0])
    ax[0].set_title("Original Full")
    sns.countplot(x = feat_name, data = train_set, ax = ax[1])
    ax[1].set_title("Train Set")
    sns.countplot(x = feat_name, data = test_set, ax = ax[2])
    ax[2].set_title("Test Set")
    plt.show()


def cat_countplots_vs_target(feat_name, target):
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    sns.countplot(x = feat_name, hue = target, data = original_full, ax = ax[0])
    ax[0].set_title("Original Full")
    sns.countplot(x = feat_name, hue = target, data = train_set, ax = ax[1])
    ax[1].set_title("Train Set")
    plt.show()


print_num_descriptions("Time_spent_Alone")


num_boxplots("Time_spent_Alone")


num_boxplots_vs_target("Time_spent_Alone", "Personality")


print_cat_descriptions("Stage_fear")


cat_countplots("Stage_fear")


cat_countplots_vs_target("Stage_fear", "Personality")


print_num_descriptions("Social_event_attendance")


num_boxplots("Social_event_attendance")


num_boxplots_vs_target("Social_event_attendance", "Personality")


print_num_descriptions("Going_outside")


num_boxplots("Going_outside")


num_boxplots_vs_target("Going_outside", "Personality")


print_cat_descriptions("Drained_after_socializing")


cat_countplots("Drained_after_socializing")


cat_countplots_vs_target("Drained_after_socializing", "Personality")


print_num_descriptions("Friends_circle_size")


num_boxplots("Friends_circle_size")


num_boxplots_vs_target("Friends_circle_size", "Personality")


print_num_descriptions("Post_frequency")


num_boxplots("Post_frequency")


num_boxplots_vs_target("Post_frequency", "Personality")


print("Original Full")
print(original_full["Personality"].value_counts(normalize = True))
print("-" * 30)
print("Train Set")
print(train_set["Personality"].value_counts(normalize = True))


fig, ax = plt.subplots(1, 2, figsize = (10, 5))
sns.countplot(x = "Personality", data = original_full, ax = ax[0])
ax[0].set_title("Original Full")
sns.countplot(x = "Personality", data = train_set, ax = ax[1])
ax[1].set_title("Train Set")
plt.show()


num_feat_matrix = original_full.select_dtypes(exclude = ["object"]).corr()
sns.heatmap(num_feat_matrix, annot = True, cmap = "coolwarm")
plt.title("Numerical Features Correlation Matrix")
plt.show()


num_feat_matrix = train_set.select_dtypes(exclude = ["object"]).drop(columns = ["id"]).corr()
sns.heatmap(num_feat_matrix, annot = True, cmap = "coolwarm")
plt.title("Numerical Features Correlation Matrix")
plt.show()


original_full.shape


train_set.shape


test_set.shape




