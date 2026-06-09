# ğŸ“¦ Loading data and packages
import pandas as pd
import warnings
import seaborn as sns
from sklearn.neighbors import KDTree

# ğŸ”§ Settings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# ğŸ“Œ Jupyter magic
%matplotlib inline

# ğŸ—‚ï¸� Get files
train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submision = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

train_data.head()


# Fill in the NaN
train_nums2 = train_data.copy()
test_num2 = test_data.copy()

def fill_nan(dataset, feat):
    # Use train set to get the weighted average of the feature
    intro = train_nums2.loc[train_nums2.Personality == "Introvert", feat].mean()
    extro = train_nums2.loc[train_nums2.Personality == "Extrovert", feat].mean()
    dataset.loc[dataset[feat].isna(), feat] = 0.5*intro + 0.5*extro

feature_names = ['Time_spent_Alone', 'Social_event_attendance',
       'Going_outside','Friends_circle_size', 'Post_frequency']
for feat in feature_names:
    fill_nan(train_nums2, feat)
    fill_nan(test_num2, feat)
    
    # Normalize with z-score 
    train_mean = train_nums2.loc[:, feat].mean()
    train_std = train_nums2.loc[:, feat].std()
    train_nums2.loc[:, feat] = (train_nums2.loc[:, feat] - train_mean)/ train_std
    train_nums2.loc[:, feat] = train_nums2.loc[:, feat].clip(-1, 1)
    test_num2.loc[:, feat] = (test_num2.loc[:, feat] - train_mean)/ train_std
    test_num2.loc[:, feat] = test_num2.loc[:, feat].clip(-1, 1)

train_nums2.isna().sum()


sns.histplot(data=train_nums2[feature_names])


from sklearn.decomposition import PCA

train_x = train_nums2[feature_names]
test_x = test_num2[feature_names]

all_reduced = PCA(n_components=2).fit_transform(pd.concat([train_x, test_x]))
train_reduced = all_reduced[:len(train_x)]
test_reduced = all_reduced[len(train_x):]

sns.scatterplot(x=train_reduced[:, 0], y=train_reduced[:, 1], hue=train_nums2.Personality)


kdt = KDTree(train_reduced)

# Test against training data
closed_neighbors = kdt.query(train_reduced, k=6, return_distance=False)

correct = 0
for i, neighbors_ids in enumerate(closed_neighbors):
    # The first neighbor itself so we skip it with [1:]
    neighbors = train_nums2.Personality[neighbors_ids[1:]].mode()[0]
    itself = train_nums2.Personality[i]
    train_nums2.loc[i, "prediction"] = neighbors
    if itself == neighbors:
        correct += 1

print(f"acc: {correct / len(train_reduced)}")


train_nums2.loc[train_nums2.Personality != train_nums2.prediction].head(15)


# Make submission
kdt = KDTree(train_reduced)

closed_neighbors = kdt.query(test_reduced, k=5, return_distance=False)

correct = 0
for i, neighbors_ids in enumerate(closed_neighbors):
    neighbors = train_nums2.Personality[neighbors_ids].mode()[0]
    test_id = test_num2.loc[i].id
    submision.loc[submision.id == test_id, "Personality"] = neighbors

submision.to_csv("submission.csv", index=False)
submision.head()

