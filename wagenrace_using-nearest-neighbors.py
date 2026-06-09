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





# Convert strings to numbers
def data_to_nums(dataset: pd.DataFrame):
    dataset = dataset.copy()
    dataset.loc[dataset.Stage_fear == "Yes", "Stage_fear"] = 1
    dataset.loc[dataset.Stage_fear == "No", "Stage_fear"] = 0
    dataset.loc[dataset.Drained_after_socializing == "Yes", "Drained_after_socializing"] = 1
    dataset.loc[dataset.Drained_after_socializing == "No", "Drained_after_socializing"] = 0
        
    return dataset

train_nums = data_to_nums(train_data)
test_nums = data_to_nums(test_data)
train_nums.head()


# Fill in the NaN
train_nums2 = train_nums.copy()
test_num2 = test_nums.copy()

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


train_x = train_nums2[feature_names]

kdt = KDTree(train_x)

# Test against training data
closed_neighbors = kdt.query(train_x, k=6, return_distance=False)

correct = 0
for i, neighbors_ids in enumerate(closed_neighbors):
    # The first neighbor itself so we skip it with [1:]
    neighbors = train_nums2.Personality[neighbors_ids[1:]].mode()[0]
    itself = train_nums2.Personality[i]
    train_nums2.loc[i, "prediction"] = neighbors
    if itself == neighbors:
        correct += 1

print(f"acc: {correct / len(train_x)}")


train_nums2.describe()


train_nums2.loc[train_nums2.Personality != train_nums2.prediction].describe()


# Make submission
kdt = KDTree(train_x)

test_x = test_num2[feature_names]
closed_neighbors = kdt.query(test_x, k=5, return_distance=False)

correct = 0
for i, neighbors_ids in enumerate(closed_neighbors):
    neighbors = train_nums2.Personality[neighbors_ids].mode()[0]
    test_id = test_num2.loc[i].id
    submision.loc[submision.id == test_id, "Personality"] = neighbors

submision.to_csv("submission.csv", index=False)
submision.head()

