import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder
import xgboost as xgb
plt.style.use("seaborn-v0_8-darkgrid")
warnings.filterwarnings("ignore")
plt.rc("font",family="SimHei",size="15")  
# import csv
train_df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
datasert_df = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


datasert_df = (
    datasert_df
    .rename(columns={'Personality': 'match_p'})
    .drop_duplicates(['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
                      'Going_outside', 'Drained_after_socializing', 
                      'Friends_circle_size', 'Post_frequency'])
)

merge_cols = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
              'Going_outside', 'Drained_after_socializing', 
              'Friends_circle_size', 'Post_frequency']

train_df = train_df.merge(datasert_df, how='left', on=merge_cols)
test_df = test_df.merge(datasert_df, how='left', on=merge_cols)

train_df.info()


train_df.info()
train_df.describe()


numeric_df = train_df.select_dtypes(include='number').drop(columns=['id'])
numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt='.2f', vmin=-1, vmax=1)
plt.title('Heatmap')
plt.show()


train_ID = train_df['id']
test_ID = test_df['id']

#Now drop the  'id' colum since it's unnecessary for  the prediction process.
train_df.drop("id", axis = 1, inplace = True)
test_df.drop("id", axis = 1, inplace = True)

ntrain = train_df.shape[0] # è®­ç»ƒé›†æ•°ç›®
ntest = test_df.shape[0] # æµ‹è¯•é›†æ•°ç›®
y_train = train_df['Personality'].map({'Extrovert': 1, 'Introvert': 0}).values # è®­ç»ƒé›†çš„Y

all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['Personality'], axis=1, inplace=True)


def fill_missing_by_quantile_group(df, group_source_col, target_col, quantiles=[0, 0.25, 0.5, 0.75, 1.0], labels=None):
    """
    Group the target_col based on quantiles of group_source_col, and fill missing values in target_col
    within each group using the group's median.
    
    Parameters:
        df (pd.DataFrame): Original dataset
        group_source_col (str): Column used for grouping (numerical)
        target_col (str): Target column to fill missing values
        quantiles (list): Quantile cut points for grouping (default is quartiles)
        labels (list): Labels for each group (default auto-generated as Q1, Q2, ...)
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled (in-place modification)
    """
    #  Automatically generate group labels
    if labels is None:
        labels = [f'Q{i+1}' for i in range(len(quantiles)-1)]

    temp_bin_col = f'{group_source_col}_bin'

    # Step 1: Create grouping column
    df[temp_bin_col] = pd.qcut(df[group_source_col], q=quantiles, labels=labels)

    # Step 2: Fill missing values within each group using the group's median
    df[target_col] = df[target_col].fillna(df.groupby(temp_bin_col)[target_col].transform('median'))

    # Step 3: Remove the temporary grouping column
    df.drop(columns=[temp_bin_col], inplace=True)

    return df

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Time_spent_Alone'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Time_spent_Alone'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Social_event_attendance'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Social_event_attendance'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Social_event_attendance'
)


all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Social_event_attendance',
    target_col='Going_outside'
)

all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Post_frequency',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Going_outside',
    target_col='Friends_circle_size'
)
all_data = fill_missing_by_quantile_group(
    df=all_data,
    group_source_col='Friends_circle_size',
    target_col='Post_frequency'
)
all_data.info()


all_data.fillna({
    'Stage_fear': 'UnKnow',
    'Drained_after_socializing': 'UnKnow'
}, inplace=True)
all_data.info()


all_data["Social_efficiency"] = all_data["Social_event_attendance"] / (all_data["Going_outside"] + 1e-3)


all_data["Alone_to_Social_ratio"] = all_data["Time_spent_Alone"] / (all_data["Friends_circle_size"] + 1e-3)


all_data["Posts_per_friend"] = all_data["Post_frequency"] / (all_data["Friends_circle_size"] + 1e-3)



all_data["Social_stress_index"] = all_data["Drained_after_socializing"].map({"UnKnow": 2,"Yes": 1, "No": 0}) * all_data["Social_event_attendance"]
all_data["Social_stress_index"].info()


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", 
            "Friends_circle_size", "Post_frequency"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(all_data[features].fillna(0))

kmeans = KMeans(n_clusters=4, random_state=42)
all_data["social_behavior_cluster"] = kmeans.fit_predict(X_scaled)



all_data["post_rank_in_circle"] = all_data.groupby("Friends_circle_size")["Post_frequency"].rank(pct=True)



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_feats = pca.fit_transform(all_data[features].fillna(0))
all_data["PCA1"], all_data["PCA2"] = pca_feats[:,0], pca_feats[:,1]


all_data = pd.get_dummies(all_data, columns=['Stage_fear', 'Drained_after_socializing','match_p'], prefix=['Stage', 'Drained','match'])
all_data.info()


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')



X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
X=X_train
y=y_train


class_0 = y_train.sum()
class_1 = len(y_train) - class_0
scale_pos_weight = class_1 / class_0


xgb = XGBClassifier(
    max_depth=4,         
    learning_rate=0.01,   
    n_estimators=1000,    
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    class_weights=[scale_pos_weight, 1],
    random_seed=42,
    verbose=0
)

lgbm = LGBMClassifier(
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight={0: scale_pos_weight, 1: 1},
    random_state=42
)


ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lgbm', lgbm)
    ],
    voting='soft'
)



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
ensemble.fit(X_train, y_train)


val_probs = ensemble.predict_proba(X_val)[:, 1]
best_threshold = 0.5
best_acc = 0

for threshold in np.arange(0.4, 0.6, 0.01):
    preds = (val_probs >= threshold).astype(int)


test_probs = ensemble.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

# Create submission
submission = pd.DataFrame({
    'id': test_ID,
    'Personality': test_preds
})
print(submission.head())
submission['Personality'] = submission['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
submission.to_csv('submission.csv', index=False)
print("Submitted successfully")

