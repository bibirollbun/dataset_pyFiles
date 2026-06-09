# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotting libraries
import matplotlib as plt 
import matplotlib.pyplot as plt
import seaborn as sns

# math libraries
from scipy import stats

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Warnings
import warnings
warnings.filterwarnings("ignore")


# === Load TRAIN data ===
train_path = "/kaggle/input/widsdatathon2025/TRAIN_NEW"
connectome_train = pd.read_csv(f"{train_path}/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
quant_meta_train = pd.read_excel(f"{train_path}/TRAIN_QUANTITATIVE_METADATA_new.xlsx")
cat_meta_train = pd.read_excel(f"{train_path}/TRAIN_CATEGORICAL_METADATA_new.xlsx")
targets_train = pd.read_excel(f"{train_path}/TRAINING_SOLUTIONS.xlsx")


# Check shapes
print("Train Connectome:", connectome_train.shape)
print("Train Quantitative metadata:", quant_meta_train.shape)
print("Train Categorical metadata:", cat_meta_train.shape)
print("Train Targets:", targets_train.shape)

# === Load TEST data ===
test_path = "/kaggle/input/widsdatathon2025/TEST"
connectome_test = pd.read_csv(f"{test_path}/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
quant_meta_test = pd.read_excel(f"{test_path}/TEST_QUANTITATIVE_METADATA.xlsx")
cat_meta_test = pd.read_excel(f"{test_path}/TEST_CATEGORICAL.xlsx")


# Check shapes
print("Test Connectome:", connectome_test.shape)
print("Test Quantitative metadata:", quant_meta_test.shape)
print("Test Categorical metadata:", cat_meta_test.shape)


#Load File
data_dict = pd.read_excel('/kaggle/input/widsdatathon2025/Data Dictionary.xlsx')


data_dict


cat_meta_train.info()


cat_meta_merge_df = cat_meta_train.loc[:,['Basic_Demos_Enroll_Year','participant_id']].merge(
    targets_train, on = 'participant_id', how = 'left'
).drop(columns = ['participant_id'])
cat_meta_merge_df.head()
sns.histplot(data=cat_meta_merge_df, x="Basic_Demos_Enroll_Year", hue="ADHD_Outcome", multiple="dodge", kde=True)


pd.set_option('display.max_colwidth', None)
data_dict.loc[data_dict["Field"]=="Basic_Demos_Study_Site","Labels"]



data_dict.loc[data_dict["Field"]=="MRI_Track_Scan_Location","Labels"]


(
    cat_meta_train
    .loc[:,['Basic_Demos_Study_Site','MRI_Track_Scan_Location']]
    .value_counts(dropna=False)
    .reset_index()
    .sort_values("Basic_Demos_Study_Site")
)


data_dict.loc[data_dict["Field"]=="PreInt_Demos_Fam_Child_Ethnicity","Labels"]



data_dict.loc[data_dict["Field"]=="PreInt_Demos_Fam_Child_Race","Labels"]


(
    cat_meta_train
    .loc[:,['PreInt_Demos_Fam_Child_Ethnicity','PreInt_Demos_Fam_Child_Race']]
    .value_counts(dropna=False)
    .reset_index()
    .sort_values(["PreInt_Demos_Fam_Child_Ethnicity","count"], ascending=[True,False])
)


pd.set_option('display.max_colwidth', None)
data_dict.loc[data_dict["Field"]=="Barratt_Barratt_P1_Occ","Labels"]



data_dict.loc[data_dict["Field"]=="Barratt_Barratt_P1_Edu","Labels"]


bb_occ = pd.concat([cat_meta_train['Barratt_Barratt_P1_Occ'],cat_meta_train['Barratt_Barratt_P2_Occ']])
print(bb_occ.describe())
bb_edu = pd.concat([cat_meta_train['Barratt_Barratt_P1_Edu'],cat_meta_train['Barratt_Barratt_P2_Edu']])
f, (ax1, ax2) = plt.subplots(1, 2)
ax1 = sns.histplot(data=bb_occ, ax=ax1)
ax2 = sns.histplot(data=bb_edu, ax=ax2)


occ_edu_df = pd.DataFrame([bb_occ.tolist(), bb_edu.tolist()],index=["occ","edu"]).transpose().dropna()
sns.regplot(data=occ_edu_df, x="occ", y="edu", x_estimator=np.mean)
sns.lmplot(data=occ_edu_df, x="occ", y="edu", x_estimator=np.mean, order=2)


bb_occ = cat_meta_train[['Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu']].max(axis=1)
bb_edu = cat_meta_train[['Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ']].max(axis=1)
occ_edu_df = pd.DataFrame([bb_occ, bb_edu],index=["occ","edu"]).transpose().dropna()
sns.regplot(data=occ_edu_df, x="occ", y="edu", x_estimator=np.mean)
sns.lmplot(data=occ_edu_df, x="occ", y="edu", x_estimator=np.mean, order=2)


cat_meta_train['single_parent'] = (
    (
        (cat_meta_train['Barratt_Barratt_P1_Edu'].notnull()) | 
        (cat_meta_train['Barratt_Barratt_P1_Occ'].notnull())
    ) &
    (cat_meta_train['Barratt_Barratt_P2_Edu'].isnull()) & 
    (cat_meta_train['Barratt_Barratt_P2_Occ'].isnull())
).astype(int)


cat_meta_merge_df = cat_meta_train.loc[:,['single_parent','participant_id']].merge(
    targets_train, on = 'participant_id', how = 'left'
).drop(columns = ['participant_id'])
cat_meta_merge_df.head()
sns.catplot(
    (cat_meta_merge_df
    .loc[cat_meta_merge_df['Sex_F'] == 1, :]
    .value_counts().reset_index()
    ), kind="bar",
    x="ADHD_Outcome", y="count", col="single_parent",
    height=4, aspect=.5,
)


print(quant_meta_train.isna().sum())
print(quant_meta_train.isna().sum().sum())


for quantcol in quant_meta_train.columns.tolist()[1:]:
    cat_meta_merge_df = quant_meta_train.loc[:,[quantcol,'participant_id']].merge(
        targets_train, on = 'participant_id', how = 'left'
    ).drop(columns = ['participant_id'])
    
    # Distribution of quantcol
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[8, 3], sharey=True)
    ax1 = sns.histplot(
        data=cat_meta_merge_df.loc[cat_meta_merge_df["Sex_F"] == 1,:], 
        x=quantcol,
        ax=ax1, stat="density", kde=True
    )
    ax1.set_title("Female")
    ax2 = sns.histplot(
        data=cat_meta_merge_df.loc[cat_meta_merge_df["Sex_F"] == 0,:], 
        x=quantcol,
        ax=ax2, stat="density", kde=True
    )
    ax2.set_title("Male")


for quantcol in quant_meta_train.columns.tolist()[1:]:
    cat_meta_merge_df = quant_meta_train.loc[:,[quantcol,'participant_id']].merge(
        targets_train, on = 'participant_id', how = 'left'
    ).drop(columns = ['participant_id'])
    
    # Distribution of quantcol
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[8, 3], sharey=True)
    ax1 = sns.histplot(
        data=cat_meta_merge_df.loc[cat_meta_merge_df["ADHD_Outcome"] == 1,:], 
        x=quantcol,
        ax=ax1, stat="density", kde=True
    )
    ax1.set_title("ADHD_Outcome = 1")
    ax2 = sns.histplot(
        data=cat_meta_merge_df.loc[cat_meta_merge_df["ADHD_Outcome"] == 0,:], 
        x=quantcol,
        ax=ax2, stat="density", kde=True
    )
    ax2.set_title("ADHD_Outcome = 0")


def custom_mode(foo_series):
    count_df = foo_series.value_counts().reset_index()
    count_df.columns = ['index','count']
    max_count = count_df["count"].max()
    return(count_df.loc[count_df["count"]== max_count,"index"].median())


def corr_heatmap(corr_df):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


cat_meta_merge_df = quant_meta_train.merge(
        targets_train, on = 'participant_id', how = 'left'
    ).drop(columns = ['participant_id'])

# Replace null values in all columns with nulls with the median of the column for training data
for col in cat_meta_merge_df.columns:
    if cat_meta_merge_df[col].isna().sum() > 0:  # Check if the column has NaN values
        if cat_meta_merge_df[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            #new_col = ("%s.nullMeanImpute" % col)
            cat_meta_merge_df[col] = cat_meta_merge_df[col].fillna(cat_meta_merge_df[col].median())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")

quant_corr_df = cat_meta_merge_df.corr()

corr_heatmap(quant_corr_df)


print(connectome_train.shape)
#connectome_corr_df = connectome_train.drop(columns = ['participant_id']).corr()
#corr_heatmap(connectome_corr_df)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# convert connectome to a matrix
train_connectome_matrix = connectome_train.drop(columns = ['participant_id']).to_numpy()
test_connectome_matrix = connectome_test.drop(columns = ['participant_id']).to_numpy()
print(train_connectome_matrix.shape)
print(test_connectome_matrix.shape)

# normalize the values (mean=0, sd=1)
connectome_scaler = StandardScaler()
connectome_scaler.fit(train_connectome_matrix)
train_connectome_scaler = connectome_scaler.transform(train_connectome_matrix)
test_connectome_scaler = connectome_scaler.transform(test_connectome_matrix)
print(train_connectome_scaler.shape)
print(test_connectome_scaler.shape)

# run PCA for enough components to get 80% of the explained variance
pca_connectome = PCA(n_components=.8)
pca_connectome.fit(train_connectome_scaler)
train_pca_connectome = pca_connectome.transform(train_connectome_scaler)
test_pca_connectome = pca_connectome.transform(test_connectome_scaler)


print('Explained variability per %i principal components: %.2f' % 
      (len(pca_connectome.explained_variance_ratio_), pca_connectome.explained_variance_ratio_.sum()))



train_pca_connectome_df = pd.DataFrame(data = train_pca_connectome
             , columns = ["PC"+str(x+1) for x in range(0,len(pca_connectome.explained_variance_ratio_))])
train_pca_connectome_df = pd.concat([connectome_train['participant_id'], train_pca_connectome_df],axis=1)
test_pca_connectome_df = pd.DataFrame(data = test_pca_connectome
             , columns = ["PC"+str(x+1) for x in range(0,len(pca_connectome.explained_variance_ratio_))])
test_pca_connectome_df = pd.concat([connectome_test['participant_id'], test_pca_connectome_df],axis=1)


plt.figure()
plt.figure(figsize=(3,3))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('PC1',fontsize=14)
plt.ylabel('PC2',fontsize=14)
plt.title("Top 2 Principal Component Analysis of Connectome",fontsize=12)
target_arr = targets_train.apply(
    lambda x: str(x["Sex_F"]) + str(x["ADHD_Outcome"]),
    axis=1)
targets = ['00', '01', '10', '11']
colors = ['r', 'g', 'b', 'k']
for target, color in zip(targets,colors):
    indicesToKeep = target_arr == target
    plt.scatter(train_pca_connectome_df.loc[indicesToKeep, 'PC1']
               , train_pca_connectome_df.loc[indicesToKeep, 'PC2'], c = color, s = 10)

plt.legend(targets,prop={'size': 15})


cat_meta_train['single_parent'] = (
    (
        (cat_meta_train['Barratt_Barratt_P1_Edu'].notnull()) | 
        (cat_meta_train['Barratt_Barratt_P1_Occ'].notnull())
    ) &
    (cat_meta_train['Barratt_Barratt_P2_Edu'].isnull()) & 
    (cat_meta_train['Barratt_Barratt_P2_Occ'].isnull())
).astype(int)


cat_meta_test['single_parent'] = (
    (
        (cat_meta_test['Barratt_Barratt_P1_Edu'].notnull()) | 
        (cat_meta_test['Barratt_Barratt_P1_Occ'].notnull())
    ) &
    (cat_meta_test['Barratt_Barratt_P2_Edu'].isnull()) & 
    (cat_meta_test['Barratt_Barratt_P2_Occ'].isnull())
).astype(int)


cat_meta_train['Barratt_Barratt_Edu_max'] = (
    cat_meta_train[['Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu']].max(axis=1))
cat_meta_train['Barratt_Barratt_Occ_max'] = (
    cat_meta_train[['Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ']].max(axis=1))
cat_meta_train['Barratt_Barratt_HomeGiver'] = (
    (cat_meta_train['Barratt_Barratt_P1_Occ'] == 0) |
     (cat_meta_train['Barratt_Barratt_P2_Occ'] == 0)
).astype(int)
cat_meta_train = cat_meta_train.drop(columns=[
    'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ'
])


cat_meta_test['Barratt_Barratt_Edu_max'] = (
    cat_meta_test[['Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu']].max(axis=1))
cat_meta_test['Barratt_Barratt_Occ_max'] = (
    cat_meta_test[['Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ']].max(axis=1))
cat_meta_test['Barratt_Barratt_HomeGiver'] = (
    (cat_meta_test['Barratt_Barratt_P1_Occ'] == 0) |
     (cat_meta_test['Barratt_Barratt_P2_Occ'] == 0)
).astype(int)
cat_meta_test = cat_meta_test.drop(columns=[
    'Barratt_Barratt_P1_Edu', 'Barratt_Barratt_P2_Edu',
    'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P2_Occ'
])


# MRI_Track_Scan_Location: set NA to 2 since all NA individuals were at Study Site 1 
# and most people who went to study site 1 did their MRI Track Scan at Location 2
cat_meta_train['MRI_Track_Scan_Location'] = cat_meta_train['MRI_Track_Scan_Location'].fillna(2.0)

# set Ethnicity values 9-11 to NA. set null value to 2.0 (Hispanic) if Ethnicity is 1.0 (Hispanic or Latino). 
cat_meta_train.loc[(
    (cat_meta_train["PreInt_Demos_Fam_Child_Ethnicity"]==1.0) &
    (cat_meta_train["PreInt_Demos_Fam_Child_Race"].isnull())),"PreInt_Demos_Fam_Child_Race"] = 2.0
cat_meta_train.loc[cat_meta_train["PreInt_Demos_Fam_Child_Race"] > 8,"PreInt_Demos_Fam_Child_Race"] = np.nan

# set Race values 2-3 to NA. set null value to 1.0 if Ethnicity is 2.0 
cat_meta_train.loc[(
    (cat_meta_train["PreInt_Demos_Fam_Child_Race"]==2.0) &
    (cat_meta_train["PreInt_Demos_Fam_Child_Ethnicity"].isnull())),"PreInt_Demos_Fam_Child_Ethnicity"] = 1.0
cat_meta_train.loc[cat_meta_train["PreInt_Demos_Fam_Child_Ethnicity"] > 1,"PreInt_Demos_Fam_Child_Ethnicity"] = np.nan



# MRI_Track_Scan_Location: set NA to 2 since all NA individuals were at Study Site 1 
# and most people who went to study site 1 did their MRI Track Scan at Location 2
cat_meta_test['MRI_Track_Scan_Location'] = cat_meta_test['MRI_Track_Scan_Location'].fillna(2.0)

# set Ethnicity values 9-11 to NA. set null value to 2.0 (Hispanic) if Ethnicity is 1.0 (Hispanic or Latino). 
cat_meta_test.loc[(
    (cat_meta_test["PreInt_Demos_Fam_Child_Ethnicity"]==1.0) &
    (cat_meta_test["PreInt_Demos_Fam_Child_Race"].isnull())),"PreInt_Demos_Fam_Child_Race"] = 2.0
cat_meta_test.loc[cat_meta_test["PreInt_Demos_Fam_Child_Race"] > 8,"PreInt_Demos_Fam_Child_Race"] = np.nan

# set Race values 2-3 to NA. set null value to 1.0 if Ethnicity is 2.0 
cat_meta_test.loc[(
    (cat_meta_test["PreInt_Demos_Fam_Child_Race"]==2.0) &
    (cat_meta_test["PreInt_Demos_Fam_Child_Ethnicity"].isnull())),"PreInt_Demos_Fam_Child_Ethnicity"] = 1.0
cat_meta_test.loc[cat_meta_test["PreInt_Demos_Fam_Child_Ethnicity"] > 1,"PreInt_Demos_Fam_Child_Ethnicity"] = np.nan



# Creating a list of all of the columns except the first
columns_to_encode = cat_meta_test.columns[1:].tolist()
columns_to_encode = [x for x in columns_to_encode if not x in ['single_parent','Barratt_Barratt_Edu_max','Barratt_Barratt_HomeGiver']]
# Print the columns to encode
print("Columns to encode:", columns_to_encode)





# encoding categorical data
for col in columns_to_encode:
    cat_meta_train[col] = cat_meta_train[col].astype('category')
drop_first_cols = [x for x in columns_to_encode if cat_meta_train[x].isnull().sum()==0]
keep_allval_cols = [x for x in columns_to_encode if cat_meta_train[x].isnull().sum()>0]

train_encoded1 = pd.get_dummies(cat_meta_train[drop_first_cols], drop_first=True)
train_encoded1 = train_encoded1.map(lambda x: 1 if x is True else (0 if x is False else x))
train_encoded2 = pd.get_dummies(cat_meta_train[keep_allval_cols])
train_encoded2 = train_encoded2.map(lambda x: 1 if x is True else (0 if x is False else x))


# Combine encoded columns with the rest of the DataFrame
cat_train_final = pd.concat([cat_meta_train.drop(columns=columns_to_encode), train_encoded1, train_encoded2], axis=1)

# Make sure it looks correct
cat_train_final.head()


# encoding categorical data
for col in columns_to_encode:
    cat_meta_test[col] = cat_meta_test[col].astype('category')
drop_first_cols = [x for x in columns_to_encode if cat_meta_test[x].isnull().sum()==0]
keep_allval_cols = [x for x in columns_to_encode if cat_meta_test[x].isnull().sum()>0]

test_encoded1 = pd.get_dummies(cat_meta_test[drop_first_cols], drop_first=True)
test_encoded1 = test_encoded1.map(lambda x: 1 if x is True else (0 if x is False else x))
test_encoded2 = pd.get_dummies(cat_meta_test[keep_allval_cols])
test_encoded2 = test_encoded2.map(lambda x: 1 if x is True else (0 if x is False else x))


# Combine encoded columns with the rest of the DataFrame
cat_test_final = pd.concat([cat_meta_test.drop(columns=columns_to_encode), test_encoded1, test_encoded2], axis=1)

# Make sure it looks correct
cat_test_final.head()


# test_pca_connectome_df
train_cat_FCM = pd.merge(cat_train_final, connectome_train, on = 'participant_id')
train_df = pd.merge(cat_train_final, quant_meta_train, on = 'participant_id')
train_df = pd.merge(train_df, train_pca_connectome_df, on = 'participant_id')
train_df.head()


test_cat_FCM = pd.merge(cat_test_final, connectome_test, on = 'participant_id')
test_df = pd.merge(cat_test_final, quant_meta_test, on = 'participant_id')
test_df = pd.merge(test_df, test_pca_connectome_df, on = 'participant_id')
test_df.head()


print(train_df.isna().sum())
print(train_df.isna().sum().sum())


# Replace null values in all columns with nulls with the median of the column for training data
for col in train_df.columns:
    if train_df[col].isna().sum() > 0:  # Check if the column has NaN values
        if train_df[col].dtype in ['float64', 'int64']:  # Ensure it's numeric
            train_df[col] = train_df[col].fillna(train_df[col].median())  # Avoid inplace
        else:
            print(f"Skipping non-numeric column: {col}")

print(train_df.isna().sum().sum()) # should now be zero


train_df.to_csv('X_train.csv', index=False)
test_df.to_csv('X_test.csv', index=False)

#X_train = train_df.drop(columns = ['participant_id'])
#Y_train = targets_train.drop(columns = ['participant_id'])





