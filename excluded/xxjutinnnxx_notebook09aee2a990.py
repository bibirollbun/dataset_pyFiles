import pandas as pd

# =====================================================
# Step 1 — Ensemble weights
# =====================================================
sub_w = [0.75, 0.25]

# =====================================================
# Step 2 — Load submissions and sample
# =====================================================
sample = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
df0 = pd.read_csv('/kaggle/input/ensemble/submission.csv')
df1 = pd.read_csv('/kaggle/input/ensemble/submission1.csv')

# =====================================================
# Step 3 — Make sure both have same number of rows/columns
# =====================================================
# If they’re longer than sample, trim them
min_len = min(len(df0), len(df1), len(sample))
df0 = df0.head(min_len)
df1 = df1.head(min_len)
sample = sample.head(min_len)

# =====================================================
# Step 4 — Weighted average of prediction columns
# =====================================================
dfs = sample.copy()
dfs.iloc[:, 1:] = df0.iloc[:, 1:] * sub_w[0] + df1.iloc[:, 1:] * sub_w[1]

# Replace row_ids with the sample’s (so Kaggle format is correct)
dfs['row_id'] = sample['row_id']

# =====================================================
# Step 5 — Save final submission file
# =====================================================
dfs.to_csv('/kaggle/working/submission.csv', index=False)

print("✅ submission.csv created successfully (no NaNs)!")
print("Shape:", dfs.shape)
print(dfs.head())



# import os
# import pandas as pd

# sub_w=[0.75, 0.25]


# list_TARGETs = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
# list_targets_0 = [f'{TARGET} 0' for TARGET in list_TARGETs]
# list_targets_1 = [f'{TARGET} 1' for TARGET in list_TARGETs]

# df0 = pd.read_csv("/kaggle/input/ensemble/submission.csv")
# df1 = pd.read_csv("/kaggle/input/ensemble/submission1.csv")

# df0 = df0.rename(columns={TARGET : f'{TARGET} 0' for TARGET in list_TARGETs})
# df1 = df1.rename(columns={TARGET : f'{TARGET} 1' for TARGET in list_TARGETs})

# dfs = pd.merge(df0,df1,on=['row_id'])

# for i in range(len(list_TARGETs)):
#     dfs[list_TARGETs[i]] = dfs[list_targets_0[i]]*sub_w[0] + sub_w[1]*dfs[list_targets_1[i]]
             
# for col0,col1 in zip(list_targets_0, list_targets_1):
#     del dfs[col0]
#     del dfs[col1]
    
    
# dfs.to_csv("submission.csv", index=False)


import os
import pandas as pd

# =====================================================
# STEP 1 — Setup
# =====================================================
sub_w = [0.75, 0.25]

# Get all target species
list_TARGETs = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))
list_targets_0 = [f'{TARGET} 0' for TARGET in list_TARGETs]
list_targets_1 = [f'{TARGET} 1' for TARGET in list_TARGETs]

# =====================================================
# STEP 2 — Read the two submission files
# =====================================================
df0 = pd.read_csv('/kaggle/input/ensemble/submission.csv')
df1 = pd.read_csv('/kaggle/input/ensemble/submission1.csv')

# Rename columns to distinguish sources
df0 = df0.rename(columns={TARGET: f'{TARGET} 0' for TARGET in list_TARGETs})
df1 = df1.rename(columns={TARGET: f'{TARGET} 1' for TARGET in list_TARGETs})

# =====================================================
# STEP 3 — Merge and compute ensemble
# =====================================================
# Merge with outer join to keep all rows
dfs = pd.merge(df0, df1, on=['row_id'], how='outer').fillna(0)

for i in range(len(list_TARGETs)):
    dfs[list_TARGETs[i]] = (
        dfs[list_targets_0[i]] * sub_w[0] +
        dfs[list_targets_1[i]] * sub_w[1]
    )

# Drop temporary columns
for col0, col1 in zip(list_targets_0, list_targets_1):
    del dfs[col0]
    del dfs[col1]


# =====================================================
# STEP 4 — Filter only valid test-like rows (_5, _10, _15)
# =====================================================
dfs = dfs[dfs['row_id'].str.endswith(('_5', '_10', '_15'))].reset_index(drop=True)

print("Filtered rows:", dfs.shape[0])
print("Sample row_ids after filtering:")
print(dfs['row_id'].head())

# =====================================================
# STEP 5 — Match the Kaggle sample submission format
# =====================================================
sample = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')

# Make sure column structure matches
sample.iloc[:, 1:] = dfs.iloc[:len(sample), 1:].values

# =====================================================
# STEP 6 — Save final valid submission
# =====================================================
sample.to_csv('/kaggle/working/submission.csv', index=False)

print("✅ submission.csv created successfully!")
print(sample.head())



import os
print(os.listdir('/kaggle/working'))



print("Before filtering:", len(pd.read_csv('/kaggle/input/ensemble/submission.csv')))
print("After merge:", dfs.shape[0])
print("After filtering (_5,_10,_15):", dfs.shape[0])



dfs.head()



sample = pd.read_csv('/kaggle/input/birdclef-2025/sample_submission.csv')
sample.head(100)





