 # ğŸ“Š Step 1: Import Libraries

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import pairwise_distances
from collections import Counter
import numpy as np
import pandas as pd
import xgboost as xgb


df_orig = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
df_orig = df_orig.rename(columns={"Personality": "match_p"}).drop_duplicates()

# Convert match_p to numeric
df_orig["match_p"] = df_orig["match_p"].astype(str)



# ğŸ§¼ 3. Combine train, test, and original for consistent encoding

# Combine train and test
full_data = pd.concat([train.drop(columns=["Personality"]), test])
full_data["source"] = ["train"] * len(train) + ["test"] * len(test)

# Select only the features used for personality matching
match_features = [
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
    'Going_outside', 'Drained_after_socializing', 'Friends_circle_size',
    'Post_frequency'
]

# Fit encoder on both original and full_data
encoder = OrdinalEncoder()
encoded_orig = encoder.fit_transform(df_orig[match_features])
encoded_full = encoder.transform(full_data[match_features])



#ğŸ”� 4. Calculate similarity to original data

# Weâ€™ll now compare each row in the current train/test to all rows in df_orig.

# Compute Hamming distance (for categorical/ordinal)
distances = pairwise_distances(encoded_full, encoded_orig, metric="hamming")  # Gives values between 0 and 1

# For each row in train/test, find the most similar row(s)
closest_idx = distances.argmin(axis=1)  # Index of best match
min_dist = distances.min(axis=1)        # Smallest distance (similarity score)



### So now you have:

    # match_p_soft: predicted personality from similar past behavior

    # match_score: how close that match is (1.0 is perfect match)


#Use It as a Feature

Now, split back into train/test and continue with encoding + model training as before:


# Restore train and test
train["match_p_soft"] = full_data.loc[full_data["source"] == "train", "match_p_soft"].values
train["match_score"] = full_data.loc[full_data["source"] == "train", "match_score"].values

test["match_p_soft"] = full_data.loc[full_data["source"] == "test", "match_p_soft"].values
test["match_score"] = full_data.loc[full_data["source"] == "test", "match_score"].values



train["match_p_soft"] = train["match_p_soft"].fillna("Unknown")
test["match_p_soft"] = test["match_p_soft"].fillna("Unknown")

ordinal = OrdinalEncoder()
train["match_p_soft"] = ordinal.fit_transform(train[["match_p_soft"]])
test["match_p_soft"] = ordinal.transform(test[["match_p_soft"]])



# ğŸš€ Final Model Training

# Add both features to your XGBoost model:

X = train.drop(columns=["id", "Personality", "Personality_encoded"])
y = train["Personality_encoded"]
X_test = test.drop(columns=["id"])


