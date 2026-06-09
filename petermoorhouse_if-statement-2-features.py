def classify_personality(p):
    
    if p["Drained_after_socializing"] == "Yes" and p["Stage_fear"] == "Yes":
            return "Introvert"
        
    return "Extrovert"


import pandas as pd

TEST_PATH = "/kaggle/input/playground-series-s5e7/test.csv"
test_df = pd.read_csv(TEST_PATH)
test_df["Personality"] = test_df.apply(classify_personality, axis=1)


submission = test_df[["id", "Personality"]]
submission.to_csv("submission.csv", index=False)
submission.head()

