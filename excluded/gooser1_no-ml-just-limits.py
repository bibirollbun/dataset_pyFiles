import pandas as pd

import operator

from sklearn.metrics import accuracy_score


train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


extro_markers = { 
    "Time_spent_Alone" : ("<", 4), 
    "Stage_fear" : "No", 
    "Social_event_attendance": (">", 3), 
    "Going_outside": (">", 3), 
    "Drained_after_socializing": "No", 
    "Friends_circle_size": (">", 5.0), 
    "Post_frequency": (">", 3)
}

intro_markers = { 
    "Time_spent_Alone" : (">", 4), 
    "Stage_fear" : "Yes", 
    "Social_event_attendance": ("<", 3), 
    "Going_outside": ("<", 3), 
    "Drained_after_socializing": "Yes", 
    "Friends_circle_size": ("<", 4.0), 
    "Post_frequency": ("<", 3)
}


ops = {
    ">": operator.gt,
    "<": operator.lt,
}

def count_satisfied_markers(row, markers):
    count = 0
    for col, rule in markers.items():
        value = row.get(col)
        if isinstance(rule, tuple):
            op, threshold = rule
            if pd.notnull(value) and ops[op](value, threshold):
                count += 1
        else:
            if pd.notnull(value) and value == rule:
                count += 1
    return count

train_df["extro_marker_count"] = train_df.apply(lambda row: count_satisfied_markers(row, extro_markers), axis=1)
train_df["intro_marker_count"] = train_df.apply(lambda row: count_satisfied_markers(row, intro_markers), axis=1)
train_df.head(3)


test_df["extro_marker_count"] = test_df.apply(lambda row: count_satisfied_markers(row, extro_markers), axis=1)
test_df["intro_marker_count"] = test_df.apply(lambda row: count_satisfied_markers(row, intro_markers), axis=1)


train_df["y_pred"] = train_df.apply(lambda row: "Introvert" if row["intro_marker_count"] > row["extro_marker_count"] 
                                    else "Extrovert", axis=1)


test_df["y_pred"] = test_df.apply(lambda row: "Introvert" if row["intro_marker_count"] > row["extro_marker_count"] 
                                    else "Extrovert", axis=1)


acc = accuracy_score(train_df["y_pred"], train_df["Personality"] )
acc


submission = pd.DataFrame(
    {
        'id': test_df["id"],
        'Personality': test_df["y_pred"]
    }
)
submission.to_csv("submission.csv", index = None)


submission

