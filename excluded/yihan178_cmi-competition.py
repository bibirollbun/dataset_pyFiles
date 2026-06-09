import os
import polars as pl
import pandas as pd
import kaggle_evaluation.cmi_inference_server


# Step 1: Load the training data from a CSV file
train = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')



# Step 2: Show the first few rows so we can see what the data looks like
print("Here's what our training data looks like:")
print(train.head())


# Step 3: Let's see all the different gestures (actions) in our data
print("\nAll unique gestures in the training data:")
print(train['gesture'].unique())


# Step 4: Count how many times each gesture appears in the data
gesture_counts = train['gesture'].value_counts()
print("\nHow many times each gesture appears:")
print(gesture_counts)


# Step 5: Find out which gesture happens the most in the training data
# Count gestures and sort
gesture_counts = train['gesture'].value_counts().sort('count', descending=True)

# Show the value_counts result for clarity
print(gesture_counts)



# Take the gesture with the highest count (the first row)
most_common_gesture = gesture_counts.row(0)[0]  # [0] is the gesture, [1] is the count
print("\nThe most common gesture is:", most_common_gesture)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # No matter what the input data is, always return the most common gesture
    return most_common_gesture


# We give our simple prediction function to the competition's grading system
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

