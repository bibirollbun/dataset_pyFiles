import json
import os

# Define the input file path
input_file_path = '/kaggle/input/arc-prize-2025/sample_submission.json'

# Load the JSON file
with open(input_file_path, 'r') as f:
    submission_data = json.load(f)

# Define the output file path
output_file_path = 'submission.json'

# Save the data to the output file
with open(output_file_path, 'w') as f:
    json.dump(submission_data, f, indent=4)

