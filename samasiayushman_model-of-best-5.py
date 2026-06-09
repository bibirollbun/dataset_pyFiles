import pandas as pd
import numpy as np

# Load the datasets
csv1 = pd.read_csv('/kaggle/input/datass/submission_prophet_enhanced.csv')
csv2 = pd.read_csv('/kaggle/input/datass/submission.csv')
csv3 = pd.read_csv('/kaggle/input/datass/submission_prophet_enhanced (1).csv')
csv4 = pd.read_csv('/kaggle/input/datass/submission_prophet_enhanced (2).csv')

# Extract the first column (IDs) and headers
id_column = csv1.iloc[:, 0]  # The ID column
headers = csv1.columns       # The headers

# Extract the categorical parts (excluding the first column)
categorical1 = csv1.iloc[:, 1:]
categorical2 = csv2.iloc[:, 1:]
categorical3 = csv3.iloc[:, 1:]
categorical4 = csv4.iloc[:, 1:]

# Initialize an empty DataFrame for results
weighted_results = pd.DataFrame()

# Weights
weight_csv1 = 6  # 60%
weight_others = 4  # remaining 40% across csv2, csv3, csv4 (each ~13.33%)

for col in categorical1.columns:
    # Repeat csv1 predictions 6 times (60% weight)
    weighted_preds = [categorical1[col]] * weight_csv1
    # Repeat other predictions once each (~13.3% each)
    weighted_preds += [categorical2[col], categorical3[col], categorical4[col]]
    
    # Concatenate predictions
    combined = pd.concat(weighted_preds, axis=1)
    # Take the mode per row
    weighted_results[col] = combined.mode(axis=1)[0]

# Reinsert the ID column
weighted_results.insert(0, headers[0], id_column)

# Save the result
weighted_results.to_csv('submission.csv', index=False)

# Compare with csv1
differences = csv1.compare(weighted_results)
print(differences)


