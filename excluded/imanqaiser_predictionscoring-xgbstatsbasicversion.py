import pandas as pd
import glob
import os


# Get all the prediction CSVs from the folder where we uploaded it
preds_dir = "/kaggle/input/d/iraj09/plaastic-test-predictions-full/preds"
csv_files = glob.glob(os.path.join(preds_dir, "*.csv"))


# Did the processing in batches so need to concatenate them back together
all_results = []

for file in csv_files:
    df = pd.read_csv(file)
    df.columns = ['object_id', 'predicted_class']
    
    df_onehot = pd.get_dummies(df['predicted_class'], prefix='class')
    result = pd.concat([df['object_id'], df_onehot], axis=1)
    result = result.groupby('object_id').sum().reset_index()
    
    all_results.append(result)

# Aforementioned concatenation
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)

     # Need class_99 to be present even tho we didn't predict, gotta match sample
    if 'class_99' not in final_df.columns:
        final_df['class_99'] = 0  # Add column with 0s

    # Sorting by object_id to match the sample submission structure
    final_df = final_df.sort_values(by='object_id').reset_index(drop=True)

    # Print class-wise prediction counts
    class_totals = final_df.drop(columns=['object_id']).sum().astype(int)
    print("ğŸ“Š Class prediction counts:")
    print(class_totals)

    # Quick preview of the thingy
    print("\nğŸ§¾ Preview of final predictions:")
    print(final_df.head())
else:
    print("âš ï¸� No CSV files !")


# Save final result to CSV for the submission !
print("ğŸ’¾  Saving...")
final_df.to_csv("submission.csv", index=False)


# Print the whole final thingy
print(final_df)


