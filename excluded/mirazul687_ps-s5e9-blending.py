import pandas as pd


def h_blend(path, fs_names, params):
    
    
    # The 'params' dictionary is assigned to a shorter alias 'dk' for convenience.
    dk = params
        
    
    def da(dk, sorting_direction):
        
        def read_subm(dk, i):      # Helper function to read a single submission file.
            
            # Get the name of the submission file from the config.
            tnm = dk["subm"][i]["name"]
            # Construct the full file path.
            FiN = dk["path"] + tnm + ".csv"
            
            # Read the CSV and rename the target column to the submission's name.
            return pd.read_csv(FiN).rename(columns={'target': tnm, dk["target"]: tnm})
            
            
        # Read all submission files into a list of DataFrames.
        dfs_subm = [read_subm(dk, i) for i in range(len(dk["subm"]))]
        
        # Merge the first two DataFrames on the ID column.
        df_subms = pd.merge(dfs_subm[0], dfs_subm[1], on=[dk['id']])
        
        # Iteratively merge the rest of the DataFrames.
        for i in range(2, len(dk["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
            
            
        # --- Dynamic Weighting Logic ---
        
        # Get the names of the submission columns.
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        
        # Get the weight correction values from the config.
        corrects = [wt for wt in dk["subwts"]]
        # Get the base weights for each submission from the config.
        weights = [subm['weight'] for subm in dk["subm"]]
        
        
        def alls(x, sd=sorting_direction, cs=cols):
            
            reverse = True if sd == 'desc' else False
            # Create a dictionary of {submission_name: prediction_value}.
            tes = {c: x[c] for c in cs}.items()
            # Sort the items and return only the sorted submission names.
            # Note: The original lambda `lambda :[1]` is a bug; it doesn't sort.
            # A correct implementation would be `lambda item: item[1]`.
            subms_sorted = [t[0] for t in sorted(tes, key=lambda item: item[1], reverse=reverse)]
            
            return subms_sorted

        
        def correct(x, cs=cols, w=weights, cw=corrects):
    
            # Find the rank (index) of each submission in the sorted list.
            ic = [x['alls'].index(c) for c in short_name_cols]
            # Calculate the dynamically weighted sum of predictions.
            # Prediction * (base_weight + correction_weight_for_its_rank)
            cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            return sum(cS)
            
            
        # Apply the functions row-wise to create the final predictions.
        df_subms['alls'] = df_subms.apply(lambda x: alls(x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        
        # --- Formatting and Saving Output (for inspection) ---
        
        # Rename columns for display purposes.
        schema_rename = {old_nc: new_shnc for old_nc, new_shnc in zip(cols, short_name_cols)}
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]: "ensemble"})
        
        # Insert decorative columns for better visual separation in the output.
        # Note: There are syntax errors in the original code here.
        # `df*subms` should be `df_subms`.
        df_subms.insert(loc=1, column=' * ', value=['   '] * len(df_subms))
        df_subms[' * '] = df_subms[' * '].astype(str)
        
        # Set pandas display options for viewing the intermediate DataFrame.
        pd.set_option('display.max_rows', 21)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        # Define the column order for the displayed DataFrame.
        # Note: `short*name_cols` is a syntax error, should be `short_name_cols`.
        vcols = [dk['id']] + [' * '] + short_name_cols + [' * '] + ['alls'] + [' * '] + ['ensemble']
        
        df_subms_view = df_subms[vcols]
        # The original code has `display(df_subms.head(4))`, which would not use `vcols`.
        # Assuming the intent was to display the formatted view.
        display(df_subms_view.head(4))
        
        # Revert pandas display options and column names for final output.
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble": dk["target"]})
        
        # Save the intermediate result to a CSV file.
        df_subms.to_csv(f'tida_alls.csv', index=False)
        
        return df_subms

    # --- Final Ensemble ---

    # Read a sample submission file to use as a template for the final output.
    sample_subm = pd.read_csv(path + fs_names[1] + ".csv")


    
    # Runs the core blending logic twice (once for descending sort, once for
    # ascending sort) and creates a final blend of these two results.
    
    def ensemble_da(dk, submission=sample_subm): 
        
        # Get key parameters from the config dictionary.
        _id, target, d, a = dk['id'], dk['target'], dk['desc'], dk['asc']
        
        # 1. Generate the blend using descending rank sort.
        dfs = da(dk, 'desc')
        dfD = dfs[[_id, target]]
        dfD.to_csv(f'tida_desc.csv', index=False)
        
        # 2. Generate the blend using ascending rank sort.
        dfs = da(dk, 'asc')
        dfA = dfs[[_id, target]]
        dfA.to_csv(f'tida_asc.csv', index=False)
        
        # 3. Create the final submission by blending the descending and ascending results.
        # Final Prediction = (desc_prediction * desc_weight) + (asc_prediction * asc_weight)
        submission[target] = dfD[target] * d + a * dfA[target]
        return submission
        
    # Execute the final ensembling process.
    da = ensemble_da(dk)

    return da


path = '/kaggle/input/music-beats-prediction/' + 'submission_'

file_short_names = ['26.38309','26.38419','26.38444']

params = {
        'path'   : path,
        'id'     : 'id',
        'target' : "BeatsPerMinute",
        'desc'   : 0.70,
        'asc'    : 0.30,
        'subwts' : [+0.121, -0.044, -0.077],
        'subm'   : [
             { 'name':file_short_names[0],'weight':0.90, },
             { 'name':file_short_names[1],'weight':0.08, },
             { 'name':file_short_names[2],'weight':0.02, },
        ]
    }

df = h_blend ( path, file_short_names, params )
df.to_csv('submission.csv', index=False)
display(df.head(5))

