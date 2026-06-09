import pandas as pd
pd.set_option('display.max_columns', None)


# train data load
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')


# train supplement load
dataset1_tc = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv').rename(columns={'TC_mean': 'Tc'})
dataset3_tg = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')
dataset4_ffv = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')


## only Tg
enriched_tg = pd.read_csv('/kaggle/input/external-polymer-data/TgSS_enriched_cleaned.csv',usecols=['SMILES', 'Tg'])
jcim_sup_bigsmiles_tg = pd.read_csv('/kaggle/input/external-polymer-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)']).rename(columns={'Tg (C)': 'Tg'})
data_tg3 = pd.read_excel('/kaggle/input/external-polymer-data/data_tg3.xlsx').rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15


## density
data_dnst = pd.read_excel('/kaggle/input/external-polymer-data/data_dnst1.xlsx').rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
data_dnst['Density'] = data_dnst['Density'].astype('float64')
data_dnst['Density'] -= 0.118


pm1m = pd.read_csv('/kaggle/input/external-polymer-data/PI1070.csv', usecols=['SMILES', 'density', 'Rg', 'thermal_conductivity']).rename(columns={"density": "Density", "thermal_conductivity": "Tc"})


import pandas as pd
import numpy as np

def update_or_add_smiles_entries(main_df, ext_df, target_column, add_new_smiles=True):
    """
    Merge polymer datasets with robust handling of new SMILES addition.
    
    Parameters:
    - main_file: Path to main CSV file (train.csv)
    - external_file: Path to external CSV file
    - target_column: Column to merge (Tg, FFV, Tc, Density, Rg)
    - add_new_smiles: Whether to add new SMILES with valid target values
    
    Returns:
    - Merged DataFrame
    - Tuple of (filled_values_count, new_smiles_added)
    """
    
    # Valid property columns
    valid_columns = {'Tg', 'FFV', 'Tc', 'Density', 'Rg'}
    
    # Validate target column
    if target_column not in valid_columns:
        raise ValueError(f"Target column must be one of: {', '.join(valid_columns)}")
    
    # Check required columns
    required_columns = {'SMILES', target_column}
    if not required_columns.issubset(ext_df.columns):
        missing = required_columns - set(ext_df.columns)
        raise ValueError(f"External dataset missing columns: {', '.join(missing)}")
    
    if 'SMILES' not in main_df.columns:
        raise ValueError("Main dataset must contain 'SMILES' column")
    
    # Initialize counters
    filled_count = 0
    new_smiles_count = 0
    
    # Process existing SMILES (fill missing values)
    merged = main_df.merge(
        ext_df[['SMILES', target_column]].dropna(subset=[target_column]),
        on='SMILES', 
        how='left',
        suffixes=('', '_ext')
    )
    
    # Fill only if main has NA and external has value
    mask = main_df[target_column].isna() & merged[f'{target_column}_ext'].notna()
    filled_count = mask.sum()
    
    if filled_count > 0:
        main_df.loc[mask, target_column] = merged.loc[mask, f'{target_column}_ext']
        print(f"Filled {filled_count} missing values for {target_column}")
    
    # Process new SMILES if requested
    if add_new_smiles:
        existing_smiles = set(main_df['SMILES'])
        
        # Get new SMILES with valid target values
        new_smiles = ext_df[
            ~ext_df['SMILES'].isin(existing_smiles) & 
            ext_df[target_column].notna()
        ].copy()
        
        if not new_smiles.empty:
            # Create template with all columns from main_df
            new_rows = pd.DataFrame(columns=main_df.columns)
            
            # Fill SMILES and target column
            new_rows['SMILES'] = new_smiles['SMILES']
            new_rows[target_column] = new_smiles[target_column]
            
            # Set other columns to NA (except ID if exists)
            for col in new_rows.columns:
                if col not in ['SMILES', target_column, 'id']:
                    new_rows[col] = np.nan
            
            # Generate new IDs if column exists
            if 'id' in main_df.columns:
                max_id = main_df['id'].max()
                new_rows['id'] = range(max_id + 1, max_id + 1 + len(new_smiles))
            
            # Safe concatenation by ensuring matching columns
            main_df = pd.concat([
                main_df,
                new_rows[main_df.columns]  # Ensure column order matches
            ], ignore_index=True)
            
            new_smiles_count = len(new_smiles)
            print(f"Added {new_smiles_count} new SMILES entries with valid {target_column} values")
        else:
            print("No new SMILES with valid target values found to add")
    
    return main_df #, (filled_count, new_smiles_count)


print(f"train_df shape before merging: {train_df.shape}")
print(f"Tg value count before merging: {train_df['Tg'].count()}")
target_column = "Tg"
print("="*30 + 'Merging dataset3_tg' + "="*30)
train_df = update_or_add_smiles_entries(train_df, dataset3_tg, target_column)
print("="*30 + 'Merging enriched_tg' + "="*30)
train_df = update_or_add_smiles_entries(train_df, enriched_tg, target_column)
print("="*30 + 'Merging jcim_sup_bigsmiles_tg' + "="*30)
train_df = update_or_add_smiles_entries(train_df, jcim_sup_bigsmiles_tg, target_column)
print("="*30 + 'Merging data_tg3' + "="*30)
train_df = update_or_add_smiles_entries(train_df, data_tg3, target_column)
print(f"train_df shape after merging: {train_df.shape}")
print(f"Tg value count after merging: {train_df['Tg'].count()}")


print(f"train_df shape before merging: {train_df.shape}")
print(f"FFV value count before merging: {train_df['FFV'].count()}")
target_column = "FFV"
print("="*30 + 'Merging dataset4_ffv' + "="*30)
train_df = update_or_add_smiles_entries(train_df, dataset4_ffv, target_column)
print(f"train_df shape after merging: {train_df.shape}")
print(f"FFV value count after merging: {train_df['FFV'].count()}")


print(f"train_df shape before merging: {train_df.shape}")
print(f"Tc value count before merging: {train_df['Tc'].count()}")
target_column = "Tc"
print("="*30 + 'Merging dataset1_tc' + "="*30)
train_df = update_or_add_smiles_entries(train_df, dataset1_tc, target_column)
print("="*30 + 'Merging dataset4_ffv' + "="*30)
train_df = update_or_add_smiles_entries(train_df, pm1m, target_column)
print(f"train_df shape: {train_df.shape}")
print(f"Tc value count: {train_df['Tc'].count()}")


print(f"train_df shape before merging: {train_df.shape}")
print(f"Density value count before merging: {train_df['Density'].count()}")
target_column = "Density"
print("="*30 + 'Merging data_dnst' + "="*30)
train_df = update_or_add_smiles_entries(train_df, data_dnst, target_column)
print("="*30 + 'Merging pm1m' + "="*30)
train_df = update_or_add_smiles_entries(train_df, pm1m, target_column)
print(f"train_df shape after merging: {train_df.shape}")
print(f"Density value count after merging: {train_df['Density'].count()}")


print(f"train_df shape before merging: {train_df.shape}")
print(f"Rg value count before merging: {train_df['Rg'].count()}")
target_column = "Rg"
print("="*30 + 'Merging pm1m' + "="*30)
train_df = update_or_add_smiles_entries(train_df, pm1m, target_column)
print(f"train_df shape after merging: {train_df.shape}")
print(f"Rg value count after merging: {train_df['Rg'].count()}")


train_df.describe()


# Example: selected columns
selected_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Create missing info DataFrame
missing_info_df = pd.DataFrame({
    'Missing Count': train_df[selected_columns].isna().sum(),
    'Missing %': (train_df[selected_columns].isna().mean() * 100).round(2)
})

# Sort by missing % (optional)
missing_info_df = missing_info_df.sort_values(by='Missing %', ascending=False)

# Display nicely in Jupyter
from IPython.display import display
display(missing_info_df)



train_df.to_csv('PolymerDataset-Merged_Tg_FFV_Tc_Density_Rg.csv', index=False)




