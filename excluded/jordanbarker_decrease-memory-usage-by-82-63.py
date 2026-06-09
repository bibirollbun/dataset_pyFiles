import numpy as np
import pandas as pd

def prepare_data(df: pd.DataFrame, is_train: bool = True):
    """
    Prepares the dataset for training or testing by renaming columns, handling missing values,
    converting categorical and numerical features.
    
    Args:
        df (pd.DataFrame): The input dataframe (train or test).
        is_train (bool): Indicates if the dataframe is training data (default is True).
        
    Returns:
        pd.DataFrame: The processed dataframe.
    """
    
    # Rename columns
    columns = [
        'id', 'brand', 'material', 'size', 'compartments', 
        'laptop_compartment', 'is_waterproof', 'style', 'color', 
        'weight_capacity'
    ]
    
    if is_train:
        columns.append('price')
    
    df.columns = columns
    
    if is_train:
        # Keep the id column on the test set since it's used for the submission
        df = df.drop(columns='id')

    # Convert categories to int
    size_mapping = {"Small": 0, "Medium": 1, "Large": 2}
    df["size"] = df["size"].map(size_mapping).fillna(-1).astype(int)

    brand_mapping = {"Adidas": 0, "Puma": 1, "Nike": 2, "Jansport": 3, "Under Armour": 4}
    df["brand"] = df["brand"].map(brand_mapping).fillna(-1).astype(int)

    material_mapping = {"Leather": 0, "Nylon": 1, "Polyester": 2, "Canvas": 3}
    df["material"] = df["material"].map(material_mapping).fillna(-1).astype(int)

    style_mapping = {"Backpack": 0, "Messenger": 1, "Tote": 2}
    df["style"] = df["style"].map(style_mapping).fillna(-1).astype(int)

    color_mapping = {"Black": 0, "Gray": 1, "Red": 2, "Pink": 3, "Green": 4, "Blue": 5}
    df["color"] = df["color"].map(color_mapping).fillna(-1).astype(int)

    binary_mapping = {"No": 0, "Yes": 1}
    df["laptop_compartment"] = df["laptop_compartment"].map(binary_mapping).fillna(-1).astype(int)
    df["is_waterproof"] = df["is_waterproof"].map(binary_mapping).fillna(-1).astype(int)
    
    df['weight_capacity'] = df['weight_capacity'].fillna(-1)
    df['compartments'] = df['compartments'].astype(int)
    
    return df
    

test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/test.csv')
train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/train.csv')
train_extra_df = pd.read_csv(r'/kaggle/input/playground-series-s5e2/training_extra.csv')
train_df = pd.concat([train_df, train_extra_df], ignore_index=True)

memory_usage_before = train_df.memory_usage(deep=True).sum()
train_df = prepare_data(train_df)
memory_usage_after = train_df.memory_usage(deep=True).sum()
perc_change = 100 * (memory_usage_after - memory_usage_before) / memory_usage_before

print(f"Percentage change in memory usage: {perc_change:.2f}%")

