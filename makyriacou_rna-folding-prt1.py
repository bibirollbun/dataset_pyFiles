import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from joblib import Parallel, delayed



def comparison_2_seq(seq_1, seq_2): 

    # sequence values 
    seq_1_x1_vals, seq_1_y1_vals, seq_1_z1_vals = seq_1['x_1'].values, seq_1['y_1'].values, seq_1['z_1'].values
    seq_2_x1_vals, seq_2_y1_vals, seq_2_z1_vals = seq_2['x_1'].values, seq_2['y_1'].values, seq_2['z_1'].values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot for first subset
    ax.scatter(seq_1_x1_vals, seq_1_y1_vals, seq_1_z1_vals, c='blue', s=50, marker='o', 
               label='Subset 1')
    ax.plot(seq_1_x1_vals, seq_1_y1_vals, seq_1_z1_vals, color='blue', 
            linestyle='--', linewidth=2)
    
    # Scatter plot for second subset
    ax.scatter(seq_2_x1_vals, seq_2_y1_vals, seq_2_z1_vals, c='orange', 
               s=50, marker='^', label='Subset 2')    
    ax.plot(seq_2_x1_vals, seq_2_y1_vals, seq_2_z1_vals, color='orange', 
            linestyle='-', linewidth=2)
    
    # Labels and title
    ax.set_title('3D Scatter Plot: Comparing Two Subsets')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    
    plt.show()



def analysis(df, drop_cols = None): 
    print(f"Data Shape: {df.shape}")
    null_count = df.isnull().sum().sum()
    print(f"Null values: {null_count}")
    if null_count / df.shape[0] < 0.3: df = df.dropna()
    else: df = df.fillna(0)

    print(f"Null values after processing: {df.isnull().sum().sum()}")
    if drop_cols: df = df.drop(columns=drop_cols)
    print(f"Data Shape: {df.shape}")
    df.info() 
    return df


def seq_map(seq):
    return [seq_dict.get(char, 4) for char in seq][0]


def fix_dataset(seq_df, labels_df=None):
    data = {}
    start_time = time.time()  # Start timing

    for indx, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Processing Sequences"):
        seq_id, seq, seq_len = row.target_id, row.sequence, row.sequence_legnth

        # Convert sequence to numerical values
        numerical_seq = [seq_map.get(nuc, 4) for nuc in seq]

        # Count nucleotide occurrences using pandas' value_counts
        counts = pd.Series(list(seq)).value_counts()
        A_count = counts.get('A', 0)
        C_count = counts.get('C', 0)
        G_count = counts.get('G', 0)
        U_count = counts.get('U', 0)

        # Create DataFrame efficiently
        df = pd.DataFrame({
            'RNA_seq': numerical_seq,
            'seq_len': range(seq_len),
            'seq_id': [f"{seq_id}_{i}" for i in range(1, seq_len+1)],
            'A_count': A_count,
            'C_count': C_count,
            'G_count': G_count,
            'U_count': U_count} )

        # Check for labels if they exist
        if labels_df is not None:
            seq_id_df = labels_df[labels_df['ID'].str.startswith(f"{seq_id}_")]

            if not seq_id_df.empty:
                coords = seq_id_df[['x_1', 'y_1', 'z_1']].to_numpy()
                
                if coords.shape[0] == seq_len:
                    # Normalize coordinates efficiently
                    mean = coords.mean(axis=0)
                    std = coords.std(axis=0) + 1e-8  # Avoid division by zero
                    coords_norm = (coords - mean) / std
                    df[['x_1', 'y_1', 'z_1']] = coords_norm
                else:
                    print(f"Warning: Mismatch for {seq_id} - coords: {coords.shape[0]}, seq_len: {seq_len}")
                    continue  
        # Store processed DataFrame
        data[seq_id] = df

    # Merge all data after loop
    merge_data = pd.concat(data.values(), ignore_index=True)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return merge_data


def clean_data(data, remove_col, std_cols): 
    print(f'BEFORE Data:{data.shape}')
    # Remove missing values
    data = data.dropna().copy()  

    # Remove and store the ID column efficiently
    id_col = data.pop(remove_col)  

    # StandardScaler
    scaler = StandardScaler()
    data[std_cols] = scaler.fit_transform(data[std_cols])
    print(f'AFTER Data:{data.shape}')

    return data, id_col


def model_prediction(df_train, df_val, target_data, **kwargs):
    # Extract target values efficiently
    y_train = df_train[target_data].to_numpy().T  # Shape (3, N)
    y_val = df_val[target_data].to_numpy().T  # Shape (3, N)
    X_train, X_val = df_train.drop(columns=target_data), df_val.drop(columns=target_data)

    print(f'... Train:{X_train.shape, y_train.shape}')
    print(f'... Validation:{X_val.shape, y_val.shape}')
    
    model_params = kwargs.get("model_params", {})

    # train a model for each coordinate
    def train_model(coord_name, coord_values):
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, coord_values)
        return coord_name, model

    # Train models with parallel training 
    models = dict(Parallel(n_jobs=-1)(
        delayed(train_model)(coord_name, y) for coord_name, y in zip(['x', 'y', 'z'], y_train)))
    print('... Models Trained')
    
    # Evaluate models
    predictions = {}
    for coord_name, model, y_true in zip(models.keys(), models.values(), y_val):
        y_pred = model.predict(X_val)
        predictions[coord_name]  = y_pred
        mse = mean_squared_error(y_true, y_pred)
        print(f'{coord_name} MSE: {mse:.4f}')

    return models, predictions




def predict_vs_true(val_data, target_data,  predictions):
    # target values 
    y_val = clean_val_data[target_data].to_numpy().T  
    X_val = clean_val_data.drop(columns=target_data)
    
    
    coordinates = ['x', 'y', 'z']
    true_values = [y_val[0], y_val[1], y_val[2]]
    pred_values = [predictions['x'], predictions['y'], predictions['z']]
    
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Loop to create the subplots for x, y, z
    for i, coord in enumerate(coordinates):
        axes[i].scatter(X_val.iloc[:, 1], true_values[i], color='blue', label=f'True {coord.upper()}')
        axes[i].scatter(X_val.iloc[:, 1], pred_values[i], color='red', label=f'Predicted {coord.upper()}', marker='x')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel(f'{coord.upper()} Coordinate')
        axes[i].set_title(f'True vs Predicted {coord.upper()}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()


drop_cols = ['temporal_cutoff', 'description', 'all_sequences']
seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


std_cols = ['seq_len', 'A_count', 'C_count', 'G_count', 'U_count']
remove_col = 'seq_id'


target_coord = ['x_1', 'y_1', 'z_1']
parmas = {'objective':'reg:squarederror',
        'n_estimators':1000,
        'max_depth':7,
        'learning_rate':0.1,
        'subsample':0.8,
        'colsample_bytree':0.8}


train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')

validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')

test_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/test_sequences.csv')
#sample_submission = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')


train_sequences = analysis(train_sequences, drop_cols=drop_cols)
train_labels = analysis(train_labels)





validation_sequences = analysis(validation_sequences, drop_cols=drop_cols)
validation_labels = analysis(validation_labels)


test_sequences = analysis(test_sequences, drop_cols=drop_cols)


train_sequences['sequence_legnth'] = train_sequences['sequence'].str.len()
validation_sequences['sequence_legnth'] = validation_sequences['sequence'].str.len()
test_sequences['sequence_legnth'] = test_sequences['sequence'].str.len()


seq1_len, seq2_len = 28, 34
x1 = train_labels[['x_1', 'y_1', 'z_1']].iloc[:seq1_len, :]
x2 = train_labels[['x_1', 'y_1', 'z_1']].iloc[seq1_len:(seq1_len+seq2_len), :]
comparison_2_seq(x1, x2)


print('.... Train Data Procesed')
train_data  = fix_dataset(seq_df=train_sequences, labels_df=train_labels)

print('.... Validation Data Procesed')
val_data  = fix_dataset(seq_df=validation_sequences, labels_df=validation_labels)

print('.... Test Data Procesed')
test_data  = fix_dataset(seq_df=test_sequences, labels_df=None)


 print('... Clean Train Data')
clean_train_data, _ = clean_data(data = train_data, 
                              remove_col = remove_col, 
                              std_cols =  std_cols)
print('... Clean Validation Data')
clean_val_data, _ = clean_data(data = val_data, 
                              remove_col = remove_col, 
                              std_cols =  std_cols)

print('... Clean Test Data')
clean_test_data, unseen_id = clean_data(data = test_data, 
                              remove_col = remove_col, 
                              std_cols =  std_cols)


train_models, predictions = model_prediction(df_train = clean_train_data, 
                                df_val = clean_val_data,  
                                target_data = target_coord, 
                                **parmas)


predict_vs_true(val_data = clean_val_data,
                target_data = target_coord, 
                predictions = predictions)


x_model, y_model, z_model  = train_models['x'], train_models['y'], train_models['z']





x_1, y_1, z_1 = x_model.predict(clean_test_data), y_model.predict(clean_test_data), z_model.predict(clean_test_data)
x_2, y_2, z_2 = x_model.predict(clean_test_data), y_model.predict(clean_test_data), z_model.predict(clean_test_data)
x_3, y_3, z_3 = x_model.predict(clean_test_data), y_model.predict(clean_test_data), z_model.predict(clean_test_data)


inverse_seq_map = {0:'A', 1:'C', 2:'G', 3:'U'}
seq_len = test_data['seq_len']
resname = clean_test_data['RNA_seq'].map(inverse_seq_map)



sample_sub = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/sample_submission.csv')
id_s, resname_seq, resid_seq = sample_sub.ID, sample_sub.resname, sample_sub.resid


submission = pd.DataFrame({'ID':id_s,
                          'resname':resname_seq, 
                          'resid':resid_seq, 
    'x_1': x_model.predict(clean_test_data),
    'y_1': y_model.predict(clean_test_data),
    'z_1': z_model.predict(clean_test_data),
                           
    'x_2': x_model.predict(clean_test_data),
    'y_2': y_model.predict(clean_test_data),
    'z_2': z_model.predict(clean_test_data),
                           
    'x_3': x_model.predict(clean_test_data),
    'y_3': y_model.predict(clean_test_data),
    'z_3': z_model.predict(clean_test_data),
                           
    'x_4': x_model.predict(clean_test_data),
    'y_4': y_model.predict(clean_test_data),
    'z_4': z_model.predict(clean_test_data),
                           
    'x_5': x_model.predict(clean_test_data),
    'y_5': y_model.predict(clean_test_data),
    'z_5': z_model.predict(clean_test_data),
})


submission.to_csv('submission.csv', index=False)

