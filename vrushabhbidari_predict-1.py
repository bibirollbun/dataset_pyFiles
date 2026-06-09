import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings

filterwarnings('ignore')

# Define your RankNet model class
class RankNet(torch.nn.Module):
    def __init__(self, input_dim):
        super(RankNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Function to normalize race strings
def normalize_race(race_str):
    """Remove underscores and hyphens from race strings for comparison."""
    if isinstance(race_str, str):
        return race_str.replace("_", " ").replace("-", " ").lower()
    return str(race_str).lower()

# Function to predict with feature alignment
def generate_predictions_with_feature_alignment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    test_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")
    submission_template = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
    
    # Store original IDs
    patient_ids = test_df['ID'].values
    
    # Load a sample training file to get original feature set
    train_df = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
    
    # Get numeric features from both datasets
    train_numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    test_numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Training features count: {len(train_numeric_cols)}")
    print(f"Testing features count: {len(test_numeric_cols)}")
    
    # Find common features
    common_features = list(set(train_numeric_cols) & set(test_numeric_cols))
    print(f"Common features count: {len(common_features)}")
    
    # Check for NaN values before preprocessing
    print(f"NaN values in test data before processing: {test_df[common_features].isna().sum().sum()}")
    
    # Prepare test data using only common features
    test_features = test_df[common_features].copy()
    
    # Handle any NaN values
    if test_features.isna().sum().sum() > 0:
        print("Filling NaN values with column means")
        test_features = test_features.fillna(test_features.mean())
    
    # Normalize the data
    scaler = StandardScaler()
    test_data_scaled = scaler.fit_transform(test_features)
    
    # Check for NaN values after scaling
    nan_count = np.isnan(test_data_scaled).sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values after scaling, replacing with zeros")
        test_data_scaled = np.nan_to_num(test_data_scaled)
    
    # Define the input dimension based on common features
    input_dim = len(common_features)
    print(f"Using input dimension: {input_dim}")
    
    # Paths to your trained models
    model_dir = "/kaggle/input/race-groups/pytorch/default/1/"
    race_models = {
        "More_than_one_race": f"{model_dir}More_than_one_race.pth",
        "Asian": f"{model_dir}Asian.pth",
        "White": f"{model_dir}White.pth",
        "American_Indian_or_Alaska_Native": f"{model_dir}American_Indian_or_Alaska_Native.pth",
        "Native_Hawaiian_or_other_Pacific_Islander": f"{model_dir}Native_Hawaiian_or_other_Pacific_Islander.pth", 
        "Black_or_African-American": f"{model_dir}Black_or_African-American.pth"
    }
    
    # Create a mapping between normalized race names and model names
    normalized_race_mapping = {normalize_race(race): race for race in race_models.keys()}
    print(f"Normalized race mapping: {normalized_race_mapping}")
    
    # Get race information if available
    race_column = 'race_group'
    
    # Initialize results
    all_predictions = np.zeros(len(test_df))
    model_count = 0
    
    if race_column in test_df.columns:
        # Apply race-specific models
        print(f"Using race-specific models based on '{race_column}' column")
        
        # Create a normalized version of the race column
        test_df['normalized_race'] = test_df[race_column].apply(normalize_race)
        
        # Show unique race values in test data
        unique_races = test_df['normalized_race'].unique()
        print(f"Unique normalized races in test data: {unique_races}")
        
        # Count number of patients by race
        race_counts = test_df['normalized_race'].value_counts()
        print(f"Patients by race: {race_counts}")
        
        for model_race, model_path in race_models.items():
            if os.path.exists(model_path):
                normalized_model_race = normalize_race(model_race)
                print(f"Processing {model_race} model (normalized: {normalized_model_race})...")
                
                # Create a new model with the correct input dimension
                model = RankNet(input_dim).to(device)
                
                # Try loading with modified first layer
                try:
                    # Apply a quick fix to the model - rebuild first layer
                    orig_model = RankNet(24).to(device)  # Original dimension
                    orig_model.load_state_dict(torch.load(model_path))
                    
                    # Copy parameters except first layer
                    for i, (new_param, old_param) in enumerate(zip(model.parameters(), orig_model.parameters())):
                        if i == 0:  # First layer weights
                            if input_dim < 24:
                                # If fewer features, take subset
                                new_param.data.copy_(old_param.data[:, :input_dim])
                            else:
                                # If more features, pad with zeros
                                new_param.data[:, :24].copy_(old_param.data)
                        else:
                            # Copy other layers directly
                            new_param.data.copy_(old_param.data)
                    
                    model.eval()
                    
                    # Filter test data for this race using normalized comparison
                    race_mask = test_df['normalized_race'] == normalized_model_race
                    if race_mask.any():
                        race_indices = np.where(race_mask)[0]
                        race_data = test_data_scaled[race_mask]
                        
                        print(f"Found {len(race_indices)} patients matching '{model_race}'")
                        
                        # Make predictions
                        with torch.no_grad():
                            inputs = torch.tensor(race_data, dtype=torch.float32).to(device)
                            outputs = model(inputs).cpu().numpy().flatten()
                            
                            # Check for NaN values in predictions
                            nan_preds = np.isnan(outputs).sum()
                            if nan_preds > 0:
                                print(f"WARNING: Found {nan_preds} NaN predictions for {model_race}")
                                # Replace NaNs with zeros
                                outputs = np.nan_to_num(outputs)
                        
                        # Store predictions
                        all_predictions[race_indices] = outputs
                        print(f"Made predictions for {len(race_indices)} patients of {model_race} race")
                        print(f"Sample predictions: {outputs[:5] if len(outputs) >= 5 else outputs}")
                        model_count += 1
                    else:
                        print(f"No patients found for race: {model_race}")
                
                except Exception as e:
                    print(f"Error with {model_race} model: {e}")
    else:
        # Ensemble approach if no race information
        print("No race information found. Using ensemble approach...")
        ensemble_predictions = np.zeros(len(test_df))
        valid_models = 0
        
        for race, model_path in race_models.items():
            if os.path.exists(model_path):
                try:
                    # Same model adaptation as above
                    model = RankNet(input_dim).to(device)
                    orig_model = RankNet(24).to(device)
                    orig_model.load_state_dict(torch.load(model_path))
                    
                    # Copy parameters
                    for i, (new_param, old_param) in enumerate(zip(model.parameters(), orig_model.parameters())):
                        if i == 0:
                            if input_dim < 24:
                                new_param.data.copy_(old_param.data[:, :input_dim])
                            else:
                                new_param.data[:, :24].copy_(old_param.data)
                        else:
                            new_param.data.copy_(old_param.data)
                    
                    model.eval()
                    
                    # Make predictions
                    with torch.no_grad():
                        inputs = torch.tensor(test_data_scaled, dtype=torch.float32).to(device)
                        outputs = model(inputs).cpu().numpy().flatten()
                        
                        # Check for NaN values
                        nan_count = np.isnan(outputs).sum()
                        if nan_count > 0:
                            print(f"WARNING: Found {nan_count} NaN values in {race} predictions")
                            outputs = np.nan_to_num(outputs)
                    
                    ensemble_predictions += outputs
                    valid_models += 1
                    print(f"Added {race} model to ensemble")
                    print(f"Sample predictions: {outputs[:5] if len(outputs) >= 5 else outputs}")
                
                except Exception as e:
                    print(f"Error with {race} model: {e}")
        
        if valid_models > 0:
            all_predictions = ensemble_predictions / (valid_models + 1e-10)  # Add small epsilon to avoid division by zero
            model_count = valid_models
    
    # Final check for NaN values
    nan_count = np.isnan(all_predictions).sum()
    if nan_count > 0:
        print(f"WARNING: Final predictions contain {nan_count} NaN values, replacing with zeros")
        all_predictions = np.nan_to_num(all_predictions)
    
    if model_count > 0:
        # Create submission file
        submission = submission_template.copy()
        submission['prediction'] = -all_predictions  # Negative sign for correct ranking
        
        # Check the final predictions
        print(f"Final predictions stats: min={submission['prediction'].min()}, max={submission['prediction'].max()}, mean={submission['prediction'].mean()}")
        
        # Save submission
        submission_path = "submission.csv"
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        return submission
    else:
        print("No valid models were able to make predictions.")
        return None

# Run the prediction function
if __name__ == "__main__":
    predictions = generate_predictions_with_feature_alignment()
    if predictions is not None:
        print(predictions.head())

