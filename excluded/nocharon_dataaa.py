# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# -*- coding: utf-8 -*-
import os
import pandas as pd

# —— 1. Check what the Kaggle mount directory is called —— 
print("Folders under the root directory /kaggle/input:", os.listdir('/kaggle/input'))

# If 'nbme-score-clinical-patient-notes' appears in the output, use it
KAGGLE_DATASET_NAME = 'nbme-score-clinical-patient-notes'
KAGGLE_PATH = os.path.join('/kaggle/input', KAGGLE_DATASET_NAME)

# If this step fails, the directory name is incorrect; please update it to the actual name
assert os.path.exists(KAGGLE_PATH), f"Cannot find dataset directory: {KAGGLE_PATH}"

print("Dataset found. Directory contents:", os.listdir(KAGGLE_PATH))


# —— 2. Force data_dir to point to this Kaggle path —— 
class NBMEProcessor:
    def __init__(self, data_dir=None, **kwargs):
        # Always use the Kaggle-mounted data path, regardless of what was passed in
        self.data_dir = KAGGLE_PATH
        print(f"[INFO] Data directory set to: {self.data_dir}")

    def load_data(self):
        print("Directory listing:", os.listdir(self.data_dir))
        self.train         = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.patient_notes = pd.read_csv(os.path.join(self.data_dir, 'patient_notes.csv'))
        self.features      = pd.read_csv(os.path.join(self.data_dir, 'features.csv'))
        print("Loading complete",
              f"train {self.train.shape},",
              f"notes {self.patient_notes.shape},",
              f"features {self.features.shape}")

    def run_full_pipeline(self):
        self.load_data()
        # …Place your preprocessing/feature engineering/model code here…
        return self.train, self.patient_notes, self.features


# —— 3. Re-instantiate and run the full pipeline —— 
processor = NBMEProcessor()
train_df, notes_df, feat_df = processor.run_full_pipeline()


import os
import re
import ast
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

class NBMEDataProcessor:
    """
    NBME Clinical Patient Notes Data Processor, adapted for Kaggle Notebook paths
    """
    def __init__(self, data_dir=None, output_dir=None, n_folds=5):
        """
        Initialize NBME data processor

        Args:
            data_dir: Input data directory, defaults to automatically detect Kaggle mount path
            output_dir: Processed data output directory, defaults to '/kaggle/working/processed'
            n_folds: Number of cross-validation folds
        """
        # Kaggle dataset default mount path
        kaggle_dir = '/kaggle/input/nbme-score-clinical-patient-notes'
        # Working directory output path
        default_output = '/kaggle/working/processed'

        # Automatically detect data directory
        if data_dir:
            self.data_dir = data_dir
        elif os.path.exists(kaggle_dir):
            self.data_dir = kaggle_dir
        else:
            raise FileNotFoundError(f"Data directory not found: {kaggle_dir}")

        # Set output directory
        self.output_dir = output_dir or default_output
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize attributes
        self.train = None
        self.test = None
        self.patient_notes = None
        self.features = None
        self.train_processed = None
        self.final_data = None
        self.n_folds = n_folds
        self.feature_female = []
        self.feature_male = []
        self.feature_year = []

        # Ignore warnings
        warnings.filterwarnings("ignore")

        # Medical abbreviation dictionary
        self.medical_abbreviations = {
            'htn': 'hypertension', 'dm': 'diabetes mellitus', 'chf': 'congestive heart failure',
            'cad': 'coronary artery disease', 'mi': 'myocardial infarction', 'afib': 'atrial fibrillation',
            'copd': 'chronic obstructive pulmonary disease', 'uti': 'urinary tract infection',
            'bph': 'benign prostatic hyperplasia', 'gerd': 'gastroesophageal reflux disease',
            'hx': 'history', 'yo': 'year old', 'y/o': 'year old', 'yo/': 'year old', 'y.o.': 'year old',
            'w/': 'with', 's/p': 'status post', 'h/o': 'history of', 'c/o': 'complains of',
            'p/w': 'presenting with', 'neg': 'negative', 'pos': 'positive', '+': 'positive', '-': 'negative',
            'w/o': 'without', 'b/l': 'bilateral', 'r/o': 'rule out', '&': 'and', 'pt': 'patient',
            'sx': 'symptoms', 'dx': 'diagnosis', 'tx': 'treatment', 'fx': 'fracture', 'vs': 'vital signs',
        }
        print(f"[INFO] Data directory: {self.data_dir}")
        print(f"[INFO] Output directory: {self.output_dir}")

    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files from:", self.data_dir)
        print("Contents:", os.listdir(self.data_dir))
        self.train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.patient_notes = pd.read_csv(os.path.join(self.data_dir, 'patient_notes.csv'))
        self.features = pd.read_csv(os.path.join(self.data_dir, 'features.csv'))
        test_path = os.path.join(self.data_dir, 'test.csv')
        if os.path.exists(test_path):
            self.test = pd.read_csv(test_path)
        print(f"Loaded train ({self.train.shape}), notes ({self.patient_notes.shape}), features ({self.features.shape})")
        if self.test is not None:
            print(f"Loaded test ({self.test.shape})")
        self.identify_feature_types()
            
    def identify_feature_types(self):
        """Identify different types of features (gender, age, etc.)"""
        print("Identifying feature types...")
        
        # Reset feature type lists
        self.feature_female = []
        self.feature_male = []
        self.feature_year = []
        
        # Iterate through features
        for idx, row in self.features.iterrows():
            feature_text = row['feature_text'].lower()
            feature_num = row['feature_num']
            
            # Identify female-related features
            if any(term in feature_text for term in ['female', 'woman', 'girl', 'mother', 'sister', 'daughter']):
                self.feature_female.append(feature_num)
                
            # Identify male-related features
            if any(term in feature_text for term in ['male', 'man', 'boy', 'father', 'brother', 'son']):
                self.feature_male.append(feature_num)
                
            # Identify age-related features
            if any(term in feature_text for term in ['age', 'year old', 'y.o', 'yo', 'y/o']):
                self.feature_year.append(feature_num)
                
        print(f"Identified {len(self.feature_female)} female-related features")
        print(f"Identified {len(self.feature_male)} male-related features")
        print(f"Identified {len(self.feature_year)} age-related features")
    
    def preprocess_features(self):
        """Process special cases in feature text (from second notebook)"""
        # Fix text for feature #27
        self.features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
        # Additional feature preprocessing logic can be added here
        return self.features
    
    def parse_annotations(self):
        """Convert string format annotations and locations to lists"""
        # Ensure annotations and locations are parsed into Python objects
        if isinstance(self.train['annotation'].iloc[0], str):
            self.train['annotation'] = self.train['annotation'].apply(ast.literal_eval)
        
        if isinstance(self.train['location'].iloc[0], str):
            self.train['location'] = self.train['location'].apply(ast.literal_eval)
            
        # Add annotation length field
        self.train['annotation_length'] = self.train['annotation'].apply(len)
        return self.train
    
    def merge_data(self):
        """Merge training data with features and patient records"""
        if self.train is None or self.features is None or self.patient_notes is None:
            print("Please load data first.")
            return None
        
        # Merge data
        self.train = self.train.merge(self.features, on=['feature_num', 'case_num'], how='left')
        self.train = self.train.merge(self.patient_notes, on=['pn_num', 'case_num'], how='left')
        return self.train
    
    def check_annotation_integrity(self):
        """
        Check the integrity and consistency of annotations
        instead of manually fixing specific errors
        """
        print("Checking annotation integrity...")
        
        # Create a copy to avoid modifying original data
        checked_train = self.train.copy()
        
        # Check for rows with empty annotations but location information
        empty_annot_with_loc = checked_train[
            (checked_train['annotation_length'] == 0) & 
            (checked_train['location'].apply(lambda x: len(x) > 0))
        ]
        
        if len(empty_annot_with_loc) > 0:
            print(f"Warning: Found {len(empty_annot_with_loc)} rows with empty annotations but location data")
            
        # Check for rows with annotations but no location information
        annot_without_loc = checked_train[
            (checked_train['annotation_length'] > 0) & 
            (checked_train['location'].apply(lambda x: len(x) == 0))
        ]
        
        if len(annot_without_loc) > 0:
            print(f"Warning: Found {len(annot_without_loc)} rows with annotations but no location data")
        
        # More integrity checks can be added...
        
        return checked_train
    
    def standardize_medical_text(self):
        """
        Standardize medical terminology (adopted and optimized from first notebook)
        """
        print("Standardizing medical text...")
        
        # Create a working copy
        train_standardized = self.train.copy()
        
        # Text standardization function
        def standardize_text(text):
            if pd.isna(text):
                return text
                
            # Replace medical abbreviations
            for abbr, full_form in self.medical_abbreviations.items():
                pattern = r'\b' + re.escape(abbr) + r'\b'
                text = re.sub(pattern, full_form, text, flags=re.IGNORECASE)
                
            return text
        
        # Standardize patient history text
        train_standardized['pn_history'] = train_standardized['pn_history'].apply(standardize_text)
        
        # Standardize feature text
        train_standardized['feature_text'] = train_standardized['feature_text'].apply(standardize_text)
        
        self.train_standardized = train_standardized
        print("Medical text standardization completed")
        return self.train_standardized
    
    def correct_offsets(self):
        """
        Correct annotation positions after text standardization (optimized from first notebook)
        """
        print("Correcting annotation offsets...")
        
        if not hasattr(self, 'train_standardized'):
            print("Standardized data not found. Running standardize_medical_text first...")
            self.standardize_medical_text()
        
        # Create a working copy
        train_offset_corrected = self.train_standardized.copy()
        
        # Since text standardization may change the text length, positions need to be updated
        # This implementation is simplified, actual application needs more complex logic
        
        def adjust_location(row):
            """Adjust position offsets"""
            if not row['location'] or pd.isna(row['pn_history']):
                return row['location']
                
            # Get original patient history text
            original_text = self.train.loc[row.name, 'pn_history']
            
            # Get standardized patient history text
            standardized_text = row['pn_history']
            
            adjusted_locations = []
            for loc_list in row['location']:
                adjusted_loc_parts = []
                
                for loc in loc_list.split(';'):
                    if ' ' in loc:
                        start, end = map(int, loc.split())
                        # Extract phrase from original text
                        if start < len(original_text) and end <= len(original_text):
                            phrase = original_text[start:end]
                            
                            # Find the phrase in standardized text
                            # Note: This is a simplified method, may need more complex handling for multiple occurrences
                            if phrase in standardized_text:
                                new_start = standardized_text.find(phrase)
                                new_end = new_start + len(phrase)
                                adjusted_loc_parts.append(f"{new_start} {new_end}")
                            else:
                                # If exact match not found, use original position
                                adjusted_loc_parts.append(loc)
                        else:
                            # If position out of range, use original position
                            adjusted_loc_parts.append(loc)
                
                if adjusted_loc_parts:
                    adjusted_locations.append([';'.join(adjusted_loc_parts)])
            
            return adjusted_locations if adjusted_locations else row['location']
        
        # Apply position adjustment logic
        for i, row in train_offset_corrected.iterrows():
            train_offset_corrected.at[i, 'location'] = adjust_location(row)
        
        self.train_offset_corrected = train_offset_corrected
        print("Offset correction completed")
        return self.train_offset_corrected
    
    def process_spaces(self, predictions=None):
        """
        Process spaces (optimized from second notebook)
        
        The purpose of this function is to clean up spaces in prediction labels through post-processing:
        - Remove unnecessary leading and trailing spaces
        - Remove middle spaces positioned before/after invalid characters (no valid characters on both sides)
        - Preserve spaces with valid characters on both sides
        
        Can be used in preprocessing stage or prediction post-processing
        """
        print("Processing spaces in text data...")
        
        if not hasattr(self, 'train_offset_corrected'):
            if hasattr(self, 'train_standardized'):
                data = self.train_standardized
            else:
                data = self.train
        else:
            data = self.train_offset_corrected
            
        # Create working copy
        processed_data = data.copy()
        
        def post_process_spaces(pred, text):
            """
            Process prediction array to handle spaces correctly.
            
            Args:
                pred: Prediction array (binary or probability values)
                text: Corresponding text
            
            Returns:
                Processed prediction array
            """
            spaces = ' \n\r\t'
            
            # Ensure lengths match
            text = text[:len(pred)]
            pred = pred[:len(text)]
            
            # Process boundary spaces
            if text[0] in spaces:
                pred[0] = 0
            if text[-1] in spaces:
                pred[-1] = 0

            # Process internal spaces
            for i in range(1, len(text) - 1):
                if text[i] in spaces:
                    if pred[i] and not pred[i - 1]:  # Space after invalid character
                        pred[i] = 0

                    if pred[i] and not pred[i + 1]:  # Space before invalid character
                        pred[i] = 0

                    if pred[i - 1] and pred[i + 1]:  # Space with valid characters on both sides
                        pred[i] = 1
            
            return pred
        
        # If predictions provided, process directly
        if predictions is not None:
            processed_predictions = []
            for i, pred in enumerate(predictions):
                if i < len(processed_data):
                    text = processed_data.iloc[i]['pn_history']
                    processed_pred = post_process_spaces(pred, text)
                    processed_predictions.append(processed_pred)
                else:
                    processed_predictions.append(pred)
            return processed_predictions
        
        # Otherwise, process existing annotation locations in the data
        # Note: This is a simplified implementation, actually needs more complex logic to handle location information
        # Since we need to convert positions to binary array, apply space processing, then convert back to positions
        
        processed_locations = []
        for i, row in processed_data.iterrows():
            text = row['pn_history']
            if pd.isna(text) or text == '':
                processed_locations.append(row['location'])
                continue
                
            # Create binary prediction array
            binary_array = np.zeros(len(text))
            
            # Convert location information to binary array
            for loc_list in row['location']:
                # Fix error: handle different types of loc_list
                if isinstance(loc_list, str):
                    # If it's a string, split by semicolon
                    locations = loc_list.split(';')
                elif isinstance(loc_list, list):
                    # If it's already a list, use directly
                    locations = loc_list
                else:
                    # Skip unknown format
                    continue
                
                for loc in locations:
                    if not isinstance(loc, str):
                        continue
                        
                    # Check if it contains semicolons, if so, need further splitting
                    if ';' in loc:
                        sub_locs = loc.split(';')
                        for sub_loc in sub_locs:
                            if ' ' in sub_loc:
                                try:
                                    start, end = map(int, sub_loc.split())
                                    if start < len(binary_array) and end <= len(binary_array):
                                        binary_array[start:end] = 1
                                except ValueError:
                                    print(f"Warning: Cannot parse location string: {sub_loc}")
                    elif ' ' in loc:
                        try:
                            start, end = map(int, loc.split())
                            if start < len(binary_array) and end <= len(binary_array):
                                binary_array[start:end] = 1
                        except ValueError:
                            print(f"Warning: Cannot parse location string: {loc}")
            
            # Apply space processing
            processed_binary = post_process_spaces(binary_array, text)
            
            # Convert processed binary array back to location information
            new_locations = []
            in_span = False
            span_start = -1
            
            for i, val in enumerate(processed_binary):
                if val == 1 and not in_span:
                    # Start new span
                    in_span = True
                    span_start = i
                elif val == 0 and in_span:
                    # End current span
                    new_locations.append(f"{span_start} {i}")
                    in_span = False
            
            # Don't forget to handle the ending span
            if in_span:
                new_locations.append(f"{span_start} {len(processed_binary)}")
            
            # Update location information
            if new_locations:
                processed_locations.append([[';'.join(new_locations)]])
            else:
                processed_locations.append([])
        
        # Update processed data
        for i, locs in enumerate(processed_locations):
            if i < len(processed_data):
                processed_data.at[i, 'location'] = locs
        
        # Update class attributes
        if hasattr(self, 'train_offset_corrected'):
            self.train_offset_corrected = processed_data
        elif hasattr(self, 'train_standardized'):
            self.train_standardized = processed_data
        else:
            self.train = processed_data
            
        print("Space processing completed")
        return processed_data
    

    
    def create_folds(self, n_folds=5):
        """
        Create cross-validation folds using GroupKFold (adopted from second notebook)
        """
        print(f"Creating {n_folds} folds using GroupKFold...")
        self.n_folds = n_folds
        
        if not hasattr(self, 'train_offset_corrected'):
            if hasattr(self, 'train_standardized'):
                data = self.train_standardized
            else:
                data = self.train
        else:
            data = self.train_offset_corrected
        
        # Use GroupKFold to group by pn_num
        fold = GroupKFold(n_splits=n_folds)
        groups = data['pn_num'].values
        
        for n, (train_index, val_index) in enumerate(fold.split(data, data['location'], groups)):
            data.loc[val_index, 'fold'] = int(n)
            
        data['fold'] = data['fold'].astype(int)
        
        # Update processed data
        if hasattr(self, 'train_offset_corrected'):
            self.train_offset_corrected = data
        elif hasattr(self, 'train_standardized'):
            self.train_standardized = data
        else:
            self.train = data
            
        print(f"Created {n_folds} folds")
        return data
    
    def create_labels_for_scoring(self, df=None):
        """
        Create label format for scoring (adopted from second notebook)
        """
        if df is None:
            if hasattr(self, 'train_offset_corrected'):
                df = self.train_offset_corrected
            elif hasattr(self, 'train_standardized'):
                df = self.train_standardized
            else:
                df = self.train
        
        # First create standard format location lists
        df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
        
        for i in range(len(df)):
            lst = df.loc[i, 'location']
            if lst:
                # Handle different formats of location data
                if isinstance(lst[0], list):
                    # If it's already in list of lists format
                    locations = []
                    for loc_list in lst:
                        for loc in loc_list:
                            locations.append(loc)
                    new_lst = ';'.join(locations)
                elif isinstance(lst[0], str):
                    # If it's in string list format
                    new_lst = ';'.join(lst)
                else:
                    # Unknown format, skip
                    continue
                    
                df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[[\"{new_lst}\"]]')
        
        # Create labels
        truths = []
        for location_list in df['location_for_create_labels'].values:
            truth = []
            if len(location_list) > 0:
                location = location_list[0]
                for loc in [s.split() for s in location.split(';')]:
                    if len(loc) >= 2:  # Ensure there are start and end positions
                        start, end = int(loc[0]), int(loc[1])
                        truth.append([start, end])
            truths.append(truth)
            
        return truths
        
    def pred_to_chars(self, token_type_logits, len_token, max_token, offset_mapping, text, feature_num):
        """
        Convert model token-level predictions to character-level predictions
        
        This function handles special medical notations like "yof" (years old female) and "yom" (years old male)
        
        Args:
            token_type_logits: Model's token-level predictions (logits)
            len_token: Actual length of token sequence
            max_token: Maximum length of token sequence
            offset_mapping: Mapping from tokens to original text characters
            text: Original text
            feature_num: Current feature number being processed
            
        Returns:
            tuple: (Character-level predictions, original text)
        """
        # Truncate to actual token length
        token_type_logits = token_type_logits[:len_token]
        offset_mapping = offset_mapping[:len_token]
        
        # Initialize character-level predictions
        char_preds = np.ones(len(text)) * -1e10
        
        # Iterate through each token mapping
        for i, (start, end) in enumerate(offset_mapping):
            # Special handling for "yof" (age + female)
            if text[start:end] == 'of' and start > 0 and text[start-1:end] == 'yof':
                if feature_num in self.feature_female:
                    # If feature is female-related, tag the last character
                    char_preds[end-1:end] = 1
                elif feature_num in self.feature_year:
                    # If feature is age-related, use previous token's prediction
                    char_preds[start:start+1] = token_type_logits[i-1]
                else:
                    # Otherwise, use current token's prediction
                    char_preds[start:end] = token_type_logits[i]
            
            # Special handling for "yom" (age + male)
            elif text[start:end] == 'om' and start > 0 and text[start-1:end] == 'yom':
                if feature_num in self.feature_male:
                    # If feature is male-related, tag the last character
                    char_preds[end-1:end] = 1
                elif feature_num in self.feature_year:
                    # If feature is age-related, use previous token's prediction
                    char_preds[start:start+1] = token_type_logits[i-1]
                else:
                    # Otherwise, use current token's prediction
                    char_preds[start:end] = token_type_logits[i]
            
            # Standard handling for other tokens
            else:
                char_preds[start:end] = token_type_logits[i]
                
        return (char_preds, text)
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create train/test split (if test set not provided)
        """
        from sklearn.model_selection import train_test_split
        
        if self.test is not None:
            print("Test data already provided, skipping split.")
            return
            
        print(f"Creating train/test split with test_size={test_size}...")
        
        if hasattr(self, 'train_offset_corrected'):
            data = self.train_offset_corrected
        elif hasattr(self, 'train_standardized'):
            data = self.train_standardized
        else:
            data = self.train
        
        # Use stratified sampling to maintain proportion of each pn_num
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=data['pn_num']
        )
        
        self.train_final = train_data.reset_index(drop=True)
        self.test = test_data.reset_index(drop=True)
        
        print(f"Split completed. Train: {len(self.train_final)} rows, Test: {len(self.test)} rows")
        return self.train_final, self.test
    
    def save_processed_data(self):
        """
        Save processed data
        """
        print("Saving processed data...")
        
        if hasattr(self, 'train_offset_corrected'):
            final_train = self.train_offset_corrected
        elif hasattr(self, 'train_standardized'):
            final_train = self.train_standardized
        else:
            final_train = self.train
        
        if not hasattr(self, 'train_final'):
            self.train_final = final_train
        
        # Save training data
        train_output_path = os.path.join(self.output_dir, "train_processed.csv")
        self.train_final.to_csv(train_output_path, index=False)
        print(f"Saved processed train data to {train_output_path}")
        
        # Save test data (if available)
        if self.test is not None:
            test_output_path = os.path.join(self.output_dir, "test_processed.csv")
            self.test.to_csv(test_output_path, index=False)
            print(f"Saved processed test data to {test_output_path}")
        
        # Save label information (for scoring)
        truths = self.create_labels_for_scoring(self.train_final)
        labels_output_path = os.path.join(self.output_dir, "train_labels.npy")
        # Use dtype=object to save irregular shaped arrays
        np.save(labels_output_path, np.array(truths, dtype=object))
        print(f"Saved labels to {labels_output_path}")
        
        # Save fold information
        folds_output_path = os.path.join(self.output_dir, "folds.npy")
        np.save(folds_output_path, self.train_final['fold'].values)
        print(f"Saved fold information to {folds_output_path}")
        
        return True
    
    def run_full_pipeline(self):
        """Run the complete data processing pipeline"""
        self.load_data()
        self.preprocess_features()
        self.parse_annotations()
        self.merge_data()
        self.check_annotation_integrity()
        self.standardize_medical_text()
        self.correct_offsets()
        self.process_spaces()
        self.create_folds(n_folds=self.n_folds)
        if self.test is None:
            self.create_train_test_split()
        self.save_processed_data()
        print("Full pipeline completed.")
        return getattr(self, 'train_final', self.train), self.test

# Usage example
if __name__ == "__main__":
    processor = NBMEDataProcessor()
    train_data, test_data = processor.run_full_pipeline()
    print(f"Processed train data shape: {train_data.shape}")
    if test_data is not None:
        print(f"Processed test data shape: {test_data.shape}")


import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from IPython.display import display
import matplotlib.pyplot as plt

def collate_fn(batch):
    """
    batch: List of tuples (inputs_dict, label_tensor)
    return: (batched_inputs_dict, batched_labels_tensor)
    """
    # input_dict keys should be consistent
    inputs_keys = batch[0][0].keys()
    batched_inputs = {
        k: torch.stack([sample[0][k] for sample in batch], dim=0)
        for k in inputs_keys
    }
    batched_labels = torch.stack([sample[1] for sample in batch], dim=0)
    return batched_inputs, batched_labels


# ====================================================
# Configuration
# ====================================================
class CFG:
    debug = False
    apex = False
    print_freq = 100
    num_workers = 4
    model = "microsoft/deberta-v3-large"
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0.1
    epochs = 3
    encoder_lr = 1e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 8
    fc_dropout = 0.2
    max_len = 512
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 5
    trn_fold = [0]  # Can specify multiple folds for training
    train = True
    
    # Data directory
    data_dir = r"C:\Users\SIMON\Desktop\NLP\nbme-score-clinical-patient-notes"
    
    # Adversarial training parameters
    adv_training = True
    adv_epsilon = 0.25
    
    # Focal Loss parameters
    focal_alpha = 1
    focal_gamma = 2
    label_smoothing = 0.1
    
    # Whether to use local model
    use_local_model = False
    local_model_path = r"C:\Users\SIMON\Desktop\NLP\models\deberta-v3-large"
    
    # Output directory
    output_dir = "./models/"

if CFG.debug:
    CFG.epochs = 1
    CFG.trn_fold = [0]
    
# Ensure output directory exists
os.makedirs(CFG.output_dir, exist_ok=True)

# —— Place at the top of the script —— 
# Automatically detect Kaggle mount path
KAGGLE_DATA_DIR = '/kaggle/input/nbme-score-clinical-patient-notes'
if os.path.exists(KAGGLE_DATA_DIR):
    CFG.data_dir = KAGGLE_DATA_DIR
    print(f"[INFO] Using Kaggle data dir: {CFG.data_dir}")

# Optional: If you want to consolidate output to /kaggle/working
KAGGLE_OUT_DIR = '/kaggle/working/models'
if os.path.exists('/kaggle/working'):
    CFG.output_dir = KAGGLE_OUT_DIR
    os.makedirs(CFG.output_dir, exist_ok=True)
    print(f"[INFO] Using Kaggle output dir: {CFG.output_dir}")

# Optional: Local model path points to HuggingFace cache in input directory
if CFG.use_local_model:
    kaggle_model_dir = os.path.join('/kaggle/input', os.path.basename(CFG.local_model_path))
    if os.path.exists(kaggle_model_dir):
        CFG.local_model_path = kaggle_model_dir
        print(f"[INFO] Using Kaggle local model path: {CFG.local_model_path}")

if os.path.exists('/kaggle/input'):
    CFG.batch_size = 2
    CFG.max_len     = 256
    CFG.apex        = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    print(f"[MEMORY OPT] batch_size={CFG.batch_size}, max_len={CFG.max_len}, apex={CFG.apex}")

# ====================================================
# Logging and Utility Functions
# ====================================================
def get_logger(filename=CFG.output_dir+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.seed)

# ====================================================
# Scoring Functions
# ====================================================
def micro_f1(preds, truths):
    """
    Micro f1 on binary arrays.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)

def spans_to_binary(spans, length=None):
    """
    Converts spans to a binary array indicating whether each character is in the span.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary

def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0)
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)

# ====================================================
# Helper Functions
# ====================================================
def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df['location_for_create_labels'] = [ast.literal_eval(f'[]')] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, 'location']
        if lst:
            new_lst = ';'.join(lst)
            df.loc[i, 'location_for_create_labels'] = ast.literal_eval(f'[[\"{new_lst}\"]]')
    # create labels
    truths = []
    for location_list in df['location_for_create_labels'].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths

def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text, 
                           add_special_tokens=True,
                           return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results

def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results

def get_predictions(results):
    predictions = []
    for result in results:
        prediction = []
        if result != "":
            for loc in [s.split() for s in result.split(';')]:
                start, end = int(loc[0]), int(loc[1])
                prediction.append([start, end])
        predictions.append(prediction)
    return predictions

def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

# ====================================================
# Focal Loss Implementation
# ====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, ignore_index=-1, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        inputs: (batch*seq_len, num_classes)
        targets: (batch*seq_len,)  where positions with pad/token_type != 0 are marked as ignore_index (e.g. -1)
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            # 1) To avoid errors in one_hot, clamp first
            targets_safe = targets.clone().clamp(0, n_classes-1)
            # 2) Construct one-hot and apply label smoothing
            target_onehot = F.one_hot(targets_safe, n_classes).float()
            smoothed = (1 - self.label_smoothing) * target_onehot + self.label_smoothing / n_classes
            # 3) Calculate smoothed cross entropy
            loss = -(smoothed * log_probs).sum(dim=-1)
            # 4) Set ignore_index positions to 0
            loss = loss.masked_fill(targets == self.ignore_index, 0.0)
        else:
            # Standard nll_loss, already supports ignore_index
            loss = F.nll_loss(
                log_probs, 
                targets, 
                weight=self.weight, 
                ignore_index=self.ignore_index, 
                reduction='none'
            )

        # Focal weight
        pt = torch.exp(-loss)
        focal = self.alpha * (1 - pt) ** self.gamma * loss
        # Average only non-ignored positions
        valid = (targets != self.ignore_index).float()
        return (focal * valid).sum() / valid.sum().clamp(min=1.0)


# ====================================================
# Adversarial Training Implementation (FGM)
# ====================================================
class FGM():
    """
    Fast Gradient Method adversarial training, perturbing the model's embedding parameters
    """
    def __init__(self, model, epsilon=0.25):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        """
        Get adversarial samples
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        """
        Restore the model's original parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ====================================================
# Dataset and DataLoader
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer
        self.feature_texts = df['feature_text'].values
        self.pn_historys = df['pn_history'].values
        self.annotation_lengths = df['annotation_length'].values
        self.locations = df['location'].values

    def __len__(self):
        return len(self.feature_texts)

    def __getitem__(self, idx):
        inputs = self._prepare_input(
            self.pn_historys[idx],
            self.feature_texts[idx]
        )
        label = self._create_label(
            self.pn_historys[idx],
            self.annotation_lengths[idx],
            self.locations[idx]
        )
        return inputs, label

    def _prepare_input(self, text, feature_text):
        """
        Encode input text, adding truncation=True to ensure maximum length
        """
        inputs = self.tokenizer(
            text,
            feature_text,
            add_special_tokens=True,
            max_length=self.cfg.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=False
        )
        # Convert to tensor
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

    def _create_label(self, text, annotation_length, location_list):
        """
        Construct token-level labels, label length matches input length
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.cfg.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )
        offset_mapping = encoded['offset_mapping']
        # Mark non-text parts as ignore
        ignore_idxs = np.where(np.array(encoded.sequence_ids()) != 0)[0]
        label = np.zeros(len(offset_mapping), dtype=np.float32)
        label[ignore_idxs] = -1

        if annotation_length != 0:
            for loc_str in location_list:
                for loc in loc_str.split(';'):
                    start_char, end_char = map(int, loc.split())
                    start_idx = end_idx = None
                    # Find corresponding token boundaries
                    for i, (s, e) in enumerate(offset_mapping):
                        if start_idx is None and s <= start_char < e:
                            start_idx = i
                        if end_idx is None and s < end_char <= e:
                            end_idx = i + 1
                    if start_idx is None:
                        start_idx = end_idx
                    if start_idx is not None and end_idx is not None:
                        label[start_idx:end_idx] = 1

        return torch.tensor(label, dtype=torch.float)

# ====================================================
# Model Definition
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        
        # —— 1. Load config with FP16 dtype —— 
        self.config = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True,
            torch_dtype=torch.float16
        )
        # —— 2. Load pretrained model with FP16 dtype —— 
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.config,
            torch_dtype=torch.float16
        )

        # The head layers still use FP32 (can be mixed for training)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 2)
        self._init_weights(self.fc)
        
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        
        # Use multiple dropout rates to improve stability
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 2)  # Binary classification: whether it's part of the annotation
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

# ====================================================
# Training Function
# ====================================================
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # Create FGM adversarial training instance (if needed)
    if CFG.adv_training:
        fgm = FGM(model, epsilon=CFG.adv_epsilon)
    
    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        # Normal forward pass
        y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 2), labels.long().view(-1))
        loss = torch.masked_select(loss, labels.view(-1) != -1).mean()
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        
        losses.update(loss.item(), batch_size)
        loss.backward()
        
        # Adversarial training
        if CFG.adv_training:
            fgm.attack()  # Add perturbation to embedding
            y_preds_adv = model(inputs)
            loss_adv = criterion(y_preds_adv.view(-1, 2), labels.long().view(-1))
            loss_adv = torch.masked_select(loss_adv, labels.view(-1) != -1).mean()
            if CFG.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / CFG.gradient_accumulation_steps
            loss_adv.backward()  # Backward pass, accumulate gradient on top of normal training
            fgm.restore()  # Restore embedding parameters
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          lr=scheduler.get_lr()[0]))
    return losses.avg

# ====================================================
# Validation Function
# ====================================================
def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 2), labels.long().view(-1))
        loss = torch.masked_select(loss, labels.view(-1) != -1).mean()
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        
        # For predictions, get the probability of the positive class
        probs = F.softmax(y_preds, dim=2)[:, :, 1].cpu().numpy()
        preds.append(probs)
        
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

# ====================================================
# Inference Function
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        probs = F.softmax(y_preds, dim=2)[:, :, 1].cpu().numpy()
        preds.append(probs)
    predictions = np.concatenate(preds)
    return predictions

# ====================================================
# Training Loop
# ====================================================
def train_loop(folds, fold, device):
    
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # Loaders
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_texts = valid_folds['pn_history'].values
    valid_labels = create_labels_for_scoring(valid_folds)
    
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers, 
        pin_memory=False, 
        drop_last=True,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers, 
        pin_memory=False, 
        drop_last=False,
        collate_fn=collate_fn
    )
    
    # Calculate warmup steps
    CFG.num_warmup_steps = int(
        CFG.num_warmup_steps * len(train_dataset) / CFG.batch_size * CFG.epochs
    )
    
    # ====================================================
    # Model and Optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG.output_dir + 'config.pth')
    model.to(device)
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        return [
            {
                'params': [
                    p for n, p in model.model.named_parameters() 
                    if not any(nd in n for nd in no_decay)
                ],
                'lr': encoder_lr, 
                'weight_decay': weight_decay
            },
            {
                'params': [
                    p for n, p in model.model.named_parameters() 
                    if any(nd in n for nd in no_decay)
                ],
                'lr': encoder_lr, 
                'weight_decay': 0.0
            },
            {
                'params': [
                    p for n, p in model.named_parameters() 
                    if "model" not in n
                ],
                'lr': decoder_lr, 
                'weight_decay': 0.0
            }
        ]

    optimizer_parameters = get_optimizer_params(
        model,
        encoder_lr=CFG.encoder_lr, 
        decoder_lr=CFG.decoder_lr,
        weight_decay=CFG.weight_decay
    )
    optimizer = AdamW(
        optimizer_parameters, 
        lr=CFG.encoder_lr, 
        eps=CFG.eps, 
        betas=CFG.betas
    )
    
    # ====================================================
    # Learning Rate Scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            return get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=cfg.num_warmup_steps, 
                num_training_steps=num_train_steps
            )
        else:  # cosine
            return get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=cfg.num_warmup_steps, 
                num_training_steps=num_train_steps, 
                num_cycles=cfg.num_cycles
            )
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # Loop
    # ====================================================
    criterion = FocalLoss(
        alpha=CFG.focal_alpha, 
        gamma=CFG.focal_gamma, 
        label_smoothing=CFG.label_smoothing
    )
    
    best_score = 0.0

    for epoch in range(CFG.epochs):
        start_time = time.time()
        
        # Training
        avg_loss = train_fn(
            fold, train_loader, model, criterion, 
            optimizer, epoch, scheduler, device
        )
        
        # Validation
        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, device
        )
        predictions = predictions.reshape((len(valid_folds), CFG.max_len))
        
        # Scoring
        char_probs = get_char_probs(valid_texts, predictions, CFG.tokenizer)
        results    = get_results(char_probs, th=0.5)
        preds      = get_predictions(results)
        score      = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  '
            f'avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s'
        )
        LOGGER.info(f'Epoch {epoch+1} - F1 Score: {score:.4f}')
        
        if best_score < score:
            best_score = score
            LOGGER.info(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model'
            )
            torch.save(
                {
                    'model': model.state_dict(),
                    'predictions': predictions
                },
                CFG.output_dir + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"
            )

    # —— Key modification: Explicitly disable weights_only mode when loading checkpoint —— 
    checkpoint_path = CFG.output_dir + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device('cpu'),
        weights_only=False
    )
    predictions = checkpoint['predictions']

    valid_folds[[i for i in range(CFG.max_len)]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds
# ====================================================
# Analysis Function for Results
# ====================================================
def analyze_results(oof_df, tokenizer, max_len):
    """
    1) Threshold → F1 curve
    2) F1 vs annotation length
    3) Show 3 FPs & 3 FNs
    """
    # prepare ground truth & probs
    true_spans  = create_labels_for_scoring(oof_df)
    preds_array = oof_df[[i for i in range(max_len)]].values
    texts       = oof_df['pn_history'].values
    char_probs  = get_char_probs(texts, preds_array, tokenizer)

    # 1) Threshold tuning
    ths = np.linspace(0.1, 0.9, 81)
    f1s = []
    for th in ths:
        res   = get_results(char_probs, th=th)
        predl = get_predictions(res)
        f1s.append(get_score(true_spans, predl))
    best_idx, best_th = int(np.argmax(f1s)), ths[np.argmax(f1s)]
    best_f1 = f1s[best_idx]

    plt.figure()
    plt.plot(ths, f1s)
    plt.scatter([best_th], [best_f1])
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Tuning Curve')
    plt.show()
    print(f"▶ Best threshold = {best_th:.2f}, F1 = {best_f1:.4f}")

    # 2) F1 vs. annotation length
    best_preds = get_predictions(get_results(char_probs, th=best_th))
    lengths    = oof_df['annotation_length'].values
    stats = []
    for L in sorted(set(lengths)):
        idx = np.where(lengths == L)[0]
        if len(idx) < 5:
            continue
        stats.append((L, span_micro_f1(
            [best_preds[i] for i in idx],
            [true_spans[i]   for i in idx]
        )))
    if stats:
        xs, ys = zip(*stats)
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel('Annotation Length')
        plt.ylabel('F1 Score')
        plt.title('F1 by Annotation Length')
        plt.show()

    # 3) Show error examples
    bp = [spans_to_binary(p, len(texts[i])) for i, p in enumerate(best_preds)]
    bt = [spans_to_binary(t, len(texts[i])) for i, t in enumerate(true_spans)]
    fp_idx = [i for i,(b1,b2) in enumerate(zip(bp,bt)) if b1.any() and not b2.any()][:3]
    fn_idx = [i for i,(b1,b2) in enumerate(zip(bp,bt)) if b2.any() and not b1.any()][:3]
    error_idx = fp_idx + fn_idx

    error_df = oof_df.iloc[error_idx][[
        'pn_history','feature_text','annotation','location'
    ]].reset_index(drop=True)

    print("▶ Example errors (3 false positives, then 3 false negatives):")
    display(error_df)
    
# ====================================================
# Main Function
# ====================================================
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data - using complete path
    train = pd.read_csv(os.path.join(CFG.data_dir, 'train.csv'))
    train['annotation'] = train['annotation'].apply(ast.literal_eval)
    train['location'] = train['location'].apply(ast.literal_eval)
    features = pd.read_csv(os.path.join(CFG.data_dir, 'features.csv'))
    
    # Preprocess features (fix any potential issues)
    def preprocess_features(features):
        # For example, fix specific feature text
        features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
        return features
    
    features = preprocess_features(features)
    patient_notes = pd.read_csv(os.path.join(CFG.data_dir, 'patient_notes.csv'))
    
    # Merge data
    train = train.merge(features, on=['feature_num', 'case_num'], how='left')
    train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
    
    # Manually correct some annotation errors
    # ... (specific corrections can be added, such as copying from the second notebook)
    
    # Add annotation length field
    train['annotation_length'] = train['annotation'].apply(len)
    
    # Setup Group K-Fold Cross Validation
    Fold = GroupKFold(n_splits=CFG.n_fold)
    groups = train['pn_num'].values
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    
    # Load tokenizer
    if CFG.use_local_model and os.path.exists(CFG.local_model_path):
        LOGGER.info(f"Loading tokenizer from local path: {CFG.local_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(CFG.local_model_path)
    else:
        LOGGER.info(f"Loading tokenizer from HuggingFace: {CFG.model}")
        tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    CFG.tokenizer = tokenizer
    
    # Determine maximum length
    # Can be adjusted based on data analysis
    
    # If training is needed
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold, device)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                
                # Evaluate results
                def get_result(oof_df):
                    labels = create_labels_for_scoring(oof_df)
                    predictions = oof_df[[i for i in range(CFG.max_len)]].values
                    char_probs = get_char_probs(oof_df['pn_history'].values, predictions, CFG.tokenizer)
                    results = get_results(char_probs, th=0.49)
                    preds = get_predictions(results)
                    score = get_score(labels, preds)
                    LOGGER.info(f'Score: {score:<.4f}')
                
                get_result(_oof_df)
                _oof_df.to_pickle(CFG.output_dir+'oof_df_{}.pkl'.format(fold))
        
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        analyze_results(oof_df, CFG.tokenizer, CFG.max_len)

if __name__ == '__main__':
    main()


