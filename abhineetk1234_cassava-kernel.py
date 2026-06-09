target_tranform = {"label_classification": "top_k(1, None)"}
from pathlib import Path


import os
import traceback
import warnings

def write_warning_to_file(message, category, filename, lineno, file=None, line=None):
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_warnings.txt', 'w') as f:
        f.write(warnings.formatwarning(message=message, category=category, filename=filename, lineno=lineno, line=line))

try:
    import os
    import pandas as pd
    # <｜fim▁begin｜>
    import os
    import pandas as pd
    # Define the base directory
    base_dir = '/kaggle/input/cassava-leaf-disease-classification'
    # List all image files in the test_images directory
    test_image_dir = os.path.join(base_dir, 'test_images')
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    # Create a DataFrame with the image IDs (including the .jpg extension) and full paths
    test_img_input_map = pd.DataFrame({
        'id': image_files,
        'img_input1': [os.path.join(test_image_dir, f) for f in image_files]
    })
    test_img_input_map.to_csv("./test_img_input_map.csv", index=False)
    print("`test_img_input_map.csv` created and saved.")
    test_img_input_map.to_csv("./test_img_input_map.csv", index=False)
    print("`test_img_input_map.csv` created and saved.")
except Exception as e:
    error_message = traceback.format_exc()
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_error.txt', 'w') as f:
        f.write(error_message)
        raise e


import os
import traceback
import warnings
def write_warning_to_file(message, category, filename, lineno, file=None, line=None):
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_warnings.txt', 'w') as f:
        f.write(warnings.formatwarning(message=message, category=category, filename=filename, lineno=lineno, line=line))
try:
    import os
    import pandas as pd
    
    # Define the base directory
    base_dir = '/kaggle/input/cassava-leaf-disease-classification'
    
    # Read the train.csv file
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    
    # Ensure the image_id column is renamed to 'id'
    train_df.rename(columns={'image_id': 'id'}, inplace=True)
    
    # Rename the label column to indicate it is a classification target
    train_df.rename(columns={'label': 'label_classification'}, inplace=True)
    
    # Create the final DataFrame
    train_tab_target_map = train_df[['id', 'label_classification']]
    
    # <｜fim▁end｜>
    # save
    train_tab_target_map.to_csv("./train_tab_target_map.csv", index=False)
    print("`train_tab_target_map.csv` created and saved.")
    
    
    # <｜fim▁end｜>
    # save
    train_tab_target_map.to_csv("./train_tab_target_map.csv", index=False)
    print("`train_tab_target_map.csv` created and saved.")
except Exception as e:
    error_message = traceback.format_exc()
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_error.txt', 'w') as f:
        f.write(error_message)
        raise e



import os
import traceback
import warnings

def write_warning_to_file(message, category, filename, lineno, file=None, line=None):
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_warnings.txt', 'w') as f:
        f.write(warnings.formatwarning(message=message, category=category, filename=filename, lineno=lineno, line=line))



try:
    import os
    import pandas as pd
    base_dir = '/kaggle/input/cassava-leaf-disease-classification'
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    train_df.rename(columns={'image_id': 'id'}, inplace=True)
    train_df['img_input1'] = train_df['id'].apply(lambda x: os.path.join(base_dir, 'train_images', x))
    train_img_input_map = train_df[['id', 'img_input1']]
    train_img_input_map.to_csv("./train_img_input_map.csv", index=False)
    print("`train_img_input_map.csv` created and saved.")
    train_img_input_map.to_csv("./train_img_input_map.csv", index=False)
    print("`train_img_input_map.csv` created and saved.")
except Exception as e:
    error_message = traceback.format_exc()
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_error.txt', 'w') as f:
        f.write(error_message)
        raise e


import atexit
import json
import os
import pathlib
import shutil
import time
import traceback
import warnings
from argparse import ArgumentParser
from functools import partial
import pytorch_lightning as L
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.optim
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
torch.manual_seed(123)


import os
import torch.optim
TAB_EMBED_LR = 1e-4
TAB_HEAD_LR = 1e-4
IMG_EMBED_LR = 1e-5
IMG_HEAD_LR = 1e-5
TXT_EMBED_LR = 1e-5
TXT_HEAD_LR = 1e-5
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
NUM_WORKERS = 4
OPTIMIZER = torch.optim.Adam
N_TRIALS = 20
TTA_ROUNDS = 4
if os.getenv("AGENT_DEBUG"):
    val_proportion = 0.05
else:
    val_proportion = 0.25 


import torch
import torchvision.models as models
from torch import nn
import pandas as pd
import os
from shutil import copyfile

import torch
import torchvision.models as models
from torch import nn
copyfile(src = "/kaggle/input/imgembedcorrect/img_embed.py", dst = "../working/img_embed.py")
from img_embed import ImageEmbedder


import pandas as pd
import torch
from torch import nn
import os.path
import pandas as pd
import torch
from torch import nn
# --- Design torch model and implement a differentiable torch loss functions
class TabularHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int):
        super(TabularHead, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)
def classification_loss(pred_logits: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(pred_logits, target_one_hot.argmax(dim=1))
    # --- Test losses
    # For regression targets:
    target = torch.rand(batch_size, output_dim, dtype=torch.float)
    reg_loss = regression_loss(output, target).mean()
    # For classification targets:
    target = torch.rand(batch_size, output_dim, dtype=torch.float)
    class_loss = classification_loss(output, target).mean()
    # @NO_MEMORY_START@
    print(f"Could compute tabular outputs and losses without error.")
    print(f"Output size: {output.shape[-1]}")
    # @NO_MEMORY_END@
class SubmissionFormatError(Exception):
    """When all attempts to create the submission format have failed"""
    pass


copyfile(src="/kaggle/input/mapdataset/map_dataset.py", dst="./map_dataset.py")
from map_dataset import Identity, MapDataset, map_dataset_collate_function
tab_fe_preprocess = None


import os
import pandas as pd
import numpy as np


import pandas as pd

def calculate_class_weights(target_df:pd.DataFrame, target_columns: list) -> np.ndarray:
    # Generate a weight vector of ones (same length as the number of samples)
    weight_vector = np.ones(len(target_df))

    return weight_vector

class_weights_fn = calculate_class_weights


"""
This script creates the transform functions for the tabular training targets.
"""
import json
from typing import Any, Union, Tuple

import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

# Read target tab file
df_train_target = pd.read_csv('./train_tab_target_map.csv')

# Create onehot encoder for transforming the target
class_names_columns_regression = [col for col in df_train_target.columns if col.endswith('_regression')]
class_names_columns_classification = [col for col in df_train_target.columns if col.endswith('_classification')]
class_names_data_classification = df_train_target[class_names_columns_classification].values
if len(class_names_columns_classification) > 0:
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(class_names_data_classification)
    for col, cat in zip(class_names_columns_classification, enc.categories_):
        if len(cat) > 0.5 * class_names_data_classification.shape[0]:
            raise ValueError(f"Trying to one-hot encode {len(class_names_columns_classification)} classification "
                             f"columns, but at least one column has {len(cat)} distinct values. Are you sure "
                             f"the column {col} is a categorical column?")
else:
    enc = None


target_columns_transform = target_tranform


def tab_target_transform(original_target: pd.DataFrame) -> pd.DataFrame:
    """
    Transform for tabular targets, maps original submission format to usable numerical format.
    It cannot return `None`, it has to at least return `original_target` if no transform is needed.
    """

    # Filter regression columns
    df_regression = original_target.filter(like='_regression', axis=1)
    if enc is None:
        df_regression.insert(loc=0, column='id', value=original_target['id'].values)
        return df_regression

    # Convert class labels to one-hot encodings
    class_name = list(enc.get_feature_names_out(class_names_columns_classification))
    x = original_target[class_names_columns_classification].values
    onehot_class = enc.transform(x)

    df_classification = pd.DataFrame([], columns=['id'] + class_name)
    df_classification['id'] = original_target['id'].values
    df_classification.iloc[:, 1:] = onehot_class.toarray()

    # Concatenate the two dataframe
    transformed_target = pd.concat([df_classification, df_regression],
                                   axis=1) if not df_regression.empty else df_classification
    transformed_target = transformed_target.infer_objects()
    return transformed_target


def onehot_to_classname(
        onehot: np.array,
        probabilities: np.array,
        classification_threshold: float,
        unknown_classname: str | None
):
    """
    This function converts a one-hot encoded array back to class names, and replaces classes with probabilities below a
    threshold with a specified 'unknown' class name.

    Parameters:
    onehot (np.array): A one-hot encoded numpy array representing class memberships.
    probabilities (np.array): A numpy array of class probabilities corresponding to the classes in 'onehot'.
    classification_threshold (float): A threshold for class probabilities. Classes with probabilities below this threshold are considered 'unknown'.
    unknown_classname (str): A string to replace the class names of 'unknown' classes (i.e., classes with probabilities below the threshold).

    Returns:
    class_names_array (np.array): A numpy array of class names. 'Unknown' classes have been replaced with 'unknown_classname'.
    """

    class_names_array = enc.inverse_transform(onehot)
    class_names_array = class_names_array[class_names_array != np.array(None)]

    if unknown_classname:
        unknown_classes_idx = np.argwhere(probabilities < classification_threshold)
        class_names_array = class_names_array.astype(object)
        class_names_array[unknown_classes_idx] = unknown_classname

    return class_names_array


def extract_values_top_k(input_string: str) -> Tuple[int, str]:
    """
    Extracts values between parentheses and the comma.

    Args:
        input_string (str): A string containing values in the format '(x, y)'.

    Returns:
        list: A list containing the extracted values.

    Example:
        extract_values('(5, new_whale)')
        ['5', 'new_whale']
    """
    result = re.findall(r'[^,()]+', input_string)
    top_k = int(result[0])
    unknown_class_name = result[1].replace(' ', '')
    unknown_class_name = None if 'none' in unknown_class_name.lower() else unknown_class_name
    return top_k, unknown_class_name


def process_classification_target(
        classification_probits: np.array,
        columns_target_map: dict,
        columns_class_name: list,
        classification_threshold: float = 1e-5

) -> pd.DataFrame:
    """
    This function processes a classification target by grouping labels, extracting features, and creating a DataFrame.

    Parameters:
    classification_target (np.array): A numpy array representing the classification probabilities.
    columns_target_map (dict): A dictionary mapping column names to targets.
    columns_class_name (list): A list of class names for the columns.
    classification_threshold (float, optional): A threshold for classification. Default is 1e-5.

    Returns:
    df_classification (pd.DataFrame): A pandas DataFrame that contains the processed classification target.
    """
    groups_label = enc.categories_
    idx_start = 0
    df_classification = pd.DataFrame()

    for group in groups_label:

        # Extract the label columns names and probits of the current classification group
        label_cols_names = columns_class_name[idx_start:idx_start + len(group)]
        group_probits = classification_probits[:, idx_start:idx_start + len(group)]

        # check that probits are indeed probits
        assert np.isclose(group_probits.sum(axis=-1), 1.0).all(), \
            f"Probits of group {group} do not sum to 1! Did you forget to apply softmax?"

        # Extract the feature column name
        feature_column_name = label_cols_names[0][:label_cols_names[0].rfind("_classification")] + "_classification"

        if columns_target_map[feature_column_name] == "proba":
            df = pd.DataFrame(group_probits, columns=label_cols_names)

        else:
            raw_top_k = columns_target_map[feature_column_name].split('top_k')[1]
            top_k, unknown_classname = extract_values_top_k(input_string=raw_top_k)
            onehot_group, proba_group = get_topk_onehot(probits=group_probits, k=top_k)

            # Format the onot hot to match shape of the onehot encoder
            onehot = np.zeros((classification_probits.shape[0] * top_k, classification_probits.shape[1]))
            onehot[:, idx_start:idx_start + len(group)] = onehot_group

            class_names_array = onehot_to_classname(
                onehot=onehot,
                probabilities=proba_group,
                classification_threshold=classification_threshold,
                unknown_classname=unknown_classname,
            )

            class_names_array = class_names_array.reshape(classification_probits.shape[0], top_k)

            if top_k > 1:
                class_names_array = pd.Series(list(class_names_array))

            df = pd.DataFrame(class_names_array, columns=[feature_column_name])

        df_classification = pd.concat([df_classification, df], axis=1)

        idx_start += len(group)

    return df_classification


def get_topk_onehot(
        probits: np.array,
        k: int = 1,
) -> np.array:
    """
    This function returns the top 'k' values and their indices from the input array 'probits'.

    Parameters:
    probits (np.array): A numpy array from which to select the top 'k' values.
    k (int, optional): The number of top values to select. Default is 1.

    Returns:
    onehot (np.array): A one-hot encoded numpy array of shape (probits.shape[0] * k, probits.shape[1]).
                       The 'i'th row of 'onehot' corresponds to the 'i'th top value in 'probits'.
    values (np.array): A flattened numpy array of the top 'k' values in 'probits'.
    """
    values, indices = torch.topk(torch.Tensor(probits), k)
    onehot = np.zeros((probits.shape[0] * k, probits.shape[1]))
    onehot[np.arange(onehot.shape[0]), indices.flatten().numpy()] = 1

    return onehot, values.flatten().numpy()


def tab_target_inverse_transform(target_values: np.ndarray, ids: Union[list[Any], np.array]) -> pd.DataFrame:
    """
    Inverse transform for tabular targets, maps back to original submission format.
    It cannot return `None`, it has to at least return the `transformed_target` if no inverse transform is needed.
    Args:
        target_values: batch of target values
        ids: list of the entry ids
    """
    if enc is None:
        regression_target = target_values
    else:
        columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))
        classification_target = target_values[:, :len(columns_class_name)]
        regression_target = target_values[:, len(columns_class_name):]

    df_regression_target = pd.DataFrame(regression_target, columns=class_names_columns_regression)
    if enc is None:
        df_regression_target.insert(0, "id", ids)
        return df_regression_target

    # We assume that if we are here, it's because at least one target column is a classification,
    # otherwise, if the task is purely a classification task, we probably shouldn't end up here ...
    df_transformed_classification_target = process_classification_target(
        classification_probits=classification_target,
        columns_target_map=target_columns_transform,
        columns_class_name=columns_class_name,
    )

    transformed_target = pd.concat([df_transformed_classification_target, df_regression_target], axis=1)
    transformed_target.insert(0, "id", ids)

    for pattern in ["_regression", "_classification"]:
        transformed_target.rename({c: c.replace(pattern, "") for c in transformed_target.columns}, axis=1, inplace=True)
    return transformed_target




custom_tab_regression_scaler = None



img_target_transform = Identity()
img_target_inverse_transform = Identity()



txt_target_transform = Identity()
txt_target_inverse_transform = Identity()


import os
import pandas as pd
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T


import torchvision.transforms as T

CustomTrainImageInputTransform: T.Compose = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.43032029271125793, 0.4967266917228699, 0.31341689825057983],
                std=[0.23058539628982544, 0.2335016131401062, 0.22131390869617462]),
    T.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x)
])



import torchvision.transforms as T

CustomTestImageInputTransform: T.Compose = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.43032029271125793, 0.4967266917228699, 0.31341689825057983],
                std=[0.23058539628982544, 0.2335016131401062, 0.22131390869617462]),
    T.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x)
])




submission_format_functions = []


submission_names = []


import os
import traceback
import warnings

def write_warning_to_file(message, category, filename, lineno, file=None, line=None):
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_warnings.txt', 'w') as f:
        f.write(warnings.formatwarning(message=message, category=category, filename=filename, lineno=lineno, line=line))




"""
This script creates a function that takes an input DataFrame and formats it to follow the df_sample_submission format
"""
# useful imports
import os
import numpy as np
import pandas as pd


### Submission format definition
def df_to_submission_format(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format the input_df to the submission format DataFrame
    Args:
        input_df: DataFrame to be formatted
    Return:
        format_submission_df: DataFrame formatted
    """
    # Rename the 'id' column to 'image_id'
    input_df = input_df.rename(columns={'id': 'image_id'})
    
    # Ensure the 'label' column contains only integer values
    if not np.issubdtype(input_df['label'].dtype, np.integer):
        input_df['label'] = input_df['label'].astype(int)
    
    # Check if the 'label' column contains only allowed values
    allowed_values = [0, 1, 2, 3, 4]
    if not input_df['label'].isin(allowed_values).all():
        raise ValueError("The 'label' column contains values outside the allowed range [0, 1, 2, 3, 4].")
    
    # Return the formatted DataFrame
    return input_df

submission_format_functions.append(df_to_submission_format)
submission_names.append('submission.csv')


import os
import traceback
import warnings

def write_warning_to_file(message, category, filename, lineno, file=None, line=None):
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_warnings.txt', 'w') as f:
        f.write(warnings.formatwarning(message=message, category=category, filename=filename, lineno=lineno, line=line))




"""
This script creates a function that takes an input Dataframe and format it to follow the df_sample_submission format
"""
# useful imports
import os
import numpy as np
import pandas as pd


### Submission format definition
def df_to_submission_format_alt(
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format the input_df to the submission format DataFrame
    Args:
        input_df: DataFrame to be formatted
    Return:
        format_submission_df: DataFrame formatted
    """
    # Step 1: Rename the 'id' column to 'image_id'
    input_df = input_df.rename(columns={'id': 'image_id'})
    
    # Step 2: Ensure the 'label' column contains integer values
    input_df['label'] = input_df['label'].astype(int)
    
    # Step 3: Reorder the columns to match the required format
    format_submission_df = input_df[['image_id', 'label']]
    
    return format_submission_df

submission_format_functions.append(df_to_submission_format_alt)
submission_names.append('submission_alt.csv')


if len(submission_format_functions) == 0:
    raise ValueError('No valid submission format functions found in setup!')


import os
import traceback
import warnings

def write_warning_to_file(message, category, filename, lineno, file=None, line=None):
    with open('/nfs/aiml/alexandre/Projects/agent/workspace/cassava-leaf-disease-classification/seed_0/_code_warnings.txt', 'w') as f:
        f.write(warnings.formatwarning(message=message, category=category, filename=filename, lineno=lineno, line=line))




"""
This script creates a function that takes the predicted output `y_pred` and the true output `y_true`
and returns the value of the metric corresponding to the task.
"""
# useful imports
import os
import numpy as np
import pandas as pd
from torch import Tensor

# metric definition
def metric_function(
        y_pred: pd.DataFrame | Tensor,
        y_true: pd.DataFrame | Tensor,
) -> float:
    """
    Computes the metric on (a batch of) inputs and returns the result.
    Args:
        y_pred: the predicted target
        y_true: the true target
    """
    # Ensure both inputs are pandas DataFrames
    if isinstance(y_pred, Tensor):
        y_pred = pd.DataFrame({'image_id': y_pred[:, 0].numpy(), 'label': y_pred[:, 1].numpy()})
    if isinstance(y_true, Tensor):
        y_true = pd.DataFrame({'image_id': y_true[:, 0].numpy(), 'label': y_true[:, 1].numpy()})

    # Drop the 'image_id' column to focus on the labels
    y_pred_labels = y_pred['label']
    y_true_labels = y_true['label']

    # Ensure the lengths of y_pred and y_true are the same
    if len(y_pred_labels) != len(y_true_labels):
        raise ValueError("The length of y_pred and y_true must be the same.")

    # Count the number of correct predictions
    correct_predictions = (y_pred_labels == y_true_labels).sum()

    # Calculate the accuracy
    accuracy = correct_predictions / len(y_true_labels)

    return accuracy




class NanLossError(Exception):
    """ raise when there is a nan loss"""
    pass


def get_optional_path(path: str) -> str | None:
    """Return path if it exists, else return None"""
    if os.path.exists(path):
        return path


tab_input_map_path = get_optional_path('train_tab_input_map.csv')


img_input_map_path = get_optional_path('train_img_input_map.csv')


txt_input_map_path = get_optional_path('train_txt_input_map.csv')


tab_target_map_path = get_optional_path('train_tab_target_map.csv')


img_target_map_path = get_optional_path('train_img_target_map.csv')


txt_target_map_path = get_optional_path('train_txt_target_map.csv')


test_tab_input_map_path = get_optional_path('test_tab_input_map.csv')


test_img_input_map_path = get_optional_path('test_img_input_map.csv')


test_txt_input_map_path = get_optional_path('test_txt_input_map.csv')


(dataset, test_dataset) = MapDataset.create_train_test_datasets(train_tab_input_map_path=tab_input_map_path, train_img_input_map_path=img_input_map_path, train_txt_input_map_path=txt_input_map_path, train_tab_target_map_path=tab_target_map_path, train_img_target_map_path=img_target_map_path, train_txt_target_map_path=txt_target_map_path, test_tab_input_map_path=test_tab_input_map_path, test_img_input_map_path=test_img_input_map_path, test_txt_input_map_path=test_txt_input_map_path, tab_input_transform=Identity(), img_input_transform=Identity(), txt_input_transform=Identity(), tab_target_transform=tab_target_transform, img_target_transform=img_target_transform, txt_target_transform=txt_target_transform, tab_target_inverse_transform=tab_target_inverse_transform, img_target_inverse_transform=img_target_inverse_transform, txt_target_inverse_transform=txt_target_inverse_transform, custom_tab_regression_scaler=custom_tab_regression_scaler, custom_img_train_input_transform=CustomTrainImageInputTransform, custom_img_test_input_transform=CustomTestImageInputTransform, tab_fe=tab_fe_preprocess.preprocess if tab_fe_preprocess else None)


if os.getenv('AGENT_DEBUG'):
    val_proportion = 0.05
else:
    val_proportion = 0.25


(train_dataset, validation_dataset) = dataset.split(frac=val_proportion)


validation_dataset.img_input_transform = test_dataset.img_input_transform


def get_data_loaders(train_batch_size: int=TRAIN_BATCH_SIZE, val_batch_size: int=TEST_BATCH_SIZE):
    if class_weights_fn is not None:
        target_df = train_dataset.tab_target_map.copy(deep=True)
        target_df = target_df.drop(columns='id', errors='ignore')
        target_columns = target_df.columns.tolist()
        weight_vector = calculate_class_weights(target_df=target_df, target_columns=target_columns)
        weight_vector = torch.from_numpy(weight_vector.astype(np.float32))
        sampler = WeightedRandomSampler(weight_vector, len(weight_vector))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle, pin_memory=True, prefetch_factor=2, persistent_workers=True, collate_fn=map_dataset_collate_function, num_workers=NUM_WORKERS, sampler=sampler)
    valid_dl = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=2, collate_fn=map_dataset_collate_function, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=map_dataset_collate_function, num_workers=NUM_WORKERS)
    return (train_dl, valid_dl, test_dl)


(train_dataloader, validation_dataloader, test_dataloader) = get_data_loaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)


train_input_sample = next(iter(train_dataloader))


(tab_inputs_batch, img_inputs_batch, txt_inputs_batch) = (None, None, None)


(tab_targets_batch, img_targets_batch, txt_targets_batch) = (None, None, None)


for (indices, (tab_inputs_batch, img_inputs_batch, txt_inputs_batch), (tab_targets_batch, img_targets_batch, txt_targets_batch)) in train_dataloader:
    break


if tab_targets_batch is not None:
    OUTPUT_DIM = tab_targets_batch.shape[1]
elif img_targets_batch is not None:
    OUTPUT_DIM = img_targets_batch.shape[1]
elif txt_targets_batch is not None:
    OUTPUT_DIM = txt_targets_batch.shape[1]


def get_tab_embedder():
    if 'TabularEmbedder' in globals():
        tab_input_dim = tab_inputs_batch.shape[-1]
        tab_embedder = TabularEmbedder(input_dim=tab_input_dim, embed_dim=TAB_EMBED_DIM)
        tab_embed_dim = TAB_EMBED_DIM
    else:
        tab_embedder = None
        tab_embed_dim = 0
    return (tab_embedder, tab_embed_dim)


def get_img_embedder():
    if 'ImageEmbedder' in globals():
        assert issubclass(ImageEmbedder, nn.Module)

        class AuxImageEmbedder(ImageEmbedder):
            """ Wrap ImageEmbedder to deal with the input dimension """

            def forward(self, x: torch.Tensor):
                """
                Args:
                    x: dimension (batch, n_images_per_id, n_channels, height, width)

                Returns:
                     y: dimension (batch, embed_dim)  --> the `n_images_per_id` are flatten
                """
                y = super().forward(x.view(-1, *x.shape[-3:]))
                return y.reshape(len(x), -1)
        img_embedder = AuxImageEmbedder()
        with torch.no_grad():
            img_embed_dim = img_embedder(img_inputs_batch[:2]).shape[-1]
    else:
        img_embedder = None
        img_embed_dim = 0
    return (img_embedder, img_embed_dim)


def get_txt_embedder():
    if 'TextEmbedder' in globals():
        assert issubclass(TextEmbedder, nn.Module)

        class AuxTextEmbedder(TextEmbedder):
            """ Wrap TextEmbedder to deal with the input dimension """

            def forward(self, x: pd.DataFrame | np.ndarray | torch.Tensor):
                """
                Args:
                    x: dimension (batch, n_texts_per_id)

                Returns:
                     y: dimension (batch, embed_dim)  --> the `n_texts_per_id` are flattened
                """
                bsz = x.shape[0]
                if isinstance(x, pd.DataFrame):
                    x = x.values
                    x = x.reshape(-1, *x.shape[1:])
                    x = x.flatten().tolist()
                y = super(AuxTextEmbedder, self).forward(x)
                return y.reshape(bsz, -1)
        txt_embedder = AuxTextEmbedder()
        with torch.no_grad():
            txt_embed_dim = txt_embedder(txt_inputs_batch).shape[-1]
    else:
        txt_embedder = None
        txt_embed_dim = 0
    return (txt_embedder, txt_embed_dim)


def get_tab_head():
    return (TabularHead, regression_loss, classification_loss)



get_img_head = None
img_loss = None



get_txt_head = None
txt_loss = None


class SubmissionModel(L.LightningModule):

    def __init__(self, learning_rate=1e-05, optimizer_choice='adam'):
        super().__init__()
        (tab_embedder, tab_embed_dim) = get_tab_embedder()
        (img_embedder, img_embed_dim) = get_img_embedder()
        (txt_embedder, txt_embed_dim) = get_txt_embedder()
        self.tab_embedder = tab_embedder
        self.img_embedder = img_embedder
        self.txt_embedder = txt_embedder
        self.embed_dim = tab_embed_dim + img_embed_dim + txt_embed_dim
        (TabularHead, regression_loss, classification_loss) = get_tab_head()
        self.tab_head = TabularHead(embed_dim=self.embed_dim, output_dim=OUTPUT_DIM)
        self.learning_rate = learning_rate
        self.optimizer_choice = optimizer_choice
        if get_img_head is None:
            self.img_head = None
        else:
            self.img_head = get_img_head()
        if get_txt_head is None:
            self.txt_head = None
        else:
            self.txt_head = get_txt_head()
        self.tab_regression_loss = regression_loss
        self.tab_classification_loss = classification_loss
        self.img_loss = img_loss
        self.txt_loss = txt_loss
        self.unfreeze_epoch = None

    def tab_loss(self, pred: torch.Tensor, target: pd.DataFrame) -> torch.Tensor:
        assert pred.shape == target.shape, (pred.shape, target.shape)
        target = torch.from_numpy(target.values).to(pred)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        loss = 0.0
        if enc:
            columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))
            classification_pred = pred[:, :len(columns_class_name)]
            classification_target = target[:, :len(columns_class_name)]
            groups_label = enc.categories_
            idx_start = 0
            for group in groups_label:
                group_pred = classification_pred[:, idx_start:idx_start + len(group)]
                group_target = classification_target[:, idx_start:idx_start + len(group)]
                idx_start += len(group)
                loss += self.tab_classification_loss(group_pred, group_target).mean()
            if pred.shape[-1] > len(columns_class_name):
                regression_pred = pred[:, len(columns_class_name):]
                regression_target = target[:, len(columns_class_name):]
                regression_target[regression_target.isnan()] = regression_pred[regression_target.isnan()]
                loss += self.tab_regression_loss(regression_pred, regression_target).mean()
        else:
            target[target.isnan()] = pred[target.isnan()]
            loss += self.tab_regression_loss(pred, target).mean()
        return loss

    def embed(self, tab, img, txt) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Get the embeddings for each modality"""
        if self.tab_embedder is None:
            tab_embed = None
        else:
            tab = torch.tensor(tab.values).to(device=self.device, dtype=self.dtype)
            tab_embed = self.tab_embedder(tab)
        if self.img_embedder is None:
            img_embed = None
        else:
            img = img.to(device=self.device, dtype=self.dtype)
            img_embed = self.img_embedder(img)
        if self.txt_embedder is None:
            txt_embed = None
        else:
            self.txt_embedder.to(self.device)
            txt_embed = self.txt_embedder(txt)
        return (tab_embed, img_embed, txt_embed)

    def decode(self, latent_embed: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Obtain logits / images / next tokens predictions given the latent embedding"""
        if self.tab_head is None:
            pred_tab = None
        else:
            pred_tab = self.tab_head(latent_embed)
        if self.img_head is None:
            pred_img = None
        else:
            pred_img = self.img_head(latent_embed)
        if self.txt_head is None:
            pred_txt = None
        else:
            pred_txt = self.txt_head(latent_embed)
        return (pred_tab, pred_img, pred_txt)

    def forward(self, tab: pd.DataFrame, img: torch.Tensor, txt: pd.DataFrame) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        (tab_embed, img_embed, txt_embed) = self.embed(tab=tab, img=img, txt=txt)
        to_fuse = [embed for embed in [tab_embed, img_embed, txt_embed] if embed is not None]
        assert len(to_fuse) > 0
        latent_embed = torch.cat(to_fuse, dim=1)
        return self.decode(latent_embed=latent_embed)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        (indices, (tab_inputs_batch, img_inputs_batch, txt_inputs_batch), (tab_targets_batch, img_targets_batch, txt_targets_batch)) = batch
        (pred_tab, pred_img, pred_txt) = self.forward(tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch)
        loss = 0.0
        if tab_targets_batch is not None:
            loss += self.tab_loss(pred=pred_tab, target=tab_targets_batch).mean()
        if self.img_loss is not None:
            loss += self.img_loss(pred_img, img_targets_batch).mean()
        if self.txt_loss:
            loss += self.txt_loss(pred_txt, txt_targets_batch).mean()
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(indices))
        if torch.isnan(loss).item():
            raise NanLossError(f'Loss value is {loss}')
        return loss

    @staticmethod
    def get_param_groups(model: nn.Module | None, **kwargs) -> dict[str, ...] | None:
        if model is None:
            return None
        else:
            return {'params': model.parameters(), **kwargs}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        groups = [self.get_param_groups(model=self.tab_embedder, lr=self.learning_rate), self.get_param_groups(model=self.tab_head, lr=self.learning_rate), self.get_param_groups(model=self.img_embedder, lr=self.learning_rate), self.get_param_groups(model=self.img_head, lr=self.learning_rate), self.get_param_groups(model=self.txt_embedder, lr=self.learning_rate), self.get_param_groups(model=self.txt_head, lr=self.learning_rate)]
        params = [group for group in groups if group is not None]
        if self.optimizer_choice == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_choice == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        elif self.optimizer_choice == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer_choice}')
        return optimizer

    def on_train_start(self) -> None:
        self.unfreeze_epoch = MAX_EPOCHS // 2

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.unfreeze_epoch and self.img_embedder:
            print(f'Unfreezing layers at epoch {self.current_epoch}')
            self.img_embedder.unfreeze(n_last_layers=3)

    def validation_step(self, batch, batch_idx):
        """Run"""
        (indices, inputs_batch, targets_batch) = batch
        (tab_inputs_batch, img_inputs_batch, txt_inputs_batch) = inputs_batch
        (tab_targets_batch, img_targets_batch, txt_targets_batch) = targets_batch
        preds_batch = self.forward(tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch)
        (tab_preds_batch, img_preds_batch, txt_preds_batch) = preds_batch
        loss = 0.0
        if self.tab_loss is not None:
            loss += self.tab_loss(tab_preds_batch, tab_targets_batch)
        if self.img_loss is not None:
            loss += self.img_loss(img_preds_batch, img_targets_batch).mean()
        if self.txt_loss:
            loss += self.txt_loss(txt_preds_batch, txt_targets_batch).mean()
        self.log('valid_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(indices))

    def get_submissions(self, dataloader: DataLoader, get_raw_preds: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """ Get submissions for each element of the dataloader

        Args:
            dataloader: dataloader on which
            get_raw_preds: whether to also return raw predictions
        """
        submission = pd.DataFrame([])
        raw_tab_preds = []
        target = pd.DataFrame([])
        img_preds = []
        txt_preds = []
        indices = []
        tab_targets_batch = None
        for batch in tqdm(dataloader):
            (indices_batch, (tab_inputs_batch, img_inputs_batch, txt_inputs_batch), (tab_targets_batch, img_targets_batch, txt_targets_batch)) = batch
            indices += list(indices_batch)
            with torch.no_grad():
                (pred_tab, pred_img, pred_txt) = self.forward(tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch)
            if pred_tab is not None:
                if enc:
                    softmax = nn.Softmax(dim=1)
                    columns_class_name = list(enc.get_feature_names_out(class_names_columns_classification))
                    classification_pred = pred_tab[:, :len(columns_class_name)].clone()
                    groups_label = enc.categories_
                    idx_start = 0
                    for group in groups_label:
                        group_pred = classification_pred[:, idx_start:idx_start + len(group)]
                        pred_tab[:, idx_start:idx_start + len(group)] = softmax(group_pred)
                        idx_start += len(group)
                pred_tab = pred_tab.cpu().numpy()
                if get_raw_preds:
                    raw_tab_preds.append(pred_tab)
                pred_tab = dataset.tab_target_inverse_transform(pred_tab, indices_batch)
                submission = pd.concat([submission, pred_tab])
            if pred_img is not None:
                assert not isinstance(dataset.img_target_inverse_transform, Identity), 'Predicted images must be transformed'
                pred_img = dataset.img_target_inverse_transform(pred_img)
                img_preds.append(pred_img)
            if pred_txt is not None:
                pred_txt = dataset.txt_target_inverse_transform(pred_txt)
                txt_preds.append(pred_txt)
            if tab_targets_batch is not None and len(tab_targets_batch) > 0:
                tab_targets_batch = dataset.tab_target_inverse_transform(tab_targets_batch.values, indices_batch)
                target = pd.concat([target, tab_targets_batch])
            if img_targets_batch is not None and len(img_targets_batch) > 0:
                img_targets_batch = dataset.img_target_inverse_transform(img_targets_batch.values, indices_batch)
                target = pd.concat([target, img_targets_batch])
            if txt_targets_batch is not None and len(txt_targets_batch.columns) > 0:
                txt_targets_batch = dataset.txt_target_inverse_transform(txt_targets_batch.values, indices_batch)
                target = pd.concat([target, txt_targets_batch])
        if dataset.custom_tab_regression_scaler is not None:
            submission[dataset.tab_regression_target_cols] = dataset.custom_tab_regression_scaler.inverse_transform(submission[dataset.tab_regression_target_cols])
            if tab_targets_batch is not None and len(tab_targets_batch) > 0:
                target[dataset.tab_regression_target_cols] = dataset.custom_tab_regression_scaler.inverse_transform(target[dataset.tab_regression_target_cols])
        if get_raw_preds:
            raw_tab_preds = pd.DataFrame(np.concatenate(raw_tab_preds), index=indices)
        else:
            raw_tab_preds = None
        return (submission, target, raw_tab_preds)

    def get_blend_submissions(self, dataloader: DataLoader) -> tuple[list[Tensor], list[Tensor], list[...]]:
        """Get submissions for each element of the dataloader"""
        embeddings_save = []
        targets_save = []
        indices = []
        for batch in tqdm(dataloader):
            (indices_batch, (tab_inputs_batch, img_inputs_batch, txt_inputs_batch), (tab_targets_batch, img_targets_batch, txt_targets_batch)) = batch
            indices += list(indices_batch)
            with torch.no_grad():
                (tab_embed, img_embed, txt_embed) = self.embed(tab=tab_inputs_batch, img=img_inputs_batch, txt=txt_inputs_batch)
                final_embedding = [embed for embed in [tab_embed, img_embed, txt_embed] if embed is not None]
                assert len(final_embedding) > 0
                latent_embed = torch.cat(final_embedding, dim=1)
                embeddings_save.append(latent_embed)
            batch_targets = []
            if tab_targets_batch is not None and len(tab_targets_batch) > 0:
                batch_targets.append(torch.tensor(tab_targets_batch.values, dtype=torch.float32))
            if img_targets_batch is not None and len(img_targets_batch) > 0:
                batch_targets.append(torch.tensor(img_targets_batch.values, dtype=torch.float32))
            if txt_targets_batch is not None and len(txt_targets_batch.columns) > 0:
                batch_targets.append(torch.tensor(txt_targets_batch.values, dtype=torch.float32))
            if batch_targets:
                targets_save.append(torch.cat(batch_targets, dim=1))
        return (embeddings_save, targets_save, indices)

    def get_score(self, dataloader: DataLoader) -> tuple[dict[str, float], pd.DataFrame]:
        """ Iterate through the dataloader to get predictions in submission format and compute the score
        Returns:
            score
            predictions in submission format
        """
        (submissions, targets, _) = self.get_submissions(dataloader=dataloader)
        scores = {}
        for (submission_format_func, submission_name) in zip(submission_format_functions, submission_names):
            try:
                formatted_submission = submission_format_func(submissions)
                formatted_targets = submission_format_func(targets)
                try:
                    score = metric_function(y_pred=formatted_submission, y_true=formatted_targets)
                    score = float(score)
                except (IndexError, TypeError):
                    traceback.print_exc()
                    score = np.nan
                scores[submission_name] = score
            except Exception as e:
                print(f'Hit exception when trying to create {submission_name}: {e}')
        return (scores, submissions)


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def objective(params: dict[str, ...], trial_num: int, accelerator: str, max_time: str, max_trials: int=10, devices: list[int] | str='auto') -> torch.Tensor:
    learning_rate = params['learning_rate'][0]
    optimizer_choice = params['optimizer'][0]
    logger = TensorBoardLogger('tb_logs', name=f'hebo_run')
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='valid_loss', mode='min', dirpath='./trials', filename=f'trial_{trial_num}')
    early_stop_callback = EarlyStopping(monitor='valid_loss', min_delta=0.0, patience=5, verbose=True, mode='min')
    train_batch_size = TRAIN_BATCH_SIZE
    test_batch_size = TEST_BATCH_SIZE
    (train_dataloader, validation_dataloader, _) = get_data_loaders(train_batch_size, test_batch_size)
    val_loss = None
    _trial = 0
    while val_loss is None and _trial < max_trials:
        model = SubmissionModel(learning_rate=learning_rate, optimizer_choice=optimizer_choice)
        try:
            extra_kwargs = dict(max_time=max_time, max_epochs=MAX_EPOCHS)
            trainer = L.Trainer(accelerator=accelerator, devices=devices, logger=logger, callbacks=[checkpoint_callback, early_stop_callback], **extra_kwargs)
            trainer.fit(model, train_dataloader, validation_dataloader)
            val_loss = trainer.checkpoint_callback.best_model_score
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if 'out of memory' in str(e).lower():
                print('| WARNING: ran out of memory, retrying with half the batch size')
                with torch.no_grad():
                    torch.cuda.empty_cache()
            train_batch_size //= 2
            test_batch_size //= 2
            if train_batch_size <= 0:
                raise RuntimeError('No batch size seems to fit on the device')
            (train_dataloader, validation_dataloader, _) = get_data_loaders(train_batch_size, test_batch_size)
        finally:
            _trial += 1
    return val_loss


def remove_trials(trials_dir: str) -> None:
    if os.path.exists(trials_dir):
        shutil.rmtree(trials_dir)
        print(f'Deleted existing {trials_dir} folder.')


def get_best_checkpoint(best_index: int, trials_dir: str='./trials') -> None:
    best_model_path = os.path.join(trials_dir, f'trial_{best_index}.ckpt')
    print('best_model_path', best_model_path)
    if os.path.exists(best_model_path):
        shutil.move(best_model_path, './best_model.ckpt')
        print(f'Copied best model ')


def _format_submission_dtypes(formatted_submission: pd.DataFrame) -> pd.DataFrame:
    sample_submission = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv')
    for column in sample_submission:
        if sample_submission[column].isna().all():
            try:
                if np.array_equal(formatted_submission[column], formatted_submission[column].astype(int)):
                    formatted_submission[column] = formatted_submission[column].round().astype(int)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
        else:
            try:
                if np.issubdtype(sample_submission[column].dtype, np.integer) and np.array_equal(formatted_submission[column], formatted_submission[column].astype(int)):
                    formatted_submission[column] = formatted_submission[column].round().astype(int)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
            try:
                if np.issubdtype(sample_submission[column].dtype, np.float32) and np.array_equal(formatted_submission[column], formatted_submission[column].astype(float)):
                    formatted_submission[column] = formatted_submission[column].astype(float)
            except (ValueError, RuntimeError) as e:
                print(e)
                pass
    return formatted_submission


def time_to_seconds(time_str: str) -> float:
    (days, hours, minutes, seconds) = map(int, time_str.split(':'))
    total_seconds = days * 3600 * 24 + hours * 3600 + minutes * 60 + seconds
    return total_seconds


def seconds_to_time_string(total_seconds: float) -> str:
    days = int(total_seconds // (3600 * 24))
    hours = int(total_seconds // 3600)
    minutes = int(total_seconds % 3600 // 60)
    seconds = int(total_seconds % 60)
    time_string = f'{days:02}:{hours:02}:{minutes:02}:{seconds:02}'
    return time_string


def graceful_exit(start_time: float, max_exec_time: float, time_file: str, run_out_time_file: str) -> None:
    time_taken = time.time() - start_time
    remaining_time = max_exec_time - time_taken
    with open(time_file, 'w') as f:
        f.write(str(remaining_time))
    if remaining_time < 0:
        print('Total execution crossed allocated time')
        with open(run_out_time_file, 'w') as f:
            f.write(str(remaining_time))


def get_max_time_limit(time_file: str, margin_time: int=900) -> tuple[float, str]:
    """
    Find maximum time allowed to execute this script
    :param margin_time: Margin time to accommodate test generation
    :param time_file: File containing allowed execution time
    :return: tuple containing maximum execution time and allowed training time
    """
    max_training_time = MAX_TIME
    max_exec_time = 2 * 24 * 3600
    if os.path.exists(time_file):
        try:
            with open(time_file, 'r') as f:
                time_str = f.read()
                if len(time_str) > 0:
                    max_exec_time = float(time_str) - margin_time
                    assert max_exec_time > 0, f'Exhausted allowed execution time {max_exec_time}'
        except Exception as e:
            raise e
    if time_to_seconds(MAX_TIME) > max_exec_time:
        max_training_time = seconds_to_time_string(max_exec_time)
    return (max_exec_time, max_training_time)


def main(accelerator: str, devices: str) -> None:
    if os.getenv('RUN_TTA_ONLY', False) in ['1', 'true', 'True'] or os.getenv('RUN_INFERENCE_ONLY', False) in ['1', 'true', 'True']:
        return generate_submissions()
    start_time: float = time.time()
    time_file = Path('./remaining_time.txt')
    run_out_time_file = time_file.parent / 'run_out_of_time.txt'
    count_free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if torch.cuda.utilization(device_id) == 0]
    accelerator = accelerator if len(count_free_gpus) else 'cpu'
    if os.path.exists(time_file):
        (max_exec_time, max_training_time) = get_max_time_limit(time_file=str(time_file))
        print(f'Maximum allowed execution time : {max_exec_time} seconds and training time {max_training_time}')
        atexit.register(partial(graceful_exit, start_time=start_time, max_exec_time=max_exec_time, time_file=time_file, run_out_time_file=run_out_time_file))
    else:
        max_exec_time = 2 * 24 * 3600
        max_training_time = MAX_TIME
    fast_model = SubmissionModel()
    fast_trainer = L.Trainer(accelerator=accelerator, devices=devices, fast_dev_run=True)
    fast_train_batch_size = TRAIN_BATCH_SIZE
    fast_test_batch_size = TEST_BATCH_SIZE
    (fast_train_dataloader, fast_validation_dataloader, _) = get_data_loaders(train_batch_size=fast_train_batch_size, val_batch_size=fast_test_batch_size)
    fast_train_val_loss = None
    _fast_trials = 0
    _max_fast_trials = 5
    print(f'BEFORE FAST TRAINING: max_exec_time: {max_exec_time}, max_training_time: {max_training_time}', flush=True)
    while fast_train_val_loss is None and _fast_trials < _max_fast_trials:
        try:
            print('STARTING FAST TRAINING', flush=True)
            _fast_trials += 1
            fast_trainer.fit(fast_model, fast_train_dataloader, fast_validation_dataloader)
            fast_train_val_loss = fast_trainer.callback_metrics['valid_loss'].item()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if 'out of memory' in str(e).lower():
                print('| WARNING: ran out of memory, retrying with half the batch size', flush=True)
                with torch.no_grad():
                    torch.cuda.empty_cache()
            elapsed_time = time.time() - start_time
            if elapsed_time > max_exec_time:
                print('Total time exceeded. Stopping trials.', flush=True)
                break
            fast_train_batch_size //= 2
            fast_test_batch_size //= 2
            if fast_train_batch_size <= 0:
                raise RuntimeError('No batch size seems to make it fit on the device')
            (fast_train_dataloader, fast_validation_dataloader, _) = get_data_loaders(fast_train_batch_size, fast_test_batch_size)
    elapsed_time = time.time() - start_time
    max_exec_time = max_exec_time - elapsed_time
    if time_to_seconds(max_training_time) > max_exec_time - elapsed_time:
        max_training_time = seconds_to_time_string(max_exec_time - elapsed_time)
    print(f'AFTER FAST TRAINING - max_exec_time: {max_exec_time}, max_training_time: {max_training_time}, elapse_time: {elapsed_time}', flush=True)
    start_time = time.time()
    if N_TRIALS == 0:
        trials_dir = './trials'
        best_index = 0
        params = {'learning_rate': [0.0001], 'optimizer': ['adam']}
        val_loss = objective(params=params, trial_num=0, accelerator=accelerator, max_time=max_training_time)
    else:
        trials_dir = './trials'
        predefined_suggestions = [{'learning_rate': [0.0001], 'optimizer': ['adam']}, {'learning_rate': [0.0001], 'optimizer': ['sgd']}, {'learning_rate': [0.0001], 'optimizer': ['adamw']}]
        opt = HEBO(HEBO_SPACE)
        remove_trials(trials_dir)
        for trial in range(N_TRIALS):
            elapsed_time = time.time() - start_time
            if elapsed_time > max_exec_time:
                print('Total time exceeded. Stopping trials.')
                break
            if time_to_seconds(max_training_time) > max_exec_time - elapsed_time:
                max_training_time = seconds_to_time_string(max_exec_time - elapsed_time)
            if is_main_process():
                if len(predefined_suggestions) > 0:
                    params = predefined_suggestions.pop(0)
                    rec = pd.DataFrame.from_dict(params, orient='columns')
                else:
                    rec = opt.suggest(n_suggestions=1)
                    params = rec.to_dict(orient='list')
            try:
                val_loss = objective(params=params, trial_num=trial, accelerator=accelerator, max_time=max_training_time, devices=devices)
                if dist.is_initialized():
                    raise NotImplementedError(f'HEBO with DDP training not supported yet - TODO: synchronize HEBO kernel params across ranks')
                    val_loss = val_loss.reshape((1,)).to(f'cuda:{dist.get_rank()}')
                    val_losses = [torch.tensor([0.0], device=f'cuda:{dist.get_rank()}') for _ in range(dist.get_world_size())]
                    dist.all_gather(val_losses, val_loss)
                    val_loss_gathered = torch.cat(val_losses).mean().cpu().detach()
                    opt.observe(rec, np.array([[val_loss_gathered.item()]]))
                else:
                    opt.observe(rec, np.array([[val_loss.item()]]))
            except NanLossError as e:
                print(f'Trial {trial} failed: {e}, stopping the trials.')
                break
            except Exception as e:
                print(f'Trial {trial} failed: {e}')
            opt.X.to_csv('./hyperopt_x.csv')
            pd.DataFrame(opt.y).to_csv('./hyperopt_y.csv')
        if opt.X.shape[0] >= 1:
            best_params = opt.X.iloc[opt.y.argmin()].to_dict()
            best_index = opt.y.argmin()
            print(f'Best hyper parameters: {best_params}')
            print(f'Best index: {best_index}')
    if is_main_process():
        get_best_checkpoint(best_index=best_index, trials_dir=trials_dir)
        generate_submissions()
        remove_trials(trial_directory)


def generate_submissions() -> None:
    model = SubmissionModel.load_from_checkpoint(checkpoint_path="/kaggle/input/solution_model/pytorch/default/1/best_model.ckpt")
    model.eval()
    free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if torch.cuda.utilization(device_id) == 0]
    print(f'Free GPUs: {free_gpus}')
    model.to(free_gpus[0] if len(free_gpus) else 'cpu')
    run_tta_only = os.getenv('RUN_TTA_ONLY', False) in ['1', 'true', 'True']
    current_dir = Path().resolve()
    if not run_tta_only:
        (val_scores, validation_submission) = model.get_score(validation_dataloader)
        save_path = str(current_dir / 'val_scores.json')
        with open(save_path, 'w') as writer:
            writer.write(json.dumps(val_scores))
    print(f'[START] Generate test predictions')
    use_tta = os.getenv('TTA', False) in ['1', 'true', 'True'] or run_tta_only
    (test_submission, _, no_tta_raw_preds) = model.get_submissions(dataloader=test_dataloader, get_raw_preds=use_tta)
    n_submissions = 0
    for (submission_format_func, submission_name) in zip(submission_format_functions, submission_names):
        try:
            formatted_sub = submission_format_func(test_submission)
            formatted_sub = _format_submission_dtypes(formatted_sub)
            formatted_sub.to_csv(os.path.join(str(current_dir), submission_name), index=False)
            n_submissions += 1
        except Exception as e:
            print(f'Hit exception when trying to create {submission_name}: {e}')
    if use_tta:
        print(f'[START] Generate test predictions using TTA')
        tta_dir = current_dir / 'tta'
        os.makedirs(tta_dir, exist_ok=True)
        predictions = {'test_transform': no_tta_raw_preds}
        from solve_params import TTA_ROUNDS
        test_dataset.img_input_transform = CustomTrainImageInputTransform
        for tta_round in tqdm(range(TTA_ROUNDS), desc='Generating TTA predictions'):
            (test_submission, _, raw_preds) = model.get_submissions(dataloader=test_dataloader, get_raw_preds=True)
            for (submission_format_func, submission_name) in zip(submission_format_functions, submission_names):
                try:
                    formatted_sub = submission_format_func(test_submission)
                    formatted_sub = _format_submission_dtypes(formatted_sub)
                    submission_name = f'tta_round_{tta_round}-' + submission_name
                    predictions[submission_name] = raw_preds.loc[no_tta_raw_preds.index]
                    formatted_sub.to_csv(tta_dir / submission_name, index=False)
                except Exception as e:
                    print(f'Hit exception when trying to create {submission_name}: {e}')
        raw_preds_mean = np.stack(list(predictions.values())).mean(0)
        tta_submission_transform = tab_target_inverse_transform(raw_preds_mean, no_tta_raw_preds.index.values)
        if test_dataset.custom_tab_regression_scaler is not None:
            tta_submission_transform[test_dataset.tab_regression_target_cols] = test_dataset.custom_tab_regression_scaler.inverse_transform(tta_submission_transform[test_dataset.tab_regression_target_cols])
        for (submission_format_func, submission_name) in zip(submission_format_functions, submission_names):
            try:
                formatted_sub = submission_format_func(tta_submission_transform)
                formatted_sub = _format_submission_dtypes(formatted_sub)
                submission_name = f'tta-' + submission_name
                formatted_sub.to_csv(current_dir / submission_name, index=False)
            except Exception as e:
                print(f'Hit exception when trying to create {submission_name}: {e}')
    if n_submissions > 0:
        print(f'[END] Generate test predictions in {current_dir}')
    else:
        print(f'[END] Failed to create a submission in {current_dir}')



accelerator = 'gpu'
devices = 'auto'
os.environ['RUN_INFERENCE_ONLY'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "1"
if os.getenv('AGENT_DEBUG', False) in ['True', 'true', '1']:
    MAX_EPOCHS = 1
    N_TRIALS = 0
    MAX_TIME = '00:00:01:00'
    (train_dataloader, validation_dataloader, test_dataloader) = get_data_loaders(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)
elif str(os.getenv('AGENT_NO_BO', False)) in ['True', 'true', '1']:
    N_TRIALS = 0
    MAX_EPOCHS = 30
    MAX_TIME = '00:10:00:00'
else:
    MAX_EPOCHS = 30
    MAX_TIME = '00:10:00:00'
main(accelerator=accelerator, devices=devices)











