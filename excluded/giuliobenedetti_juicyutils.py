import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def merge_mdata(qmeta, cmeta):

    meta = qmeta.merge(
        cmeta,
        on="participant_id",
        how="left"
    )

    meta = meta.set_index("participant_id").sort_index()
    return meta


def get_sample_weights(labels):

    # Convert to numpy
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()
        
    # Find indices for ADHD females
    adhd_females = (labels[:, 0] == 1) & (labels[:, 1] == 1)
    # Assign weights 2 to ADHD females and 1 otherwise
    sample_weights = np.where(adhd_females, 2, 1)

    return sample_weights


def juicy_f1(y_true, y_pred, label=None, average="micro"):
    """
    Calculate the weighted F1 score with specific emphasis on ADHD females.

    Parameters:
    y_true (array-like or pd.DataFrame): 
        True labels for the samples, can be a 2D array or a pandas DataFrame. 
        The first column should represent the ADHD classification, 
        and the second column should represent the sex classification.
    
    y_pred (array-like): 
        Predicted labels for the samples. This should be a 1D or 2D array 
        that matches the shape of y_true by number of rows (samples).
    
    label (str, optional): 
        Specifies which class to evaluate. If "adhd", only the ADHD classification 
        (first column of y_true) will be considered; if "sex", only the sex classification 
        (second column of y_true) will be evaluated. 
        Defaults to None, which evaluates both classes.
    
    average (str, optional): 
        The type of averaging to be performed on the data. 
        Options include "micro", "macro", "weighted", or "samples". 
        Defaults to "micro", which calculates metrics globally by counting the total 
        true positives, false negatives, and false positives.

    Returns:
    float: 
        The weighted F1 score calculated with sample weights, giving extra importance 
        to ADHD females (labelled as 1 in the both columns of y_true).
    """

    # Convert to numpy
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    # Check dimensions of y_true
    if y_true.ndim == 1:
        raise ValueError("y_true must be a 2D array-like structure.")
    
    # Check dimensions of y_pred
    if y_pred.ndim == 1 and label is None:
        raise ValueError("y_pred is 1D, please specify a label ('adhd' or 'sex').")

    # Compute sample weights
    sample_weights = get_sample_weights(y_true)
    
    if label == "adhd":
        y_true = y_true[:, 0]
    if label == "sex":
        y_true = y_true[:, 1]

    # Compute weighted F1 score
    f1 = f1_score(y_true, y_pred, average=average, sample_weight=sample_weights)

    return f1


def optimise_threshold(y_true, y_prob, label):
    thresholds = np.arange(0.0, 1.0, 0.01)
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = juicy_f1(y_true, y_pred, label=label)
        f1_scores.append(f1)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold, max(f1_scores)


def lowvarfilter(data, keep_percent = 1):
    '''Dataframe, float -> Dataframe, List
    
    Takes a dataframe and removes the features with the least variance,
    keeping up to a defined percentage of the original features. Returns
    the filtered data and the column names of the retained features.'''
    #how many features to keep
    feature_count = data.shape[1] 
    keep_count = int(round(keep_percent * feature_count))
    
    fmri_var = X.var() #find variance
    fmri_var = fmri_var.sort_values(ascending = False, na_position = 'last').reset_index() #sort by var
    filt_cols = fmri_var.iloc[:keep_count] #cut to desired size
    filt_cols = filt_cols['index'].to_list() #take the column names
    filt_data = data[filt_cols]
    return(filt_data, filt_cols)


def threshold_probs(y_prob, adhd_threshold, sex_threshold):

    y_prob = np.array(y_prob)[:, :, 1].T

    adhd_pred = (y_prob[:, 0] >= adhd_threshold).astype(int)
    sex_pred = (y_prob[:, 1] >= sex_threshold).astype(int)

    y_pred = np.vstack([adhd_pred, sex_pred]).T
    return y_pred

