import numpy as np
import pandas as pd

import pandas.api.types

import sklearn.metrics


import glob
import cv2
import matplotlib.pyplot as plt

import re


import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)



# Competition Metric
class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, group_column_name: str = "group") -> float:
    '''
    Mean of the Average Precision AP. AP is calculated for each grouped by wrapping
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    and then the mean of the APs of all grouped is computed.

    AP summarizes a precision-recall curve as the weighted mean of precisions
    achieved at each threshold, with the increase in recall from the previous
    threshold used as the weight:

    .. math::
    \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    where :math:`P_n` and :math:`R_n` are the precision and recall at the nth
    threshold [1]_. This implementation is not interpolated and is different
    from computing the area under the precision-recall curve with the
    trapezoidal rule, which uses linear interpolation and can be too
    optimistic.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    solution : ndarray of shape (n_samples,) or (n_samples, n_classes)
    True binary labels or binary label indicators.

    submission : ndarray of shape (n_samples,) or (n_samples, n_classes)
    Target scores, can either be probability estimates of the positive
    class, confidence values, or non-thresholded measure of decisions
    (as returned by :term:`decision_function` on some classifiers).


    Examples
    --------

    >>> import pandas as pd
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 0, 0] + [1,0,0,1] + [1,0,1,1])
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true["id"] = range(len(y_true))
    >>> y_true["group"] = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    >>> y_pred = np.array([0.1, 0.4, 0.35, 0.8] * 3)
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred["id"] = range(len(y_pred))
    >>> score(y_true.copy(), y_pred.copy(), "id", "group")
    0.6018518518518519
    '''

    # Skip sorting and equality checks for the row_id_column since that should already be handled
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not group_column_name in solution.columns:
        raise ParticipantVisibleError('Missing group column in solution')

    group = solution[group_column_name]
    del solution[group_column_name]
    groups = group.unique()

    if not((len(submission.columns) == 1) or (len(submission.columns) == len(solution.columns))):
        raise ParticipantVisibleError(f'Invalid number of submission columns. Found {len(submission.columns)}')

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {x: submission[x].dtype  for x in submission.columns if not pandas.api.types.is_numeric_dtype(submission[x])}
        raise ParticipantVisibleError(f'Invalid submission data types found: {bad_dtypes}')

    if submission.max().max() > 1 or submission.min().min() < 0:
        raise ParticipantVisibleError('Submitted values were not valid probabilities')

    solution = solution.values
    submission = submission.values

    score_result = np.mean([
        sklearn.metrics.average_precision_score(solution[group == g], submission[group == g])
        for g in groups
    ])

    return score_result


train = pd.read_csv('/kaggle/input/nexar-collision-prediction/train.csv')
test = pd.read_csv('/kaggle/input/nexar-collision-prediction/test.csv')

ss = pd.read_csv('/kaggle/input/nexar-collision-prediction/sample_submission.csv')


train = train.sort_values(by='id')
test = test.sort_values(by='id')
ss = ss.sort_values(by='id')


train.head()


test.head()


ss.head()


# Read the image locations
train_filenames = glob.glob('/kaggle/input/nexar-collision-prediction/train/*.mp4')
test_filenames = glob.glob('/kaggle/input/nexar-collision-prediction/test/*.mp4')

# Sort by id
train_filenames = sorted(train_filenames)
test_filenames = sorted(test_filenames)


# video_path = '/kaggle/input/nexar-collision-prediction/train/00000.mp4'

# def get_id(video_path):
#     match = re.search(r'(\d+)\.mp4$', video_path)
#     return match.group(1) if match else None

# get_id(video_path)


WIDTH = 1280
HEIGHT = 720


def get_metadata(video_paths):
    fps_values = []
    frame_counts = []
    total_durations = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Error: Cannot open video file.')
            exit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps if fps > 0 else 0
        
        fps_values.append(fps)
        frame_counts.append(frame_count)
        total_durations.append(total_duration)
        cap.release()

    results = {
        'fps': fps_values,
        'frame_count': frame_counts,
        'total_duration': total_durations
        }
    
    return results


train_metadata = get_metadata(train_filenames)
for key in train_metadata:
    train[key] = train_metadata[key]

test_metadata = get_metadata(test_filenames)
for key in test_metadata:
    test[key] = test_metadata[key]
    


# Load video
video_path = train_filenames[0]  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print('Error: Cannot open video file.')
    exit()

frame_count = 0

# Read and process frames
while frame_count < 5:  
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display frame using matplotlib
    plt.imshow(gray_frame, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show()
    
    frame_count += 1

cap.release()



train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

