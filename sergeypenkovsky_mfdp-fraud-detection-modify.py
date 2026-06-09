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


!pip install dotenv


import os
from dotenv import load_dotenv

#load_dotenv()

#aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
#aws_key_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
#s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL")

#os.environ['MLFLOW_TRACKING_URI']= os.getenv('MLFLOW_URL')

#os.environ["AWS_ACCESS_KEY_ID"] = aws_key_id
#os.environ["AWS_SECRET_ACCESS_KEY"] = aws_key_secret
#os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url



import requests

def get_jwt_token(url: str, client_id: str, client_secret: str, username: str, password: str) -> str | None:
    """
    ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµÑ‚ JWT Ñ‚Ğ¾ĞºĞµĞ½ Ñ� Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸ĞµĞ¼ grant_type=password.
    
    :param url: URL Ğ´Ğ»Ñ� Ğ¾Ñ‚Ğ¿Ñ€Ğ°Ğ²ĞºĞ¸ POST-Ğ·Ğ°Ğ¿Ñ€Ğ¾Ñ�Ğ°
    :param client_id: ID ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ° OAuth2
    :param client_secret: Ğ¡ĞµĞºÑ€ĞµÑ‚ ĞºĞ»Ğ¸ĞµĞ½Ñ‚Ğ° OAuth2
    :param username: Ğ˜Ğ¼Ñ� Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»Ñ�
    :param password: ĞŸĞ°Ñ€Ğ¾Ğ»ÑŒ Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ñ‚ĞµĞ»Ñ�
    :return: JWT Ñ‚Ğ¾ĞºĞµĞ½ (access_token) Ğ¸Ğ»Ğ¸ None Ğ² Ñ�Ğ»ÑƒÑ‡Ğ°Ğµ Ğ¾ÑˆĞ¸Ğ±ĞºĞ¸
    """
    data = {
        "grant_type": "password",
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password,
        "scope": "openid"
    }

    try:
        response = requests.post(url, data=data)
        response.raise_for_status()  # Ğ²Ñ‹Ğ±Ñ€Ğ¾Ñ�Ğ¸Ñ‚ Ğ¸Ñ�ĞºĞ»Ñ�Ñ‡ĞµĞ½Ğ¸Ğµ Ğ¿Ñ€Ğ¸ Ğ¾ÑˆĞ¸Ğ±ĞºĞµ HTTP

        token = response.json().get("access_token")
        if token:
            print(f"JWT Token: {token}")
            return token
        else:
            print("Token not found in response.")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request exception occurred: {req_err}")
    except ValueError as json_err:
        print(f"Error decoding JSON: {json_err}")

    return None


def create_token():
    client_id = os.getenv("OAUTH_CLIENT_ID")
    client_secret = os.getenv("OAUTH_CLIENT_SECRET")
    oauth_endpoint_url = os.getenv("OAUTH_URL")
    username = os.getenv("MLFLOW_USERNAME")
    password = os.getenv("MLFLOW_PASSWORD")

    return get_jwt_token(
        url=oauth_endpoint_url,
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password

    )



# Ğ‘Ğ°Ğ·Ğ¾Ğ²Ñ‹Ğµ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ğ§Ñ‚Ğ¾Ğ±Ñ‹ ĞºÑ€Ğ°Ñ�Ğ¸Ğ²Ğ¾ Ğ¾Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶Ğ°Ğ»Ğ¸Ñ�ÑŒ Ğ³Ñ€Ğ°Ñ„Ğ¸ĞºĞ¸
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
#plt.style.use("seaborn-dark")


N_TRIALS = 5
EPOCHS = 40


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    start_mem = df.memory_usage().sum() / 1024**2    
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # Ñ‚Ğ¸Ğ¿ float16, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğ¹ Ğ½Ğµ Ğ¿Ğ¾Ğ´Ğ´ĞµÑ€Ğ¶Ğ¸Ğ²Ğ°ĞµÑ‚Ñ�Ñ� scikit-learn Ğ¸ pandas Ğ² Ğ½ĞµĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ñ… Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ñ�Ñ… â€” Ğ¾Ñ�Ğ¾Ğ±ĞµĞ½Ğ½Ğ¾ Ğ¿Ñ€Ğ¸ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğ¸ Ğ¸ Ğ¸Ğ½Ğ´ĞµĞºÑ�Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğ¸.
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose: print('Mem. usage decreased to {:5.2f} => {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


import pandas as pd

def load_and_merge_data(path_transaction, path_identity, reduce_mem_usage_func):
    # Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
    transaction = pd.read_csv(path_transaction)
    identity = pd.read_csv(path_identity)
    print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ train_transaction: {transaction.shape}")
    print(f"Ğ Ğ°Ğ·Ğ¼ĞµÑ€ train_identity: {identity.shape}")

    # Ğ£Ğ¼ĞµĞ½ÑŒÑˆĞµĞ½Ğ¸Ğµ Ğ¿Ğ°Ğ¼Ñ�Ñ‚Ğ¸
    transaction = reduce_mem_usage_func(transaction)
    identity = reduce_mem_usage_func(identity)

    # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿Ğ¾ TransactionID
    train = transaction.merge(identity, how='left', on='TransactionID')
    print(f"Ğ˜Ñ‚Ğ¾Ğ³Ğ¾Ğ²Ñ‹Ğ¹ Ñ€Ğ°Ğ·Ğ¼ĞµÑ€ train: {train.shape}")

    return train


def check_column_consistency(train, test, target_column='isFraud'):
    """
    ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµÑ‚ Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´ĞµĞ½Ğ¸Ğµ Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğ¹ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ¾Ğ² Ğ¼ĞµĞ¶Ğ´Ñƒ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰ĞµĞ¹ Ğ¸ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ğ¾Ğ¹ Ğ²Ñ‹Ğ±Ğ¾Ñ€ĞºĞ°Ğ¼Ğ¸.

    Ğ�Ñ€Ğ³ÑƒĞ¼ĞµĞ½Ñ‚Ñ‹:
        train (pd.DataFrame): Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰Ğ¸Ğ¹ Ğ½Ğ°Ğ±Ğ¾Ñ€ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
        test (pd.DataFrame): Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğ¹ Ğ½Ğ°Ğ±Ğ¾Ñ€ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
        target_column (str): Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ³Ğ¾ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ° (Ğ¿Ğ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� 'isFraud')

    Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚:
        train_only_cols (set): Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ² train
        test_only_cols (set): Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ñ‹ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ² test
    """
    print('ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ñ�Ğ¾Ğ²Ğ¿Ğ°Ğ´ĞµĞ½Ğ¸Ñ� ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº Ğ¼ĞµĞ¶Ğ´Ñƒ train Ğ¸ test...')
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    train_only_cols = train_cols - test_cols
    test_only_cols = test_cols - train_cols

    if train_only_cols:
        print(f"Ğ’Ğ½Ğ¸Ğ¼Ğ°Ğ½Ğ¸Ğµ: Ğ² train ĞµÑ�Ñ‚ÑŒ {len(train_only_cols)} ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ñ… Ğ½ĞµÑ‚ Ğ² test:")
        print(sorted(train_only_cols))
    else:
        print("Ğ’ train Ğ½ĞµÑ‚ Ğ»Ğ¸ÑˆĞ½Ğ¸Ñ… ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº.")

    if test_only_cols:
        print(f"Ğ’Ğ½Ğ¸Ğ¼Ğ°Ğ½Ğ¸Ğµ: Ğ² test ĞµÑ�Ñ‚ÑŒ {len(test_only_cols)} ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº, ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ñ… Ğ½ĞµÑ‚ Ğ² train:")
        print(sorted(test_only_cols))
    else:
        print("Ğ’ test Ğ½ĞµÑ‚ Ğ»Ğ¸ÑˆĞ½Ğ¸Ñ… ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº.")

    common_cols = train_cols & test_cols
    common_ratio = len(common_cols) / max(len(train_cols), len(test_cols))
    print(f"Ğ”Ğ¾Ğ»Ñ� Ğ¿ĞµÑ€ĞµÑ�ĞµĞºĞ°Ñ�Ñ‰Ğ¸Ñ…Ñ�Ñ� ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº: {common_ratio:.2%}")

    if target_column in train_only_cols:
        print(f"ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ°: Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ñ�Ñ‚Ğ¾Ğ»Ğ±ĞµÑ† '{target_column}' ĞµÑ�Ñ‚ÑŒ Ñ‚Ğ¾Ğ»ÑŒĞºĞ¾ Ğ² train (Ğ¾Ğ¶Ğ¸Ğ´Ğ°ĞµĞ¼Ğ¾)")

    print("ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° ĞºĞ¾Ğ»Ğ¾Ğ½Ğ¾Ğº Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ°.")
    return train_only_cols, test_only_cols


def unify_column_names(df, from_char='-', to_char='_'):
    """
    Ğ—Ğ°Ğ¼ĞµĞ½Ñ�ĞµÑ‚ Ñ�Ğ¸Ğ¼Ğ²Ğ¾Ğ» from_char Ğ½Ğ° to_char Ğ²Ğ¾ Ğ²Ñ�ĞµÑ… Ğ½Ğ°Ğ·Ğ²Ğ°Ğ½Ğ¸Ñ�Ñ… Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ¾Ğ² Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼Ğ°.

    Ğ�Ñ€Ğ³ÑƒĞ¼ĞµĞ½Ñ‚Ñ‹:
        df (pd.DataFrame): Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼, Ğ² ĞºĞ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ¼ Ñ‚Ñ€ĞµĞ±ÑƒĞµÑ‚Ñ�Ñ� ÑƒĞ½Ğ¸Ñ„Ğ¸Ñ†Ğ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒ Ğ¸Ğ¼ĞµĞ½Ğ° Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ¾Ğ²
        from_char (str): Ñ�Ğ¸Ğ¼Ğ²Ğ¾Ğ», ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğ¹ Ğ½ÑƒĞ¶Ğ½Ğ¾ Ğ·Ğ°Ğ¼ĞµĞ½Ğ¸Ñ‚ÑŒ (Ğ¿Ğ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� '-')
        to_char (str): Ñ�Ğ¸Ğ¼Ğ²Ğ¾Ğ», Ğ½Ğ° ĞºĞ¾Ñ‚Ğ¾Ñ€Ñ‹Ğ¹ Ğ·Ğ°Ğ¼ĞµĞ½Ñ�ĞµĞ¼ (Ğ¿Ğ¾ ÑƒĞ¼Ğ¾Ğ»Ñ‡Ğ°Ğ½Ğ¸Ñ� '_')

    Ğ’Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚:
        pd.DataFrame: ĞºĞ¾Ğ¿Ğ¸Ñ� Ğ´Ğ°Ñ‚Ğ°Ñ„Ñ€ĞµĞ¹Ğ¼Ğ° Ñ� Ğ¾Ğ±Ğ½Ğ¾Ğ²Ğ»Ñ‘Ğ½Ğ½Ñ‹Ğ¼Ğ¸ Ğ¸Ğ¼ĞµĞ½Ğ°Ğ¼Ğ¸ Ñ�Ñ‚Ğ¾Ğ»Ğ±Ñ†Ğ¾Ğ²
    """
    df = df.copy()
    df.columns = [col.replace(from_char, to_char) for col in df.columns]
    return df


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion(y_true, y_pred, threshold=None, labels=None, title="Confusion Matrix"):
    """
    Ğ¡Ñ‚Ñ€Ğ¾Ğ¸Ñ‚ confusion matrix Ğ¸ ĞµÑ‘ heatmap-Ğ³Ñ€Ğ°Ñ„Ğ¸Ğº.
    y_true â€” Ğ¸Ñ�Ñ‚Ğ¸Ğ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸ (array-like)
    y_pred â€” Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚ĞºĞ¸ (array-like)
    labels â€” Ğ¿Ğ¾Ñ€Ñ�Ğ´Ğ¾Ğº ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ¾Ñ�ĞµĞ¹ (Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€, [0, 1])
    """
    if threshold is not None:
        # Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ğ¸Ñ‚ÑŒ Ğ¿Ğ¾Ñ€Ğ¾Ğ³ Ğº Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚Ñ�Ğ¼
        y_pred = (y_pred >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.show()


def plot_feature_correlation_matrix(X, y=None, target_name='target', figsize=(12,10), annot=False):
    """
    Ğ¡Ñ‚Ñ€Ğ¾Ğ¸Ñ‚ Ğ¸ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµÑ‚ ĞºĞ¾Ñ€Ñ€ĞµĞ»Ñ�Ñ†Ğ¸Ğ¾Ğ½Ğ½ÑƒÑ� Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñƒ, Ğ²ĞºĞ»Ñ�Ñ‡Ğ°Ñ� target, ĞµÑ�Ğ»Ğ¸ Ğ·Ğ°Ğ´Ğ°Ğ½.
    X â€” DataFrame Ñ� Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°Ğ¼Ğ¸.
    y â€” Series/array Ñ� Ñ†ĞµĞ»ĞµĞ²Ğ¾Ğ¹ Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ¾Ğ¹ (ĞµÑ�Ğ»Ğ¸ Ğ½Ğ°Ğ´Ğ¾ Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ¸Ñ‚ÑŒ Ğ² Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñƒ).
    """
    if y is not None:
        X = X.copy()
        X[target_name] = y
    corr = X.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Feature Correlation Matrix" + (f" (+{target_name})" if y is not None else ''))
    plt.show()
    return corr



def analyse_folds(X, y, cv):
    # ĞŸÑ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ�
    fold_idx = 1
    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]


        print(f"Fold {fold_idx}")
        print("Train class distribution:", Counter(y_train_fold))
        print("Test class distribution:", Counter(y_test_fold))

        # Ğ’Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ�
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(y_train_fold, kde=False, bins=3).set_title(f'Fold {fold_idx} - Train')
        plt.subplot(1, 2, 2)
        sns.histplot(y_test_fold, kde=False, bins=3).set_title(f'Fold {fold_idx} - Test')
        plt.show()

        fold_idx += 1


# Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ±Ğ¸Ğ±Ğ»Ğ¸Ğ¾Ñ‚ĞµĞºĞ¸
import pandas as pd
import numpy as np

# ĞŸÑƒÑ‚ÑŒ Ğº Ğ´Ğ°Ğ½Ğ½Ñ‹Ğ¼ /kaggle/input/ieee-fraud-detection
BASE = '/kaggle/input/ieee-fraud-detection'
#BASE = '../data/raw/ieee-fraud-detection'


PATH_TRANSACTION_TRAIN = f'{BASE}/train_transaction.csv'
PATH_IDENTITY_TRAIN = f'{BASE}/train_identity.csv'

train = load_and_merge_data(PATH_TRANSACTION_TRAIN, PATH_IDENTITY_TRAIN, reduce_mem_usage)
train.head()


PATH_TRANSACTION_TEST = f'{BASE}/test_transaction.csv'
PATH_IDENTITY_TEST = f'{BASE}/test_identity.csv'

test = load_and_merge_data(PATH_TRANSACTION_TEST, PATH_IDENTITY_TEST, reduce_mem_usage)
test.head()


check_column_consistency(train, test)


train = unify_column_names(train)
test = unify_column_names(test)


check_column_consistency(train, test)


#train.to_csv('../data/raw/ieee-fraud-detection/test_transactions.csv', index=False)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = train.copy()


df.shape


df.dtypes


df.head()


missing_col = df.isnull().mean().sort_values(ascending=False)
missing_col.head(10)


summary = (df['isFraud']
   .value_counts()
   .to_frame('count')
)
summary['share(%)'] = 100 * summary['count'] / summary['count'].sum()
print(summary)


cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique; most common: {df[col].value_counts(dropna=False).head()}\n")



num_cols = df.select_dtypes(include='number').columns
df[num_cols].describe().T



ts = df["TransactionDT"]
plt.hist(ts, bins=100)
plt.title("TransactionDT distribution")
plt.show()


v_cols = [col for col in df.columns if col.startswith('V')]
corr = df[v_cols].corr().abs()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlations between V features")
plt.show()


#from sklearn.model_selection import TimeSeriesSplit


#tscv = TimeSeriesSplit(n_splits=5)
#analyse_folds(X_train, y_train, tscv)


# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ñ�Ğ¸Ğ»ÑŒĞ½Ğ¾ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ‰ĞµĞ½Ğ½Ñ‹Ğµ ĞºĞ¾Ğ»Ğ¾Ğ½ĞºĞ¸:
drop_col_threshold = 0.3
to_drop = df.columns[df.isnull().mean() > drop_col_threshold].tolist()
df = df.drop(columns=to_drop)

# Ğ£Ğ´Ğ°Ğ»Ğ¸Ğ¼ Ñ�Ñ‚Ñ€Ğ¾ĞºĞ¸ Ñ� Ğ¼Ğ½Ğ¾Ğ¶ĞµÑ�Ñ‚Ğ²ĞµĞ½Ğ½Ñ‹Ğ¼Ğ¸ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ°Ğ¼Ğ¸:
drop_row_threshold = 0.5
row_mask = df.isnull().mean(axis=1) <= drop_row_threshold
df = df.loc[row_mask]


num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    median = df[col].median()
    df[col] = df[col].fillna(median)

df[num_cols].head()


from sklearn.preprocessing import LabelEncoder

cat_cols = df.select_dtypes(include=['object', 'category']).columns
# Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ¸, ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ
for col in cat_cols:
    df[col] = df[col].astype(str).fillna('unknown')
    # Count encoding (Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğ° Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ�)
    df[f"{col}_count"] = df[col].map(df[col].value_counts())
    # Label encoding
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


df[cat_cols].head()


min_dt = df['TransactionDT'].min()
rel_trx = df['TransactionDT'] - min_dt
df['Relative_TransactionDT'] = rel_trx
df['Transaction_day'] = (rel_trx // (24 * 60 * 60)).astype(int)
df['Transaction_hour'] = ((rel_trx // 3600) % 24).astype(int)
df['Transaction_weekday'] = ((rel_trx // (3600*24)) % 7).astype(int)

df[['TransactionDT', 'Relative_TransactionDT', 'Transaction_day', 'Transaction_hour', 'Transaction_weekday']].head()



amt = df['TransactionAmt']
q1 = amt.quantile(0.25)
q3 = amt.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
df['isOutlier'] = ((amt < lower) | (amt > upper)).astype(int)
df['TransactionAmt_binned'] = pd.cut(
    amt, bins=[0, 100, 1000, 5000, 10000, np.inf],
    labels=['Low', 'Medium', 'High', 'Very High', 'Extremely High']
).astype(str).fillna('unknown')
le_amt = LabelEncoder()
df['TransactionAmt_binned'] = le_amt.fit_transform(df['TransactionAmt_binned'])


df[['TransactionAmt', 'log_TransactionAmt', 'isOutlier', 'TransactionAmt_binned']].head()


for col in ['card1', 'card4']:
    if col in df.columns and 'TransactionAmt' in df.columns:
        group_mean = df.groupby(col)['TransactionAmt'].transform('mean')
        group_std = df.groupby(col)['TransactionAmt'].transform('std').replace(0, 1)
        df[f'TransactionAmt_to_mean_{col}'] = df['TransactionAmt'] / group_mean
        df[f'TransactionAmt_to_std_{col}'] = df['TransactionAmt'] / group_std

group_cols = []
for col in ['card1', 'card4']:
    group_cols.append(f'TransactionAmt_to_mean_{col}')
    group_cols.append(f'TransactionAmt_to_std_{col}')

df[group_cols].head()


from sklearn.preprocessing import StandardScaler

scale_cols = [col for col in df.select_dtypes(include='number').columns if col not in ['isFraud', 'TransactionID', 'TransactionDT']]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

df[scale_cols].head()


from sklearn.decomposition import PCA

v_cols = [col for col in df.columns if col.startswith('V')]
if v_cols:
    X_v = df[v_cols].fillna(-999)
    v_scaler = StandardScaler()
    X_v_scaled = v_scaler.fit_transform(X_v)
    pca = PCA(n_components=0.90, random_state=42)
    X_v_pca = pca.fit_transform(X_v_scaled)
    for i in range(X_v_pca.shape[1]):
        df[f'V_PCA_{i}'] = X_v_pca[:, i]
    df = df.drop(columns=v_cols)


v_cols = [col for col in df.columns if col.startswith('V')]
df[v_cols].head()


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

class FraudDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Ğ£Ğ»ÑƒÑ‡ÑˆĞµĞ½Ğ½Ñ‹Ğ¹ Ğ¿Ğ°Ğ¹Ğ¿Ğ»Ğ°Ğ¹Ğ½ Ğ´Ğ»Ñ� anti-fraud Ğ´Ğ°Ñ‚Ğ°Ñ�ĞµÑ‚Ğ¾Ğ² Ñ� Ğ²Ğ¾Ğ·Ğ¼Ğ¾Ğ¶Ğ½Ğ¾Ñ�Ñ‚ÑŒÑ� Ñ�Ğ²Ğ½Ğ¾Ğ³Ğ¾ ÑƒĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ².
    """
    def __init__(
        self,
        drop_threshold_col=0.3,
        drop_threshold_row=0.5,
        skip_cols=['isFraud', 'TransactionID', 'TransactionDT'],
        categorical_features=None    # <-- Ğ½Ğ¾Ğ²Ñ‹Ğ¹ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€
    ):
        self.drop_threshold_col = drop_threshold_col
        self.drop_threshold_row = drop_threshold_row
        self.skip_cols = skip_cols if skip_cols is not None else []
        self.cols_to_drop_ = []
        self.label_encoders = {}
        self.numeric_medians = {}
        self.cat_cols_ = []
        self.time_features = ['Transaction_day', 'Transaction_hour', 'Transaction_weekday']
        self.min_transactiondt = None
        self.rare_label = 'RARE_CAT'
        self.rare_thresh = 10
        self.rare_labels_map = {}
        self.cat_count_maps = {}
        self.group_stats = {}
        self.lower_bound = None
        self.upper_bound = None
        self.base_num_cols_ = set()
        self.extra_num_cols_ = set()
        self.scaler = StandardScaler()
        self.v_cols_ = []
        self.v_pca = None
        self.v_pca_n_components_ = 0
        self.v_pca_scaler = None
        self.categorical_features = categorical_features

    def fit(self, X, y=None):
        df = X.copy()
        # ĞŸÑ€Ğ¸Ğ²ĞµĞ´ĞµĞ½Ğ¸Ğµ Ñ‚Ğ¸Ğ¿Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
        if self.categorical_features is not None:
            for col in self.categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        if 'isFraud' in df.columns:
            df = df.drop(columns=['isFraud'])

        self._drop_columns_by_missing_ratio(df)
        df = df.drop(columns=self.cols_to_drop_)
        df = self._drop_rows_by_missing_ratio(df)

        self._fit_categorical(df)
        self._fit_numerical(df)
        self._fit_time_features(df)
        self._fit_sum_features(df)

        if 'TransactionAmt' in df.columns:
            self.extra_num_cols_.add('log_TransactionAmt')
            self.extra_num_cols_.add('isOutlier')
        if 'TransactionAmt_binned' in df.columns:
            self.extra_num_cols_.add('TransactionAmt_binned')
        for col in ['card1', 'card4']:
            if col in df.columns and 'TransactionAmt' in df.columns:
                self.extra_num_cols_.add(f'TransactionAmt_to_mean_{col}')
                self.extra_num_cols_.add(f'TransactionAmt_to_std_{col}')

        self._fit_transaction_group_features(df)
        self._fit_v_pca_features(df)
        self._fit_v_pca_scaler(df)

        df_cat_trans = self._transform_categorical(df.copy())
        identity_nums = [col for col in df_cat_trans.columns if col.endswith('_count')]
        if identity_nums is not None:
            for col in identity_nums:
                self.extra_num_cols_.add(col)
        df_trans = df_cat_trans[identity_nums].copy()
        print(f'## df columns: {df_cat_trans.columns}')
        print(f'## identity_nums: {identity_nums}')
        print(f'## extra_num_cols_: {self.extra_num_cols_}')

        df_trans = self._transform_sum_features(df.copy())
        df_trans = self._transform_transaction_group_features(df_trans)
        self.full_num_cols_ = [col for col in list(self.base_num_cols_ | self.extra_num_cols_)
                               if col not in self.skip_cols and col in df_trans.columns]

        df_trans = self._fillna_numeric(df_trans, self.full_num_cols_)

        self.scaler.fit(df_trans[self.full_num_cols_])

        print(self.full_num_cols_)
        return self

    def transform(self, X):
        df = X.copy()
        # ĞŸÑ€Ğ¸Ğ²ĞµĞ´ĞµĞ½Ğ¸Ğµ Ñ‚Ğ¸Ğ¿Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ·Ğ°Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
        if self.categorical_features is not None:
            for col in self.categorical_features:
                if col in df.columns:
                    df[col] = df[col].astype('category')

        y = None
        if 'isFraud' in df.columns:
            y = df['isFraud']
            df = df.drop(columns=['isFraud'])
        df = df.drop(columns=[col for col in self.cols_to_drop_ if col in df.columns], errors='ignore')
        df = self._drop_rows_by_missing_ratio(df)
        df = self._transform_categorical(df)
        df = self._transform_numerical(df)
        df = self._transform_time_features(df)
        df = self._transform_sum_features(df)
        df = self._transform_transaction_group_features(df)

        applied_num_cols = [col for col in list(self.full_num_cols_) if col in df.columns]
        print(f"### applied_num_cols: {applied_num_cols}")
        df = self._fillna_numeric(df, applied_num_cols)

        if np.isinf(df[applied_num_cols].to_numpy()).any():
            raise RuntimeError("Ğ�Ñ�Ñ‚Ğ°Ğ»Ğ¸Ñ�ÑŒ inf Ğ² Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°Ñ…!")
        if np.isnan(df[applied_num_cols].to_numpy()).any():
            raise RuntimeError("Ğ�Ñ�Ñ‚Ğ°Ğ»Ğ¸Ñ�ÑŒ nan Ğ² Ñ‡Ğ¸Ñ�Ğ»Ğ¾Ğ²Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ°Ñ…!")

        df[applied_num_cols] = self.scaler.transform(df[applied_num_cols])

        df = self._transform_v_pca_features(df)
        if hasattr(self, 'v_pca_scaler') and self.v_pca_scaler is not None:
            v_pca_cols = [col for col in df.columns if col.startswith('V_PCA_')]
            if v_pca_cols:
                df[v_pca_cols] = pd.DataFrame(
                    self.v_pca_scaler.transform(df[v_pca_cols]),
                    columns=v_pca_cols,
                    index=df.index
                )

        if y is not None:
            df['isFraud'] = y.loc[df.index]

        print("Inf in columns:", df[list(self.extra_num_cols_)].isin([np.inf, -np.inf]).any())
        print("NaN in columns:", df[list(self.extra_num_cols_)].isna().any())
        return df

    def _fillna_numeric(self, df, num_cols):
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(0)
        return df

    def _drop_columns_by_missing_ratio(self, df):
        process_cols = [col for col in df.columns if col not in self.skip_cols]
        missing_ratio_col = df[process_cols].isnull().mean()
        self.cols_to_drop_ = missing_ratio_col[missing_ratio_col > self.drop_threshold_col].index.tolist()

    def _drop_rows_by_missing_ratio(self, df):
        process_cols = [col for col in df.columns if col not in self.skip_cols]
        missing_ratio_row = df[process_cols].isnull().mean(axis=1)
        return df.loc[missing_ratio_row <= self.drop_threshold_row]

    def _fit_categorical(self, df):
        process_cols = [col for col in df.columns if col not in self.skip_cols]
        self.cat_cols_ = df[process_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        self.cat_count_maps = {}
        self.rare_labels_map = {}
        df[self.cat_cols_] = df[self.cat_cols_].astype(str)
        for col in self.cat_cols_:
            col_values = df[col].fillna('unknown')
            self.extra_num_cols_.add(f"{col}_count")

            value_counts = col_values.value_counts()
            rare_labels = value_counts[value_counts <= self.rare_thresh].index
            self.rare_labels_map[col] = set(rare_labels)
            col_values = col_values.replace(rare_labels, self.rare_label)
            if self.rare_label not in col_values.unique():
                col_values = pd.concat([col_values, pd.Series([self.rare_label])], ignore_index=True)
            le = LabelEncoder()
            le.fit(col_values)
            self.label_encoders[col] = le
            self.cat_count_maps[col] = col_values.value_counts(dropna=False)

    def _transform_categorical(self, df):
        for col in self.cat_cols_:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('unknown')
                df[col] = df[col].apply(lambda x: self.rare_label if x in self.rare_labels_map[col] else x)
                df[f"{col}_count"] = df[col].map(self.cat_count_maps[col]).fillna(0)
                le = self.label_encoders[col]
                df[col] = df[col].where(df[col].isin(le.classes_), self.rare_label)
                df[col] = le.transform(df[col])
        return df

    def _fit_numerical(self, df):
        process_cols = [col for col in df.columns if col not in self.skip_cols]
        num_cols = df[process_cols].select_dtypes(include='number').columns.tolist()
        for col in num_cols:
            median = df[col].median()
            self.numeric_medians[col] = median
            df[col] = df[col].fillna(median)
        self.base_num_cols_ = set(num_cols)

    def _transform_numerical(self, df):
        for col in self.base_num_cols_:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(self.numeric_medians[col])
        return df

    def _fit_time_features(self, df):
        if 'TransactionDT' in df.columns:
            self.min_transactiondt = df['TransactionDT'].min()
            rel_trx = df['TransactionDT'] - self.min_transactiondt
            df['Relative_TransactionDT'] = rel_trx
            df['Transaction_day'] = (rel_trx // (24 * 60 * 60)).astype(int)
            df['Transaction_hour'] = ((rel_trx // 3600) % 24).astype(int)
            df['Transaction_weekday'] = ((rel_trx // (3600*24)) % 7).astype(int)
            for f in self.time_features:
                if f in df.columns:
                    self.base_num_cols_.add(f)

    def _transform_time_features(self, df):
        if 'TransactionDT' in df.columns and self.min_transactiondt is not None:
            rel_trx = df['TransactionDT'] - self.min_transactiondt
            df['Relative_TransactionDT'] = rel_trx
            df['Transaction_day'] = (rel_trx // (24 * 60 * 60)).astype(int)
            df['Transaction_hour'] = ((rel_trx // 3600) % 24).astype(int)
            df['Transaction_weekday'] = ((rel_trx // (3600*24)) % 7).astype(int)
        return df

    def _fit_sum_features(self, df):
        if 'TransactionAmt' in df.columns:
            q1 = df['TransactionAmt'].quantile(0.25)
            q3 = df['TransactionAmt'].quantile(0.75)
            iqr = q3 - q1
            self.lower_bound = q1 - 1.5 * iqr
            self.upper_bound = q3 + 1.5 * iqr
            df['TransactionAmt_binned'] = pd.cut(
                df['TransactionAmt'],
                bins=[0, 100, 1000, 5000, 10000, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High', 'Extremely High']
            ).astype(str).fillna('unknown')
            le = LabelEncoder()
            le.fit(df['TransactionAmt_binned'])
            self.label_encoders['TransactionAmt_binned'] = le

    def _transform_sum_features(self, df):
        if 'TransactionAmt' in df.columns and self.lower_bound is not None:
            df['log_TransactionAmt'] = np.log1p(df['TransactionAmt'])
            df['isOutlier'] = ((df['TransactionAmt'] < self.lower_bound) | (df['TransactionAmt'] > self.upper_bound)).astype(int)
            df['TransactionAmt_binned'] = pd.cut(
                df['TransactionAmt'],
                bins=[0, 100, 1000, 5000, 10000, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High', 'Extremely High']
            ).astype(str).fillna('unknown')
            le = self.label_encoders.get('TransactionAmt_binned')
            if le is not None:
                df['TransactionAmt_binned'] = df['TransactionAmt_binned'].where(
                    df['TransactionAmt_binned'].isin(le.classes_), 'unknown')
                df['TransactionAmt_binned'] = le.transform(df['TransactionAmt_binned'])
        return df

    def _fit_transaction_group_features(self, df):
        self.group_stats = {}
        for col in ['card1', 'card4']:
            if col in df.columns and 'TransactionAmt' in df.columns:
                g = df.groupby(col)['TransactionAmt']
                self.group_stats[f'{col}_mean'] = g.mean()
                self.group_stats[f'{col}_std'] = g.std()

    def _transform_transaction_group_features(self, df):
        for col in ['card1', 'card4']:
            if col in df.columns and 'TransactionAmt' in df.columns:
                df[f'TransactionAmt_to_mean_{col}'] = df['TransactionAmt'] / df[col].map(self.group_stats.get(f'{col}_mean')).replace([np.inf, -np.inf], 0)
                df[f'TransactionAmt_to_std_{col}'] = df['TransactionAmt'] / df[col].map(self.group_stats.get(f'{col}_std')).replace([np.inf, -np.inf], 0)
                df[[f'TransactionAmt_to_mean_{col}', f'TransactionAmt_to_std_{col}']] = df[[f'TransactionAmt_to_mean_{col}', f'TransactionAmt_to_std_{col}']].fillna(0)
        return df

    def _fit_v_pca_features(self, df):
        self.v_cols_ = [col for col in df.columns if col.startswith('V')]
        common_cols = self.v_cols_
        if len(common_cols) > 0:
            X_v = df[common_cols].fillna(-999)
            v_scaler = StandardScaler()
            X_v_scaled = v_scaler.fit_transform(X_v)
            pca = PCA(n_components=0.90, random_state=42)
            X_v_pca = pca.fit_transform(X_v_scaled)
            self.v_pca = (pca, v_scaler)
            self.v_pca_n_components_ = X_v_pca.shape[1]

    def _transform_v_pca_features(self, df):
        if hasattr(self, "v_pca") and self.v_pca is not None and self.v_cols_:
            common_cols = [col for col in self.v_cols_ if col in df.columns]
            if len(common_cols) == 0:
                return df
            v_scaler = self.v_pca[1]
            pca = self.v_pca[0]
            X_v = df[common_cols].fillna(-999)
            X_v_scaled = v_scaler.transform(X_v)
            X_v_pca = pca.transform(X_v_scaled)
            for i in range(X_v_pca.shape[1]):
                df[f'V_PCA_{i}'] = X_v_pca[:, i]
            df = df.drop(columns=common_cols)
        return df

    def _fit_v_pca_scaler(self, df):
        if hasattr(self, 'v_pca') and self.v_pca is not None and self.v_cols_:
            common_cols = [col for col in self.v_cols_ if col in df.columns]
            if len(common_cols) == 0:
                return
            v_scaler = self.v_pca[1]
            pca = self.v_pca[0]
            X_v = df[common_cols].fillna(-999)
            X_v_scaled = v_scaler.transform(X_v)
            X_v_pca = pca.transform(X_v_scaled)
            v_pca_cols = [f'V_PCA_{i}' for i in range(X_v_pca.shape[1])]
            tmp = pd.DataFrame(X_v_pca, columns=v_pca_cols)
            self.v_pca_scaler = StandardScaler().fit(tmp)


categorical_features = [
    'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'
]
identity_categoricals = [col for col in train.columns if col.startswith('id_')]
categorical_features += identity_categoricals
m_features = [col for col in train.columns if col.startswith('M')]
categorical_features += m_features
categorical_features = [col for col in categorical_features if col in train.columns]
categorical_features


import pandas as pd
from sklearn.utils import resample

def temporal_split_and_balance(df, time_col, target_col, test_size=0.2):
    # 1. Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    # 2. Ğ”ĞµĞ»Ğ¸Ğ¼ Ğ½Ğ° train Ğ¸ test Ğ¿Ğ¾ Ğ²Ñ€ĞµĞ¼ĞµĞ½Ğ¸
    n_test = int(len(df_sorted) * test_size)
    test = df_sorted.iloc[-n_test:]
    train = df_sorted.iloc[:-n_test]
    
    def balance_classes(data, target_col):
        min_count = data[target_col].value_counts().min()
        # Ğ‘ĞµÑ€Ñ‘Ğ¼ min_count Ğ¾Ğ±ÑŠĞµĞºÑ‚Ğ¾Ğ² ĞºĞ°Ğ¶Ğ´Ğ¾Ğ³Ğ¾ ĞºĞ»Ğ°Ñ�Ñ�Ğ°
        fraud = data[data[target_col] == 1].sample(n=min_count, random_state=42)
        not_fraud = data[data[target_col] == 0].sample(n=min_count, random_state=42)
        balanced = pd.concat([fraud, not_fraud]).sample(frac=1, random_state=42)
        return balanced

    # 3. Ğ‘Ğ°Ğ»Ğ°Ğ½Ñ�Ğ¸Ñ€ÑƒĞµĞ¼ train Ğ¸ test
    train_bal = balance_classes(train, target_col)
    test_bal  = balance_classes(test, target_col)
    
    print("Train balanced shape:", train_bal.shape)
    print("Test  balanced shape:", test_bal.shape)
    print('Train isFraud mean:', train_bal[target_col].mean())
    print('Test  isFraud mean:', test_bal[target_col].mean())
    return train_bal, test_bal

# Ğ˜Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ
data_train, data_test = temporal_split_and_balance(train, time_col="TransactionDT", target_col="isFraud")


#from sklearn.model_selection import train_test_split
#
#data_train, data_test = train_test_split(
#    train,
#    test_size=0.2,     # 20% Ğ² Ñ‚ĞµÑ�Ñ‚, 80% Ğ² train
#    random_state=42,   # Ğ´Ğ»Ñ� Ğ²Ğ¾Ñ�Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸
#    #stratify=y         # Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ğ¿Ñ€Ğ¾Ğ¿Ğ¾Ñ€Ñ†Ğ¸Ğ¸ ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² (Ğ²Ğ°Ğ¶Ğ½Ğ¾ Ğ´Ğ»Ñ� Ğ°Ğ½Ñ‚Ğ¸Ñ„Ñ€Ğ¾Ğ´Ğ°!)
#)


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', FraudDataPreprocessor(categorical_features=categorical_features))
])

pipeline.fit(data_train)





import joblib

# Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾Ñ�Ğ»Ğµ fit
joblib.dump(pipeline, 'fraud_pipeline.joblib')


base_train = pipeline.transform(data_train)
base_train.head()



base_train.shape


base_train.columns


base_test = pipeline.transform(data_test)
base_test.head()


base_test.shape


base_test.columns


!pip install mlflow


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    recall_score, precision_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone
import mlflow
from mlflow.models import infer_signature

def compute_metrics(y_true, y_pred, y_prob):
    """Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµÑ‚ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ binary classification."""
    
    print(f"[DEBUG] y_true={len(y_true)}, y_pred={len(y_pred)}, y_prob={len(y_prob)}")

    metrics = {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        print(f"âš ï¸� ROC AUC Ğ½Ğµ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½: {e}")
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception as e:
        print(f"âš ï¸� PR AUC Ğ½Ğµ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½: {e}")
        metrics["pr_auc"] = 0.0

    return metrics

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

def plot_confusion(y_true, y_pred, threshold=None, labels=[0, 1], filename=None):
    if threshold is not None:
        # Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ğ¸Ñ‚ÑŒ Ğ¿Ğ¾Ñ€Ğ¾Ğ³ Ğº Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚Ñ�Ğ¼
        y_pred = (y_pred >= threshold).astype(int)


    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.xticks(labels)
    plt.yticks(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
    if filename:
        plt.savefig(filename)
        mlflow.log_artifact(filename)
    plt.show()
    plt.close()

def plot_roc_pr_curves(y_true, y_pred_proba, roc_auc, pr_auc, filename=None):
    plt.figure(figsize=(10, 4))
    # ROC curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # PR curve
    plt.subplot(1, 2, 2)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall_vals, precision_vals, label=f'PR AUC={pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        mlflow.log_artifact(filename)
    plt.show()
    plt.close()

def fit_model(
    model, X_train, y_train, X_val=None, y_val=None,
    cat_features=None, verbose=False
):
    """Ğ�Ğ±ÑƒÑ‡Ğ°ĞµÑ‚ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ñ� ÑƒÑ‡Ñ‘Ñ‚Ğ¾Ğ¼ CatBoost-Ñ�Ğ¾Ğ²Ğ¼ĞµÑ�Ñ‚Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸."""
    fit_params = {}
    if hasattr(model, "__class__") and model.__class__.__name__ == "CatBoostClassifier":
        fit_params['cat_features'] = cat_features
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
                verbose=verbose,
                **fit_params
            )
        else:
            model.fit(X_train, y_train, verbose=verbose, **fit_params)
    else:
        model.fit(X_train, y_train)
    return model

def cross_val_metrics_oof(model, X, y, cv, cat_features=None, verbose=False):
    oof_pred = np.full(len(y), np.nan)
    oof_pred_proba = np.full(len(y), np.nan)
    metrics_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        local_model = clone(model)
        fit_params = {}
        if hasattr(local_model, "__class__") and local_model.__class__.__name__ == "CatBoostClassifier" and cat_features is not None:
            fit_params["cat_features"] = cat_features
            local_model.fit(X_tr, y_tr, eval_set=(X_te, y_te), use_best_model=True, verbose=verbose, **fit_params)
        else:
            local_model.fit(X_tr, y_tr)
        oof_pred[test_idx] = local_model.predict(X_te)
        oof_pred_proba[test_idx] = local_model.predict_proba(X_te)[:, 1]
        fold_metrics = compute_metrics(y_te, oof_pred[test_idx], oof_pred_proba[test_idx])
        metrics_per_fold.append(fold_metrics)
        print(f"Fold {fold+1}: {[f'{k}: {v:.4f}' for k,v in fold_metrics.items()]}")

    valid_idx = ~np.isnan(oof_pred)
    avg_metrics = compute_metrics(y[valid_idx], oof_pred[valid_idx], oof_pred_proba[valid_idx])
    print(f"CV mean metrics: {[f'{k}: {v:.4f}' for k,v in avg_metrics.items()]}")
    return oof_pred, oof_pred_proba, avg_metrics

def train_antifraud_model(
    X_train, y_train,
    X_test=None, y_test=None,
    model=None,
    cat_features=None,
    plot_curves=True,
    verbose=False,
    cv=None,
    random_state=None
):
    """
    Ğ£Ğ½Ğ¸Ğ²ĞµÑ€Ñ�Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ°Ğ½Ñ‚Ğ¸Ñ„Ñ€Ğ¾Ğ´-Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ñ� ĞºÑ€Ğ¾Ñ�Ñ�-Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸ĞµĞ¹ Ğ¸ Ğ»Ğ¾Ğ³Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸ĞµĞ¼ Ğ² MLflow.
    """
    #os.environ['MLFLOW_TRACKING_TOKEN'] = create_token()
    mlflow.set_experiment("antifraud_experiment")
    with mlflow.start_run():

        # Auto-create CatBoost ĞµÑ�Ğ»Ğ¸ Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ½Ğµ Ğ·Ğ°Ğ´Ğ°Ğ½Ğ°
        if model is None:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                loss_function='Logloss',
                eval_metric='AUC',
                iterations=1000,
                early_stopping_rounds=50,
                cat_features=cat_features,
                verbose=verbose
            )

        # Ğ›Ğ¾Ğ³Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ² Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        if cv is not None:
            from sklearn.model_selection import TimeSeriesSplit
            is_ts_split = isinstance(cv, TimeSeriesSplit)

            if not is_ts_split and hasattr(cv, 'split') \
                and hasattr(cv, 'get_n_splits') \
                and getattr(cv, 'shuffle', True):

                is_catboost = hasattr(model, "__class__") and model.__class__.__name__ == "CatBoostClassifier"
                fit_params = {"cat_features": cat_features} if is_catboost and cat_features is not None else {}
                y_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict', fit_params=fit_params)
                y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba', fit_params=fit_params)[:, 1]
                metrics = compute_metrics(y_train, y_pred, y_pred_proba)

                print(f"[CV={cv.get_n_splits()} folds | Out-of-fold eval]:")
                print_metrics(metrics)

                if plot_curves:
                    plot_confusion(y_train, y_pred, filename="confusion_matrix.png")
                    plot_roc_pr_curves(y_train, y_pred_proba, metrics["roc_auc"], metrics["pr_auc"], filename="roc_pr_curves.png")

                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", metric_value)

                if model.__class__.__name__ == "CatBoostClassifier":
                    signature = infer_signature(X_train, y_pred)
                    mlflow.catboost.log_model(model, "model", signature=signature)
                else:
                    mlflow.sklearn.log_model(model, "model")

                return model, metrics
            else:
                oof_pred, oof_pred_proba, metrics = cross_val_metrics_oof(model, X_train, y_train, cv, cat_features, verbose)
                valid_idx = ~np.isnan(oof_pred)

                if plot_curves:
                    plot_confusion(y_train.iloc[valid_idx], oof_pred[valid_idx], filename="confusion_matrix.png")
                    plot_roc_pr_curves(y_train.iloc[valid_idx], oof_pred_proba[valid_idx], metrics["roc_auc"], metrics["pr_auc"], filename="roc_pr_curves.png")

                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", metric_value)


                # ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ²Ñ…Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
                input_example = X_train.iloc[:3]

                if model.__class__.__name__ == "CatBoostClassifier":
                    signature = infer_signature(X_train, oof_pred)
                    mlflow.catboost.log_model(model, "model", signature=signature, input_example=input_example)
                else:
                    # Ğ˜Ğ½Ñ„ĞµÑ€ĞµĞ½Ñ� Ñ�Ğ¸Ğ³Ğ½Ğ°Ñ‚ÑƒÑ€Ñ‹ (Ñ‚Ğ¸Ğ¿Ğ¾Ğ² Ğ²Ñ…Ğ¾Ğ´Ğ¾Ğ² Ğ¸ Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ¾Ğ²)
                    signature = infer_signature(X_train, oof_pred)
                    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

                return model, metrics

        # Hold-out Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ
        model = fit_model(model, X_train, y_train, X_test, y_test, cat_features, verbose)

        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        metrics_train = compute_metrics(y_train, y_train_pred, y_train_pred_proba)

        y_val_pred = model.predict(X_test)
        y_val_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics_val = compute_metrics(y_test, y_val_pred, y_val_pred_proba)

        print("[Hold-out] TRAIN:")
        print_metrics(metrics_train)

        print("[Hold-out] VAL:")
        print_metrics(metrics_val)

        if plot_curves:
            plot_confusion(y_test, y_val_pred, filename="confusion_matrix.png")
            plot_roc_pr_curves(y_test, y_val_pred_proba, metrics_val["roc_auc"], metrics_val["pr_auc"], filename="roc_pr_curves.png")

        for metric_name, metric_value in metrics_train.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)

        for metric_name, metric_value in metrics_val.items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)


        # ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ²Ñ…Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
        input_example = X_test.iloc[:3]

        if model.__class__.__name__ == "CatBoostClassifier":
            signature = infer_signature(X_test, y_test)
            mlflow.catboost.log_model(model, "model", signature=signature, input_example=input_example)
        else:
            # Ğ˜Ğ½Ñ„ĞµÑ€ĞµĞ½Ñ� Ñ�Ğ¸Ğ³Ğ½Ğ°Ñ‚ÑƒÑ€Ñ‹ (Ñ‚Ğ¸Ğ¿Ğ¾Ğ² Ğ²Ñ…Ğ¾Ğ´Ğ¾Ğ² Ğ¸ Ğ²Ñ‹Ñ…Ğ¾Ğ´Ğ¾Ğ²)
            signature = infer_signature(X_test, y_test)
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

        return model, metrics_val


base_train.columns


X_train = base_train.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
y_train = base_train['isFraud']

X_test = base_test.drop(columns=['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
y_test = base_test['isFraud']


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model, metrics = train_antifraud_model(X_train, y_train, X_test=X_test, y_test=y_test, model=model, verbose=True)
metrics


# ĞŸĞ¾Ğ»ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
importances = model.feature_importances_

# Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ğµ Ñ�Ğ¿Ğ¸Ñ�ĞºĞ° Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² Ñ� Ğ¸Ñ… Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ñ�Ğ¼Ğ¸
feature_names = X_train.columns  # Ğ¸Ğ»Ğ¸ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾ Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº Ğ¸Ğ¼ĞµĞ½ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
important_features = list(zip(feature_names, importances))

# Ğ¡Ğ¾Ñ€Ñ‚Ğ¸Ñ€Ğ¾Ğ²ĞºĞ° Ğ¿Ğ¾ Ğ²Ğ°Ğ¶Ğ½Ğ¾Ñ�Ñ‚Ğ¸
important_features.sort(key=lambda x: x[1], reverse=True)

# Ğ�Ñ‚Ğ¾Ğ±Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ğ²Ğ°Ğ¶Ğ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
for feature in important_features:
    print(f"Feature: {feature[0]}, Importance: {feature[1]}")



#from sklearn.model_selection import train_test_split

# X â€” Ğ²Ğ°ÑˆĞ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ (pandas.DataFrame Ğ¸Ğ»Ğ¸ numpy.array)
# y â€” Ñ†ĞµĞ»ĞµĞ²Ğ°Ñ� Ğ¿ĞµÑ€ĞµĞ¼ĞµĞ½Ğ½Ğ°Ñ� (Series Ğ¸Ğ»Ğ¸ array)
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y,
#    test_size=0.2,     # 20% Ğ² Ñ‚ĞµÑ�Ñ‚, 80% Ğ² train
#    random_state=42,   # Ğ´Ğ»Ñ� Ğ²Ğ¾Ñ�Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ²Ğ¾Ğ´Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸
#    stratify=y         # Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ğ¿Ñ€Ğ¾Ğ¿Ğ¾Ñ€Ñ†Ğ¸Ğ¸ ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² (Ğ²Ğ°Ğ¶Ğ½Ğ¾ Ğ´Ğ»Ñ� Ğ°Ğ½Ñ‚Ğ¸Ñ„Ñ€Ğ¾Ğ´Ğ°!)
#)


cat_features = [
    'ProductCD',
    'card1', 
    'card2', 
    'card3', 
    'card4', 
    'card5', 
    'card6', 
    'addr1', 
    'addr2',
    'P_emaildomain',
    #'TransactionAmt_binned', 
]


!pip install boto3


from catboost import CatBoostClassifier

my_model = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='AUC',
            #eval_metric='Logloss', 
            iterations=1000,
            early_stopping_rounds=50,
            cat_features=cat_features,
            verbose=False,
            #task_type="GPU",
)

#best_params = {'learning_rate': 0.03574712922600244, 'depth': 10, 'l2_leaf_reg': 5.395030966670228, 'bagging_temperature': 0.5986584841970366, 'random_strength': 2.4041677639819286, 'border_count': 66, 'scale_pos_weight': 6.750277604651747}
#my_model = CatBoostClassifier(verbose=False, **best_params)
#cv = TimeSeriesSplit(n_splits=5)

model, metrics = train_antifraud_model(X_train, y_train, X_test=X_test, y_test=y_test, model=my_model, verbose=True)
metrics


import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'depth': trial.suggest_int("depth", 4, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 1),
        'random_strength': trial.suggest_float("random_strength", 1, 10),
        'border_count': trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0),
        'iterations': 500,
        'early_stopping_rounds': 10,
        #'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',  
        'cat_features': cat_features,
        'verbose': True,
        'random_state': 42,
        #'task_type': "GPU", 
    }

    model = CatBoostClassifier(**params)
    cv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    return scores.mean()

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True, n_jobs=-1)

print("Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:", study.best_params)
print("Ğ›ÑƒÑ‡ÑˆĞ¸Ğ¹ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚ AUC:", study.best_value)


study.best_trial.params


#best_params = {'learning_rate': 0.03574712922600244, 'depth': 10, 'l2_leaf_reg': 5.395030966670228, 'bagging_temperature': 0.5986584841970366, 'random_strength': 2.4041677639819286, 'border_count': 66, 'scale_pos_weight': 6.750277604651747}
#best_params = {'learning_rate': 0.03574712922600244, 'depth': 10, 'l2_leaf_reg': 5.395030966670228, 'bagging_temperature': 0.5986584841970366, 'random_strength': 2.4041677639819286, 'border_count': 66, 'scale_pos_weight': 6.750277604651747}
#best_params ={'learning_rate': 0.03574712922600244, 'depth': 10, 'l2_leaf_reg': 5.395030966670228, 'bagging_temperature': 0.5986584841970366, 'random_strength': 2.4041677639819286, 'border_count': 66, 'scale_pos_weight': 6.750277604651747}

best_params = study.best_trial.params
model = CatBoostClassifier(**best_params)


model, metrics = train_antifraud_model(X_train, y_train, X_test, y_test, model=my_model)
metrics


import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score
import numpy as np

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'depth': trial.suggest_int("depth", 4, 10),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 1),
        'random_strength': trial.suggest_float("random_strength", 1, 10),
        'border_count': trial.suggest_int("border_count", 32, 255),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 100.0),
        'iterations': 500,
        'early_stopping_rounds': 10,
        #'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',  
        'cat_features': cat_features,
        'verbose': True,
        'random_state': 42,
        #'task_type': "GPU", 
    }
    
    cv = TimeSeriesSplit(n_splits=3)
    precisions = []

    for train_idx, valid_idx in cv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, verbose=0)
        y_proba = model.predict_proba(X_val)[:,1]
        
        # Ğ�Ğ°Ğ¹Ğ´Ñ‘Ğ¼ Ğ¼Ğ¸Ğ½Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ threshold, Ğ¿Ñ€Ğ¸ ĞºĞ¾Ñ‚Ğ¾Ñ€Ğ¾Ğ¼ recall >= 0.95
        from sklearn.metrics import precision_recall_curve
        precisions_curve, recalls_curve, thresholds = precision_recall_curve(y_val, y_proba)
        
        mask = recalls_curve >= 0.95
        if np.any(mask):
            precision_max = precisions_curve[mask].max()
        else:
            precision_max = 0.0  # Ğ�Ğ° Ñ�Ñ‚Ğ¾Ğ¼ Ñ„Ğ¾Ğ»Ğ´Ğµ Ñ€ĞµĞºĞ¾Ğ»Ğ» Ğ½Ğµ Ğ´Ğ¾Ñ�Ñ‚Ğ¸Ğ³Ğ½ÑƒÑ‚, ÑˆÑ‚Ñ€Ğ°Ñ„ÑƒĞµĞ¼
        
        precisions.append(precision_max)
    
    return np.mean(precisions)

# ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ½Ğ°Ñ�Ñ‚Ñ€Ğ¾Ğ¹ĞºĞ¸ Optuna
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True, n_jobs=-1)

print("Ğ›ÑƒÑ‡ÑˆĞ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹:", study.best_params)
print("Ğ›ÑƒÑ‡ÑˆĞµĞµ precision Ğ¿Ñ€Ğ¸ recall >= 0.95:", study.best_value)


study.best_trial.params


#best_params = {'learning_rate': 0.03574712922600244, 'depth': 10, 'l2_leaf_reg': 5.395030966670228, 'bagging_temperature': 0.5986584841970366, 'random_strength': 2.4041677639819286, 'border_count': 66, 'scale_pos_weight': 6.750277604651747}
best_params = study.best_trial.params
model = CatBoostClassifier(**best_params)


model, metrics = train_antifraud_model(X_train, y_train, X_test, y_test, model=my_model)
metrics


import torch
print(torch.__version__)





!pip install torch_geometric


import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)                     # Ğ’Ñ�Ñ‚Ñ€Ğ¾ĞµĞ½Ğ½Ñ‹Ğ¹ random
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)          # GPU (Ğ¾Ğ´Ğ¸Ğ½)
        torch.cuda.manual_seed_all(seed)      # GPU (Ğ²Ñ�Ğµ, ĞµÑ�Ğ»Ğ¸ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·ÑƒĞµÑ‚Ñ�Ñ� multi-GPU)
        torch.backends.cudnn.deterministic = True  # Ğ�Ğ±ĞµÑ�Ğ¿ĞµÑ‡Ğ¸Ğ²Ğ°ĞµÑ‚ Ğ´ĞµÑ‚ĞµÑ€Ğ¼Ğ¸Ğ½Ğ¸Ğ·Ğ¼
        torch.backends.cudnn.benchmark = False     # Ğ�Ñ‚ĞºĞ»Ñ�Ñ‡Ğ°ĞµÑ‚ autotuner, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ Ğ¸Ğ·Ğ±ĞµĞ¶Ğ°Ñ‚ÑŒ Ñ�Ğ»ÑƒÑ‡Ğ°Ğ¹Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹

# ĞŸÑ€Ğ¸Ğ¼ĞµÑ€ Ğ¸Ñ�Ğ¿Ğ¾Ğ»ÑŒĞ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ�
set_seed(42)


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    recall_score, precision_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import random
import copy

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

def compute_metrics(y_true, y_pred, y_prob):
    """Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ñ�ĞµÑ‚ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ binary classification."""
    
    print(f"[DEBUG] y_true={len(y_true)}, y_pred={len(y_pred)}, y_prob={len(y_prob)}")

    metrics = {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        print(f"âš ï¸� ROC AUC Ğ½Ğµ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½: {e}")
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception as e:
        print(f"âš ï¸� PR AUC Ğ½Ğµ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½: {e}")
        metrics["pr_auc"] = 0.0

    return metrics

def print_metrics(metrics):
    """ĞŸĞµÑ‡Ğ°Ñ‚Ğ°ĞµÑ‚ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ñ� Ğ¼ĞµÑ‚Ñ€Ğ¸Ğº Ğ¿Ğ¾Ñ�Ñ‚Ñ€Ğ¾Ñ‡Ğ½Ğ¾."""
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

def print_val_metrics_by_epoch(history: dict, filename=None):
    """Ğ¡Ñ‚Ñ€Ğ¾Ğ¸Ñ‚ Ğ³Ñ€Ğ°Ñ„Ğ¸Ğº Ğ¿Ğ¾ Ñ�Ğ¿Ğ¾Ñ…Ğ°Ğ¼."""
    plt.figure(figsize=(8, 3))
    plt.plot(history["val_f1"], label='Val F1')
    plt.plot(history["val_pr_auc"], label='Val PR AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Ğ’Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¾Ğ½Ğ½Ñ‹Ğµ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞ¸ Ğ¿Ğ¾ Ñ�Ğ¿Ğ¾Ñ…Ğ°Ğ¼')
    plt.legend()
    if filename:
        plt.savefig(filename)
        mlflow.log_artifact(filename)
    plt.show()
    plt.close()

def plot_confusion(y_true, y_pred, threshold=0.5, labels=[0, 1], filename=None):
    """Ğ¡Ñ‚Ñ€Ğ¾Ğ¸Ñ‚ confusion matrix Ğ¿Ğ¾ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�Ğ¼."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.title(f'Confusion matrix (thr={threshold:.2f})')
    plt.xlabel('ĞŸÑ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¾')
    plt.ylabel('Ğ˜Ñ�Ñ‚Ğ¸Ğ½Ğ½Ğ¾Ğµ Ğ·Ğ½Ğ°Ñ‡ĞµĞ½Ğ¸Ğµ')
    plt.colorbar()
    plt.xticks(labels)
    plt.yticks(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
    if filename:
        plt.savefig(filename)
        mlflow.log_artifact(filename)
    plt.show()
    plt.close()

def plot_roc_pr_curves(y_true, y_pred_proba, roc_auc, pr_auc, filename=None):
    """Ğ¡Ñ‚Ñ€Ğ¾Ğ¸Ñ‚ ROC-ĞºÑ€Ğ¸Ğ²ÑƒÑ� Ğ¸ Precision-Recall curve."""
    plt.figure(figsize=(10, 4))
    # ROC-ĞºÑ€Ğ¸Ğ²Ğ°Ñ�
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-ĞºÑ€Ğ¸Ğ²Ğ°Ñ�')
    plt.legend(loc='lower right')
    # Precision-Recall ĞºÑ€Ğ¸Ğ²Ğ°Ñ�
    plt.subplot(1, 2, 2)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall_vals, precision_vals, label=f'PR AUC={pr_auc:.4f}')
    plt.xlabel('Recall (ĞŸĞ¾Ğ»Ğ½Ğ¾Ñ‚Ğ°)')
    plt.ylabel('Precision (Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ)')
    plt.title('Precision-Recall ĞºÑ€Ğ¸Ğ²Ğ°Ñ�')
    plt.legend(loc='upper right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        mlflow.log_artifact(filename)
    plt.show()
    plt.close()


def find_best_threshold_f1(y_true, y_prob):
    """ĞŸĞ¾Ğ´Ğ±Ğ¸Ñ€Ğ°ĞµÑ‚ Ñ‚Ğ°ĞºĞ¾Ğ¹ Ğ¿Ğ¾Ñ€Ğ¾Ğ³ Ğ´Ğ»Ñ� Ğ²ĞµÑ€Ğ¾Ñ�Ñ‚Ğ½Ğ¾Ñ�Ñ‚ĞµĞ¹, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ f1 Ğ±Ñ‹Ğ» Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¼."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    idx = np.argmax(f1)
    if idx == len(thresholds):  # edge case
        best_thr = 0.5
        best_f1 = f1[-1]
    else:
        best_thr = thresholds[idx]
        best_f1 = f1[idx]
    return float(best_thr), float(best_f1)

def find_best_threshold_recall(y_true, y_prob, min_precision=0.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision, recall = precision[1:], recall[1:]
    valid = (precision >= min_precision)
    if np.any(valid):
        idx = np.argmax(recall[valid])
        idx_global = np.where(valid)[0][idx]
        best_thr = thresholds[idx_global]
        best_recall = recall[valid][idx]
    else:
        idx = np.argmax(recall)
        best_thr = thresholds[idx]
        best_recall = recall[idx]
    return float(best_thr), float(best_recall)

def find_threshold_for_fixed_recall(y_true, y_prob, target_recall=0.95):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[1:]
    recall = recall[1:]
    valid = recall >= target_recall
    if np.any(valid):
        idx = np.argmax(valid)
        best_thr = thresholds[idx]
        actual_recall = recall[idx]
        actual_precision = precision[idx]
    else:
        idx = np.argmax(recall)
        best_thr = thresholds[idx]
        actual_recall = recall[idx]
        actual_precision = precision[idx]
        print(f"[WARN] Ğ�Ğµ ÑƒĞ´Ğ°Ğ»Ğ¾Ñ�ÑŒ Ğ´Ğ¾Ñ�Ñ‚Ğ¸Ñ‡ÑŒ Ğ¶ĞµĞ»Ğ°ĞµĞ¼Ğ¾Ğ³Ğ¾ recall={target_recall:.2f}, Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼ÑƒĞ¼={actual_recall:.3f}")
    return float(best_thr), float(actual_recall), float(actual_precision)

def select_threshold(y_true, y_prob, strategy='f1', **kwargs):
    """
    strategy: 'f1', 'recall', 'fixed_recall'
    kwargs: Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ´Ğ»Ñ� Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²ÑƒÑ�Ñ‰Ğ¸Ñ… Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ğ¹
    """
    if strategy == 'f1':
        thr, _ = find_best_threshold_f1(y_true, y_prob)
        return thr
    elif strategy == 'recall':
        min_precision = kwargs.get('min_precision', 0.0)
        thr, _ = find_best_threshold_recall(y_true, y_prob, min_precision)
        return thr
    elif strategy == 'fixed_recall':
        target_recall = kwargs.get('target_recall', 0.95)
        thr, _, _ = find_threshold_for_fixed_recall(y_true, y_prob, target_recall)
        return thr
    else:
        raise ValueError(f"Unknown threshold selection strategy: {strategy}")

def get_example_input(loader, device):
    for i, batch in enumerate(loader):
        print(f"[DEBUG] Batch {i}: type={type(batch)}, has_x={hasattr(batch, 'x')}, x_is_none={batch.x is None if hasattr(batch, 'x') else 'N/A'}")
        if batch is not None and hasattr(batch, "x") and batch.x is not None:
            return batch.to(device)
    raise ValueError("Ğ�Ğµ ÑƒĞ´Ğ°Ğ»Ğ¾Ñ�ÑŒ Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ¸Ñ‚ÑŒ Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€ Ğ²Ñ…Ğ¾Ğ´Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ğ¸Ğ· train_loader")


def train_gnn_antifraud_model(
    model,
    train_loader,
    val_loader=None,
    device=None,
    n_epochs=10,
    lr=1e-3,
    weight_decay=0.0004,
    criterion=None,
    plot_curves=True,
    verbose=True,
    seed=42,
    early_stopping_patience=5,
    key_metric='pr_auc',
    threshold_selection_strategy='f1',  # 'f1', 'recall', 'fixed_recall'
    threshold_params=None               # dict, Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€ {'target_recall': 0.9}
):
    """
    Ğ£Ğ½Ğ¸Ğ²ĞµÑ€Ñ�Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹ Ñ†Ğ¸ĞºĞ» Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ñ� Ğ´Ğ»Ñ� GNN Ğ² Ğ·Ğ°Ğ´Ğ°Ñ‡Ğµ Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ğ¾Ğ¹ ĞºĞ»Ğ°Ñ�Ñ�Ğ¸Ñ„Ğ¸ĞºĞ°Ñ†Ğ¸Ğ¸ anti-fraud Ñ� Ğ¿Ğ¾Ğ´Ğ±Ğ¾Ñ€Ğ¾Ğ¼ Ğ¾Ğ¿Ñ‚Ğ¸Ğ¼Ğ°Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ°.
    Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµÑ‚ Ğ»ÑƒÑ‡ÑˆÑƒÑ� Ğ¿Ğ¾ ĞºĞ»Ñ�Ñ‡ĞµĞ²Ğ¾Ğ¹ Ğ¼ĞµÑ‚Ñ€Ğ¸ĞºĞµ (pr_auc/F1) Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ.
    """

    set_seed(seed)

    if len(train_loader) == 0:
        raise ValueError("train_loader Ğ¿ÑƒÑ�Ñ‚Ğ¾Ğ¹ â€” Ğ½ĞµÑ‚ Ğ±Ğ°Ñ‚Ñ‡ĞµĞ¹ Ğ´Ğ»Ñ� Ğ²Ñ‹Ğ±Ğ¾Ñ€Ğ° example_input")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if threshold_params is None:
        threshold_params = {}

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
    )

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    first_train_y = []
    for batch in train_loader:
        first_train_y.append(batch.y.cpu().numpy())
        if len(first_train_y) > 10:
            break
    first_train_y = np.concatenate(first_train_y)
    frac_pos = np.mean(first_train_y)
    print(f"[INFO] ĞŸÑ€Ğ¸Ğ¼ĞµÑ€Ğ½Ñ‹Ğ¹ Ğ´Ğ¸Ñ�Ğ±Ğ°Ğ»Ğ°Ğ½Ñ� ĞºĞ»Ğ°Ñ�Ñ�Ğ¾Ğ² Ğ² train: {frac_pos:.4f} positive ({np.sum(first_train_y==1)}/{len(first_train_y)})")


    best_metric = -np.inf
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())
    best_thr = 0.5
    history = {"train_loss": [], "train_f1": [], "val_f1": [], "val_pr_auc": []}
    eval_metrics = None

    #os.environ['MLFLOW_TRACKING_TOKEN'] = create_token()
    mlflow.set_experiment("antifraud_experiment")
    with mlflow.start_run():
        mlflow.log_params({
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "early_stopping_patience": early_stopping_patience,
            "key_metric": key_metric,
            "threshold_strategy": threshold_selection_strategy,
            "seed": seed,
        })
        mlflow.log_param("frac_pos", frac_pos)

        for epoch in range(n_epochs):
            model.train()
            tr_loss = 0
            tr_true, tr_prob, tr_pred = [], [], []
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                y = data.y.float().view(-1)
                o = out.view(-1)
                loss = criterion(o, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * y.size(0)
                proba = torch.sigmoid(o).detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                pred = (proba > 0.5).astype(int)
                tr_true.append(y_np)
                tr_prob.append(proba)
                tr_pred.append(pred)

            tr_true = np.concatenate(tr_true)
            tr_prob = np.concatenate(tr_prob)
            tr_pred = np.concatenate(tr_pred)
            tr_loss /= len(tr_true)
            tr_metrics = compute_metrics(tr_true, tr_pred, tr_prob)

            history["train_loss"].append(tr_loss)
            history["train_f1"].append(tr_metrics.get("f1", 0.0))

            if verbose:
                print(f"[Ğ­Ğ¿Ğ¾Ñ…Ğ° {epoch+1}] Train loss={tr_loss:.4f} | ", end="")
                print_metrics(tr_metrics)

            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            for k, v in tr_metrics.items():
                mlflow.log_metric(f"train_{k}", v, step=epoch)

            # ===== Ğ’Ğ�Ğ›Ğ˜Ğ”Ğ�Ğ¦Ğ˜Ğ¯ =====
            if val_loader is not None:
                model.eval()
                val_true, val_prob, val_pred = [], [], []
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        out = model(data).view(-1)
                        proba = torch.sigmoid(out).cpu().numpy()
                        y = data.y.cpu().numpy()
                        pred = (proba > 0.5).astype(int)
                        val_true.append(y)
                        val_prob.append(proba)
                        val_pred.append(pred)

                val_true = np.concatenate(val_true)
                val_prob = np.concatenate(val_prob)
                val_pred = np.concatenate(val_pred)
                best_thr_epoch = select_threshold(
                    val_true, val_prob, 
                    strategy=threshold_selection_strategy, 
                    **threshold_params
                )
                val_metrics = compute_metrics(val_true, val_pred, val_prob)

                history["val_f1"].append(val_metrics["f1"])
                history["val_pr_auc"].append(val_metrics["pr_auc"])

                if verbose:
                    print(f"[Ğ­Ğ¿Ğ¾Ñ…Ğ° {epoch+1}] [Ğ’Ğ�Ğ› thr={best_thr_epoch:.3f}]: ", end="")
                    print_metrics(val_metrics)

                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v, step=epoch)

                current_metric = val_metrics.get(key_metric, 0)
                scheduler.step(current_metric)

                # Early stopping Ğ¸ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ¸Ğµ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ğ¾Ğ²
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_thr = select_threshold(val_true, val_prob, strategy=threshold_selection_strategy, **threshold_params)
                    eval_metrics = val_metrics
                    epochs_no_improve = 0
                    if verbose:
                        print(f"[INFO] ğŸ”¥ Ğ�Ğ¾Ğ²Ğ°Ñ� Ğ»ÑƒÑ‡ÑˆĞ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ğ¿Ğ¾ {key_metric}: {best_metric:.4f} (Ñ�Ğ¿Ğ¾Ñ…Ğ° {epoch})")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        if verbose:
                            print(f"[INFO] Ğ Ğ°Ğ½Ğ½Ñ�Ñ� Ğ¾Ñ�Ñ‚Ğ°Ğ½Ğ¾Ğ²ĞºĞ° Ğ½Ğ° Ñ�Ğ¿Ğ¾Ñ…Ğµ {epoch}. {key_metric} Ğ½Ğµ ÑƒĞ»ÑƒÑ‡ÑˆĞ°Ğ»Ğ°Ñ�ÑŒ {early_stopping_patience} Ñ�Ğ¿Ğ¾Ñ….")
                        break
            else:
                # Ğ‘ĞµĞ· Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ğ¸ â€” Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ğ¾ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ğ¿Ğ¾Ñ�Ğ»ĞµĞ´Ğ½Ğ¸Ğµ Ğ²ĞµÑ�Ğ°
                best_model_state = copy.deepcopy(model.state_dict())
                best_thr = 0.5
                val_metrics = tr_metrics # Ğ”Ğ»Ñ� Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‚Ğ°

                scheduler.step(tr_metrics["f1"])

        # ======= Ğ—Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ° Ğ»ÑƒÑ‡ÑˆĞµĞ¹ Ğ¼Ğ¾Ğ´ĞµĞ»Ğ¸ Ğ¸ Ñ„Ğ¸Ğ½Ğ°Ğ»ÑŒĞ½Ñ‹Ğµ ĞºÑ€Ğ¸Ğ²Ñ‹Ğµ =======
        model.load_state_dict(best_model_state)


        eval_true, eval_prob = [], []
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data)
                    y = data.y.float().view(-1)
                    o = out.view(-1)
                    proba = torch.sigmoid(o).cpu().numpy()
                    eval_true.append(y.cpu().numpy())
                    eval_prob.append(proba)
            eval_true = np.concatenate(eval_true)
            eval_prob = np.concatenate(eval_prob)
            eval_pred = (eval_prob > best_thr).astype(int)
            eval_metrics = compute_metrics(eval_true, eval_pred, eval_prob)
            eval_metrics["best_thr"] = float(best_thr)
        else:
            eval_true = tr_true
            eval_prob = tr_prob
            eval_pred = (eval_prob > best_thr).astype(int)
            eval_metrics = compute_metrics(eval_true, eval_pred, eval_prob)
            eval_metrics["best_thr"] = float(best_thr)


        if plot_curves:
            plot_confusion(eval_true, eval_pred, threshold=best_thr, filename="confusion_matrix.png")
            plot_roc_pr_curves(eval_true, eval_prob, eval_metrics['roc_auc'], eval_metrics['pr_auc'], filename="roc_pr_curves.png")

            # Ğ”Ğ¸Ğ½Ğ°Ğ¼Ğ¸ĞºĞ° Ğ¿Ğ¾ Ñ�Ğ¿Ğ¾Ñ…Ğ°Ğ¼, ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ Ğ²Ğ°Ğ»Ğ¸Ğ´Ğ°Ñ†Ğ¸Ñ�
            if val_loader is not None and len(history["val_f1"]) > 1:
                print_val_metrics_by_epoch(history, filename="hisotry_metrics.png")


        # ğŸ’¡ Ğ“ĞµĞ½ĞµÑ€Ğ°Ñ†Ğ¸Ñ� input_example Ğ¸ Ñ�Ğ¸Ğ³Ğ½Ğ°Ñ‚ÑƒÑ€Ñ‹
        example_input = get_example_input(train_loader, device)

        with torch.no_grad():
            example_output = model(example_input)
        
        signature = infer_signature(
            example_input.x.cpu().numpy(),
            example_output.detach().cpu().numpy()
        )

        #input_example= {"x": example_input.x.cpu().numpy()}
        input_example = {
            "x": example_input.x.cpu().numpy(),
            "edge_index": example_input.edge_index.cpu().numpy(),
            "batch": example_input.batch.cpu().numpy() if hasattr(example_input, "batch") else None,
        }

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        mlflow.log_metric("best_val_metric", best_metric)
        mlflow.log_metric("best_threshold", best_thr)

        print(f"[INFO] âœ… Ğ�Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ Ğ·Ğ°Ğ²ĞµÑ€ÑˆĞµĞ½Ğ¾. Ğ›ÑƒÑ‡ÑˆĞ°Ñ� Ğ¼Ğ¾Ğ´ĞµĞ»ÑŒ Ñ�Ğ¾Ñ…Ñ€Ğ°Ğ½ĞµĞ½Ğ°.")

    return model, eval_metrics


import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd

def to_graph_data(
    df: pd.DataFrame,
    edge_keys=['card1'],
    target_col='isFraud',
    ignore_cols=None,
    num_cols=None,        # Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº float Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
    cat_cols=None,        # Ñ�Ğ¿Ğ¸Ñ�Ğ¾Ğº ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ… Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ² (label-encoded!)
    edge_features_fn=None # Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ�, ĞµÑ�Ğ»Ğ¸ Ğ½ÑƒĞ¶Ğ½Ñ‹ edge_attr
):
    if ignore_cols is None:
        ignore_cols = [target_col]
    if num_cols is None or cat_cols is None:
        num_cols = df.select_dtypes(include=['float', 'float32', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['int', 'int32', 'int64']).columns.tolist()
        num_cols = [c for c in num_cols if c not in ignore_cols]
        cat_cols = [c for c in cat_cols if c not in ignore_cols]

    # Ğ¢ĞµĞ½Ğ·Ğ¾Ñ€Ñ‹ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²
    x_num = torch.tensor(df[num_cols].values, dtype=torch.float32) if num_cols else None
    x_cat = torch.tensor(df[cat_cols].values, dtype=torch.long) if cat_cols else None
    y = torch.tensor(df[target_col].values, dtype=torch.float32) if target_col in df.columns else None

    #print("cat_features:", cat_cols)
    #print("df[cat_features].columns:", df[cat_cols].columns.tolist())
    #print("df[cat_features].shape:", df[cat_cols].shape)

    # Ğ Ñ‘Ğ±Ñ€Ğ° Ğ¸ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¸ Ñ€ĞµĞ±ĞµÑ€ (ĞµÑ�Ğ»Ğ¸ Ğ¿Ğ¾Ğ½Ğ°Ğ´Ğ¾Ğ±Ğ¸Ñ‚Ñ�Ñ�)
    edges_src, edges_dst, edge_attrs = [], [], []
    for edge_key in edge_keys:
        groups = df.groupby(edge_key).indices
        for indxs in groups.values():
            indxs = np.asarray(list(indxs))
            if len(indxs) > 1:
                src, dst = np.meshgrid(indxs, indxs)
                mask = src != dst
                src, dst = src[mask], dst[mask]
                edges_src.append(src)
                edges_dst.append(dst)
                # Ğ•Ñ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ Ñ„ÑƒĞ½ĞºÑ†Ğ¸Ñ� Ğ´Ğ»Ñ� edge_attr â€” Ğ½Ğ°Ğ¿Ñ€Ğ¸Ğ¼ĞµÑ€ Ñ€Ğ°Ğ·Ğ½Ğ¾Ñ�Ñ‚Ğ¸ ĞºĞ°ĞºĞ¸Ñ…-Ğ½Ğ¸Ğ±ÑƒĞ´ÑŒ Ğ¿Ñ€Ğ¸Ğ·Ğ½Ğ°ĞºĞ¾Ğ²:
                if edge_features_fn is not None:
                    edge_feat = edge_features_fn(df.iloc[src], df.iloc[dst]) # shape: [num_edges, n_edge_feats]
                    edge_attrs.append(edge_feat)
    if len(edges_src) > 0:
        edges_src = np.concatenate(edges_src)
        edges_dst = np.concatenate(edges_dst)
        edge_index = torch.tensor(np.stack([edges_src, edges_dst]), dtype=torch.long)
        if len(edge_attrs) > 0:
            edge_attr = np.concatenate(edge_attrs, axis=0)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_attr = None
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = None


    if x_num is not None and x_cat is not None:
        x = torch.cat([x_num, x_cat.float()], dim=1)
    elif x_num is not None:
        x = x_num
    elif x_cat is not None:
        x = x_cat.float()
    else:
        x = None

    return Data(
        x=x,
        x_num=x_num, 
        x_cat=x_cat, 
        edge_index=edge_index, 
        edge_attr=edge_attr, # (None ĞµÑ�Ğ»Ğ¸ Ğ½ĞµÑ‚)
        y=y
    )


from torch.utils.data import Dataset

class FraudGraphChunkDataset(Dataset):
    def __init__(self, df, chunk_size=2000, edge_keys=['card1'],
                 target_col='isFraud', ignore_cols=None,
                 num_cols=None, cat_cols=None,
                 to_graph_data_fn=None):
        self.df = df
        self.chunk_size = chunk_size
        self.edge_keys = edge_keys
        self.target_col = target_col
        self.ignore_cols = ignore_cols if ignore_cols is not None else [target_col]
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.to_graph_data_fn = to_graph_data_fn or to_graph_data

        self.num_chunks = (len(df) + chunk_size - 1) // chunk_size

        first_chunk = self.df.iloc[:chunk_size].reset_index(drop=True)
        data_example = self._build_data(first_chunk)
        self.num_num_features = data_example.x_num.shape[1] if data_example.x_num is not None else 0
        self.num_cat_features = data_example.x_cat.shape[1] if data_example.x_cat is not None else 0

    def _build_data(self, df_chunk):
        return self.to_graph_data_fn(
            df_chunk,
            edge_keys=self.edge_keys,
            target_col=self.target_col,
            ignore_cols=self.ignore_cols,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
        )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        stop = min(len(self.df), (idx + 1) * self.chunk_size)
        df_chunk = self.df.iloc[start:stop].reset_index(drop=True)
        return self._build_data(df_chunk)

    def get_feature_dim(self):
        return self.num_num_features, self.num_cat_features


def suggest_emb_dim(cardinality):
    return min(600, round(1.6 * cardinality ** 0.56))

# 3. Ğ�Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚ĞºĞ° Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
def make_dataset(df, config):
    return FraudGraphChunkDataset(
        df[config["columns"]],
        chunk_size=config["chunk_size"],
        edge_keys=config["edge_keys"],
        target_col=config["target_col"],
        ignore_cols=config["ignore_cols"],
        num_cols=config["num_features"],
        cat_cols=config["cat_features"]
    )


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm
from torch_geometric.nn import GATv2Conv, JumpingKnowledge

class FraudGANClassifierV1(nn.Module):
    def __init__(self, num_num_features, cat_cardinalities, emb_dims, hidden_dim, output_dim=1, dropout=0.3):
        super().__init__()
        # Ğ­Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ¸ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ…
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, emb_dims)
        ])
        input_dim = num_num_features + sum(emb_dims)

        self.gat1 = GATv2Conv(input_dim,  hidden_dim, heads=4, dropout=dropout)
        self.bn1 = LayerNorm(hidden_dim * 4)
        self.gat2 = GATv2Conv(hidden_dim*4, hidden_dim, heads=2, dropout=dropout)
        self.bn2 = LayerNorm(hidden_dim*2)
        self.gat3 = GATv2Conv(hidden_dim*2, hidden_dim, heads=1, dropout=dropout)
        self.bn3 = LayerNorm(hidden_dim)
        
        # Jumping Knowledge aggregator (concat all layers)
        self.jk = JumpingKnowledge("cat")


        self.fc1 = nn.Linear(hidden_dim*(4+2+1), 32)   # 4+2+1 = 7
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x_num = data.x_num
        x_cat = data.x_cat

        #print(f"x_cat.shape: {x_cat.shape}")  # Ğ´Ğ¾Ğ»Ğ¶Ğ½Ğ¾ Ğ±Ñ‹Ñ‚ÑŒ (batch_size, 11)
        #print("emb_layers:", len(self.emb_layers))  # Ğ´Ğ¾Ğ»Ğ¶Ğ½Ğ¾ Ğ±Ñ‹Ñ‚ÑŒ 11


        emb_list = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x_cat_emb = torch.cat(emb_list, dim=1) if emb_list else None

        if x_cat_emb is not None:
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        edge_index = data.edge_index


        # GAT Stack + residuals
        h1 = F.elu(self.bn1(self.gat1(x, edge_index)))
        h2 = F.elu(self.bn2(self.gat2(h1, edge_index)))
        h3 = F.elu(self.bn3(self.gat3(h2, edge_index)))

        # Jumping Knowledge concatenation
        h = self.jk([h1, h2, h3])


        x = self.dropout(h)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm
from torch_geometric.nn import GATv2Conv, JumpingKnowledge

class FraudGANClassifierV2(nn.Module):
    def __init__(self, num_num_features, cat_cardinalities, emb_dims, hidden_dim, output_dim=1, dropout=0.3):
        super().__init__()
        # Ğ­Ğ¼Ğ±ĞµĞ´Ğ´Ğ¸Ğ½Ğ³Ğ¸ ĞºĞ°Ñ‚ĞµĞ³Ğ¾Ñ€Ğ¸Ğ°Ğ»ÑŒĞ½Ñ‹Ñ…
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, emb_dims)
        ])
        input_dim = num_num_features + sum(emb_dims)

        self.gat1 = GATv2Conv(input_dim,  hidden_dim, heads=8, dropout=dropout)
        self.bn1 = LayerNorm(hidden_dim * 8)
        self.gat2 = GATv2Conv(hidden_dim*8,  hidden_dim, heads=4, dropout=dropout)
        self.bn2 = LayerNorm(hidden_dim * 4)
        self.gat3 = GATv2Conv(hidden_dim*4, hidden_dim, heads=2, dropout=dropout)
        self.bn3 = LayerNorm(hidden_dim*2)
        self.gat4 = GATv2Conv(hidden_dim*2, hidden_dim, heads=1, dropout=dropout)
        self.bn4 = LayerNorm(hidden_dim)
        
        # Jumping Knowledge aggregator (concat all layers)
        self.jk = JumpingKnowledge("cat")


        self.fc1 = nn.Linear(hidden_dim*(8+4+2+1), 64)   # 4+2+1 = 7
        self.fc2 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x_num = data.x_num
        x_cat = data.x_cat

        #print(f"x_cat.shape: {x_cat.shape}")  # Ğ´Ğ¾Ğ»Ğ¶Ğ½Ğ¾ Ğ±Ñ‹Ñ‚ÑŒ (batch_size, 11)
        #print("emb_layers:", len(self.emb_layers))  # Ğ´Ğ¾Ğ»Ğ¶Ğ½Ğ¾ Ğ±Ñ‹Ñ‚ÑŒ 11


        emb_list = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x_cat_emb = torch.cat(emb_list, dim=1) if emb_list else None

        if x_cat_emb is not None:
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        edge_index = data.edge_index


        # GAT Stack + residuals
        h1 = F.elu(self.bn1(self.gat1(x, edge_index)))
        h2 = F.elu(self.bn2(self.gat2(h1, edge_index)))
        h3 = F.elu(self.bn3(self.gat3(h2, edge_index)))
        h4 = F.elu(self.bn4(self.gat4(h3, edge_index)))

        # Jumping Knowledge concatenation
        h = self.jk([h1, h2, h3, h4])


        x = self.dropout(h)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze()





CONFIG = {
    "num_features": ["log_TransactionAmt"],
    "cat_features": ["card1", "card4"],
    "columns": [
        "TransactionID", "TransactionDT", "isFraud", "card1", "card4", "log_TransactionAmt"
    ],
    "chunk_size": 512,
    "edge_keys": ["card1", "card4"],
    "target_col": "isFraud",
    "ignore_cols": ["TransactionID", "TransactionDT", "isFraud"],
    "hidden_dim": 64,
    "output_dim": 1,
    "lr": 1e-3,
    "epochs": EPOCHS,
    "batch_size": 4
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_dataset = make_dataset(base_train, CONFIG)
test_dataset = make_dataset(base_test, CONFIG)


cat_cardinalities = [len(pipeline['preprocessor'].label_encoders[col].classes_) for col in CONFIG["cat_features"]]
emb_dims = [suggest_emb_dim(card) for card in cat_cardinalities]


model = FraudGANClassifierV1(
    len(CONFIG["num_features"]),
    cat_cardinalities,
    emb_dims,
    hidden_dim=CONFIG["hidden_dim"],
    output_dim=CONFIG["output_dim"]
).to(DEVICE)


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)



from collections import Counter
import torch

cnt = Counter(train_dataset.df.isFraud)
num_pos, num_neg = cnt[1], cnt[0]
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])



trained_model, val_metrics = train_gnn_antifraud_model(
    model, 
    train_loader=train_loader, 
    val_loader=test_loader,
    n_epochs=CONFIG["epochs"],
    lr=CONFIG["lr"],
    criterion=criterion,
    plot_curves=True,
    device=DEVICE
)

print('Training complete. Validation metrics:', val_metrics)


columns = [
    "TransactionID",    # Ğ¾Ğ±Ñ�Ğ·Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾
    "TransactionDT",    # Ğ¾Ğ±Ñ�Ğ·Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾
    "isFraud",          # Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚ (Ğ½Ğµ Ñ‡Ğ°Ñ�Ñ‚ÑŒ x)
    'ProductCD',
    'card1', 
    'card2', 
    'card3', 
    'card4', 
    'card5', 
    'card6', 
    'addr1', 
    'addr2',
    'P_emaildomain',
    #'TransactionAmt_binned', 
    'Transaction_day', 
    'Transaction_hour',
    'Transaction_weekday', 
    #'isOutlier',
    'TransactionAmt',
    "log_TransactionAmt",
    'TransactionAmt_to_mean_card1',
       'TransactionAmt_to_std_card1', 'TransactionAmt_to_mean_card4',
       'TransactionAmt_to_std_card4',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
       'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15',
    'V_PCA_0', 'V_PCA_1', 'V_PCA_2',
       'V_PCA_3', 'V_PCA_4', 'V_PCA_5', 'V_PCA_6', 'V_PCA_7', 'V_PCA_8',
       'V_PCA_9', 'V_PCA_10', 'V_PCA_11', 'V_PCA_12', 'V_PCA_13', 'V_PCA_14',
       'V_PCA_15', 'V_PCA_16', 'V_PCA_17', 'V_PCA_18', 'V_PCA_19', 'V_PCA_20',
       'V_PCA_21', 'V_PCA_22', 'V_PCA_23', 'V_PCA_24', 'V_PCA_25', 'V_PCA_26',
       'V_PCA_27', 'V_PCA_28', 'V_PCA_29', 'V_PCA_30', 'V_PCA_31', 'V_PCA_32',
       'V_PCA_33', 'V_PCA_34', 'V_PCA_35', 'V_PCA_36', 'V_PCA_37'
]

cat_features = [
    'ProductCD',
    'card1', 
    'card2', 
    'card3', 
    'card4', 
    'card5', 
    'card6', 
    'addr1', 
    'addr2',
    'P_emaildomain',
    #'TransactionAmt_binned', 
]

num_features = [
    'Transaction_day', 
    'Transaction_hour',
    'Transaction_weekday',
    #'isOutlier',
    'TransactionAmt',
    "log_TransactionAmt",
    'TransactionAmt_to_mean_card1',
       'TransactionAmt_to_std_card1', 'TransactionAmt_to_mean_card4',
       'TransactionAmt_to_std_card4',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
       'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15',
    'V_PCA_0', 'V_PCA_1', 'V_PCA_2',
       'V_PCA_3', 'V_PCA_4', 'V_PCA_5', 'V_PCA_6', 'V_PCA_7', 'V_PCA_8',
       'V_PCA_9', 'V_PCA_10', 'V_PCA_11', 'V_PCA_12', 'V_PCA_13', 'V_PCA_14',
       'V_PCA_15', 'V_PCA_16', 'V_PCA_17', 'V_PCA_18', 'V_PCA_19', 'V_PCA_20',
       'V_PCA_21', 'V_PCA_22', 'V_PCA_23', 'V_PCA_24', 'V_PCA_25', 'V_PCA_26',
       'V_PCA_27', 'V_PCA_28', 'V_PCA_29', 'V_PCA_30', 'V_PCA_31', 'V_PCA_32',
       'V_PCA_33', 'V_PCA_34', 'V_PCA_35', 'V_PCA_36', 'V_PCA_37'
]

edge_keys = ['card1', 'card4', 'addr1', 'P_emaildomain']


CONFIG = {
    "num_features": num_features,
    "cat_features": cat_features,
    "columns": columns,
    "chunk_size": 512,
    "edge_keys": edge_keys,
    "target_col": "isFraud",
    "ignore_cols": ["TransactionID", "TransactionDT", "isFraud"],
    "hidden_dim": 256,
    "output_dim": 1,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "epochs": EPOCHS,
    "batch_size": 4
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


base_train.columns


base_train[['TransactionAmt_binned']].head()


train_dataset = make_dataset(base_train, CONFIG)
test_dataset = make_dataset(base_test, CONFIG)


cat_cardinalities = [len(pipeline['preprocessor'].label_encoders[col].classes_) for col in CONFIG["cat_features"]]
emb_dims = [suggest_emb_dim(card) for card in cat_cardinalities]


model = FraudGANClassifierV1(
    len(CONFIG["num_features"]),
    cat_cardinalities,
    emb_dims,
    hidden_dim=CONFIG["hidden_dim"],
    output_dim=CONFIG["output_dim"]
).to(DEVICE)


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)


from collections import Counter
import torch

cnt = Counter(train_dataset.df.isFraud)
num_pos, num_neg = cnt[1], cnt[0]
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])


trained_model, val_metrics = train_gnn_antifraud_model(
    model, 
    train_loader=train_loader, 
    val_loader=test_loader,
    n_epochs=CONFIG["epochs"],
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
    criterion=criterion,
    plot_curves=True,
    device=DEVICE
)

print('Training complete. Validation metrics:', val_metrics)


trained_model, val_metrics = train_gnn_antifraud_model(
    model, 
    train_loader=train_loader, 
    val_loader=test_loader,
    n_epochs=CONFIG["epochs"],
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
    criterion=criterion,
    key_metric='precision',
    threshold_selection_strategy='fixed_recall',
    threshold_params={'target_recall': 0.95},
    plot_curves=True,
    device=DEVICE
)

print('Training complete. Validation metrics:', val_metrics)


from torch_geometric.nn import SAGEConv, LayerNorm, JumpingKnowledge

class FraudSAGEClassifierV1(nn.Module):
    def __init__(self, num_num_features, cat_cardinalities, emb_dims, hidden_dim, output_dim=1, dropout=0.3):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim)
            for cardinality, emb_dim in zip(cat_cardinalities, emb_dims)
        ])
        input_dim = num_num_features + sum(emb_dims)

        self.sage1 = SAGEConv(input_dim,  hidden_dim * 4)
        self.bn1 = LayerNorm(hidden_dim * 4)
        self.sage2 = SAGEConv(hidden_dim * 4,  hidden_dim * 2)
        self.bn2 = LayerNorm(hidden_dim * 2)
        self.sage3 = SAGEConv(hidden_dim * 2, hidden_dim)
        self.bn3 = LayerNorm(hidden_dim)
        
        # Jumping Knowledge aggregator (concat all layers)
        self.jk = JumpingKnowledge("cat")

        self.fc1 = nn.Linear(hidden_dim*(4+2+1), 32)   # 4+2+1 = 7
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x_num = data.x_num
        x_cat = data.x_cat

        emb_list = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x_cat_emb = torch.cat(emb_list, dim=1) if emb_list else None

        if x_cat_emb is not None:
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        edge_index = data.edge_index

        h1 = F.elu(self.bn1(self.sage1(x, edge_index)))
        h2 = F.elu(self.bn2(self.sage2(h1, edge_index)))
        h3 = F.elu(self.bn3(self.sage3(h2, edge_index)))

        h = self.jk([h1, h2, h3])

        x = self.dropout(h)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze()


columns = [
    "TransactionID",    # Ğ¾Ğ±Ñ�Ğ·Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾
    "TransactionDT",    # Ğ¾Ğ±Ñ�Ğ·Ğ°Ñ‚ĞµĞ»ÑŒĞ½Ğ¾
    "isFraud",          # Ñ‚Ğ°Ñ€Ğ³ĞµÑ‚ (Ğ½Ğµ Ñ‡Ğ°Ñ�Ñ‚ÑŒ x)
    'ProductCD',
    'card1', 
    'card2', 
    'card3', 
    'card4', 
    'card5', 
    'card6', 
    'addr1', 
    'addr2',
    'P_emaildomain',
    'TransactionAmt_binned', 
    'Transaction_day', 
    'Transaction_hour',
    'Transaction_weekday', 
    #'isOutlier',
    'TransactionAmt',
    "log_TransactionAmt",
    'TransactionAmt_to_mean_card1',
       'TransactionAmt_to_std_card1', 'TransactionAmt_to_mean_card4',
       'TransactionAmt_to_std_card4',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
       'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15',
    'V_PCA_0', 'V_PCA_1', 'V_PCA_2',
       'V_PCA_3', 'V_PCA_4', 'V_PCA_5', 'V_PCA_6', 'V_PCA_7', 'V_PCA_8',
       'V_PCA_9', 'V_PCA_10', 'V_PCA_11', 'V_PCA_12', 'V_PCA_13', 'V_PCA_14',
       'V_PCA_15', 'V_PCA_16', 'V_PCA_17', 'V_PCA_18', 'V_PCA_19', 'V_PCA_20',
       'V_PCA_21', 'V_PCA_22', 'V_PCA_23', 'V_PCA_24', 'V_PCA_25', 'V_PCA_26',
       'V_PCA_27', 'V_PCA_28', 'V_PCA_29', 'V_PCA_30', 'V_PCA_31', 'V_PCA_32',
       'V_PCA_33', 'V_PCA_34', 'V_PCA_35', 'V_PCA_36', 'V_PCA_37'
]

cat_features = [
    'ProductCD',
    'card1', 
    'card2', 
    'card3', 
    'card4', 
    'card5', 
    'card6', 
    'addr1', 
    'addr2',
    'P_emaildomain',
    #'TransactionAmt_binned', 
]

num_features = [
    'Transaction_day', 
    'Transaction_hour',
    'Transaction_weekday',
    #'isOutlier',
    'TransactionAmt',
    "log_TransactionAmt",
    'TransactionAmt_to_mean_card1',
       'TransactionAmt_to_std_card1', 'TransactionAmt_to_mean_card4',
       'TransactionAmt_to_std_card4',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
       'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D4', 'D10', 'D15',
    'V_PCA_0', 'V_PCA_1', 'V_PCA_2',
       'V_PCA_3', 'V_PCA_4', 'V_PCA_5', 'V_PCA_6', 'V_PCA_7', 'V_PCA_8',
       'V_PCA_9', 'V_PCA_10', 'V_PCA_11', 'V_PCA_12', 'V_PCA_13', 'V_PCA_14',
       'V_PCA_15', 'V_PCA_16', 'V_PCA_17', 'V_PCA_18', 'V_PCA_19', 'V_PCA_20',
       'V_PCA_21', 'V_PCA_22', 'V_PCA_23', 'V_PCA_24', 'V_PCA_25', 'V_PCA_26',
       'V_PCA_27', 'V_PCA_28', 'V_PCA_29', 'V_PCA_30', 'V_PCA_31', 'V_PCA_32',
       'V_PCA_33', 'V_PCA_34', 'V_PCA_35', 'V_PCA_36', 'V_PCA_37'
]

edge_keys = ['card1', 'card4', 'addr1', 'P_emaildomain']


CONFIG = {
    "num_features": num_features,
    "cat_features": cat_features,
    "columns": columns,
    "chunk_size": 512,
    "edge_keys": edge_keys,
    "target_col": "isFraud",
    "ignore_cols": ["TransactionID", "TransactionDT", "isFraud"],
    "hidden_dim": 64,
    "output_dim": 1,
    "lr": 1e-3,
    "epochs": EPOCHS,
    "batch_size": 4
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset = make_dataset(base_train, CONFIG)
test_dataset = make_dataset(base_test, CONFIG)


cat_cardinalities = [len(pipeline['preprocessor'].label_encoders[col].classes_) for col in CONFIG["cat_features"]]
emb_dims = [suggest_emb_dim(card) for card in cat_cardinalities]




model = FraudSAGEClassifierV1(
    len(CONFIG["num_features"]),
    cat_cardinalities,
    emb_dims,
    hidden_dim=CONFIG["hidden_dim"],
    output_dim=CONFIG["output_dim"]
).to(DEVICE)


from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)


from collections import Counter
import torch

cnt = Counter(train_dataset.df.isFraud)
num_pos, num_neg = cnt[1], cnt[0]
pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float, device=DEVICE)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])


trained_model, val_metrics = train_gnn_antifraud_model(
    model, train_loader, val_loader=test_loader,
    n_epochs=CONFIG["epochs"],
    lr=CONFIG["lr"],
    criterion=criterion,
    plot_curves=True,
    device=DEVICE
)

print('Training complete. Validation metrics:', val_metrics)

