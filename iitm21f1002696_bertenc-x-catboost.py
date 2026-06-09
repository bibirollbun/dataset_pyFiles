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


from transformers import AutoTokenizer, AutoModel


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from lightgbm import early_stopping,log_evaluation,LGBMClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
tqdm.pandas()


path="/kaggle/input/wsdm-cup-multilingual-chatbot-arena/"
train = pd.read_parquet(path+"train.parquet")
test = pd.read_parquet(path+"test.parquet")
sub = pd.read_csv(path+"sample_submission.csv")


train.head()


test.head()


# 10% as validation split, this percentage could be changed
train,valid=train_test_split(train,test_size=0.1,stratify=train["winner"],random_state=42)

# Train set can be inverted (and winner too) to get twice the data from the available training dataset
train_inv=train.copy()
train_inv["response_a"],train_inv["response_b"]=train_inv["response_b"],train_inv["response_a"]
train_inv["winner"]=train_inv["winner"].apply(lambda x: "model_a" if "b" in x else "model_b")


# Here I compute some features
def compute_feats(df):
    for col in tqdm(["response_a","response_b","prompt"]):
        # response lenght is a key factor when choosing between two responses
        df[f"{col}_len"]=df[f"{col}"].str.len()

        # Some characters counting features 
        df[f"{col}_spaces"]=df[f"{col}"].str.count("\s")
        df[f"{col}_punct"]=df[f"{col}"].str.count(",|\.|!")
        df[f"{col}_question_mark"]=df[f"{col}"].str.count("\?")
        df[f"{col}_quot"]=df[f"{col}"].str.count("'|\"")
        df[f"{col}_formatting_chars"]=df[f"{col}"].str.count("\*|\_")
        df[f"{col}_math_chars"]=df[f"{col}"].str.count("\-|\+|\=")
        df[f"{col}_curly_open"]=df[f"{col}"].str.count("\{")
        df[f"{col}_curly_close"]=df[f"{col}"].str.count("}")
        df[f"{col}_round_open"]=df[f"{col}"].str.count("\(")
        df[f"{col}_round_close"]=df[f"{col}"].str.count("\)")
        df[f"{col}_accent_chars"]=df[f"{col}"].str.count("Ã¨|Ã²|Ã |Ã¹|Ã©|Ã¬")
        df[f"{col}_special_chars"]=df[f"{col}"].str.count("\W")
        df[f"{col}_digits"]=df[f"{col}"].str.count("\d")/df[f"{col}_len"]
        df[f"{col}_lower"]=df[f"{col}"].str.count("[a-z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_upper"]=df[f"{col}"].str.count("[A-Z]").astype("float32")/df[f"{col}_len"]
        df[f"{col}_chinese"]=df[f"{col}"].str.count(r'[\u4e00-\u9fff]+').astype("float32")/df[f"{col}_len"]
        df[f"{col}_tild"]=df[f"{col}"].str.count("~")>0

        # Feature that show how balanced are curly and round brackets
        df[f"{col}_round_balance"]=df[f"{col}_round_open"]-df[f"{col}_round_close"]
        df[f"{col}_curly_balance"]=df[f"{col}_curly_open"]-df[f"{col}_curly_close"]

        # Feature that tells if the string json is present somewhere (e.g. asking a json response or similar)
        df[f"{col}_json"]=df[f"{col}"].str.lower().str.count("json")
        df[f"{col}_yaml"]=df[f"{col}"].str.lower().str.count("yaml")

    return df
train=compute_feats(train)
train_inv=compute_feats(train_inv)

train=pd.concat([train,train_inv])
valid=compute_feats(valid)
test=compute_feats(test)


train.head()


vectorizer_char = TfidfVectorizer(sublinear_tf=True, analyzer='char', ngram_range=(1,2), max_features=100_000)
vectorizer_word = TfidfVectorizer(sublinear_tf=True, analyzer='word', min_df=3)
preprocessor = ColumnTransformer(
    transformers=[
        ('prompt_feats', FeatureUnion([
            ('prompt_char', vectorizer_char),
            ('prompt_word', vectorizer_word)
        ]), 'prompt'),
        ('response_a_feats', FeatureUnion([
            ('response_a_char', vectorizer_char),
            ('response_a_word', vectorizer_word)
        ]), 'response_a'),
        ('response_b_feats', FeatureUnion([
            ('response_b_char', vectorizer_char),
            ('response_b_word', vectorizer_word)
        ]), 'response_b')
    ]
)
train_feats = preprocessor.fit_transform(train[["response_a","response_b","prompt"]])
test_feats = preprocessor.transform(test[["response_a","response_b","prompt"]])
valid_feats = preprocessor.transform(valid[["response_a","response_b","prompt"]])


model = LogisticRegression(C=0.1, solver='liblinear', dual=True, random_state=42)
model.fit(train_feats, train.winner)


model.predict_proba(test_feats)


# !pip install pandarallel


from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_cosine_similarity_feature(df, feature1, feature2, new_feature_name):
    def compute_cosine_similarity(row):
        try:
            if not row[feature1] or not row[feature2]:
                return 0.0  

            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([row[feature1], row[feature2]])            

            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
        except:
            return 0
    
    df[new_feature_name] = df.parallel_apply(compute_cosine_similarity, axis=1)
    return df


train = add_cosine_similarity_feature(train, 'prompt', 'response_a', 'response_a_sim')
train = add_cosine_similarity_feature(train, 'prompt', 'response_b', 'response_b_sim')

valid = add_cosine_similarity_feature(valid, 'prompt', 'response_a', 'response_a_sim')
valid = add_cosine_similarity_feature(valid, 'prompt', 'response_b', 'response_b_sim')


train.columns


feats=list(train.columns)[8:]
train["winner"]=(train["winner"]=="model_a").astype("int")
valid["winner"]=(valid["winner"]=="model_a").astype("int")


text_cols = ['prompt', 'response_a', 'response_b']


train[text_cols]


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm


if torch.cuda.device_count() > 1:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
else:
    print("Multiple GPUs not available.")



def encode_and_store_embeddings(df, text_columns, batch_size=16):
    """
    Encode texts and store embeddings directly in the dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_columns (list): List of column names to encode
        model_name (str): Name of the pretrained model
        batch_size (int): Batch size for processing
    
    Returns:
        pd.DataFrame: DataFrame with new embedding columns
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_with_embeddings = df.copy()
    
    # Initialize tokenizer and model
    model = AutoModel.from_pretrained("/kaggle/input/bert-multilingual/transformers/bert/1/bert_multilingual_model")
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/bert-multilingual-tokenizer/bert_multilingual_tokenizer")
    
    # Move model to GPU and enable parallel processing
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    
    # Process each text column
    for col in text_columns:
        print(f"Processing column: {col}")
        embeddings_list = []
        
        # Create dataloader for current column
        texts = df[col].tolist()
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to GPU
            input_ids = encoded['input_ids'].cuda()
            attention_mask = encoded['attention_mask'].cuda()
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Get CLS token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.extend(batch_embeddings)
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings_list)
        
        # Store embeddings in dataframe
        embedding_column_name = f"{col}_embeddings"
        df_with_embeddings[embedding_column_name] = embeddings_list
        
        print(f"Added embeddings column: {embedding_column_name}")
        print(f"Embedding shape for {col}: {embeddings_array.shape}")
    
    return df_with_embeddings


train_df_with_embeddings = encode_and_store_embeddings(
    df=train,
    text_columns=text_cols,
    batch_size=24
)


val_train_df_with_embeddings = encode_and_store_embeddings(
    df=valid,
    text_columns=text_cols,
    batch_size=24
)


train_df_with_embeddings.head()


from catboost import CatBoostClassifier


train_df_with_embeddings.columns


relevant_cols = train_df_with_embeddings.columns[8:]


X_train = train_df_with_embeddings[relevant_cols]
y_train = train_df_with_embeddings['winner']


X_val = val_train_df_with_embeddings[relevant_cols]
y_val = val_train_df_with_embeddings['winner']


def prepare_bert_embeddings_for_catboost(df, embedding_columns, relevant_cols, target_column=None, processing_test_data=False):
    """
    Prepare features for CatBoost (by exploding the embedding vectors into their own columns)

    Each BERT embedded text is a vector of dim 768. These 768 values (in each vector) will be placed in their own columns 
    (thus there will be 768 columns for each embedding vector). Here, there are 3 such text embedding vectors in each row, 
    thus in total there will be 768 * 3 total extra columns per row.
    
    Args:
        df: DataFrame containing the data
        embedding_columns: List of column names containing BERT embeddings
        relevant_cols: List of column names of all feature (including embedding columns)
        target_column: Name of the target variable column
        processing_test_data: Bool; whether preprocessing test data or not
    
    Returns:
        Processed features DataFrame, target series, feature names list
    """
    processed_features = {}
    feature_names = []
    
    # Process each column in relevant_cols
    for col in relevant_cols:
        if col in embedding_columns:
            # Handle embedding columns
            embeddings = np.vstack(df[col].values) # vertically stack the embeddings present in 'col'; row i -> Embedding vec of the ith text; col i -> ith entries of Embedding vec of all texts present in 'col' 
            # vstack ed embeddings ðŸ‘‡
            # embeddings = [[embedding_vec_1], => each vec is of dim 768 (BERT's d_model) 
            #               [embedding_vec_2],
            #               .................
            #               [embedding_vec_n]] ; n -> total no of data
            
            # Create feature names for this embedding
            col_feature_names = [f"{col}_dim_{i}" for i in range(embeddings.shape[1])] # Shape is <Total_num_data, 768 (bert hidden layer embedding size)>
            feature_names.extend(col_feature_names)
            # Iterate over all 768 dims and store the columns, that is store the ith entries of all the embedded text vectors of the entire data
            # 0th iter of 'prompt' -> 'prompt'_dim_'0' = [embedded_text_1[0], embedded_text_2[0], ..... , embedded_text_n[0]]
            # ...
            # 80th iter of 'response_b' -> 'prompt'_dim_'80' = [embedded_text_1[80], embedded_text_2[80], ..... , embedded_text_n[80]]
            for i, name in enumerate(col_feature_names):
                processed_features[name] = embeddings[:, i]
        else:
            # Handle non-embedding columns
            feature_names.append(col)
            processed_features[col] = df[col].values
    
    # Create DataFrame with all processed features
    X = pd.DataFrame(processed_features)
    
    # Get target variable
    if not processing_test_data:
        y = df[target_column]
    else:
        y = None
    
    return X, y, feature_names


# np.vstack(train_df_with_embeddings['prompt_embeddings'].values)


# np.vstack(train_df_with_embeddings['prompt_embeddings'].values)[:,0]


embedding_columns = [
    'prompt_embeddings',
    'response_a_embeddings',
    'response_b_embeddings'
]


X_train, y_train, feature_names = prepare_bert_embeddings_for_catboost(
    df=train_df_with_embeddings,
    embedding_columns=embedding_columns,
    relevant_cols=relevant_cols,  # includes both embedding and other feature columns
    target_column='winner'
)


X_val, y_val, _ = prepare_bert_embeddings_for_catboost(
    df=val_train_df_with_embeddings,
    embedding_columns=embedding_columns,
    relevant_cols=relevant_cols,
    target_column='winner'
)


# import optuna
from sklearn.metrics import accuracy_score, classification_report


# def train_catboost(
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_val: pd.DataFrame,
#     y_val: pd.Series,
#     depth: int,
#     l2_leaf_reg: float,
#     random_strength: float,
#     od_wait: int,
#     od_type: str,
#     leaf_estimation_iterations: int,
#     grow_policy: str,
#     min_data_in_leaf: int,
#     leaf_estimation_method: str,
#     num_trees: int,
# ) -> float:
#     """
#     This function trains a Catboost model on the given data and returns the trained model.

#     Args:
#         X_train (pd.DataFrame): The training features
#         y_train (pd.Series): The training target
#         X_val (pd.DataFrame): The test features
#         y_val (pd.Series): The test target
#         depth (int): The depth of the trees
#         l2_leaf_reg (float): The L2 regularization coefficient
#         random_strength (float): The random strength
#         od_wait (int): The number of iterations to wait for the metric to improve
#         od_type (str): The type of the overfitting detector
#         leaf_estimation_iterations (int): The number of iterations to build the leaves
#         grow_policy (str): The grow policy
#         min_data_in_leaf (int): The minimum number of samples in a leaf
#         leaf_estimation_method (str): The method to estimate the leaves
#         num_trees (int): The number of trees in the model

#     Returns:
#         CatBoostClassifier: The trained Catboost model
#     """
#     # Catboost model
#     clf = CatBoostClassifier(
#         depth=depth,
#         l2_leaf_reg=l2_leaf_reg,
#         random_strength=random_strength,
#         od_wait=od_wait,
#         od_type=od_type,
#         leaf_estimation_iterations=leaf_estimation_iterations,
#         grow_policy=grow_policy,
#         min_data_in_leaf=min_data_in_leaf,
#         leaf_estimation_method=leaf_estimation_method,
#         num_trees=num_trees,
#         verbose=5000,
#         task_type="GPU",
#         devices='0:1',
#         loss_function="MultiClass",
#     )

#     clf.fit(X_train, y_train)

#     # ? Predict the labels of the test set and compute the accuracy score
#     y_pred = clf.predict(X_val)
#     acc = accuracy_score(y_val, y_pred)

#     return acc


# def objective(trial: optuna.Trial) -> float:
#     """
#     This function defines the objective of the optimization problem.
#     It takes a trial object and returns the value of the objective function
#     for the given hyperparameters.


#     Args:
#         trial (optuna.Trial): A trial object that contains the hyperparameters to be sampled

#     Returns:
#         float: The function returns the accuracy score of the model with the sampled hyperparameters
#     """
#     # ? Sample the hyperparameters from the trial object
#     depth = trial.suggest_int("depth", 6, 10)
#     l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 5)
#     random_strength = trial.suggest_float("random_strength", 0, 1)
#     od_wait = trial.suggest_int("od_wait", 10, 30)
#     od_type = trial.suggest_categorical("od_type", ["IncToDec"])
#     leaf_estimation_iterations = trial.suggest_int("leaf_estimation_iterations", 1, 10)
#     grow_policy = trial.suggest_categorical(
#         "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
#     )
#     min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 3)
#     leaf_estimation_method = trial.suggest_categorical(
#         "leaf_estimation_method", ["Newton", "Gradient"]
#     )
#     num_trees = trial.suggest_int("num_trees", 8500, 10500)
#     acc = train_catboost(
#         X_train,
#         y_train,
#         X_val,
#         y_val,
#         depth,
#         l2_leaf_reg,
#         random_strength,
#         od_wait,
#         od_type,
#         leaf_estimation_iterations,
#         grow_policy,
#         min_data_in_leaf,
#         leaf_estimation_method,
#         num_trees,
#     )

#     # ? Return the accuracy score as the objective value
#     return acc


# # ? Create a study object and optimize the objective function
# study = optuna.create_study(
#     direction="maximize",
#     pruner=optuna.pruners.HyperbandPruner(
#         min_resource=1, max_resource=100, reduction_factor=3
#     ),
# )


# study.optimize(objective, n_trials=50)


# trial = study.best_trial
# print("  Value: {}".format(trial.value))


# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))


# type(trial.params)


# params = trial.params


# Best Hyperparams obtained from optuna
params = {
    "depth": 7,
    "l2_leaf_reg": 1.6310394443532703,
    "random_strength": 0.4003459196395534,
    "od_wait": 18,
    "od_type": "IncToDec",
    "leaf_estimation_iterations": 1,
    "grow_policy": "Lossguide",
    "min_data_in_leaf": 1,
    "leaf_estimation_method": "Gradient",
    "num_trees": 9167,
}


# !pip install --upgrade catboost


clf = CatBoostClassifier(
    **params,
    loss_function='CrossEntropy',
    random_seed=42,
    task_type='GPU',
    devices='0:1',
    verbose=1000,
)


clf.fit(
    X_train,
    y_train,
)


y_pred_val = clf.predict(X_val)


print(classification_report(y_val, y_pred_val, digits=6))





test = add_cosine_similarity_feature(test, 'prompt', 'response_a', 'response_a_sim')
test = add_cosine_similarity_feature(test, 'prompt', 'response_b', 'response_b_sim')


test_df_with_embeddings = encode_and_store_embeddings(
    df=test,
    text_columns=text_cols,
    batch_size=24
)


X_test, _, _ = prepare_bert_embeddings_for_catboost(
    df=test_df_with_embeddings,
    embedding_columns=embedding_columns,
    relevant_cols=relevant_cols,  # includes both embedding and other feature columns
    target_column='winner',
    processing_test_data=True
)


X_test


X_test['winner'] = clf.predict(X_test)


X_test["id"] = test_df_with_embeddings["id"]


X_test["winner"]=X_test["winner"].apply(lambda x: "model_a" if x==1 else "model_b")

sub=X_test[["id","winner"]]


sub.head()


sub.to_csv("submission.csv",index=False)




