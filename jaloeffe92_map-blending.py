%%capture
!pip install /kaggle/input/lightgbm-cuda-compile/LightGBM/dist/lightgbm-4.6.0.99-py3-none-linux_x86_64.whl


%load_ext cudf.pandas
from out_of_fold_helper import kfold_split
from path_helper import generate_base_path, generate_fold_path_for_model, joblib_dump_model
import numpy as np
import cudf
import cuml
import pandas as pd
import sklearn
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import cupyx
from cuml.ensemble import RandomForestClassifier as cuRFC
from lightgbm import LGBMClassifier
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DebertaTokenizer, DebertaForSequenceClassification, DebertaV2ForSequenceClassification
#from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import os
import torch
from sklearn.model_selection import train_test_split
import joblib
import itertools
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import cloudpickle as cp

import scipy.sparse
from scipy.sparse.linalg import norm
from scipy.sparse import vstack
import cloudpickle

print('RAPIDS',cuml.__version__)


import gc
import os


TARGET_COLUMN_ONE="target1"
CALCULATE_BLENDING=True
TARGET_COLUMN_TWO="target2"
EMBEDDING="emb"
TEXT_FOR_LLM="text"
MODELS="models"
DEBERTA_SMALL_LOAD_PATH="/kaggle/input/huggingfacedebertav3variants/deberta-v3-xsmall"
DEBERTA_CHECKPOINT = "/kaggle/input/map-llm-models/map-blending-models/kaggle/input/huggingfacedebertav3variants/deberta-v3-xsmall"
DEBERTA_MODEL_NAME="deberta-v3-xsmall"
QWEN_MODEL_NAME="qwn"
QWEN_LOAD_PATH = "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL"
K_FOLD_RESULT="kfold_result.bin"
LGB="lgb"
XGB="xgb"
ETTIN="etting"
LOGISTIC="logistic_regression"
DEBERTA_SMALL="deberta-v3-xsmall"
BASE_PATH="/kaggle/input/map-models/archive"
DEBERTA_LOAD_PATH="/kaggle/input/ettin-encoder-1b-cv943"
MODEL_PATH=f"{BASE_PATH}/models"
N_SPLITS = 5
EPOCHS = 10
VER=1
#DIR = f"ver_{VER}"
BEST_CV_VAL=0.9264088729016768
TRAIN_OTHER=False
TRAIN_LLM=False
TRAIN_XGB=False
DETERMINE_BEST_COMBINATION=True
#os.makedirs(DIR, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)


#!rsync -r /kaggle/input/map-llm-models/map-blending-models/kaggle /kaggle/working/map-blending-models


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


ETTIN_TOKENIZER=AutoTokenizer.from_pretrained(DEBERTA_LOAD_PATH)


DEBERTA_TOKENIZER = AutoTokenizer.from_pretrained(DEBERTA_SMALL_LOAD_PATH)
#QWEN_TOKENIZER = AutoTokenizer.from_pretrained(QWEN_LOAD_PATH)
MAX_LEN = 256


train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)


def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train[TEXT_FOR_LLM] = train.apply(format_input,axis=1)
#test[TEXT_FOR_LLM] = test.apply(format_input,axis=1)


test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test[TEXT_FOR_LLM] = test.apply(format_input,axis=1)


train['Misconception'] = train['Misconception'].fillna('NA')
train['Misconception'] = train['Misconception'].map(str)
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(train.shape, test.shape)
#train.head()


map_target1 = train['Category'].value_counts().to_frame()
map_target1['count'] = np.arange(len(map_target1))
map_target1 = map_target1.to_dict()['count']

map_target2 = train['Misconception'].value_counts().to_frame()
map_target2['count'] = np.arange(len(map_target2))
map_target2 = map_target2.to_dict()['count']

#map_target1, map_target2


train[TARGET_COLUMN_ONE] = train['Category'].map(map_target1)
train[TARGET_COLUMN_TWO] = train['Misconception'].map(map_target2)

#train['Category'].value_counts()


def tokenize_deberta(batch):
    return DEBERTA_TOKENIZER(batch["text"], padding="max_length", truncation=True, max_length=256)


def transformer_acc(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def load_deberta_model(num_classes: int):
    return DebertaV2ForSequenceClassification.from_pretrained(
            DEBERTA_SMALL_LOAD_PATH,
            num_labels=num_classes
    )


%%capture

transformer_test_pred = {}
transformer_test_pred[TARGET_COLUMN_ONE] = np.zeros((len(test), len(map_target1)))
transformer_test_pred[TARGET_COLUMN_TWO] = np.zeros((len(test), len(map_target2)))

def transformer_func(
    df_train: pd.DataFrame
    ,df_valid: pd.DataFrame
    ,target_column:str
    ,fold: int
    ,**kwargs: dict[str, str]
)-> pd.DataFrame:

    model_name = kwargs['model_name']

    #tokenizer = kwargs['tokenizer']

    load_model_func = kwargs['load_model_func']

    #MODEL_PATH_CP = kwargs['load_path_checkpoint']

    print(f"model_name: {model_name}")

    print(f"fold: {fold}")

    

    if TRAIN_LLM:
        COLS = ['text','label']
        num_classes = len(df_train[target_column].unique())
        
        train_df = df_train.rename(columns={target_column: "label"})
    
        train_df = Dataset.from_pandas(train_df[COLS])

        train_df = train_df.map(tokenize_deberta, batched=True)

        # Set format for PyTorch
        columns = ['input_ids', 'attention_mask', 'label']
        train_df.set_format(type='torch', columns=columns)
        
        training_args = TrainingArguments(
            output_dir=generate_fold_path_for_model(model_name, fold),
            do_train=True,
            do_eval=False,
            eval_strategy="no",
            save_strategy="steps",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=8,
            learning_rate=5e-5,
            logging_dir="./logs",
            logging_steps=50,
            save_steps=200,
            save_total_limit=1,
            report_to="none"
        )

    
        model = load_model_func(num_classes)


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_df,
            compute_metrics=None,
        )

        trainer.train()   
    else:
        training_args = TrainingArguments(
            do_train=False,
            do_eval=False,
            per_device_eval_batch_size=64,
            dataloader_drop_last=False,
            report_to="none",
            logging_dir=None
        )
        
        load_path = f"{MODEL_PATH_CP}/{target_column}/folds/{fold}/checkpoint-2295"
        model = AutoModelForSequenceClassification.from_pretrained(load_path, local_files_only=True)

        trainer = Trainer(model=model, args=training_args)
    

    ds_valid = Dataset.from_pandas(df_valid[['text']])
    ds_valid = ds_valid.map(tokenize_deberta, batched=True)

    predictions = trainer.predict(ds_valid)

    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()

    test_predictions = trainer.predict(Dataset.from_pandas(test[['text']]).map(tokenize_deberta, batched=True))

    test_probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()

    transformer_test_pred[target_column] += ((test_probs) / N_SPLITS)

    return probs


def custom_acc(y_true, y_pred):
    return np.mean( y_true == np.argmax(y_pred, 1) ) 


def split_func(X: pd.DataFrame, y: pd.DataFrame):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=None)
    return skf.split(X, y)


if TRAIN_LLM:
    MODEL_NAME=f"{DEBERTA_SMALL}_{TARGET_COLUMN_ONE}"
    res_one_transformer = kfold_split(
        train=train
        ,target_column=TARGET_COLUMN_ONE
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=transformer_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=6
        ,load_from_checkpoint=False
        ,model_name=MODEL_NAME
        ,load_path_hf=DEBERTA_SMALL_LOAD_PATH
        ,load_path_checkpoint=DEBERTA_CHECKPOINT
        ,load_model_func=load_deberta_model
    )

    #joblib.dump(res_one_transformer, 'res_one_transformer.bin')


if TRAIN_LLM:
    MODEL_NAME=f"{DEBERTA_SMALL}_{TARGET_COLUMN_TWO}"
    res_two_transformer = kfold_split(
        train=train
        ,target_column=TARGET_COLUMN_TWO
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=transformer_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=36
        ,load_from_checkpoint=True
        ,model_name=MODEL_NAME
        ,load_path_hf=DEBERTA_SMALL_LOAD_PATH
        ,load_path_checkpoint=DEBERTA_CHECKPOINT
        ,tokenizer=tokenize_deberta
        ,load_model_func=load_deberta_model
    )

    #joblib.dump(res_two_transformer, 'res_two_transformer.bin')


train['sentence'] = train.apply(lambda x: f"Question: {x['QuestionText']}\nAnswer: {x['MC_Answer']}\nExplanation: {x['StudentExplanation']}", axis=1)
test['sentence'] = test.apply(lambda x: f"Question: {x['QuestionText']}\nAnswer: {x['MC_Answer']}\nExplanation: {x['StudentExplanation']}", axis=1)


model = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word', max_df=0.95, min_df=2)
model.fit(pd.concat([train, test]).sentence)

train_embeddings = model.transform(train.sentence)
print('Train sparse shape is',train_embeddings.shape)

test_embeddings = model.transform(test.sentence)
print('Test sparse shape is',test_embeddings.shape)


model_two = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), analyzer='word', max_df=0.95, min_df=2)
model_two.fit(pd.concat([train, test]).sentence)

train_embeddings_target_two = model_two.transform(train.sentence)
print('Train sparse shape is',train_embeddings.shape)

test_embeddings_target_two = model_two.transform(test.sentence)
print('Test sparse shape is',test_embeddings.shape)


train_inp = pd.DataFrame({EMBEDDING: train_embeddings, TARGET_COLUMN_ONE: train[TARGET_COLUMN_ONE]})
train_inp_two = pd.DataFrame({EMBEDDING: train_embeddings_target_two, TARGET_COLUMN_TWO: train[TARGET_COLUMN_TWO]})


def convert_to_scipy_sparse_matrix(inp: pd.DataFrame):
    return cupyx.scipy.sparse.vstack(inp.values.tolist())


def cupy_csr_to_scipy_csr(cupy_csr):
    #print(cupy_csr.data.sum())
    data = cupy_csr.data.get()
    indices = cupy_csr.indices.get()
    indptr = cupy_csr.indptr.get()
    shape = cupy_csr.shape
    sparse_train_as_scipy = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    #print(sparse_train_as_scipy.data.sum())
    return sparse_train_as_scipy


test_embeddings_as_scipy = vstack([cupy_csr_to_scipy_csr(e) for e in test_embeddings])


test_embeddings_target_two_as_scipy = vstack([cupy_csr_to_scipy_csr(e) for e in test_embeddings_target_two])


def get_test_embeddings_for_target_one():
    return test_embeddings_as_scipy


def get_test_embeddings_for_target_two():
    return test_embeddings_target_two_as_scipy


def to_stacked_vector(inp_df):
    return vstack(
        np.array([cupy_csr_to_scipy_csr(e) for e in inp_df[EMBEDDING].values])
    )


all_test_preds = {}

all_test_preds[DEBERTA_SMALL] = {}
all_test_preds[DEBERTA_SMALL][TARGET_COLUMN_ONE] = np.zeros((len(test), len(map_target1)))
all_test_preds[DEBERTA_SMALL][TARGET_COLUMN_TWO] = np.zeros((len(test), len(map_target2)))

all_test_preds[LOGISTIC] = {}
all_test_preds[LOGISTIC][TARGET_COLUMN_ONE] = np.zeros((len(test), len(map_target1)))
all_test_preds[LOGISTIC][TARGET_COLUMN_TWO] = np.zeros((len(test), len(map_target2)))

all_test_preds[LGB] = {}
all_test_preds[LGB][TARGET_COLUMN_ONE] = np.zeros((len(test), len(map_target1)))
all_test_preds[LGB][TARGET_COLUMN_TWO] = np.zeros((len(test), len(map_target2)))

all_test_preds[XGB] = {}
all_test_preds[XGB][TARGET_COLUMN_ONE] = np.zeros((len(test), len(map_target1)))
all_test_preds[XGB][TARGET_COLUMN_TWO] = np.zeros((len(test), len(map_target2)))

all_test_preds[ETTIN] = {}
all_test_preds[ETTIN][TARGET_COLUMN_ONE] = np.zeros((len(test), len(map_target1)))
all_test_preds[ETTIN][TARGET_COLUMN_TWO] = np.zeros((len(test), len(map_target2)))




def xgb_func(
   df_train: pd.DataFrame
    ,df_valid: pd.DataFrame
    ,target_column:str
    ,fold: int
    ,**kwargs: dict[str, str] 
)-> pd.DataFrame:
    global xgb_test_pred
    #train_as_scipy = np.array([cupy_csr_to_scipy_csr(e) for e in df_train[EMBEDDING].values])
    #valid_as_scipy = np.array([cupy_csr_to_scipy_csr(e) for e in df_valid[EMBEDDING].values])
    get_test_embeddings = kwargs["get_test_embeddings"]
    MODEL_NAME = kwargs["model_name"] 
    
    train_stacked_vectors = to_stacked_vector(df_train)
    valid_stacked_vectors = to_stacked_vector(df_valid)


    params = {'max_depth': 10, 'learning_rate': 0.0010974684673250828, 'min_child_weight': 1,
            'subsample': 0.9800969106980701, 'colsample_bytree': 0.6305990340548516, 'gamma': 2.4140963859555216, 'random_state' : 42,'objective': 'multi:softprob',
            'num_class': kwargs["num_classes_for_xgb"] , 'eval_metric': 'mlogloss', 'device': 'cuda:0'
           }


    dtrain = xgb.DMatrix(train_stacked_vectors, label=df_train[target_column].values)
    dvalid = xgb.DMatrix(valid_stacked_vectors, label=df_valid[target_column].values)

    model = xgb.train(
            params,
            dtrain,
            num_boost_round=100, 
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50, 
            verbose_eval=False
        )

    joblib_dump_model(model, MODEL_NAME, fold)

    
    dtest_as_xgb_matrix = xgb.DMatrix(get_test_embeddings())
    all_test_preds[XGB][target_column] += (model.predict(dtest_as_xgb_matrix) / N_SPLITS)

    return model.predict(dvalid)




def lgb_func(
    df_train: pd.DataFrame
    ,df_valid: pd.DataFrame
    ,target_column:str
    ,fold: int
    ,**kwargs: dict[str, str]
)-> pd.DataFrame:
    global lgb_test_pred
    train_as_scipy = np.array([cupy_csr_to_scipy_csr(e) for e in df_train[EMBEDDING].values])
    valid_as_scipy = np.array([cupy_csr_to_scipy_csr(e) for e in df_valid[EMBEDDING].values])
    get_test_embeddings = kwargs["get_test_embeddings"] 
    MODEL_NAME = kwargs["model_name"] 
    
    train_stacked_vectors = vstack(train_as_scipy)
    valid_stacked_vectors = vstack(valid_as_scipy)
    model = LGBMClassifier(device="cuda")
    model.fit(train_stacked_vectors, np.array(df_train[target_column].values))


    joblib_dump_model(model, MODEL_NAME, fold)
    
    all_test_preds[LGB][target_column] += (model.predict_proba(get_test_embeddings()) / N_SPLITS)

    return model.predict_proba(valid_stacked_vectors)




def logistic_regression_func(
    df_train: pd.DataFrame
    ,df_valid: pd.DataFrame
    ,target_column:str
    ,fold: int
    ,**kwargs: dict[str, str]) -> pd.DataFrame:
    X = convert_to_scipy_sparse_matrix(df_train[EMBEDDING])
    Y = convert_to_scipy_sparse_matrix(df_valid[EMBEDDING])
    get_test_embeddings = kwargs["get_test_embeddings"]
    MODEL_NAME = kwargs["model_name"] 
    if "class_weight" in kwargs.keys():
        class_weight = kwargs["class_weight"]
    else:
        class_weight = None
    
    model = cuml.LogisticRegression(class_weight=class_weight)
    model.fit(X, df_train[target_column])

    joblib_dump_model(model, MODEL_NAME, fold)

    all_test_preds[LOGISTIC][target_column] += (model.predict_proba(get_test_embeddings()) / N_SPLITS)

    return model.predict_proba(Y).get()


def train_xgb_target_one():
    MODEL_NAME = f"{XGB}_{TARGET_COLUMN_TWO}"
    res_two_xgb = kfold_split(
        train=train_inp_two
        ,target_column=TARGET_COLUMN_TWO
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=xgb_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=36
        ,get_test_embeddings=get_test_embeddings_for_target_two
        ,model_name=MODEL_NAME
        ,num_classes_for_xgb=36
    )


def train_xgb_target_two():
    MODEL_NAME = f"{XGB}_{TARGET_COLUMN_ONE}"
    res_one_xgb = kfold_split(
        train=train_inp
        ,target_column=TARGET_COLUMN_ONE
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=xgb_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=6
        ,get_test_embeddings=get_test_embeddings_for_target_one
        ,model_name=MODEL_NAME
        ,num_classes_for_xgb=6
    )


def train_lgb_target_one():
    MODEL_NAME = f"{LGB}_{TARGET_COLUMN_ONE}"
    res_one_lgb = kfold_split(
        train=train_inp
        ,target_column=TARGET_COLUMN_ONE
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=lgb_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=6
        ,get_test_embeddings=get_test_embeddings_for_target_one
        ,model_name=MODEL_NAME
    )


def train_lgb_target_two():
    MODEL_NAME = f"{LGB}_{TARGET_COLUMN_TWO}"
    res_two_lgb = kfold_split(
        train=train_inp_two
        ,target_column=TARGET_COLUMN_TWO
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=lgb_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=36
        ,get_test_embeddings=get_test_embeddings_for_target_two
        ,model_name=MODEL_NAME
    )


def train_logistic_regression_target_one():
    MODEL_NAME = f"{LOGISTIC}_{TARGET_COLUMN_ONE}"
    res_one_logistic = kfold_split(
        train=train_inp
        ,target_column=TARGET_COLUMN_ONE
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=logistic_regression_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=6
        ,get_test_embeddings=get_test_embeddings_for_target_one
        ,model_name=MODEL_NAME
    )


def train_logistic_regression_target_two():
    MODEL_NAME = f"{LOGISTIC}_{TARGET_COLUMN_TWO}"
    res_two_logistic = kfold_split(
        train=train_inp_two
        ,target_column=TARGET_COLUMN_TWO
        ,eval_func=custom_acc
        ,kfold_split_func=split_func
        ,model_func=logistic_regression_func
        ,should_delete_target_column_for_valid_data=False
        ,additional_dimension_for_oof=36
        ,get_test_embeddings=get_test_embeddings_for_target_two
        ,class_weight="balanced"
        ,model_name=MODEL_NAME
    )


%%capture
if TRAIN_OTHER:
    train_logistic_regression_target_one()
    train_logistic_regression_target_two()


%%capture
if TRAIN_XGB:
    train_xgb_target_one()
    train_xgb_target_two()


%%capture
if TRAIN_OTHER:
    train_lgb_target_one()
    train_lgb_target_two()


res_one_lgb = joblib.load(f"{MODEL_PATH}/lgb_target1/kfold_result.bin")
res_one_logistic = joblib.load(f"{MODEL_PATH}/logistic_regression_target1/kfold_result.bin")
res_one_transformer = joblib.load(f"{MODEL_PATH}/deberta-v3-xsmall_target1/kfold_result.bin")
res_one_xgb = joblib.load(f"{MODEL_PATH}/xgb_target1/kfold_result.bin")
res_one_ettin = joblib.load("/kaggle/input/map-ettin-tpu/models/tpu_ettin_target1/kfold_result.bin")

res_two_lgb = joblib.load(f"{MODEL_PATH}/lgb_target2/kfold_result.bin")
res_two_logistic = joblib.load(f"{MODEL_PATH}/logistic_regression_target2/kfold_result.bin")
res_two_transformer = joblib.load(f"{MODEL_PATH}/deberta-v3-xsmall_target2/kfold_result.bin")
res_two_xgb = joblib.load(f"{MODEL_PATH}/xgb_target2/kfold_result.bin")
res_two_ettin = joblib.load("/kaggle/input/map-tpu-setup/models/tpu_ettin_target2/kfold_result.bin")

res_two_logistic.oof_pred[:, 0] = 0


def etin_tokenize(batch):
    print(type(batch))
    return ETTIN_TOKENIZER(batch['text'], padding="max_length", truncation=True,max_length=MAX_LEN)


from torch.utils.data import DataLoader
from datasets import Dataset

# Assuming 'etin_tokenize' is a function that tokenizes the 'text' column

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        COLS = ['text']
        #train_ds = Dataset.from_pandas(texts[['text']])
        
        # 1. Tokenize the dataset
        tokenized_frame = Dataset.from_pandas(texts[['text']]).map(etin_tokenize, batched=True)

        # 2. Set format for PyTorch
        columns = ['input_ids', 'attention_mask']
        tokenized_frame.set_format(type='torch', columns=columns)
        
        # 3. Store the prepared, PyTorch-formatted dataset instance
        self._data = tokenized_frame 
        
    def __len__(self):
        # Delegate length to the underlying dataset object
        return len(self._data)
        
    def __getitem__(self, i):
        # Delegate item retrieval to the underlying dataset object, 
        # which is already formatted to return PyTorch tensors.
        # This handles both single index (int) and batch index (list) requests.
        return self._data[i]

test_ds = TestDataset(test, etin_tokenize)


dl = DataLoader(
    dataset=test_ds,
    batch_size=1,
    shuffle=False 
)


from dataclasses import dataclass


@dataclass
class CFG:
    SEED = 42 
    MODEL_NAME = '/kaggle/input/deepseek-math-7b-instruct/transformers/main/1'
    NUM_EPOCHS = 2
    BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 1
     
    MAX_LENGTH = 256
    WARMUP_RATIO = 0.01
    LR = 4e-4 #lora likes high
    
    # Lora configs
    NUM_LABELS = 65 
    LORA_RANK = 8
    LORA_ALPHA = 16
    DROPOUT = 0.001
    LORA_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] 
    
    # Gradient accumulation
    GRADACCUM = 1
    
    # Evaluation strategy
    VAL_SPLIT = 0.1 
    EVAL_STRATEGY = "steps"  # "epoch", "steps", "no"
    EVAL_STEPS = 4000  # Only used if EVAL_STRATEGY = "steps"
    
    # Save strategy  
    SAVE_STRATEGY = "epoch"  # "epoch", "steps", "no"
    SAVE_STEPS = 1000  # Only used if SAVE_STRATEGY = "steps"
    SAVE_TOTAL_LIMIT = 2  # Keep only 2 most recent checkpoints
    
    # Output directory
    OUT_DIR = "checkpoints"
    
    # Additional training parameters
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0 


from peft import  PeftModel,  get_peft_model, LoraConfig, TaskType



import torch.nn.functional as F


def to_lora(base_model):
    lora_config = LoraConfig(
        r=CFG.LORA_RANK,  # the dimension of the low-rank matrices
        lora_alpha = CFG.LORA_ALPHA, # scaling factor for LoRA activations vs pre-trained weight activations
        lora_dropout= CFG.DROPOUT, 
        bias='none',
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        target_modules=["Wqkv", "Wo", "Wi"] 
    ) 

    # Create LoRa Model
    model = get_peft_model(base_model, lora_config)
    return model


target_column_to_number_of_classes = {
    TARGET_COLUMN_ONE: 6,
    TARGET_COLUMN_TWO: 36
}

target_column_to_base_path = {
    TARGET_COLUMN_ONE: f"/kaggle/input/map-ettin-tpu/models/tpu_ettin_{TARGET_COLUMN_ONE}",
    TARGET_COLUMN_TWO: f"/kaggle/input/map-tpu-setup/models/tpu_ettin_{TARGET_COLUMN_TWO}"
}

#/kaggle/input/map-tpu-setup/models/tpu_ettin_target2/folds/0/tpu_ettin_target2.pt

def generate_test_ettin_pred(fold, target_column):
    BASE_PATH=target_column_to_base_path[target_column]
    model = torch.load(f"{BASE_PATH}/folds/0/tpu_ettin_{target_column}.pt", weights_only=False,  map_location='cuda')
    torch.save(model.state_dict(), "clean_state_dict.pt")
    del model; gc.collect(); torch.cuda.empty_cache()

    ckpt = torch.load("/kaggle/working/clean_state_dict.pt", map_location="cuda")
    model = AutoModelForSequenceClassification.from_pretrained(
        "/kaggle/input/ettin-encoder-1b-cv943",
        num_labels=target_column_to_number_of_classes[target_column],
        reference_compile=False,
        ignore_mismatched_sizes=True
    )
    
    peft_model = to_lora(model)

    peft_model.load_state_dict(ckpt)
    
    peft_model.eval()
    
    dl = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False 
    )

    peft_model.to("cuda")

    preds = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to("cuda") for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = peft_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            logits = outputs.logits.to(dtype=torch.float32)
            probs = F.softmax(logits, dim=-1)
            preds.extend(probs)
    #print(preds)
    #test_probs =  / N_SPLITS
    #print(f"preds per split: {preds_per_split}")
    all_test_preds[ETTIN][target_column] += ((torch.stack(preds).cpu().numpy()) / N_SPLITS)


for fold in range(5):
    generate_test_ettin_pred(fold, TARGET_COLUMN_TWO)
    generate_test_ettin_pred(fold, TARGET_COLUMN_ONE)
    


def generate_test_transformer_pred(fold, target_column):
    training_args = TrainingArguments(
            do_train=False,
            do_eval=False,
            per_device_eval_batch_size=8,
            dataloader_drop_last=False,
            report_to="none",
            logging_dir=None
        )
        
    load_path = f"{MODEL_PATH}/{DEBERTA_SMALL}_{target_column}/folds/{fold}/checkpoint-4590"
    model = AutoModelForSequenceClassification.from_pretrained(load_path, local_files_only=True)

    trainer = Trainer(model=model, args=training_args)

    test_predictions = trainer.predict(Dataset.from_pandas(test[['text']]).map(tokenize_deberta, batched=True))

    test_probs = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()

    all_test_preds[DEBERTA_SMALL][target_column] += ((test_probs) / N_SPLITS)




def generate_test_prediction_for_logistic_regression(fold, target_column, get_test_embeddings):
    load_path = f"{MODEL_PATH}/{LOGISTIC}_{target_column}/folds/{fold}/{LOGISTIC}_{target_column}.bin"
    model = joblib.load(load_path)

    all_test_preds[LOGISTIC][target_column] += (model.predict_proba(get_test_embeddings()) / N_SPLITS)




def generate_test_prediction_for_lgb(fold, target_column, get_test_embeddings):
    load_path = f"{MODEL_PATH}/{LGB}_{target_column}/folds/{fold}/{LGB}_{target_column}.bin"
    model = joblib.load(load_path)

    all_test_preds[LGB][target_column] += (model.predict_proba(get_test_embeddings()) / N_SPLITS)





def generate_test_prediction_for_xgb(fold, target_column, get_test_embeddings):
    load_path = f"{MODEL_PATH}/{XGB}_{target_column}/folds/{fold}/{XGB}_{target_column}.bin"
    model = joblib.load(load_path)
    
    dtest_as_xgb_matrix = xgb.DMatrix(get_test_embeddings())
    all_test_preds[XGB][target_column] += (model.predict(dtest_as_xgb_matrix) / N_SPLITS)



for fold in range(5):
    #generate_test_prediction_for_logistic_regression(fold, TARGET_COLUMN_ONE, get_test_embeddings_for_target_one)
    #generate_test_prediction_for_logistic_regression(fold, TARGET_COLUMN_TWO, get_test_embeddings_for_target_two)
    
    #generate_test_prediction_for_xgb(fold, TARGET_COLUMN_ONE, get_test_embeddings_for_target_one)
    #generate_test_prediction_for_xgb(fold, TARGET_COLUMN_TWO, get_test_embeddings_for_target_two)

    #generate_test_prediction_for_lgb(fold, TARGET_COLUMN_ONE, get_test_embeddings_for_target_one)
    #generate_test_prediction_for_lgb(fold, TARGET_COLUMN_TWO, get_test_embeddings_for_target_two)    

    generate_test_transformer_pred(fold, TARGET_COLUMN_ONE)
    generate_test_transformer_pred(fold, TARGET_COLUMN_TWO)


first = np.arange(0.01,1,0.01)
second = np.arange(0.01,1,0.01)

np.random.shuffle(first)
np.random.shuffle(second)
#first
#second = np.arange(start,0.51,0.01)


def arg_sort(inp):
    return np.argsort(-inp, 1)[:,:3]


map_inverse1 = {map_target1[k]:k for k in map_target1}
map_inverse2 = {map_target2[k]:k for k in map_target2}


def generate_pred_string(predicted1, predicted2, generate_test_prediction=False):
    predict = []
    for i in range(len(predicted1)):
        pred = []
        for j in range(3):
            p1 = map_inverse1[predicted1[i, j]]
            p2 = map_inverse2[predicted2[i, j]]        
            if 'Misconception' in p1:
                pred.append(p1 + ":" + p2 )
            else:
                pred.append(p1 + ":NA")
        
        if generate_test_prediction:
            predict.append(" ".join(pred))
        else:
            predict.append(pred)    

    return predict


def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score+=1.
        elif t == p[1]:
            score+=1/2
        elif t == p[2]:
            score+=1/3
    return score / len(target_list)


targets = [(res_one_ettin, res_two_ettin, ETTIN),(res_one_xgb, res_two_xgb, XGB), (res_one_lgb, res_two_lgb, LGB), (res_one_logistic, res_two_logistic, LOGISTIC), (res_one_transformer, res_two_transformer, DEBERTA_SMALL)]


from collections import namedtuple

def combine_targets(model_one_target_one, model_one_target_two, model_name_one, model_two_target_one, model_two_target_two, model_name_two):
    

    first = np.arange(0.01,1,0.01)
    second = np.arange(0.01,1,0.01)

    np.random.shuffle(first)
    np.random.shuffle(second)
    
    best_score = 0
    best_prob_target_one = None
    best_prob_target_two = None
    
    for i,j in zip(first, second):
        first_model_weight_target_one = i
        second_model_weight_target_one = 1 - i

        first_model_weight_target_two = j
        second_model_weight_target_two = 1-j
    
        target_one = (first_model_weight_target_one * model_one_target_one.oof_pred + second_model_weight_target_one *  model_two_target_one.oof_pred)
        target_two = (first_model_weight_target_two * model_one_target_two.oof_pred + second_model_weight_target_two *  model_two_target_two.oof_pred)

        tof_sorted = arg_sort(target_one)
        m_sorted = arg_sort(target_two)

        pred_strings = generate_pred_string(tof_sorted, m_sorted)
        map3_score = map3(train['target_cat'], pred_strings)

        if map3_score > best_score:
            first_model_best_weight_target_one = first_model_weight_target_one
            second_model_best_weight_target_one = second_model_weight_target_one
            
            first_model_best_weight_target_two = first_model_weight_target_two
            second_model_best_weight_target_two = second_model_weight_target_two
            
            #print(f"new best_score {map3_score}")
            best_score = map3_score
            best_prob_target_one = target_one
            best_prob_target_two = target_two

    Prediction = namedtuple('Prediction', ['oof_pred', 'model_name', 'weight'])
    OofPred = namedtuple('OofPred', ['oof_pred'])
    ModelNameAndWeight = namedtuple('ModelNameAndWeight', ['model_name', 'weight'])
    model_combination = namedtuple('model_combination', ['first_model', 'second_model'])
    #models_for_target_two = namedtuple('models_for_target_two', ['first_model', 'second_model'])
    Score = namedtuple('Score', ['score'])
    Run = namedtuple('Run', ['best_probs_for_target_one', 'best_probs_for_target_two', 'models_for_target_one', 'models_for_target_two', 'score'])

    #oof_pred = best_prob_target_one, 
    first_for_target_one = ModelNameAndWeight(model_name=model_name_one, weight=first_model_best_weight_target_one)
    second_for_target_one = ModelNameAndWeight(model_name=model_name_two, weight=second_model_best_weight_target_one)

    first_for_target_two = ModelNameAndWeight(model_name=model_name_one, weight=first_model_best_weight_target_two)
    second_for_target_two = ModelNameAndWeight(model_name=model_name_two, weight=second_model_best_weight_target_two)
    
    first_models = model_combination(first_model = first_for_target_one, second_model = second_for_target_one)
    second_models = model_combination(first_model = first_for_target_two, second_model = second_for_target_two)

    run = Run( best_probs_for_target_one = OofPred(oof_pred=best_prob_target_one)
        ,best_probs_for_target_two = OofPred(oof_pred=best_prob_target_two)
        ,models_for_target_one = first_models
        ,models_for_target_two = second_models
        ,score=best_score
       )
    return run


with open(f"{BASE_PATH}/best_run.bin", "rb") as f:
    data = f.read()
best_run = cp.loads(data)


if DETERMINE_BEST_COMBINATION:
    current_best_score = 0
    best_run = []
    for idx_combo in itertools.combinations(range(len(targets)), 2):
        current_run = []
        t = [targets[idx_combo[0]], targets[idx_combo[1]]]
    
        stack = [s for s in range(len(targets)) if s not in list(idx_combo)]
        first_not_blended_model = targets[stack[0]]
        #print(targets[stack[0]][2])
        print("--- blending ---")
        run_result_blending = combine_targets(t[0][0], t[0][1], t[0][2], t[1][0], t[1][1], t[1][2])
        #print(f"run_result: {run_result_blending}")
        current_run.append(run_result_blending)
        print("--- stacking 1 ---")
    
        run_result_stacking = combine_targets(first_not_blended_model[0], first_not_blended_model[1], first_not_blended_model[2],  run_result_blending.best_probs_for_target_one, run_result_blending.best_probs_for_target_two, "stacking 1")
        #print(f"run_result_stacking: {run_result_stacking}")
    
        current_run.append(run_result_stacking)
        print("--- stacking 2 ---")
        second_not_blended_model = targets[stack[1]]
        run_result_stack_of_stack = combine_targets(second_not_blended_model[0], second_not_blended_model[1],second_not_blended_model[2], run_result_stacking.best_probs_for_target_one, run_result_stacking.best_probs_for_target_two, "stacking 2")

        print("--- stacking 3 ---")
        third_not_blended_model = targets[stack[2]]
        run_result_stack_of_stack_of_stack = combine_targets(third_not_blended_model[0], third_not_blended_model[1], third_not_blended_model[2], run_result_stack_of_stack.best_probs_for_target_one, run_result_stack_of_stack.best_probs_for_target_two, "stacking 3")

        #print(run_result_stack_of_stack)
        current_run.append(run_result_stack_of_stack)
        if run_result_stack_of_stack_of_stack.score > current_best_score:
            current_best_score = run_result_stack_of_stack_of_stack.score
            print(f"new best score: {current_best_score}")
            best_run = []

            best_run.append(run_result_blending)
            best_run.append(run_result_stacking)
            best_run.append(run_result_stack_of_stack)
            best_run.append(run_result_stack_of_stack_of_stack)

        with open("best_run.bin", "wb") as f:
            cloudpickle.dump(best_run, f)


if CALCULATE_BLENDING:
	first_model_name = best_run[0].models_for_target_one.first_model.model_name
	first_model_weight = best_run[0].models_for_target_one.first_model.weight
	
	second_model_name = best_run[0].models_for_target_one.second_model.model_name
	second_model_weight = best_run[0].models_for_target_one.second_model.weight
	
	blending_result = all_test_preds[first_model_name][TARGET_COLUMN_ONE] * first_model_weight + all_test_preds[second_model_name][TARGET_COLUMN_ONE] * second_model_weight
	
	blending_weight = best_run[1].models_for_target_one.second_model.weight
	
	second_model_name = best_run[1].models_for_target_one.first_model.model_name
	second_model_weight = best_run[1].models_for_target_one.first_model.weight
	
	stacking_result = blending_weight * blending_result + all_test_preds[second_model_name][TARGET_COLUMN_ONE] * second_model_weight
	
	stacking_weight = best_run[2].models_for_target_one.second_model.weight
	
	second_model_name = best_run[2].models_for_target_one.first_model.model_name
	second_model_weight = best_run[2].models_for_target_one.first_model.weight
	
	target_one_result = stacking_result * stacking_weight + all_test_preds[second_model_name][TARGET_COLUMN_ONE] * second_model_weight
	
	
	
	first_model_name = best_run[0].models_for_target_two.first_model.model_name
	first_model_weight = best_run[0].models_for_target_two.first_model.weight
	
	second_model_name = best_run[0].models_for_target_two.second_model.model_name
	second_model_weight = best_run[0].models_for_target_two.second_model.weight
	
	blending_result = all_test_preds[first_model_name][TARGET_COLUMN_TWO] * first_model_weight + all_test_preds[second_model_name][TARGET_COLUMN_TWO] * second_model_weight
	
	blending_weight = best_run[1].models_for_target_two.second_model.weight
	
	second_model_name = best_run[1].models_for_target_two.first_model.model_name
	second_model_weight = best_run[1].models_for_target_two.first_model.weight
	
	stacking_result = blending_weight * blending_result + all_test_preds[second_model_name][TARGET_COLUMN_TWO] * second_model_weight
	
	stacking_weight = best_run[2].models_for_target_two.second_model.weight
	
	second_model_name = best_run[2].models_for_target_two.first_model.model_name
	second_model_weight = best_run[2].models_for_target_two.first_model.weight
	
	target_two_result = stacking_result * stacking_weight + all_test_preds[second_model_name][TARGET_COLUMN_TWO] * second_model_weight
	
	
	
	one_sorted = arg_sort(target_one_result)
	
	two_sorted = arg_sort(target_two_result)
	
	predict = generate_pred_string(one_sorted, two_sorted, True)
else:
    sample_submission = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

    predict = len(sample_submission) * [sample_submission.iloc[0][1]]
    


test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

submission = pd.DataFrame({
    'row_id': test_df.row_id.values,
    'Category:Misconception': predict
})

submission.to_csv('submission.csv', index=False)
submission

