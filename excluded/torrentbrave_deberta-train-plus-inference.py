%load_ext cudf.pandas


import os
import gc
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import log_loss
import torch
from functools import partial
import warnings
from transformers import logging as transformers_logging
from transformers import EarlyStoppingCallback
import json
from pprint import pformat
from tqdm import trange
warnings.simplefilter('ignore')

USE_QLORA = False
ADD_33K = False
DEBUG = True

TYPE = "base"
VER= 1
DATE = "0120"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
# Set up logging
transformers_logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, filename=f'logs_v{VER}.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PATHS:
    train_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet'
    test_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet'
    sub_path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/sample_submission.csv'
    model_name = f"deberta-v3-{TYPE}"
    model_path = f"/kaggle/input/huggingfacedebertav3variants/{model_name}"
    tokenizer_path = f"/kaggle/input/wsdm-{TYPE}{VER}-{DATE}/fold_0/tokenizer"
    general_tokenizer = "/kaggle/input/lmsys-base3-0704/fold_0/tokenizer"
    #general_tokenizer = "/kaggle/input/wsdm-base1-0120/fold_0/tokenizer"

class CFG:
    seed = 42
    max_length = 2048
    lr = 5e-5  # 学习率
    weight_decay = 0.01  # 权重衰减
    warmup_ratio = 0 # 学习率预热比例
    max_grad_norm = 1000  # 梯度裁剪最大范数
    lr_scheduler_type = 'linear'  # 学习率调度类型
    frozen_embedding = True # 冻结前面的层
    frozen_num = 6
    train_batch_size = 32  # 训练批量大小
    eval_batch_size = 64  # 评估批量大小
    evaluation_strategy = 'steps'  # 更改为 steps 评估策略
    metric_for_best_model = "eval_log_loss"  # 用于选择最佳模型的度量标准
    save_strategy = 'steps'  # 更改为 steps 保存策略
    save_steps = 200  # 每 步保存一次模型
    save_total_limit = 1  # 保存检查点总数限制
    train_epochs = 5  # 训练周期数
    num_labels = 6
    output_dir = f'/kaggle/working/wsdm-{TYPE}{VER}-{DATE}'  # 输出目录
    fp16 = True  # 使用混合精度训练
    load_best_model_at_end = True  # 训练结束时加载最佳模型
    report_to = 'none'  # 不报告训练日志到外部工具
    optim = 'adamw_torch'  # 优化器类型
    logging_first_step = True  # 记录第一步的日志
    logging_steps = 200  # 每 步记录一次日志
    logging_dir =f'logs_v{VER}'  # 日志保存目录
    n_splits = 5
    model_name = PATHS.model_name
    greater_is_better = False
    early_stop = False
    early_stopping_patience = 3  # Number of evaluation calls with no improvement after which training will be stopped
    early_stopping_threshold = 0.001  # Minimum change to qualify as an improvement

def seed_everything(seed):
    import random
    import os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG.seed)

# tokenizer = AutoTokenizer.from_pretrained(PATHS.tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(PATHS.general_tokenizer)
sep_token = tokenizer.sep_token_id

def log_parameters(logger):
    """Log all parameters from PATHS and CFG classes."""
    logger.info("=== Parameter Settings ===")
    
    logger.info("PATHS:")
    for key, value in PATHS.__dict__.items():
        if not key.startswith('__'):
            logger.info(f"  {key}: {value}")
    
    logger.info("CFG:")
    for key, value in CFG.__dict__.items():
        if not key.startswith('__'):
            logger.info(f"  {key}: {value}")
    
    logger.info("=*100")

def tokenize_function(row, tokenizer):
    max_len = CFG.max_length - 2 # We need 2 separator tokens
    tokens_prompt = tokenizer(row['prompt'], truncation=True, max_length=max_len//4, add_special_tokens=False)['input_ids']
    remaining_length = max_len - len(tokens_prompt)
    
    tokens_response_a = tokenizer(row['response_a'], truncation=True, max_length=remaining_length//2, add_special_tokens=False)['input_ids']
    remaining_length -= len(tokens_response_a)
    tokens_response_b = tokenizer(row['response_b'], truncation=True, max_length=remaining_length, add_special_tokens=False)['input_ids']
    
    input_ids = [tokenizer.cls_token_id] + tokens_prompt + [sep_token] + tokens_response_a + [sep_token] + tokens_response_b
    token_type_ids = [0] * (len(tokens_prompt) + 2) + [1] * (len(tokens_response_a) + 1) + [2] * len(tokens_response_b)
    attention_mask = [1] * len(input_ids)
    
    padding_length = CFG.max_length - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length
    
    return {
        'input_ids': input_ids[:CFG.max_length],
        'token_type_ids': token_type_ids[:CFG.max_length],
        'attention_mask': attention_mask[:CFG.max_length],
    }

def add_label(df):
    #labels = np.zeros(len(df), dtype=np.int32)
    df['labels'] = df['winner'].map({'model_a': 0, 'model_b': 1})
    return df

def process_data(df, mode='train'):
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=False)#batch_size=CFG.train_batch_size)
    remove_cols = ['id', 'prompt', 'response_a', 'response_b', 'scored']
    if mode == 'train':
        remove_cols += ['model_a', 'model_b', 'winner', 'language']
    for col in remove_cols:
        if col in tokenized_dataset.column_names:
            tokenized_dataset = tokenized_dataset.remove_columns([col])
    return tokenized_dataset

def split_train_val(dataset, train_fraction):
    np.random.seed(0)
    ixs = np.arange(len(dataset))
    cutoff = int(len(ixs) * train_fraction)
    np.random.shuffle(ixs)
    ixs_train = ixs[:cutoff]
    ixs_val = ixs[cutoff:]
    fit_train = dataset.select(ixs_train)
    fit_val = dataset.select(ixs_val)
    return fit_train, fit_val

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    return {
        'eval_log_loss': log_loss(labels, probabilities),
        'eval_accuracy': (np.argmax(logits, axis=1) == labels).mean()
    }

def memory_cleanup(model=None):
    """
    Cleans up GPU memory but avoids deleting model parameters.
    """
    try:
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                # Only delete tensors that are not part of the model's parameters
                if model and obj not in model.parameters():
                    del obj
    except ReferenceError:
        # If there is an error due to weak references (like already deleted objects), pass
        pass

    gc.collect()
    torch.cuda.empty_cache()
    
def train_model():
    log_parameters(logger)
    train_df = pd.read_parquet(PATHS.train_path)
    #train_table = pd.read_table(PATHS.train_path)
    #train_df = pd.DataFrame.from_dict(train_table.to_pydict())
    if DEBUG:
        train_df = train_df.iloc[:64]
    train_df = add_label(train_df)
    train_tokenized = process_data(train_df, mode='train')
    
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_tokenized, train_tokenized['labels'])):
        print(f"Training fold {fold + 1}")
        logger.info(f"Training fold {fold + 1}")
        
        fit_train = train_tokenized.select(train_idx)
        fit_val = train_tokenized.select(val_idx)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            PATHS.model_path,
            num_labels=2,
            problem_type="single_label_classification"
        )

        # 使用DataParallel实现多GPU训练
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        else:
            raise RuntimeError("Multiple GPUs are required for training.")
        
        training_args = TrainingArguments(
            output_dir=f"{CFG.output_dir}/fold_{fold}",  # 模型和检查点的输出目录
            fp16=CFG.fp16,  # 使用混合精度训练
            learning_rate=CFG.lr,  # 学习率
            per_device_train_batch_size=1, #CFG.train_batch_size,  # 每个设备上的训练批量大小
            per_device_eval_batch_size=1, #CFG.eval_batch_size,  # 每个设备上的评估批量大小
            num_train_epochs=CFG.train_epochs,  # 训练的总周期数
            weight_decay=CFG.weight_decay,  # 权重衰减（L2正则化）
            evaluation_strategy=CFG.evaluation_strategy,  # 评估策略
            metric_for_best_model=CFG.metric_for_best_model,  # 用于选择最佳模型的度量标准
            save_strategy=CFG.save_strategy,  # 保存策略
            save_total_limit=CFG.save_total_limit,  # 保存的检查点总数限制
            load_best_model_at_end=CFG.load_best_model_at_end,  # 在训练结束时加载最佳模型
            report_to=CFG.report_to,  # 不报告训练日志到外部工具
            warmup_ratio=CFG.warmup_ratio,  # 学习率预热比例
            lr_scheduler_type=CFG.lr_scheduler_type,  # 学习率调度类型
            optim=CFG.optim,  # 使用的优化器类型
            logging_first_step=CFG.logging_first_step,  # 记录第一步的日志
            greater_is_better=CFG.greater_is_better,
            
            max_grad_norm=CFG.max_grad_norm,  # 设置梯度裁剪
            
            logging_steps=CFG.logging_steps,  # 每 500 步记录一次日志
            logging_dir=CFG.logging_dir,  # 日志保存目录
        
            save_steps=CFG.save_steps,  # 每  步保存一次模型
            eval_steps=CFG.save_steps,  # 添加 eval_steps 参数,与 save_steps 保持一致

            remove_unused_columns=False
        )

         # Log training arguments
        logger.info("Training arguments:")
        logger.info(pformat(training_args.to_dict()))

        if CFG.frozen_embedding:
            n = CFG.frozen_num
            # 冻结嵌入层
            for i, layer in enumerate(model.module.deberta.encoder.layer[:n]):
                for param in layer.parameters():
                    param.requires_grad = False # True False
            for param in model.module.deberta.embeddings.parameters():
                param.requires_grad = False

        # 初始化 tokenizer
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Create EarlyStoppingCallback
        if CFG.early_stop:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=CFG.early_stopping_patience,
                early_stopping_threshold=CFG.early_stopping_threshold,
            )
        
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=fit_train,
                data_collator=data_collator,
                eval_dataset=fit_val,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping_callback],  # Add the early stopping callback
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=fit_train,
                data_collator=data_collator,
                eval_dataset=fit_val,
                compute_metrics=compute_metrics,
            )
        
        trainer.train()
        
        # Save the model
        trainer.save_model(f"{CFG.output_dir}/fold_{fold}/best_model")
        tokenizer.save_pretrained(f"{CFG.output_dir}/fold_{fold}/tokenizer")
        
        # Log the results
        eval_result = trainer.evaluate()
        logger.info(f"Fold {fold + 1} - Evaluation result: {eval_result}")
        logger.info("=*100")

        #clear memory
        memory_cleanup()

def predict_test():
    test_df = pd.read_parquet(PATHS.test_path)
    test_tokenized = process_data(test_df, mode='test')
    
    predictions = []
    
    for fold in trange(CFG.n_splits):
        model = AutoModelForSequenceClassification.from_pretrained(f"{CFG.output_dir}/fold_{fold}/best_model")
        model.eval()
        
        trainer = Trainer(model=model)
        fold_preds = trainer.predict(test_tokenized).predictions
        fold_preds = np.exp(fold_preds) / np.sum(np.exp(fold_preds), axis=1, keepdims=True)
        predictions.append(fold_preds)
    
    # Average predictions across folds
    final_preds = np.mean(predictions, axis=0)
    display(predictions)
    logger.info(f"Final_preds: {final_preds}")
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner': ['model_a' if a > b else 'model_b' for a, b in zip(final_preds[:, 0], final_preds[:, 1])],
    })
    
    submission.to_csv('submission.csv', index=False)
    display(submission)


%time
if __name__ == "__main__":
     train_model()
    #predict_test()


train_df = pd.read_parquet(PATHS.train_path)
if DEBUG:
    train_df = train_df.iloc[:64]
train_df = add_label(train_df)
train_tokenized = process_data(train_df, mode='train')


train_tokenized


train_tokenized['labels'][:4]


train_df = pd.read_parquet(PATHS.train_path)


train_df.columns


train_df = add_label(train_df)


a = train_df.copy()


train_df.columns




