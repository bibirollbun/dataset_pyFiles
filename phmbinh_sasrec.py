!pip install recbole


import polars as pl
import pandas as pd
import torch
import gc
from tqdm import tqdm
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


# ===========================
# 1. DATA PREPARATION (FIXED)
# ===========================
print("Loading data...")
train = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test.parquet')
test = pl.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')

# âœ… FIX: Chá»‰ dÃ¹ng train data Ä‘á»ƒ train model
df = train.clone()

# âœ… FIX: Sáº¯p xáº¿p Ä�ÃšNG - chá»‰ theo session vÃ  timestamp
df = df.sort(['session', 'ts'])

# Convert timestamp to nanoseconds
df = df.with_columns((pl.col('ts') * 1e9).alias('ts'))

# Rename columns theo format RecBole
df = df.rename({
    'session': 'session:token', 
    'aid': 'aid:token', 
    'ts': 'ts:float'
})

# Save data
!mkdir -p /kaggle/working/recbox_data
df.select(['session:token', 'aid:token', 'ts:float']).write_csv(
    '/kaggle/working/recbox_data/recbox_data.inter', 
    separator='\t'
)

print(f"âœ… Data prepared: {len(df)} interactions")


# ===========================
# 2. MODEL CONFIGURATION (IMPROVED)
# ===========================
MAX_ITEM = 20  # Giá»¯ nguyÃªn theo yÃªu cáº§u

parameter_dict = {
    'data_path': '/kaggle/working/',
    'USER_ID_FIELD': 'session',
    'ITEM_ID_FIELD': 'aid',
    'TIME_FIELD': 'ts',
    
    # âœ… FIX: Giáº£m filtering Ä‘á»ƒ giá»¯ nhiá»�u data hÆ¡n
    'user_inter_num_interval': "[2,Inf)",
    'item_inter_num_interval': "[3,Inf)",
    
    'load_col': {'inter': ['session', 'aid', 'ts']},
    'train_neg_sample_args': None,
    
    # âœ… FIX: TÄƒng epochs vÃ  early stopping
    'epochs': 50,
    'stopping_step': 5,
    
    # âœ… ADD: Hyperparameters quan trá»�ng
    'hidden_size': 64,
    'inner_size': 256,
    'n_layers': 2,
    'n_heads': 2,
    'hidden_dropout_prob': 0.5,
    'attn_dropout_prob': 0.5,
    'learning_rate': 0.001,
    'train_batch_size': 2048,
    'eval_batch_size': 1024,
    
    'MAX_ITEM_LIST_LENGTH': MAX_ITEM,
    'eval_args': {
        'split': {'RS': [8, 1, 1]},  # âœ… FIX: ThÃªm test set Ä‘á»ƒ evaluate
        'group_by': 'user',
        'order': 'TO',
        'mode': 'full'
    }
}

# Initialize config
config = Config(model='SASRec', dataset='recbox_data', config_dict=parameter_dict)
init_seed(config['seed'], config['reproducibility'])
init_logger(config)
logger = getLogger()

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

logger.info(config)


# ===========================
# 3. DATASET & MODEL TRAINING
# ===========================
print("\nCreating dataset...")
dataset = create_dataset(config)
logger.info(dataset)

print("\nPreparing data splits...")
train_data, valid_data, test_data = data_preparation(config, dataset)

print("\nInitializing model...")
model = SASRec(config, train_data.dataset).to(config['device'])
logger.info(model)

print("\nTraining model...")
trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

# âœ… ADD: Evaluate model (trÃªn valid data thay vÃ¬ test)
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"Best validation score: {best_valid_score}")
print(f"Best validation result: {best_valid_result}")

# Náº¿u muá»‘n evaluate trÃªn test data, cáº§n load model vá»›i weights_only=False
try:
    # Evaluate trá»±c tiáº¿p khÃ´ng load láº¡i model
    test_result = trainer.evaluate(test_data, load_best_model=False)
    logger.info(f"Test results (current model): {test_result}")
except Exception as e:
    print(f"âš ï¸� Cannot evaluate on test data: {e}")
    print("Continuing with best validation results...")


# ===========================
# 4. PREDICTION (OPTIMIZED)
# ===========================
def get_predictions_batch(session_ids, model, dataset, top_k=20):
    """
    Batch prediction vá»›i error handling tá»‘t hÆ¡n
    """
    results = {}
    batch_data = []
    
    for external_session_id in session_ids:
        try:
            internal_session_id = dataset.token2id(dataset.uid_field, str(external_session_id))
        except (KeyError, ValueError):
            results[external_session_id] = ""
            continue
        
        # Láº¥y interaction history
        inter_feat = dataset.inter_feat
        session_mask = (inter_feat[dataset.uid_field] == internal_session_id)
        session_indices = session_mask.nonzero(as_tuple=True)[0]
        
        if len(session_indices) == 0:
            results[external_session_id] = ""
            continue
        
        # Láº¥y item IDs
        internal_item_ids = inter_feat[dataset.iid_field][session_indices].cpu().tolist()
        max_len = dataset.config['MAX_ITEM_LIST_LENGTH']
        internal_item_ids = internal_item_ids[-max_len:]
        
        if len(internal_item_ids) == 0:
            results[external_session_id] = ""
            continue
        
        # Padding
        padded_items = internal_item_ids + [0] * (max_len - len(internal_item_ids))
        
        batch_data.append({
            'session_id': external_session_id,
            'internal_session_id': internal_session_id,
            'padded_items': padded_items,
            'item_length': len(internal_item_ids),
            'original_items': internal_item_ids
        })
    
    if len(batch_data) == 0:
        return results
    
    # Táº¡o batch tensors
    item_list_field = dataset.iid_field + '_list'
    
    batch_interaction = Interaction({
        dataset.uid_field: torch.tensor([d['internal_session_id'] for d in batch_data], dtype=torch.long),
        item_list_field: torch.tensor([d['padded_items'] for d in batch_data], dtype=torch.long),
        'item_length': torch.tensor([d['item_length'] for d in batch_data], dtype=torch.long),
    })
    
    # Batch prediction
    model.eval()
    try:
        with torch.no_grad():
            batch_interaction = batch_interaction.to(model.device)
            scores = model.full_sort_predict(batch_interaction)
            
            for idx, data in enumerate(batch_data):
                session_id = data['session_id']
                internal_item_ids = data['original_items']
                
                session_scores = scores[idx].clone()
                
                # Loáº¡i bá»� items Ä‘Ã£ tÆ°Æ¡ng tÃ¡c
                for item_id in internal_item_ids:
                    if item_id < len(session_scores):
                        session_scores[item_id] = -float('inf')
                
                # Get top-k
                k = min(top_k, len(session_scores))
                top_k_indices = torch.topk(session_scores, k=k).indices.cpu().tolist()
                
                # Convert to external IDs
                external_item_ids = [dataset.id2token(dataset.iid_field, idx) for idx in top_k_indices]
                results[session_id] = " ".join(external_item_ids)
    
    except RuntimeError as e:
        print(f"âš ï¸� CUDA Error in batch: {e}")
        print(f"Falling back to single prediction...")
        
        # Fallback: xá»­ lÃ½ tá»«ng session
        for data in batch_data:
            session_id = data['session_id']
            try:
                single_interaction = Interaction({
                    dataset.uid_field: torch.tensor([data['internal_session_id']], dtype=torch.long),
                    item_list_field: torch.tensor([data['padded_items']], dtype=torch.long),
                    'item_length': torch.tensor([data['item_length']], dtype=torch.long),
                })
                
                with torch.no_grad():
                    single_interaction = single_interaction.to(model.device)
                    session_scores = model.full_sort_predict(single_interaction)[0]
                    
                    for item_id in data['original_items']:
                        if item_id < len(session_scores):
                            session_scores[item_id] = -float('inf')
                    
                    k = min(top_k, len(session_scores))
                    top_k_indices = torch.topk(session_scores, k=k).indices.cpu().tolist()
                    external_item_ids = [dataset.id2token(dataset.iid_field, idx) for idx in top_k_indices]
                    results[session_id] = " ".join(external_item_ids)
            except Exception as e2:
                print(f"â�Œ Error for session {session_id}: {e2}")
                results[session_id] = ""
    
    return results


# ===========================
# 5. GENERATE SUBMISSION
# ===========================
print("\n" + "="*50)
print("GENERATING SUBMISSION")
print("="*50)

test_session_ids = test['session'].unique().to_list()
print(f"Total sessions to predict: {len(test_session_ids)}")

submission_data = []
batch_size = 64  # TÄƒng batch size cho nhanh hÆ¡n

print(f"Processing in batches of {batch_size}...")

for i in tqdm(range(0, len(test_session_ids), batch_size)):
    batch_sessions = test_session_ids[i:i+batch_size]
    
    predictions = get_predictions_batch(batch_sessions, model, dataset, top_k=20)
    
    for session_id in batch_sessions:
        predicted_aids = predictions.get(session_id, "")
        
        submission_data.append({
            'session_type': f'{session_id}_clicks', 
            'labels': predicted_aids
        })
        submission_data.append({
            'session_type': f'{session_id}_carts', 
            'labels': predicted_aids
        })
        submission_data.append({
            'session_type': f'{session_id}_orders', 
            'labels': predicted_aids
        })
    
    # Clear cache Ä‘á»‹nh ká»³
    if (i // batch_size) % 10 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Save submission
df_submission = pd.DataFrame(submission_data)
df_submission.to_csv('submission.csv', index=False)

print("\n" + "="*50)
print("âœ… COMPLETED!")
print("="*50)
print(f"ğŸ“Š Total rows: {len(df_submission)}")
print(f"ğŸ“‹ Sample rows:")
print(df_submission.head(9))
print(f"\nğŸ’¾ File saved: submission.csv")

