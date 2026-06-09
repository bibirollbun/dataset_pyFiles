!pip install recbole


import polars as pl


# RecBole expects an atomic file, which we create here
# more information: https://recbole.io/atomic_files.html

train = pl.read_parquet('/kaggle/input/otto-train-and-test-data-for-local-validation/test.parquet')
test = pl.read_parquet('/kaggle/input/otto-full-optimized-memory-footprint/test.parquet')

df = pl.concat([train, test])

df = df.sort(['session', 'aid', 'ts'])
df = df.with_columns((pl.col('ts') * 1e9).alias('ts')) # unix nanoseconds are expected by RecBole
df = df.rename({'session': 'session:token', 'aid': 'aid:token', 'ts': 'ts:float'}) # the columnname including :[type] is expected by RecBole

!mkdir /kaggle/working/recbox_data
df['session:token', 'aid:token', 'ts:float'].write_csv('/kaggle/working/recbox_data/recbox_data.inter', separator='\t')


df


df.filter(pl.col("session:token") == 11098530), df.filter(pl.col("session:token") == 11098556), df.filter(pl.col("session:token") == 12899775)


import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger

from recbole.utils.case_study import full_sort_topk


MAX_ITEM = 20  # limit the input sequence to the last 20 items a user interacted with

parameter_dict = {
    'data_path': '/kaggle/working/',              # path where the data (e.g. inter.csv) is stored
    'USER_ID_FIELD': 'session',                   # field used to identify users (sessions in this case)
    'ITEM_ID_FIELD': 'aid',                       # field used to identify items
    'TIME_FIELD': 'ts',                           # timestamp field for ordering interactions
    'user_inter_num_interval': "[5,Inf)",         # keep users with at least 5 interactions
    'item_inter_num_interval': "[5,Inf)",         # keep items with at least 5 interactions
    'load_col': {'inter': ['session', 'aid', 'ts']},  # load only these columns from the interaction file
    'train_neg_sample_args': None,                # no negative sampling (use full item ranking)
    'epochs': 10,                                 # number of training epochs
    'stopping_step': 3,                           # stop early if no improvement after 3 valid steps

    'eval_batch_size': 1024,                      # batch size during evaluation
    # 'train_batch_size': 1024,                   # (optional) batch size for training
    # 'enable_amp': True,                         # (optional) enable mixed-precision training
    'MAX_ITEM_LIST_LENGTH': MAX_ITEM,             # max number of past items used in sequence
    'eval_args': {
        'split': {'RS': [9, 1, 0]},               # random split: 90% train, 10% valid, 0% test
        'group_by': 'user',                       # group data per user/session
        'order': 'TO',                            # respect temporal order
        'mode': 'full'                            # use full item list for evaluation
    }
}

# initialize RecBole with the GRU4Rec model, dataset name, and custom parameters 
config = Config(model='GRU4Rec', dataset='recbox_data', config_dict=parameter_dict)

# init random seed
init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()

# create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# write config info into log
logger.info(config)


# create the dataset based on our configurations
dataset = create_dataset(config)
logger.info(dataset)


# prepare the dataset by splitting it into train, valid and test chunks
train_data, valid_data, test_data = data_preparation(config, dataset)


# model loading and initialization
model = GRU4Rec(config, train_data.dataset).to(config['device'])
logger.info(model)

# trainer loading and initialization
trainer = Trainer(config, model)

# model training
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)


import torch
from recbole.data.interaction import Interaction
from recbole.utils.case_study import full_sort_scores

def recommend_for_sessions(external_session_id, model, dataset, top_k=10):
     # Convert external session ID to internal
    internal_session_id = dataset.token2id('session', external_session_id)

    # load interaction data
    inter_feat = dataset.inter_feat
    inter_df = pl.DataFrame({
        'session': inter_feat['session'].tolist(),
        'aid': inter_feat['aid'].tolist(),
        'ts': inter_feat['ts'].tolist(),
    })

    # filter and sort session history
    session_history = (
        inter_df
        .filter(pl.col('session') == internal_session_id)
        .sort('ts')
    )
    internal_item_ids = session_history['aid'].to_list()

    # prepare interaction
    item_list_field = dataset.iid_field + '_list'
    user_field = dataset.uid_field

    interaction = Interaction({
        user_field: torch.tensor([internal_session_id]),
        item_list_field: torch.tensor([internal_item_ids]),
        'item_length': torch.tensor([len(internal_item_ids)]),
    })

    # predict scores
    model.eval()
    scores = model.full_sort_predict(interaction.to(model.device))
    top_k_indices = torch.topk(scores[0], k=top_k).indices.tolist()
    external_item_ids = dataset.id2token(dataset.iid_field, top_k_indices)

    # print result
    print(f"Top {top_k} recommended items for session {external_session_id}:")
    for internal_id, external_id in zip(top_k_indices, external_item_ids):
        print(f"  Internal ID: {internal_id}  →  External ID: {external_id}")


recommend_for_sessions('11098556', model, dataset, top_k=10)


recommend_for_sessions('11098530', model, dataset, top_k=10)


recommend_for_sessions('11098533', model, dataset, top_k=10)

