import os
import logging
import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch

from data_utils import load_data, create_feature_dataframe, merge_target
from train_catboost import train_model, make_predictions, get_feature_importance, plot_feature_importance, get_topk_features, save_models
from generate_smart_money_list_leak_free import generate_smart_money_list_strict_past

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Set up plotly to display plots properly when the notebook is viewed
pio.renderers.default = 'iframe'


TRANSACTIONS_CSV_PATH = "/kaggle/input/pumpfun-30s-september-2025"

TARGETS_CSV_PATH = 'https://drive.usercontent.google.com/u/0/uc?id=1EsqpZXPBU-6m0djDmccCrtUX07jV2fHA'

EVAL_CSV_PATH = '/kaggle/input/alpha-radar-solana-sprint'

# Load and Preprocess Data
transactions_df, targets_df, eval_df = load_data(TRANSACTIONS_CSV_PATH, TARGETS_CSV_PATH, EVAL_CSV_PATH)


# Taking a look at how a single token's info change from first to last transaction timestamp

rand_mint_token_id = transactions_df.sample(n=1).mint_token_id.iloc[0]

token_data = transactions_df.loc[transactions_df['mint_token_id']==rand_mint_token_id].sort_values(by='timestamp', ascending=True)
token_data['is_target'] = token_data.iloc[0].mint_token_id in targets_df['Target Token Addresses'].values

token_data[['timestamp','mint_token_id', 'market_cap_usd', 'sol_volume', 'trade_mode', 'is_target']].head()



fig = px.line(token_data, x='timestamp', y='market_cap_usd', title='Market cap (USD) vs Timestamp', color='is_target')
fig.show()


# create smart money features helper files

if not (os.path.isfile("./smart_money_list_strict_past.csv") and os.path.isfile("./token_order.csv")):
    logging.info('Creating smart money and target tokens CSVs')
    generate_smart_money_list_strict_past(TRANSACTIONS_CSV_PATH, TARGETS_CSV_PATH)
else:
    logging.info('Smart money and token order CSV files already exist. Skipping creation...')


# Path for the final output file
OUTPUT_FEATURES_PATH_TRAIN = './processed_token_features_train.csv'
OUTPUT_FEATURES_PATH_EVAL = './processed_token_features_eval.csv'

create_features = True # Recreate new training and evaluation features dataframes

if create_features:
    logging.info('create_features is enabled. Creating train and eval features dataframes. This may take a while...\n')
    features_df = create_feature_dataframe(transactions_df)
    eval_df = create_feature_dataframe(eval_df, save_path=OUTPUT_FEATURES_PATH_EVAL)
    
    # Merge Target Variable
    train_df = merge_target(features_df, targets_df)
    
    # Save to CSV
    try:
        logging.info(f"Saving train data to {OUTPUT_FEATURES_PATH_TRAIN}...")
        train_df.to_csv(OUTPUT_FEATURES_PATH_TRAIN, index=False)
        logging.info("--- Process Complete ---")
    except Exception as e:
        logging.error(f"Error during saving to {OUTPUT_FEATURES_PATH_TRAIN}: {e}", exc_info=True)
        
    logging.info("\nPositive class distribution:")
    logging.info(train_df['is_target'].value_counts(normalize=True))
    
elif os.path.isdir(OUTPUT_FEATURES_PATH_TRAIN) and os.path.exists(OUTPUT_FEATURES_PATH_EVAL):
    logging.info('create_features is False and train and eval features CSV files already exist. Skipping their creation...')


device = 'GPU' if torch.cuda.is_available() else 'CPU'
device


models, best_threshold = train_model(task_type=device)


feature_importance = get_feature_importance(models)

feature_importance.head()


plot_feature_importance(feature_importance)


top10_features = get_topk_features(feature_importance, 10)
top10_features


make_predictions(models, eval_df, optimal_threshold=best_threshold)



save_models(models, best_threshold, './models')

