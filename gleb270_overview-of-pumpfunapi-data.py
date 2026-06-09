import polars as pl


pump_info = pl.read_parquet('/kaggle/input/pump-fun-api-solana-tokens-info/pump_fun_api_info.parquet')
dune_info = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/dune_token_info_v2.csv')
onchain_divers_info = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/token_info_onchain_divers_v2.csv')
train = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/train.csv')
test = pl.read_csv('/kaggle/input/pump-fun-graduation-february-2025/test_unlabeled.csv')


pump_info.head()


pump_info.group_by('symbol').len().sort('len', descending=True).head(10)


pump_mints = set(pump_info['mint'])
train_mints = set(train['mint'])
test_mints = set(test['mint'])
onchain_mints = set(onchain_divers_info['mint'].str.strip_chars_end('\x00'))
dune_mints = set(dune_info['token_mint_address'])


len(train_mints - dune_mints), len(test_mints - dune_mints)


len(train_mints - onchain_mints), len(test_mints - onchain_mints)


len(train_mints - pump_mints), len(test_mints - pump_mints)


len(train_mints - pump_mints - dune_mints), len(test_mints - pump_mints - dune_mints)


len(train_mints - pump_mints - onchain_mints), len(test_mints - pump_mints - onchain_mints)

