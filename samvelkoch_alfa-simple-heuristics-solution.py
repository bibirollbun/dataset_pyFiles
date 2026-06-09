import pandas as pd

train = pd.read_parquet('/kaggle/input/alpha-summer-challenge/train.pa')
transactions = pd.read_parquet('/kaggle/input/alpha-summer-challenge/df_transaction.pa')

train_clients = set(train['client_num'])
test_clients = set(transactions['client_num']) - train_clients

train_txns = transactions[transactions['client_num'].isin(train_clients)].merge(train, on='client_num')


# Find exclusive merchants for each target class
exclusive_by_target = {}

for target in range(7):
    target_merchants = set(train_txns[train_txns['target'] == target]['merchant_name'])
    other_merchants = set(train_txns[train_txns['target'] != target]['merchant_name'])
    exclusive_by_target[target] = target_merchants - other_merchants

test_txns = transactions[transactions['client_num'].isin(test_clients)]
test_features = test_txns.groupby('client_num').agg({
    'amount': 'sum',
    'merchant_name': lambda x: set(x)
}).reset_index()

test_features.columns = ['client_num', 'total_amount', 'merchants']


def predict(row):
    client_merchants = row['merchants']
    
    # Check exclusive merchants first
    for target, exclusive_merchants in exclusive_by_target.items():
        if client_merchants & exclusive_merchants:
            return target
    
    # Fallback to spending buckets (based on training data analysis)
    amount = row['total_amount']
    if amount < 130000: return 0      # Class 0: mean 199k
    elif amount < 200000: return 1    # Class 1: mean 171k  
    elif amount < 240000: return 2    # Class 2: mean 219k
    elif amount < 310000: return 3    # Class 3: mean 260k
    elif amount < 435000: return 4    # Class 4: mean 360k
    elif amount < 670000: return 5    # Class 5: mean 509k
    else: return 6                    # Class 6: mean 825k


test_features['target'] = test_features.apply(predict, axis=1)


submission = pd.DataFrame({
    'client_num': list(test_clients),
    'target': 1 
})
submission = submission.merge(test_features[['client_num', 'target']], on='client_num', how='left', suffixes=['', '_pred'])
submission['target'] = submission['target_pred'].fillna(submission['target']).astype(int)
submission = submission[['client_num', 'target']].sort_values('client_num')

submission.to_csv('submission.csv', index=False)

