import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

train = pd.read_csv('/kaggle/input/bakery-sales-prediction-summer-2025/train.csv')
test = pd.read_csv('/kaggle/input/bakery-sales-prediction-summer-2025/test.csv')
wetter = pd.read_csv('/kaggle/input/bakery-sales-prediction-summer-2025/wetter.csv')
kiwo = pd.read_csv('/kaggle/input/bakery-sales-prediction-summer-2025/kiwo.csv')

for df in [train, test, wetter, kiwo]:
    df['Datum'] = pd.to_datetime(df['Datum'])

train = train.merge(wetter, on='Datum', how='left').merge(kiwo, on='Datum', how='left')
test = test.merge(wetter, on='Datum', how='left').merge(kiwo, on='Datum', how='left')

for df in [train, test]:
    df['dayofweek'] = df['Datum'].dt.dayofweek
    df['month'] = df['Datum'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

for col in ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode']:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

feature_cols = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
                'KielerWoche', 'dayofweek', 'month', 'is_weekend']

submission = pd.DataFrame()
mape_list = []

for wg in train['Warengruppe'].unique():
    train_wg = train[train['Warengruppe'] == wg]
    test_wg = test[test['Warengruppe'] == wg]
    
    X = train_wg[feature_cols]
    y = train_wg['Umsatz']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {'objective': 'regression', 'metric': 'mape', 'verbosity': -1}
    model = lgb.train(params, lgb_train, valid_sets=[lgb_val])
    
    val_preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, val_preds)
    mape_list.append(mape)
    print(f"Warengruppe {wg} Validation MAPE: {mape:.4f}")
    
    X_test = test_wg[feature_cols]
    preds = model.predict(X_test)
    
    temp_sub = pd.DataFrame({
        'id': test_wg['id'],
        'Umsatz': preds
    })
    
    submission = pd.concat([submission, temp_sub], axis=0)

avg_mape = sum(mape_list) / len(mape_list)
print(f"Average Validation MAPE across all Warengruppen: {avg_mape:.4f}")

submission = submission.sort_values('id')
submission.to_csv('/kaggle/working/submission_lightgbm.csv', index=False)

submission.head()


