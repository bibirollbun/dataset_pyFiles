import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import os


TARGET_COLS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

DEMO_COLS = [
    'ncodpers', 'fecha_dato', 
    'sexo', 'age', 'antiguedad', 'renta', 'ind_nuevo', 
    'ind_actividad_cliente', 'tiprel_1mes', 'pais_residencia', 'canal_entrada'
]

DTYPE_DICT = {
    'ncodpers': int, 
    'ind_nom_pens_ult1': 'float32', 
    'ind_nomina_ult1': 'float32',
    'conyuemp': 'object'
}
for col in TARGET_COLS:
    DTYPE_DICT[col] = 'float32'


def get_data(file_name, dates_to_keep=None):
    if 'test' in file_name.lower():
        cols_to_use = DEMO_COLS
    else:
        cols_to_use = DEMO_COLS + TARGET_COLS

    filtered_chunks = []
    
    if dates_to_keep:
        target_dates = pd.to_datetime(dates_to_keep)
  
    chunk_iter = pd.read_csv(
        file_name, 
        dtype=DTYPE_DICT, 
        na_values=[' NA', 'NA', '   NA'], 
        usecols=cols_to_use,  
        chunksize=50_000
    )
    
    for i, chunk in enumerate(chunk_iter):
        chunk['fecha_dato'] = pd.to_datetime(chunk['fecha_dato'], format='%Y-%m-%d')
        
        if dates_to_keep:
            chunk = chunk[chunk['fecha_dato'].isin(target_dates)]
        
        if len(chunk) > 0:
            filtered_chunks.append(chunk)
            
        if i % 10 == 0:
            gc.collect()
            
    if len(filtered_chunks) > 0:
        df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Loaded {len(df)} rows from {file_name}.")
    else:
        print(f"Warning: No data loaded from {file_name}!")
        return pd.DataFrame(columns=cols_to_use)
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(40).astype('int8')
    df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce').fillna(0).astype('int16')
    df['renta'] = pd.to_numeric(df['renta'], errors='coerce').fillna(100000).astype('int32')
    df['ind_nuevo'] = pd.to_numeric(df['ind_nuevo'], errors='coerce').fillna(0).astype('int8')
    df['ind_actividad_cliente'] = pd.to_numeric(df['ind_actividad_cliente'], errors='coerce').fillna(0).astype('int8')
    
    mapping_dict = {
        'sexo': {'V': 0, 'H': 1, -99: 0},
        'tiprel_1mes': {'A': 0, 'I': 1, 'P': 2, 'R': 3, -99: 1}
    }
    df['sexo'] = df['sexo'].map(mapping_dict['sexo']).fillna(0).astype('int8')
    df['tiprel_1mes'] = df['tiprel_1mes'].map(mapping_dict['tiprel_1mes']).fillna(1).astype('int8')
    
    for col in ['pais_residencia', 'canal_entrada']:
        if col in df.columns:
            df[col] = pd.factorize(df[col])[0].astype('int16')

    # Якщо це train файл, заповнюємо пропуски в таргетах
    # Якщо test - таргетів тут немає, пропускаємо
    existing_targets = [col for col in TARGET_COLS if col in df.columns]
    if existing_targets:
        df[existing_targets] = df[existing_targets].fillna(0).astype('int8')
    
    return df


def make_prev_month_features(df, date_current, date_prev):
    print(f"Creating lag features...")
    df_curr = df[df['fecha_dato'] == date_current][['ncodpers'] + TARGET_COLS].copy()
    df_prev = df[df['fecha_dato'] == date_prev].copy()
    
    df_curr.drop_duplicates('ncodpers', inplace=True)
    df_prev.drop_duplicates('ncodpers', inplace=True)
    
    joined = df_prev.merge(df_curr, on='ncodpers', how='inner', suffixes=('', '_target'))
    return joined


if __name__ == "__main__":
    df_train_raw = get_data('train_ver2.csv', ['2015-05-28', '2015-06-28'])
    train_df = make_prev_month_features(df_train_raw, '2015-06-28', '2015-05-28')
    
    del df_train_raw
    gc.collect()

    X_cols = ['sexo', 'age', 'antiguedad', 'renta', 'ind_nuevo', 
              'ind_actividad_cliente', 'tiprel_1mes', 'pais_residencia', 'canal_entrada'] + TARGET_COLS
    
    X = []
    y = []
    
    for i, prod in enumerate(TARGET_COLS):
        target_col = prod + '_target'
        new_buyers = train_df[(train_df[target_col] == 1) & (train_df[prod] == 0)]
        if not new_buyers.empty:
            X.append(new_buyers[X_cols].values)
            y.append(np.full(len(new_buyers), i))
            
    if len(X) > 0:
        X = np.vstack(X)
        y = np.concatenate(y)
        
        del train_df
        gc.collect()
        
        
        params = {
            'objective': 'multiclass',
            'num_class': len(TARGET_COLS),
            'metric': 'multi_logloss',
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 60,
            'verbosity': -1,
            'seed': 42
        }
        d_train = lgb.Dataset(X, label=y)
        model = lgb.train(params, d_train, num_boost_round=120)
        
        del X, y, d_train
        gc.collect()
        
        
        df_may16 = get_data('train_ver2.csv', ['2016-05-28'])
        
        df_test = get_data('test_ver2.csv') 
        
        df_test = df_test.merge(df_may16[['ncodpers'] + TARGET_COLS], on='ncodpers', how='left')
        
        df_test[TARGET_COLS] = df_test[TARGET_COLS].fillna(0)
        
        X_test = df_test[X_cols].values
        preds = model.predict(X_test)
        
        
        prev_products = X_test[:, -24:]
        
        result = []
        for i in range(len(preds)):
            p = preds[i, :]
            p = p * (1 - prev_products[i, :])
            
            top7_idx = p.argsort()[::-1][:7]
            top7_names = [TARGET_COLS[j] for j in top7_idx]
            result.append(" ".join(top7_names))
            
        submission = pd.DataFrame({
            'ncodpers': df_test['ncodpers'].astype(int),
            'added_products': result
        })
        submission.to_csv('submission.csv', index=False)
        

