import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import zipfile
import gc

for filename in ['train_ver2.csv', 'test_ver2.csv']:
    with zipfile.ZipFile(f"/kaggle/input/santander-product-recommendation/{filename}.zip", 'r') as z:
        z.extractall("/kaggle/working/")

# Типи даних
dtype_dict = {
    'ncodpers': 'int32',
    'ind_actividad_cliente': 'float32',
    'renta': 'float32'
}

target_cols = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 
    'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 
    'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 
    'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
    'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
    'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
    'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
]

target_cols = target_cols[2:]

for col in target_cols:
    dtype_dict[col] = 'float32'

df = pd.read_csv(
    "/kaggle/working/train_ver2.csv", 
    usecols=['ncodpers', 'fecha_dato'] + target_cols,
    dtype=dtype_dict
)

# Беремо тільки ТРАВЕНЬ 2016 
df_may16 = df[df['fecha_dato'] == '2016-05-28'].copy()
del df
gc.collect()

print(f"Дані за травень 2016: {df_may16.shape}")

# Заповнюємо пропуски нулями
df_may16[target_cols] = df_may16[target_cols].fillna(0)


# ПІДГОТОВКА МАТРИЦІ (User-Item Matrix)

# Встановлюємо ID клієнта як індекс
user_item_matrix = df_may16.set_index('ncodpers')[target_cols]

#  SVD 

svd = TruncatedSVD(n_components=10, random_state=42)


matrix_reduced = svd.fit_transform(user_item_matrix)
matrix_reconstructed = svd.inverse_transform(matrix_reduced)


preds_df = pd.DataFrame(
    matrix_reconstructed, 
    index=user_item_matrix.index, 
    columns=target_cols
)


# ---  ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ---

# Беремо 5 випадкових клієнтів для демонстрації
sample_users = preds_df.sample(5).index

for user_id in sample_users:
    # Що клієнт вже має (реальні дані)
    existing_products = user_item_matrix.loc[user_id]
    owned_list = existing_products[existing_products == 1].index.tolist()
    
    #  SVD (ймовірності)
    user_scores = preds_df.loc[user_id].sort_values(ascending=False)
    
    # Фільтруємо: прибираємо те, що вже є
    recommendations = []
    for prod, score in user_scores.items():
        if prod not in owned_list:
            recommendations.append(prod)
        if len(recommendations) >= 5: # Топ 5 рекомендацій
            break
            

print("\n Таблиця рекомендацій (Топ-10 рядків):")
final_results = []

for user_id in preds_df.index[:10]:
    user_scores = preds_df.loc[user_id].sort_values(ascending=False)
    existing = user_item_matrix.loc[user_id]
    owned = existing[existing == 1].index
    
    recs = [prod for prod in user_scores.index if prod not in owned][:7]
    final_results.append({'ncodpers': user_id, 'recommendations': " ".join(recs)})

results_df = pd.DataFrame(final_results)

pd.set_option('display.max_colwidth', None) 
display(results_df)

