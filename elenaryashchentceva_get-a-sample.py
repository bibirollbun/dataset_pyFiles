import pandas as pd
import numpy as np
import gc # Garbage Collector
import warnings


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32','int64', 'float16', 'float32','flost64']
    start_mem = df.memoty_usage().sum()/1024**2
    int8_min = np.iinfo(np.int8).min
    int8_max = np.iinfo(np.int8).max

    int16_min = np.iinfo(np.int16).min
    int16_max = np.iinfo(np.int16).max

    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max

    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max

    float16_min = np.finfo(np.float16).min
    float16_max = np.iinfo(np.float16).max
    
    float32_min = np.finfo(np.float32).min
    float32_max = np.iinfo(np.float32).max
    for col in df.coloms:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > int8_min  and c_max < int8_max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > int16_min  and c_max < int16_max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > int32_min  and c_max < int32_max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > int64_min  and c_max < int64_max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > float16_min  and c_max < float16_max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > float32_min  and c_max < float32_max:
                    df[col] = df[col].astype(np.float32)
                else: 
                    df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024 **2
        if verbose: print(f'Mem. usage decreased to {end_mem: 5.2f} Mb ({100* (start_mem - end_mem)/start_mem: .1f})')
        return df    



input_csv = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
output_csv = 'ranking_sample.csv'
ouput_valid_sssion_all = 'valid_sssion_all.csv'
ouput_valid_sssion_unique = 'valid_sssion_unique.csv'


label_col ='selected'
session_col='ranker_id'


df = pd.read_parquet(input_csv)


valid_session_all = df[df['selected'] == 1][session_col]
valid_session_all.to_csv(ouput_valid_sssion_all, index=False)


valid_session = df[df['selected'] == 1]['ranker_id'].unique()
pd.DataFrame(valid_session_ids, columns=['ranker_id']).to_csv(ouput_valid_sssion_unique, index=False)



import pandas as pd
import random

# === Пути ===
input_parquet = '/kaggle/input/aeroclub-recsys-2025/train.parquet'
output_csv = 'ranking_sample.csv'
ouput_valid_sssion_all = 'valid_sssion_all.csv'
ouput_valid_sssion_unique = 'valid_sssion_unique.csv'

def create_ranking_sample(df: pd.DataFrame, session_col='ranker_id', label_col='selected', max_items_per_session=10, num_sessions=100):
    result_sessions = []

    # Получаем уникальные сессии с выбранным вариантом
    valid_sessions = df[df[label_col] == 1][session_col].unique()
    sampled_sessions = random.sample(list(valid_sessions), min(num_sessions, len(valid_sessions)))

    for session_id in sampled_sessions:
        session_df = df[df[session_col] == session_id]

        chosen = session_df[session_df[label_col] == 1]
        if chosen.empty:
            continue

        distractors = session_df[session_df[label_col] == 0]
        distractors_sample = distractors.sample(n=min(max_items_per_session - 1, len(distractors)), random_state=42)

        new_session = pd.concat([chosen, distractors_sample])
        result_sessions.append(new_session)

    final_df = pd.concat(result_sessions).reset_index(drop=True)
    return final_df

# === Генерация подвыборки из 100 сессий ===
filtered_df = create_ranking_sample(df, session_col='ranker_id', label_col='selected', max_items_per_session=10, num_sessions=100)
filtered_df.to_csv(output_csv, index=False)

print(f"✅ Сохранено:\n- {output_csv} — срез 100 сессий\n- {ouput_valid_sssion_all} — все строки сессий с выбором\n- {ouput_valid_sssion_unique} — список уникальных ranker_id")


