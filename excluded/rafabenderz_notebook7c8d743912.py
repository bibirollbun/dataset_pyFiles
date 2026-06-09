import pandas as pd
import numpy as np

# Wczytanie danych z Kaggle
print("Wczytywanie danych...")

# Ścieżka do danych na Kaggle
data_path = "/kaggle/input/linking-writing-processes-to-writing-quality/"

train_logs = pd.read_csv(f"{data_path}train_logs.csv")
train_scores = pd.read_csv(f"{data_path}train_scores.csv")

print(f"Wczytano {len(train_logs)} wierszy logów")
print(f"Wczytano {len(train_scores)} esejów")


# Sortowanie po id i event_id
train_logs = train_logs.sort_values(['id', 'event_id']).reset_index(drop=True)

# Funkcja do identyfikacji czy aktywność przerywa R-Burst
def is_r_burst_interruptor(activity):
    """Sprawdza czy aktywność przerywa R-Burst"""
    if pd.isna(activity):
        return False
    activity_str = str(activity)
    return (activity_str == 'Nonproduction' or 
            activity_str == 'Remove/Cut' or 
            activity_str == 'Replace' or 
            activity_str.startswith('Move'))

print("Funkcja is_r_burst_interruptor zdefiniowana")


def calculate_r_burst_features_fast(df):
    """Szybka wersja obliczania cech R-Burst"""
    results = []
    
    for essay_id, essay_data in df.groupby('id'):
        essay_data = essay_data.sort_values('event_id').reset_index(drop=True)
        
        # Identyfikacja R-Bursts używając wektoryzacji
        is_input = (essay_data['activity'] == 'Input').values
        is_interruptor = essay_data['activity'].apply(is_r_burst_interruptor).values
        
        # Znajdź granice R-Bursts
        r_bursts = []
        current_burst_start = None
        
        for i in range(len(essay_data)):
            if is_input[i]:
                if current_burst_start is None:
                    current_burst_start = i
            else:
                if current_burst_start is not None:
                    r_bursts.append((current_burst_start, i))
                    current_burst_start = None
                if is_interruptor[i]:
                    continue
        
        # Dodaj ostatni burst jeśli istnieje
        if current_burst_start is not None:
            r_bursts.append((current_burst_start, len(essay_data)))
        
        # Obliczanie cech
        if len(r_bursts) == 0:
            words_no_pause_r_burst = 0
            writing_time_no_pause_r_burst = 0
            group_r_burst = 0
        else:
            max_words = 0
            max_time = 0
            
            for start_idx, end_idx in r_bursts:
                burst_data = essay_data.iloc[start_idx:end_idx]
                
                if len(burst_data) > 0:
                    # Liczba wyrazów w burst
                    first_word_count = burst_data.iloc[0]['word_count']
                    last_word_count = burst_data.iloc[-1]['word_count']
                    words_in_burst = max(0, last_word_count - first_word_count)
                    max_words = max(max_words, words_in_burst)
                    
                    # Czas burst - różnica między ostatnim up_time a pierwszym down_time
                    if len(burst_data) > 0:
                        first_time = burst_data.iloc[0]['down_time']
                        last_time = burst_data.iloc[-1]['up_time']
                        time_in_burst = last_time - first_time
                    else:
                        time_in_burst = 0
                    max_time = max(max_time, time_in_burst)
            
            words_no_pause_r_burst = max_words
            writing_time_no_pause_r_burst = max_time
            group_r_burst = len(r_bursts)
        
        results.append({
            'id': essay_id,
            'words_no_pause_r_burst': words_no_pause_r_burst,
            'writing_time_no_pause_r_burst': writing_time_no_pause_r_burst,
            'group_r_burst': group_r_burst
        })
    
    return pd.DataFrame(results)

print("Funkcja calculate_r_burst_features_fast zdefiniowana")


# Obliczanie cech R-Burst
print("Konstruowanie R-Bursts...")
r_burst_features = calculate_r_burst_features_fast(train_logs)
print(f"Obliczono cechy R-Burst dla {len(r_burst_features)} esejów")


# Tworzenie grouped_train_logs (grupowanie po id)
print("Tworzenie grouped_train_logs...")
grouped_train_logs = train_logs.groupby('id').agg({
    'event_id': 'count',
    'word_count': 'max',
    'action_time': 'sum',
    'down_time': 'min',
    'up_time': 'max'
}).reset_index()
grouped_train_logs.columns = ['id', 'total_events', 'max_word_count', 'total_action_time', 'start_time', 'end_time']

# Dodanie cech R-Burst
grouped_train_logs = grouped_train_logs.merge(r_burst_features, on='id', how='left')

# Wypełnienie NaN wartości zerami
grouped_train_logs['words_no_pause_r_burst'] = grouped_train_logs['words_no_pause_r_burst'].fillna(0)
grouped_train_logs['writing_time_no_pause_r_burst'] = grouped_train_logs['writing_time_no_pause_r_burst'].fillna(0)
grouped_train_logs['group_r_burst'] = grouped_train_logs['group_r_burst'].fillna(0)

print("Dodatkowe cechy R-Burst dodane do grouped_train_logs")
print(f"Liczba esejów: {len(grouped_train_logs)}")


def calculate_additional_features_fast(df):
    """Szybka wersja obliczania dodatkowych cech"""
    results = []
    
    for essay_id, essay_data in df.groupby('id'):
        essay_data = essay_data.sort_values('event_id').reset_index(drop=True)
        
        # Identyfikacja R-Bursts (używamy tej samej logiki)
        is_input = (essay_data['activity'] == 'Input').values
        is_interruptor = essay_data['activity'].apply(is_r_burst_interruptor).values
        
        r_bursts = []
        current_burst_start = None
        
        for i in range(len(essay_data)):
            if is_input[i]:
                if current_burst_start is None:
                    current_burst_start = i
            else:
                if current_burst_start is not None:
                    r_bursts.append((current_burst_start, i))
                    current_burst_start = None
                if is_interruptor[i]:
                    continue
        
        if current_burst_start is not None:
            r_bursts.append((current_burst_start, len(essay_data)))
        
        # 1. Średnia długość R-Burst
        avg_words_per_r_burst = 0
        if len(r_bursts) > 0:
            words_per_burst = []
            for start_idx, end_idx in r_bursts:
                burst_data = essay_data.iloc[start_idx:end_idx]
                if len(burst_data) > 0:
                    first_word_count = burst_data.iloc[0]['word_count']
                    last_word_count = burst_data.iloc[-1]['word_count']
                    words_in_burst = max(0, last_word_count - first_word_count)
                    words_per_burst.append(words_in_burst)
            if len(words_per_burst) > 0:
                avg_words_per_r_burst = np.mean(words_per_burst)
        
        # 2. Stosunek czasu Input do całkowitego czasu eseju
        input_mask = essay_data['activity'] == 'Input'
        total_input_time = essay_data.loc[input_mask, 'action_time'].sum()
        total_essay_time = essay_data['action_time'].sum()
        input_time_ratio = total_input_time / total_essay_time if total_essay_time > 0 else 0
        
        # 3. Liczba operacji Remove/Cut per 1000 wyrazów
        max_words = essay_data['word_count'].max()
        remove_cut_count = (essay_data['activity'] == 'Remove/Cut').sum()
        remove_cut_per_1000_words = (remove_cut_count / max_words * 1000) if max_words > 0 else 0
        
        # 4. Średni czas między kolejnymi akcjami Input
        input_actions = essay_data[input_mask].copy()
        avg_inter_key_interval = 0
        if len(input_actions) > 1:
            input_actions = input_actions.sort_values('event_id')
            prev_up = input_actions['up_time'].values[:-1]
            curr_down = input_actions['down_time'].values[1:]
            intervals = curr_down - prev_up
            intervals = intervals[intervals >= 0]
            if len(intervals) > 0:
                avg_inter_key_interval = intervals.mean()
        
        # 5. Liczba operacji Replace
        replace_count = (essay_data['activity'] == 'Replace').sum()
        
        # 6. Stosunek liczby R-Burst do maksymalnej liczby wyrazów
        r_burst_count = len(r_bursts)
        r_burst_frequency = (r_burst_count / max_words) if max_words > 0 else 0
        
        # 7. Maksymalna długość ciągłej sekwencji Input
        max_consecutive_input = 0
        current_consecutive = 0
        for val in is_input:
            if val:
                current_consecutive += 1
                max_consecutive_input = max(max_consecutive_input, current_consecutive)
            else:
                current_consecutive = 0
        
        results.append({
            'id': essay_id,
            'avg_words_per_r_burst': avg_words_per_r_burst,
            'input_time_ratio': input_time_ratio,
            'remove_cut_per_1000_words': remove_cut_per_1000_words,
            'avg_inter_key_interval': avg_inter_key_interval,
            'replace_count': replace_count,
            'r_burst_frequency': r_burst_frequency,
            'max_consecutive_input': max_consecutive_input
        })
    
    return pd.DataFrame(results)

print("Funkcja calculate_additional_features_fast zdefiniowana")


# Obliczanie dodatkowych cech
print("Tworzenie dodatkowych cech...")
additional_features = calculate_additional_features_fast(train_logs)

# Dodanie dodatkowych cech do grouped_train_logs
grouped_train_logs = grouped_train_logs.merge(additional_features, on='id', how='left')

# Wypełnienie NaN wartości zerami
additional_cols = ['avg_words_per_r_burst', 'input_time_ratio', 'remove_cut_per_1000_words', 
                   'avg_inter_key_interval', 'replace_count', 'r_burst_frequency', 'max_consecutive_input']
for col in additional_cols:
    grouped_train_logs[col] = grouped_train_logs[col].fillna(0)

print(f"Liczba esejów: {len(grouped_train_logs)}")
print(f"Dodatkowe cechy dodane: {len(additional_cols)}")


essay_ffb8c745 = grouped_train_logs[grouped_train_logs['id'] == 'ffb8c745']

if len(essay_ffb8c745) > 0:
    words_value = essay_ffb8c745['words_no_pause_r_burst'].values[0]
    print(f"Odpowiedź 1: words_no_pause_r_burst = {words_value}")
    # Zapisanie wartości do zmiennej dla dalszego użycia
    answer_1 = words_value
else:
    print("Esej o ID ffb8c745 nie został znaleziony")
    answer_1 = None


if len(essay_ffb8c745) > 0:
    time_value = essay_ffb8c745['writing_time_no_pause_r_burst'].values[0]
    print(f"Odpowiedź 2: writing_time_no_pause_r_burst = {time_value}")
    # Zapisanie wartości do zmiennej dla dalszego użycia
    answer_2 = time_value
else:
    print("Esej o ID ffb8c745 nie został znaleziony")
    answer_2 = None


if len(essay_ffb8c745) > 0:
    group_value = essay_ffb8c745['group_r_burst'].values[0]
    print(f"Odpowiedź 3: group_r_burst = {group_value}")
    # Zapisanie wartości do zmiennej dla dalszego użycia
    answer_3 = group_value
else:
    print("Esej o ID ffb8c745 nie został znaleziony")
    answer_3 = None


if len(essay_ffb8c745) > 0:
    row = essay_ffb8c745.iloc[0]
    print("="*60)
    print("Wszystkie cechy dla eseju ffb8c745:")
    print("="*60)
    print(f"\nCECHY R-BURST:")
    print(f"  words_no_pause_r_burst: {row['words_no_pause_r_burst']}")
    print(f"  writing_time_no_pause_r_burst: {row['writing_time_no_pause_r_burst']}")
    print(f"  group_r_burst: {row['group_r_burst']}")
    
    print(f"\nDODATKOWE CECHY:")
    print(f"  avg_words_per_r_burst: {row['avg_words_per_r_burst']:.2f}")
    print(f"  input_time_ratio: {row['input_time_ratio']:.4f}")
    print(f"  remove_cut_per_1000_words: {row['remove_cut_per_1000_words']:.2f}")
    print(f"  avg_inter_key_interval: {row['avg_inter_key_interval']:.2f}")
    print(f"  replace_count: {row['replace_count']}")
    print(f"  r_burst_frequency: {row['r_burst_frequency']:.4f}")
    print(f"  max_consecutive_input: {row['max_consecutive_input']}")
else:
    print("Esej o ID ffb8c745 nie został znaleziony")


# Zapisanie wyników do pliku CSV
grouped_train_logs.to_csv('grouped_train_logs.csv', index=False)
print("Wyniki zapisane w pliku: grouped_train_logs.csv")
print(f"\nLiczba esejów: {len(grouped_train_logs)}")
print(f"\nCechy R-Burst:")
print("  - words_no_pause_r_burst")
print("  - writing_time_no_pause_r_burst")
print("  - group_r_burst")
print(f"\nDodatkowe cechy ({len(additional_cols)}):")
for col in additional_cols:
    print(f"  - {col}")

