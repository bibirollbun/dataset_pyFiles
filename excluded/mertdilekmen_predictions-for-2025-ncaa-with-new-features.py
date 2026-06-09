import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

path = "/kaggle/input/march-machine-learning-mania-2025/"

def add_basketball_features(df):
    """
    Adds advanced stats (eFG, TOV, ORB, FT_R, OE, DE, Poss) for both
    the winning (W_...) and losing (L_...) teams in the given DataFrame.
    Returns the updated DataFrame.
    """
    # 1) W_eFG
    df['W_eFG'] = df.apply(
        lambda row: (row['WFGM'] + 0.5 * row['WFGM3']) / row['WFGA'] if row['WFGA'] != 0 else 0,
        axis=1
    )

    # 2) W_TOV
    df['W_TOV'] = df.apply(
        lambda row: row['WTO'] / (row['WFGA'] + 0.44 * row['WFTA'] + row['WTO'])
                    if (row['WFGA'] + 0.44 * row['WFTA'] + row['WTO']) != 0 else 0,
        axis=1
    )

    # 3) W_ORB
    df['W_ORB'] = df.apply(
        lambda row: row['WOR'] / (row['WOR'] + row['LDR']) if (row['WOR'] + row['LDR']) != 0 else 0,
        axis=1
    )

    # 4) W_FT_R
    df['W_FT_R'] = df.apply(
        lambda row: row['WFTA'] / row['WFGA'] if row['WFGA'] != 0 else 0,
        axis=1
    )

    # 5) W_OE
    df['W_OE'] = df.apply(
        lambda row: row['WScore'] / row['WFGA'] if row['WFGA'] != 0 else 0,
        axis=1
    )

    # 6) W_DE
    df['W_DE'] = df.apply(
        lambda row: row['LScore'] / (row['LFGA'] - row['LOR'] + row['LTO'] + 0.44 * row['LFTA'])
                    if (row['LFGA'] - row['LOR'] + row['LTO'] + 0.44 * row['LFTA']) != 0 else 0,
        axis=1
    )

    # 7) W_Poss
    df['W_Poss'] = df.apply(
        lambda row: 0.96 * (row['WFGA'] - row['WOR'] - row['WTO'] + 0.475 * row['WFTA'])
                    if (row['WFGA'] != 0 or row['WFTA'] != 0) else 0,
        axis=1
    )

    # L_eFG
    df['L_eFG'] = df.apply(
        lambda row: (row['LFGM'] + 0.5 * row['LFGM3']) / row['LFGA'] if row['LFGA'] != 0 else 0,
        axis=1
    )

    # L_TOV
    df['L_TOV'] = df.apply(
        lambda row: row['LTO'] / (row['LFGA'] + 0.44 * row['LFTA'] + row['LTO'])
                    if (row['LFGA'] + 0.44 * row['LFTA'] + row['LTO']) != 0 else 0,
        axis=1
    )

    # L_ORB
    df['L_ORB'] = df.apply(
        lambda row: row['LOR'] / (row['LOR'] + row['WDR']) if (row['LOR'] + row['WDR']) != 0 else 0,
        axis=1
    )

    # L_FT_R
    df['L_FT_R'] = df.apply(
        lambda row: row['LFTA'] / row['LFGA'] if row['LFGA'] != 0 else 0,
        axis=1
    )

    # L_OE
    df['L_OE'] = df.apply(
        lambda row: row['LScore'] / row['LFGA'] if row['LFGA'] != 0 else 0,
        axis=1
    )

    # L_DE
    df['L_DE'] = df.apply(
        lambda row: row['WScore'] / (row['WFGA'] - row['WOR'] + row['WTO'] + 0.44 * row['WFTA'])
                    if (row['WFGA'] - row['WOR'] + row['WTO'] + 0.44 * row['WFTA']) != 0 else 0,
        axis=1
    )

    # L_Poss
    df['L_Poss'] = df.apply(
        lambda row: 0.96 * (row['LFGA'] - row['LOR'] - row['LTO'] + 0.475 * row['LFTA'])
                    if (row['LFGA'] != 0 or row['LFTA'] != 0) else 0,
        axis=1
    )

    return df


# 2) MRegularSeasonDetailedResults + MNCAATourneyDetailedResults (Men)
df_m_reg = pd.read_csv(path + "MRegularSeasonDetailedResults.csv")
df_m_tour = pd.read_csv(path + "MNCAATourneyDetailedResults.csv")
df_m_combined = pd.concat([df_m_reg, df_m_tour], ignore_index=True)
df_m_combined = add_basketball_features(df_m_combined)

print("=== Men Combined Detailed with new features: ===")
print(df_m_combined.shape)
print(df_m_combined.head(10))

# 3) WRegularSeasonDetailedResults + WNCAATourneyDetailedResults (Women)
df_w_reg = pd.read_csv(path + "WRegularSeasonDetailedResults.csv")
df_w_tour = pd.read_csv(path + "WNCAATourneyDetailedResults.csv")
df_w_combined = pd.concat([df_w_reg, df_w_tour], ignore_index=True)
df_w_combined = add_basketball_features(df_w_combined)

print("\n=== Women Combined Detailed with new features: ===")
print(df_w_combined.shape)
print(df_w_combined.head(10))

print("\nAll Columns (Men):")
print(df_m_combined.columns)

print("\nAll Columns (Women):")
print(df_w_combined.columns)




"""
"It is necessary to test which of the generated features can be used for prediction. 
Features with an accuracy rate of over approximately ~0.7 will be utilized."
"""

def check_feature_hypotheses(df_control, label="Men"):
    """
    Checks each hypothesis (W_eFG > L_eFG, W_TOV < L_TOV, etc.) for the given df_control,
    prints the correctness and accuracy percentages, and labels the output with 'label'.
    """

    total_games = len(df_control)

    # 1) Hypothesis: W_eFG > L_eFG
    cond_eFG = df_control['W_eFG'] > df_control['L_eFG']
    correct_eFG = cond_eFG.sum()
    incorrect_eFG = total_games - correct_eFG
    ratio_eFG = correct_eFG / total_games * 100

    # 2) Hypothesis: W_TOV < L_TOV
    cond_TOV = df_control['W_TOV'] < df_control['L_TOV']
    correct_TOV = cond_TOV.sum()
    incorrect_TOV = total_games - correct_TOV
    ratio_TOV = correct_TOV / total_games * 100

    # 3) Hypothesis: W_ORB > L_ORB
    cond_ORB = df_control['W_ORB'] > df_control['L_ORB']
    correct_ORB = cond_ORB.sum()
    incorrect_ORB = total_games - correct_ORB
    ratio_ORB = correct_ORB / total_games * 100

    # 4) Hypothesis: W_FT_R > L_FT_R
    cond_FT_R = df_control['W_FT_R'] > df_control['L_FT_R']
    correct_FT_R = cond_FT_R.sum()
    incorrect_FT_R = total_games - correct_FT_R
    ratio_FT_R = correct_FT_R / total_games * 100

    # 5) Hypothesis: W_OE > L_OE
    cond_OE = df_control['W_OE'] > df_control['L_OE']
    correct_OE = cond_OE.sum()
    incorrect_OE = total_games - correct_OE
    ratio_OE = correct_OE / total_games * 100

    # 6) Hypothesis: W_DE < L_DE
    cond_DE = df_control['W_DE'] < df_control['L_DE']
    correct_DE = cond_DE.sum()
    incorrect_DE = total_games - correct_DE
    ratio_DE = correct_DE / total_games * 100

    # 7) Hypothesis: W_Poss > L_Poss
    cond_Poss = df_control['W_Poss'] > df_control['L_Poss']
    correct_Poss = cond_Poss.sum()
    incorrect_Poss = total_games - correct_Poss
    ratio_Poss = correct_Poss / total_games * 100

    print(f"\n=== Feature Hypothesis Check for {label} ===")
    print(f"Total number of matches: {total_games}\n")

    print(f"1) Hypothesis: W_eFG > L_eFG")
    print(f"   Correct:  {correct_eFG} | Wrong: {incorrect_eFG} | Accuracy: {ratio_eFG:.2f}%\n")

    print(f"2) Hypothesis: W_TOV < L_TOV")
    print(f"   Correct:  {correct_TOV} | Wrong: {incorrect_TOV} | Accuracy: {ratio_TOV:.2f}%\n")

    print(f"3) Hypothesis: W_ORB > L_ORB")
    print(f"   Correct:  {correct_ORB} | Wrong: {incorrect_ORB} | Accuracy: {ratio_ORB:.2f}%\n")

    print(f"4) Hypothesis: W_FT_R > L_FT_R")
    print(f"   Correct:  {correct_FT_R} | Wrong: {incorrect_FT_R} | Accuracy: {ratio_FT_R:.2f}%\n")

    print(f"5) Hypothesis: W_OE > L_OE")
    print(f"   Correct:  {correct_OE} | Wrong: {incorrect_OE} | Accuracy: {ratio_OE:.2f}%\n")

    print(f"6) Hypothesis: W_DE < L_DE")
    print(f"   Correct:  {correct_DE} | Wrong: {incorrect_DE} | Accuracy: {ratio_DE:.2f}%\n")

    print(f"7) Hypothesis: W_Poss > L_Poss")
    print(f"   Correct:  {correct_Poss} | Wrong: {incorrect_Poss} | Accuracy: {ratio_Poss:.2f}%\n")


# === Usage Example ===

# For men (df_m_combined)
check_feature_hypotheses(df_m_combined, label="Men")

# For women (df_w_combined)
check_feature_hypotheses(df_w_combined, label="Women")



import re

def compute_weighted_seeds(csv_path):
    """
    Reads the seeds CSV file (e.g., MNCAATourneySeeds.csv or WNCAATourneySeeds.csv),
    extracts the numeric part of the 'Seed' column, calculates 'Weight' by season,
    and returns a DataFrame with 'TeamID' and 'weighted_seed'.
    """

    def extract_seed_num(seed_str):
        """
        Example:
        'W01'   -> 1
        'X16a'  -> 16
        'Z08b'  -> 8
        'W15'   -> 15
        """
        nums = re.findall(r'\d+', seed_str)
        if nums:
            return int(nums[0])  # convert the first numeric part found to an integer
        else:
            return None          # if it doesn't find any number

    # 1) Read the seeds CSV
    df_seeds = pd.read_csv(csv_path)

    # 2) Convert the Seed column to an integer value
    df_seeds['SeedNum'] = df_seeds['Seed'].apply(extract_seed_num)

    # 3) Weight calculation (example: earliest season -> weight=1, next -> weight=2, etc.)
    df_seeds['Weight'] = df_seeds['Season'] - df_seeds['Season'].min() + 1 
     #Here, 1 is a completely arbitrary number. It is a parameter and can be updated to improve the model's performance.

    # 4) Calculate the weighted average Seed by team
    # Weighted Average = sum(SeedNum * Weight) / sum(Weight)
    df_performance_seeds = df_seeds.groupby('TeamID', as_index=False).apply(
        lambda d: pd.Series({
            'weighted_seed': (d['SeedNum'] * d['Weight']).sum() / d['Weight'].sum()
        })
    )

    # 5) Sort and reset index
    df_performance_seeds = df_performance_seeds.sort_values(by='TeamID', ascending=True)
    df_performance_seeds.reset_index(drop=True, inplace=True)

    return df_performance_seeds

# For men's seeds
m_Teams_performance_seeds = compute_weighted_seeds(path + "MNCAATourneySeeds.csv")
print("Men's Weighted Seeds:")
print(m_Teams_performance_seeds.head(9))

# For women's seeds
w_Teams_performance_seeds = compute_weighted_seeds(path + "WNCAATourneySeeds.csv")
print("\nWomen's Weighted Seeds:")
print(w_Teams_performance_seeds.head())



from collections import defaultdict


def compute_team_performance(df):
    """
    For the given combined detailed DataFrame (df),
    computes avg_eFG, avg_FT_R, avg_OE, avg_DE for each TeamID.
    The losing team's stats are multiplied by 0.8 (penalty).
    Also, the losing team uses the winner's DE (W_DE) multiplied by 0.8.

    Returns a DataFrame with columns: [TeamID, avg_eFG, avg_FT_R, avg_OE, avg_DE].
    """

    # Copy the original DataFrame to avoid modifying it directly
    temp_df = df.copy()

    # Dictionary to store cumulative stats and match counts
    stats_sum = defaultdict(lambda: {'eFG': 0.0, 'FT_R': 0.0, 'OE': 0.0, 'DE': 0.0, 'count': 0})

    # Iterate over each match
    for idx, row in temp_df.iterrows():
        wteam = row['WTeamID']
        lteam = row['LTeamID']

        # Winner team stats (added as is)
        stats_sum[wteam]['eFG'] += row['W_eFG']
        stats_sum[wteam]['FT_R'] += row['W_FT_R']
        stats_sum[wteam]['OE'] += row['W_OE']
        stats_sum[wteam]['DE'] += row['W_DE']
        stats_sum[wteam]['count'] += 1

        # Losing team stats (multiplied by 0.8)
        stats_sum[lteam]['eFG'] += 0.8 * row['L_eFG']
        stats_sum[lteam]['FT_R'] += 0.8 * row['L_FT_R']
        stats_sum[lteam]['OE'] += 0.8 * row['L_OE']
        # The losing team uses W_DE (with 0.8 multiplier)
        stats_sum[lteam]['DE'] += 0.8 * row['W_DE']
        stats_sum[lteam]['count'] += 1

    # Build the performance DataFrame
    records = []
    for team_id, vals in stats_sum.items():
        c = vals['count']
        if c > 0:
            records.append({
                'TeamID': team_id,
                'avg_eFG': vals['eFG'] / c,
                'avg_FT_R': vals['FT_R'] / c,
                'avg_OE': vals['OE'] / c,
                'avg_DE': vals['DE'] / c
            })

    df_performance = pd.DataFrame(records)
    df_performance = df_performance.sort_values(by='TeamID', ascending=True).reset_index(drop=True)

    return df_performance


# --- Example usage for men and women ---

# For men's DataFrame
m_Teams_performance = compute_team_performance(df_m_combined)
print("Men's Teams Performance:")
print(m_Teams_performance.head(10))

# For women's DataFrame
w_Teams_performance = compute_team_performance(df_w_combined)
print("\nWomen's Teams Performance:")
print(w_Teams_performance.head())



def merge_team_features(seeds_df, performance_df):
    """
    Merges two DataFrames (e.g., seeds performance and match performance)
    on 'TeamID', sorts by 'TeamID', and returns the merged result.
    """
    merged_df = pd.merge(
        seeds_df,
        performance_df,
        on='TeamID',   # Merge key
        how='inner'    # or 'left', 'outer', etc. if needed
    )
    merged_df = merged_df.sort_values(by='TeamID', ascending=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


#  For men's
m_teams_features = merge_team_features(m_Teams_performance_seeds, m_Teams_performance)
print("Men Teams Features:")
print(m_teams_features.head(10))

# 2) For women's
w_teams_features = merge_team_features(w_Teams_performance_seeds, w_Teams_performance)
print("\nWomen Teams Features:")
print(w_teams_features.head(10))


import pandas as pd

# 1) Load MTeams.csv and WTeams.csv (example)
df_mteams = pd.read_csv(path + "MTeams.csv")  # Male teams
df_wteams = pd.read_csv(path + "WTeams.csv")  # Female teams

# 2) m_teams_features and w_teams_features DataFrames
#    You already have these DataFrames, for example:
#    m_teams_features = merge_team_features(m_Teams_performance_seeds, m_Teams_performance)
#    w_teams_features = merge_team_features(w_Teams_performance_seeds, w_Teams_performance)
#    (These DataFrames contain columns such as TeamID, weighted_seed, avg_eFG, avg_FT_R, avg_OE, avg_DE, etc.)

# 3) Merge male teams with male features
men_teams_merged = pd.merge(
    df_mteams,            # MTeams.csv -> columns: [TeamID, TeamName, FirstD1Season, LastD1Season]
    m_teams_features,     # m_teams_features -> columns: [TeamID, weighted_seed, avg_eFG, ...]
    on="TeamID",          # Merge key
    how="left"            # Include all male teams; if a team has no features, NaN is used
)

# 4) Merge female teams with female features
women_teams_merged = pd.merge(
    df_wteams,            # WTeams.csv -> columns: [TeamID, TeamName]
    w_teams_features,     # w_teams_features -> columns: [TeamID, weighted_seed, avg_eFG, ...]
    on="TeamID",
    how="left"
)

# 5) Examine the results
print("=== Men Teams + Features ===")
print(men_teams_merged.head(10))

print("\n=== Women Teams + Features ===")
print(women_teams_merged.head(10))

# If you want to merge into a single DataFrame (typically male and female team IDs do not overlap):
combined_all = pd.concat([men_teams_merged, women_teams_merged], ignore_index=True)
combined_all.to_csv("all_teams_with_features.csv", index=False)

# Save men_teams_merged DataFrame as an Excel file
men_teams_merged.to_excel("men_teams_merged.xlsx", index=False)

# Save women_teams_merged DataFrame as an Excel file
women_teams_merged.to_excel("women_teams_merged.xlsx", index=False)

print("Men teams features saved to men_teams_merged.xlsx")
print("Women teams features saved to women_teams_merged.xlsx")



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    accuracy_score,
    f1_score,
    roc_auc_score
)

def unify_id(row, year=2025):
    """
    ID'yi zorunlu olarak TeamA < TeamB olacak şekilde düzenler,
    böylece '2025_1108_1378' ve '2025_1378_1108' aynı ID haline gelir.
    """
    tA = min(row["TeamA"], row["TeamB"])
    tB = max(row["TeamA"], row["TeamB"])
    return f"{year}_{int(tA)}_{int(tB)}"

def train_predict_from_basic_matches(df_matches, df_teams_features, df_sample_submission, year=2025, label="Men"):
    """
    1) df_matches: Sadece Season, WTeamID, LTeamID sütunlarını içerir.
    2) Her maç için iki satır oluşturulur (TeamA = kazanan => y=1, TeamA = kaybeden => y=0).
    3) TeamA ve TeamB özellikleri df_teams_features kullanılarak birleştirilir.
    4) RandomForest modeli eğitilir; Brier Skoru ve diğer metrikler aynı veri üzerinde hesaplanır.
    5) Örnek gönderim dosyasındaki sıraya göre potansiyel 2025 maç tahminlerini (sadece 'ID' ve 'Pred' sütunları) döndürür.
    """
    
    # 0) Gelişmiş istatistik sütunlarını kaldır.
    advanced_cols = [
        "W_eFG", "W_TOV", "W_ORB", "W_FT_R", "W_OE", "W_DE", "W_Poss",
        "L_eFG", "L_TOV", "L_ORB", "L_FT_R", "L_OE", "L_DE", "L_Poss"
    ]
    df_matches = df_matches.drop(columns=advanced_cols, errors='ignore').copy()
    
    # Sadece temel sütunları al.
    keep_cols = ["Season", "WTeamID", "LTeamID"]
    all_cols_in_df = [c for c in keep_cols if c in df_matches.columns]
    df_matches = df_matches[all_cols_in_df]
    
    # 1) Tarihsel maçları genişlet (model eğitimi için).
    rows = []
    for _, row in df_matches.iterrows():
        rows.append({
            'TeamA': row['WTeamID'],
            'TeamB': row['LTeamID'],
            'Season': row['Season'],
            'y': 1
        })
        rows.append({
            'TeamA': row['LTeamID'],
            'TeamB': row['WTeamID'],
            'Season': row['Season'],
            'y': 0
        })
    df_expanded = pd.DataFrame(rows)
    
    # 2) Takım özelliklerinde her takım için tek satır olduğundan emin olun.
    df_teams_features_unique = df_teams_features.drop_duplicates(subset=["TeamID"]).copy()
    
    # 3) TeamA özelliklerini ekle.
    dfA = df_teams_features_unique.copy()
    dfA = dfA.add_prefix("A_")
    dfA.rename(columns={"A_TeamID": "TeamA"}, inplace=True)
    df_merged = pd.merge(df_expanded, dfA, on="TeamA", how="inner")
    
    # 4) TeamB özelliklerini ekle.
    dfB = df_teams_features_unique.copy()
    dfB = dfB.add_prefix("B_")
    dfB.rename(columns={"B_TeamID": "TeamB"}, inplace=True)
    df_merged = pd.merge(df_merged, dfB, on="TeamB", how="inner")
    
    # 5) Model eğitimi için X ve y'yi tanımla.
    exclude_cols = ["Season", "TeamA", "TeamB", "y"]
    X = df_merged.drop(columns=exclude_cols, errors='ignore')
    y = df_merged["y"].values
    
    # RandomForest modelini eğit.
    rfc = RandomForestClassifier(
        n_estimators=350,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    rfc.fit(X, y)
    
    # (Opsiyonel) Eğitim verisi üzerinde değerlendirme.
    y_prob = rfc.predict_proba(X)[:, 1]
    y_pred = rfc.predict(X)
    brier = brier_score_loss(y, y_prob)
    ll = log_loss(y, y_prob)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    print(f"\n=== RandomForest Metrics ({label}, Same Data) ===")
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss:    {ll:.4f}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC AUC:     {auc:.4f}")
    
    # ================== 2025 Tahminleri (Örnek Gönderim Dosyasına Göre) ==================
    df_sample = df_sample_submission.copy()
    # ID sütunundan Year, TeamA ve TeamB bilgilerini ayır.
    df_sample[['Year','TeamA','TeamB']] = df_sample['ID'].str.split('_', expand=True)
    df_sample["TeamA"] = df_sample["TeamA"].astype(int)
    df_sample["TeamB"] = df_sample["TeamB"].astype(int)
    
    # Takım özelliklerinde de 2025 için her takımın tek satır olduğundan emin olun.
    dfA_2025 = df_teams_features_unique.copy()
    dfA_2025 = dfA_2025.add_prefix("A_")
    dfA_2025.rename(columns={"A_TeamID": "TeamA"}, inplace=True)
    
    dfB_2025 = df_teams_features_unique.copy()
    dfB_2025 = dfB_2025.add_prefix("B_")
    dfB_2025.rename(columns={"B_TeamID": "TeamB"}, inplace=True)
    
    # Örnek gönderim dosyası ile özellikleri birleştir.
    df_merged_2025 = pd.merge(df_sample, dfA_2025, on="TeamA", how="left")
    df_merged_2025 = pd.merge(df_merged_2025, dfB_2025, on="TeamB", how="left")
    
    # Tahmin için kullanılacak özellikleri belirle (gereksiz sütunları kaldır).
    exclude_cols_2025 = ["Year", "TeamA", "TeamB", "ID", "Pred"]
    X_future = df_merged_2025.drop(columns=exclude_cols_2025, errors='ignore')
    
    # NaN (eksik) değerleri sıfır ile dolduruyoruz.
    X_future = X_future.fillna(0)
    
    # Model tahminleri.
    y_future_prob = rfc.predict_proba(X_future)[:, 1]
    df_merged_2025["Pred"] = y_future_prob
    
    # Sadece ID ve Pred sütunlarını al.
    df_preds_2025 = df_merged_2025[["ID", "Pred"]].copy()
    
    # Sıralama ve indeks sıfırlama.
    df_preds_2025.sort_values(by="ID", inplace=True)
    df_preds_2025.reset_index(drop=True, inplace=True)
    
    print(f"\n=== 2025 Potansiyel Maç Tahminleri [shape: {df_preds_2025.shape[0]}] ===")
    print(df_preds_2025.head(20))
    
    return df_preds_2025

# Örnek kullanım:
# df_m_combined: Erkek maçları için tarihsel veriler
# m_teams_features: Erkek takımların özellikleri
# df_sample_submission: Kaggle tarafından sağlanan örnek gönderim dosyası (örneğin "SampleSubmissionStage2.csv")
df_sample_submission = pd.read_csv(path + "SampleSubmissionStage2.csv")
df_preds_men_2025 = train_predict_from_basic_matches(df_m_combined, m_teams_features, df_sample_submission, year=2025, label="Men")
print("\nMen's 2025 Predictions :")
print(df_preds_men_2025.head(20))

# Kadınlar verisi için:
df_sample_submission_w = pd.read_csv(path + "SampleSubmissionStage2.csv")
df_preds_women_2025 = train_predict_from_basic_matches(df_w_combined, w_teams_features, df_sample_submission_w, year=2025, label="Women")
print("\nWomen's 2025 Predictions :")
print(df_preds_women_2025.head(20))



# Select the first 66067 rows from the men's predictions (0-index: 0-66066)
men_predictions_subset = df_preds_men_2025.iloc[:66067]

# Select rows from the women's predictions from the 66068th row (0-index: 66067) up to the 131408th row (66067-131407)
women_predictions_subset = df_preds_women_2025.iloc[66067:131408]

# Concatenate both subsets vertically
df_combined_predictions = pd.concat([men_predictions_subset, women_predictions_subset], ignore_index=True)

# Save the new dataframe as a CSV file
df_combined_predictions.to_csv("All_Predictions_Data_2025.csv", index=False)
print("Predictions saved to 'All_Predictions_Data_2025.csv'")


