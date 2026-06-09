import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, T5Tokenizer
import sentencepiece as spm
import warnings
warnings.filterwarnings("ignore")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def load_teams():
    """
    Load and combine men's and women's team data.
    CSV Files: MTeams.csv and WTeams.csv
    """
    mteams = pd.read_csv("MTeams.csv")
    wteams = pd.read_csv("WTeams.csv")
    teams = pd.concat([mteams, wteams], ignore_index=True)
    return teams

def load_detailed_results():
    """
    Load and combine men's and women's detailed game results.
    CSV Files: MRegularSeasonDetailedResults.csv and WRegularSeasonDetailedResults.csv
    """
    mresults = pd.read_csv("MRegularSeasonDetailedResults.csv")
    wresults = pd.read_csv("WRegularSeasonDetailedResults.csv")
    results = pd.concat([mresults, wresults], ignore_index=True)
    return results

def load_game_cities():
    """
    Load and combine men's and women's game cities data.
    CSV Files: MGameCities.csv and WGameCities.csv
    Also load Cities.csv for city information.
    """
    m_gc = pd.read_csv("MGameCities.csv")
    w_gc = pd.read_csv("WGameCities.csv")
    game_cities = pd.concat([m_gc, w_gc], ignore_index=True)
    cities = pd.read_csv("Cities.csv")
    return game_cities, cities

def load_public_rankings():
    """
    Load public rankings from MMasseyOrdinals.csv (men's only).
    Filter for final pre-tournament rankings (RankingDayNum==133 when available,
    otherwise use latest available such as DayNum==128).
    Returns a DataFrame with columns: Season, TeamID, SystemName, OrdinalRank.
    """
    rankings = pd.read_csv("MMasseyOrdinals.csv")
    final_rankings = rankings[rankings['RankingDayNum'] == 133]
    if final_rankings.empty:
        final_rankings = rankings[rankings['RankingDayNum'] == rankings['RankingDayNum'].max()]
    final_rankings = final_rankings.sort_values(by=['SystemName'])
    final_rankings = final_rankings.groupby(['Season', 'TeamID']).first().reset_index()
    return final_rankings

def load_tourney_seeds():
    """
    Load tournament seeds for men's and women's tournaments.
    CSV Files: MNCAATourneySeeds.csv and WNCAATourneySeeds.csv.
    Extract numeric seed and build a dictionary keyed by (Season, TeamID).
    """
    mseeds = pd.read_csv("MNCAATourneySeeds.csv")
    wseeds = pd.read_csv("WNCAATourneySeeds.csv")
    seeds = pd.concat([mseeds, wseeds], ignore_index=True)
    def extract_seed(s):
        s = str(s)
        num = ''.join([c for c in s if c.isdigit()])
        try:
            return int(num)
        except:
            return 99
    seeds['SeedNum'] = seeds['Seed'].apply(extract_seed)
    seeds_dict = {}
    for _, row in seeds.iterrows():
        season = int(row['Season'])
        team = int(row['TeamID'])
        seeds_dict[(season, team)] = row['SeedNum']
    return seeds_dict

def load_team_conferences():
    """
    Load team conferences for men's and women's teams.
    CSV Files: MTeamConferences.csv and WTeamConferences.csv.
    Returns a dictionary mapping TeamID to ConfAbbrev for the current season (2025).
    """
    mconf = pd.read_csv("MTeamConferences.csv")
    wconf = pd.read_csv("WTeamConferences.csv")
    conf = pd.concat([mconf, wconf], ignore_index=True)
    # Filter for current season (2025)
    conf_current = conf[conf['Season'] == 2025]
    conf_map = conf_current.groupby('TeamID').first()['ConfAbbrev'].to_dict()
    return conf_map



def merge_game_context(detailed_df, game_cities):
    """
    Merge game cities (Data Section 3) into the detailed results.
    Adds CRType and CityID to each game.
    """
    detailed_df['Season'] = detailed_df['Season'].astype(int)
    detailed_df['DayNum'] = detailed_df['DayNum'].astype(int)
    game_cities['Season'] = game_cities['Season'].astype(int)
    game_cities['DayNum'] = game_cities['DayNum'].astype(int)
    merge_keys = ['Season', 'DayNum', 'WTeamID', 'LTeamID']
    merged = pd.merge(detailed_df, game_cities[merge_keys + ['CRType', 'CityID']], 
                      on=merge_keys, how='left')
    return merged


def create_team_history(merged_df, final_rankings, seeds_dict):
    """
    Build a dictionary mapping each TeamID to a list of game records.
    Each record includes:
      - Basic game info (Season, DayNum, outcome).
      - Box score stats.
      - Derived efficiency metrics (shooting_pct, three_pt_pct, ft_pct, total_reb, ast_to_to).
      - Contextual features (CRType, CityID).
      - Public ranking (final_rank) for men's teams; default 999 for women's.
      - Tournament seed (if available; otherwise default 99) for tournament games.
    """
    team_history = {}
    def derived_metrics(fgm, fga, fgm3, wfga3, ftm, fta, or_, dr, ast, to):
        shooting_pct = fgm / fga if fga > 0 else 0
        three_pt_pct = fgm3 / wfga3 if wfga3 > 0 else 0
        ft_pct = ftm / fta if fta > 0 else 0
        total_reb = or_ + dr
        ast_to_to = ast / to if to > 0 else 0
        return shooting_pct, three_pt_pct, ft_pct, total_reb, ast_to_to

    def get_final_rank(team, season):
        if 1000 <= team < 2000:
            rank_row = final_rankings[(final_rankings['Season'] == season) & (final_rankings['TeamID'] == team)]
            if not rank_row.empty:
                return rank_row.iloc[0]['OrdinalRank']
            else:
                return 999
        else:
            return 999

    for idx, row in merged_df.iterrows():
        season = row['Season']
        daynum = row['DayNum']
        wteam = row['WTeamID']
        lteam = row['LTeamID']
        
        # Derived metrics for winning team
        win_shooting, win_three_pt, win_ft, win_reb, win_ast_to = derived_metrics(
            row['WFGM'], row['WFGA'], row['WFGM3'], row['WFGA3'], row['WFTM'], row['WFTA'],
            row['WOR'], row['WDR'], row['WAst'], row['WTO'])
        # Derived metrics for losing team
        lose_shooting, lose_three_pt, lose_ft, lose_reb, lose_ast_to = derived_metrics(
            row['LFGM'], row['LFGA'], row['LFGM3'], row['LFGA3'], row['LFTM'], row['LFTA'],
            row['LOR'], row['LDR'], row['LAst'], row['LTO'])
        
        # Contextual features
        crtype = row.get('CRType', None)
        cityid = row.get('CityID', None)
        
        # Public ranking
        win_final_rank = get_final_rank(wteam, season)
        lose_final_rank = get_final_rank(lteam, season)
        
        # Tournament seed: if this is a tournament game (CRType == "NCAA"), get seed from seeds_dict.
        if str(crtype).upper() == "NCAA":
            win_seed = seeds_dict.get((season, wteam), 99)
            lose_seed = seeds_dict.get((season, lteam), 99)
        else:
            win_seed = None
            lose_seed = None
        
        record_win = {
            'Season': season,
            'DayNum': daynum,
            'is_win': 1,
            'FGM': row['WFGM'],
            'FGA': row['WFGA'],
            'FGM3': row['WFGM3'],
            'WFGA3': row['WFGA3'],
            'WFTM': row['WFTM'],
            'WFTA': row['WFTA'],
            'WOR': row['WOR'],
            'WDR': row['WDR'],
            'WAst': row['WAst'],
            'WTO': row['WTO'],
            'WStl': row['WStl'],
            'WBlk': row['WBlk'],
            'WPF': row['WPF'],
            'shooting_pct': win_shooting,
            'three_pt_pct': win_three_pt,
            'ft_pct': win_ft,
            'total_reb': win_reb,
            'ast_to_to': win_ast_to,
            'CRType': crtype,
            'CityID': cityid,
            'final_rank': win_final_rank,
            'seed': win_seed if win_seed is not None else 99
        }
        record_loss = {
            'Season': season,
            'DayNum': daynum,
            'is_win': 0,
            'FGM': row['LFGM'],
            'FGA': row['LFGA'],
            'FGM3': row['LFGM3'],
            'WFGA3': row['LFGA3'],
            'WFTM': row['LFTM'],
            'WFTA': row['LFTA'],
            'WOR': row['LOR'],
            'WDR': row['LDR'],
            'WAst': row['LAst'],
            'WTO': row['LTO'],
            'WStl': row['LStl'],
            'WBlk': row['LBlk'],
            'WPF': row['LPF'],
            'shooting_pct': lose_shooting,
            'three_pt_pct': lose_three_pt,
            'ft_pct': lose_ft,
            'total_reb': lose_reb,
            'ast_to_to': lose_ast_to,
            'CRType': crtype,
            'CityID': cityid,
            'final_rank': lose_final_rank,
            'seed': lose_seed if lose_seed is not None else 99
        }
        team_history.setdefault(wteam, []).append(record_win)
        team_history.setdefault(lteam, []).append(record_loss)
    
    # Sort each team's history by Season and DayNum
    for team in team_history:
        team_history[team] = sorted(team_history[team], key=lambda x: (x['Season'], x['DayNum']))
    return team_history


candidate_features = ['is_win', 'FGM', 'FGA', 'FGM3', 'WFGA3', 'WFTM', 'WFTA',
                      'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                      'shooting_pct', 'three_pt_pct', 'ft_pct', 'total_reb', 'ast_to_to',
                      'final_rank']

def analyze_team_feature_correlation(team_history, sample_team_ids, feature_keys, threshold=0.1):
    """
    For each team in sample_team_ids, compute correlations of features with 'is_win'.
    Plot the correlation matrix and average the absolute correlations across teams.
    Features with average correlation below threshold are marked for removal.
    """
    correlations = {}
    for team_id in sample_team_ids:
        if team_id not in team_history:
            print(f"Team {team_id} not found in history.")
            continue
        df = pd.DataFrame(team_history[team_id])
        if df.empty or len(df) < 5:
            print(f"Team {team_id} does not have enough games for analysis.")
            continue
        corr_matrix = df[feature_keys].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Correlation Matrix for Team {team_id}")
        plt.show()
        team_corr = corr_matrix['is_win'].drop('is_win').abs()
        correlations[team_id] = team_corr
    corr_df = pd.DataFrame(correlations)
    avg_corr = corr_df.mean(axis=1)
    print("Average absolute correlation with 'is_win' across sample teams:")
    print(avg_corr)
    features_to_keep = avg_corr[avg_corr >= threshold].index.tolist()
    features_to_remove = avg_corr[avg_corr < threshold].index.tolist()
    print(f"\nFeatures to keep (>= {threshold}): {features_to_keep}")
    print(f"Features to remove (< {threshold}): {features_to_remove}")
    return features_to_keep, features_to_remove, avg_corr



def get_team_sequence(team_history, team_id, current_season, current_daynum, seq_len=10, feature_keys=None):
    """
    Retrieve a fixed-length sequence (last seq_len games) for the team prior to the given game.
    If not enough games exist, pad with zeros.
    """
    if feature_keys is None:
        feature_keys = [f for f in candidate_features if f != 'is_win']
    history = team_history.get(team_id, [])
    filtered = [game for game in history if (game['Season'] < current_season) or 
                (game['Season'] == current_season and game['DayNum'] < current_daynum)]
    seq = filtered[-seq_len:]
    if len(seq) < seq_len:
        pad = [{key: 0 for key in feature_keys} for _ in range(seq_len - len(seq))]
        seq = pad + seq
    arr = np.array([[game[key] for key in feature_keys] for game in seq], dtype=np.float32)
    return arr

def get_matchup_text(team1_id, team2_id, teams_df):
    """
    Build a text string with team metadata for the matchup.
    Incorporates team name and conference info.
    """
    team1 = teams_df[teams_df['TeamID'] == team1_id]
    team2 = teams_df[teams_df['TeamID'] == team2_id]
    team1_name = team1.iloc[0]['TeamName'] if not team1.empty else "unknown_team"
    team2_name = team2.iloc[0]['TeamName'] if not team2.empty else "unknown_team"
    team1_conf = team1.iloc[0]['ConfAbbrev'] if 'ConfAbbrev' in team1.columns else "NA"
    team2_conf = team2.iloc[0]['ConfAbbrev'] if 'ConfAbbrev' in team2.columns else "NA"
    text = f"{team1_name} ({team1_conf}) vs {team2_name} ({team2_conf})"
    return text



class NCAADataset(Dataset):
    def __init__(self, results_df, team_history, teams_df, seeds_dict, seq_len=10, feature_keys=None):
        """
        Build training samples from historical games.
        For each game, reorient the matchup so that the lower TeamID is first.
        Sequential features are built by concatenating the last seq_len games for each team.
        Also compute a seed feature using the rudimentary formula: 0.5 + 0.03 * (seed difference),
        clipped between 0.05 and 0.95.
        """
        self.results_df = results_df
        self.team_history = team_history
        self.teams_df = teams_df
        self.seeds_dict = seeds_dict
        self.seq_len = seq_len
        if feature_keys is None:
            self.feature_keys = [f for f in candidate_features if f != 'is_win']
        else:
            self.feature_keys = feature_keys
        self.samples = []
        for idx, row in self.results_df.iterrows():
            season = row['Season']
            daynum = row['DayNum']
            wteam = row['WTeamID']
            lteam = row['LTeamID']
            lower_team = min(wteam, lteam)
            higher_team = max(wteam, lteam)
            label = 1 if wteam == lower_team else 0
            seq_team1 = get_team_sequence(team_history, lower_team, season, daynum, seq_len, self.feature_keys)
            seq_team2 = get_team_sequence(team_history, higher_team, season, daynum, seq_len, self.feature_keys)
            combined_seq = np.concatenate([seq_team1, seq_team2], axis=0)
            
            # Compute seed feature:
            seed_low = self.seeds_dict.get((season, lower_team), 99)
            seed_high = self.seeds_dict.get((season, higher_team), 99)
            seed_diff = abs(seed_high - seed_low)
            seed_feature = 0.5 + 0.03 * seed_diff
            seed_feature = max(0.05, min(seed_feature, 0.95))
            
            sample = {
                'seq': combined_seq,
                'text': get_matchup_text(lower_team, higher_team, teams_df),
                'target': label,
                'seed_feature': seed_feature,
                'season': season,
                'lower_team': lower_team,
                'higher_team': higher_team
            }
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq_tensor = torch.tensor(sample['seq'], dtype=torch.float32)
        target_tensor = torch.tensor([sample['target']], dtype=torch.float32)
        seed_feature_tensor = torch.tensor([sample['seed_feature']], dtype=torch.float32)
        return {'seq': seq_tensor, 'text': sample['text'], 'target': target_tensor, 'seed_feature': seed_feature_tensor}



class NCAA_HybridTransformer(nn.Module):
    def __init__(self,
                 seq_input_dim,           # number of features per time step
                 seq_hidden_dim=128,      # hidden dim for temporal module
                 seq_num_layers=2,        # transformer layers for sequential data
                 fusion_hidden_dim=128,   # dimension for fusion
                 num_heads=8,             # number of attention heads
                 t5_model_name="t5-base"):  # T5 model name
        super(NCAA_HybridTransformer, self).__init__()
        # Temporal module
        self.seq_input_fc = nn.Linear(seq_input_dim, seq_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=seq_hidden_dim, nhead=num_heads, batch_first=True)
        self.seq_transformer = nn.TransformerEncoder(encoder_layer, num_layers=seq_num_layers)
        
        # Textual module (T5 Encoder)
        self.t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.t5_fc = nn.Linear(self.t5_encoder.config.d_model, fusion_hidden_dim)
        
        # Fusion: project temporal output to fusion space
        self.seq_proj_fc = nn.Linear(seq_hidden_dim, fusion_hidden_dim)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=fusion_hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Final prediction head: incorporate seed feature
        self.fc_seed = nn.Linear(fusion_hidden_dim + 1, 1)
        
    def forward(self, seq_input, text_inputs, seed_feature=None):
        batch_size = seq_input.size(0)
        # Temporal branch
        x_seq = self.seq_input_fc(seq_input)            # (B, seq_len, seq_hidden_dim)
        x_seq = self.seq_transformer(x_seq)               # (B, seq_len, seq_hidden_dim)
        x_seq_pooled = x_seq.mean(dim=1)                  # (B, seq_hidden_dim)
        x_seq_proj = self.seq_proj_fc(x_seq_pooled)       # (B, fusion_hidden_dim)
        
        # Textual branch
        encoding = self.t5_tokenizer(text_inputs, padding=True, truncation=True, return_tensors="pt")
        encoding = {k: v.to(seq_input.device) for k, v in encoding.items()}
        t5_outputs = self.t5_encoder(input_ids=encoding['input_ids'],
                                     attention_mask=encoding['attention_mask'])
        x_text = t5_outputs.last_hidden_state            # (B, text_seq_len, t5_hidden_dim)
        x_text_pooled = x_text.mean(dim=1)                 # (B, t5_hidden_dim)
        x_text_proj = self.t5_fc(x_text_pooled)            # (B, fusion_hidden_dim)
        
        # Fusion: stack temporal and textual representations as tokens
        fusion_tokens = torch.stack([x_seq_proj, x_text_proj], dim=1)  # (B, 2, fusion_hidden_dim)
        fusion_output, _ = self.fusion_attn(fusion_tokens, fusion_tokens, fusion_tokens)
        fusion_pooled = fusion_output.mean(dim=1)         # (B, fusion_hidden_dim)
        
        # Concatenate seed feature (if provided) and predict
        if seed_feature is not None:
            seed_feature = seed_feature.view(batch_size, 1)
            fused = torch.cat([fusion_pooled, seed_feature], dim=1)
        else:
            fused = fusion_pooled
        logits = self.fc_seed(fused)
        pred = torch.sigmoid(logits)
        return pred



def brier_loss(preds, targets):
    return torch.mean((preds - targets) ** 2)

def train_model(model, dataloader, optimizer, num_epochs=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            seq_input = batch['seq'].to(device)
            text_inputs = batch['text']
            targets = batch['target'].to(device)
            seed_feature = batch['seed_feature'].to(device)
            optimizer.zero_grad()
            preds = model(seq_input, text_inputs, seed_feature)
            loss = brier_loss(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

def generate_submission(model, matchup_data, season="2025", device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate predictions for hypothetical matchups.
    Each matchup dict must contain:
      - 'seq': tensor of sequential features.
      - 'text': matchup meta-data string.
      - 'seed_feature': computed seed feature.
      - 'team_ids': tuple (lower_team, higher_team).
    """
    model.eval()
    submission = []
    with torch.no_grad():
        for matchup in matchup_data:
            seq_input = matchup['seq'].unsqueeze(0).to(device)
            text_input = [matchup['text']]
            seed_feature = matchup['seed_feature'].unsqueeze(0).to(device)
            pred = model(seq_input, text_input, seed_feature)
            pred_val = pred.item()
            team1, team2 = matchup['team_ids']
            matchup_id = f"{season}_{team1:04d}_{team2:04d}"
            submission.append({"ID": matchup_id, "Pred": pred_val})
    submission_df = pd.DataFrame(submission)
    return submission_df


print("Loading teams, detailed results, game cities, public rankings, tournament seeds, and team conferences...")
teams_df = load_teams()
detailed_results_df = load_detailed_results()
game_cities, cities_df = load_game_cities()
final_rankings = load_public_rankings()
seeds_dict = load_tourney_seeds()
conf_map = load_team_conferences()


teams_df['ConfAbbrev'] = teams_df['TeamID'].map(conf_map)

print("Merging game context (CRType, CityID) into detailed results...")
merged_df = merge_game_context(detailed_results_df, game_cities)


print("Creating team history with derived features, updated rankings, and tournament seeds...")
team_history = create_team_history(merged_df, final_rankings, seeds_dict)


sample_team_ids = [1101, 1181, 1201, 1301, 1401, 1501]
    
print("Performing correlation analysis for selected teams...")
features_to_keep, features_to_remove, avg_corr = analyze_team_feature_correlation(
        team_history=team_history,
        sample_team_ids=sample_team_ids,
        feature_keys=candidate_features,
        threshold=0.1  # adjust as needed
    )

if 'is_win' in features_to_keep:
    features_for_model = [f for f in features_to_keep if f != 'is_win']
else:
    features_for_model = [f for f in candidate_features if f != 'is_win']
    
print(f"Selected features for modeling: {features_for_model}")


print("Building training dataset...")
dataset = NCAADataset(merged_df, team_history, teams_df, seeds_dict, seq_len=10, feature_keys=features_for_model)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)


for i in range(5):
    print(dataset[i])


sample_item = dataset[-1]
print("\nSample Training Data:")
print("Sequential Input Shape:", sample_item['seq'].shape)
print("Sequential Input Shape:", sample_item['seq'])
print("Text Input:", sample_item['text'])
print("Seed Feature:", sample_item['seed_feature'].item())
print("Target:", sample_item['target'].item())


seq_input_dim = len(features_for_model)
model = NCAA_HybridTransformer(seq_input_dim=seq_input_dim,
                                   seq_hidden_dim=128,
                                   seq_num_layers=2,
                                   fusion_hidden_dim=128,
                                   num_heads=8,
                                   t5_model_name="t5-base")
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


print(len(dataloader))

for batch in dataloader:
    print(batch.keys())
    break


print("Training the model...")
train_model(model, dataloader, optimizer, num_epochs=5)


print("Generating hypothetical matchups for 2025...")
# Get list of all team IDs from the combined teams data.
team_ids = teams_df['TeamID'].unique()
team_ids = sorted(team_ids)


# Filter teams to include only those active in 2025.
# For men's teams, use LastD1Season == 2025.
# For women's teams, if 'LastD1Season' exists, filter similarly; otherwise, assume they are active.
if 'LastD1Season' in teams_df.columns:
    active_teams = teams_df[teams_df['LastD1Season'] == 2025]
else:
    active_teams = teams_df.copy()

# Get the sorted list of active team IDs.
team_ids = sorted(active_teams["TeamID"].unique())
print("Number of active teams:", len(team_ids))

# Generate matchup data using only these active teams.
matchup_data = []
for i in range(len(team_ids)):
    for j in range(i+1, len(team_ids)):
        lower_team = team_ids[i]
        higher_team = team_ids[j]
        seq_team1 = get_team_sequence(team_history, lower_team, current_season=2025, current_daynum=1, seq_len=10, feature_keys=features_for_model)
        seq_team2 = get_team_sequence(team_history, higher_team, current_season=2025, current_daynum=1, seq_len=10, feature_keys=features_for_model)
        combined_seq = np.concatenate([seq_team1, seq_team2], axis=0)
        text_str = get_matchup_text(lower_team, higher_team, teams_df)

        # Compute seed feature for 2025 matchups:
        seed_low = seeds_dict.get((2025, lower_team), 99)
        seed_high = seeds_dict.get((2025, higher_team), 99)
        seed_diff = abs(seed_high - seed_low)
        seed_feature = 0.5 + 0.03 * seed_diff
        seed_feature = max(0.05, min(seed_feature, 0.95))
        
        matchup_data.append({
                'seq': torch.tensor(combined_seq, dtype=torch.float32),
                'text': text_str,
                'seed_feature': torch.tensor([seed_feature], dtype=torch.float32),
                'team_ids': (lower_team, higher_team)
        })



print("Generating submission file...")
submission_df = generate_submission(model, matchup_data, season="2025")
print(submission_df.head())
submission_df.to_csv("submission_1.csv", index=False)


import pandas as pd
# Load submission
submission_df = pd.read_csv('submission_1.csv')

# Extract TeamA and TeamB from the ID (assuming the format is like "2025_1100_3200")
submission_df[['Season', 'TeamA', 'TeamB']] = submission_df['ID'].str.split('_', expand=True)
submission_df['TeamA'] = submission_df['TeamA'].astype(int)
submission_df['TeamB'] = submission_df['TeamB'].astype(int)

# Remove invalid matchups where a men's team (1000-1999) is matched with a women's team (3000-3999)
valid_submission_df = submission_df[~(
    ((submission_df['TeamA'].between(1000, 1999)) & (submission_df['TeamB'].between(3000, 3999))) |
    ((submission_df['TeamA'].between(3000, 3999)) & (submission_df['TeamB'].between(1000, 1999))) |
    (submission_df['TeamA'] == submission_df['TeamB'])
)]

# Drop extra columns
valid_submission_df = valid_submission_df.drop(columns=['Season', 'TeamA', 'TeamB'])

# Save cleaned submission
valid_submission_df.to_csv('cleaned_submission.csv', index=False)
print("Cleaned submission saved with valid matchups.")

