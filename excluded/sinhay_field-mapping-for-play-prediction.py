#@title Loading Packages
# general
import os
import zipfile
import pandas as pd
import glob # for loading tracking data easier
import numpy as np
import math
import torch
import tqdm
from copy import deepcopy
# data processing and formatting
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split # splitting dataset
# PyTorch packages
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
# model optimization
from sklearn.model_selection import KFold
# model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


#@title individual categories
print(f'Loading individual category csv:')
games_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/games.csv')
print(f' - loaded games_df: {games_df.shape}')
player_play_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/player_play.csv')
print(f' - loaded player_play_df: {player_play_df.shape}')
players_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/players.csv')
print(f' - loaded players_df: {players_df.shape}')
plays_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2025/plays.csv')
print(f' - loaded plays_df: {plays_df.shape}')


#@title loading tracking data
folder_path = '/kaggle/input/nfl-big-data-bowl-2025'
file_list = glob.glob(folder_path + '/tracking_week_*.csv')
print(f'Loading Tracking Week data:')
# loading first df in chunks - better on memory
iter_tracking_csv = pd.read_csv(file_list[0], iterator=True, chunksize=1000)
# intial load of tracking data - filtering data to only include non-null pre-snap info
tracking_df = pd.concat([
    chunk[
        (chunk['event'].notna()) &
        (chunk['event'] != 'huddle_break_offense') &
        (chunk['event'] != 'huddle_start_offense') &
        (chunk['frameType'] != 'AFTER_SNAP')
    ] for chunk in iter_tracking_csv])
print(f' - Tracking Week 1 loaded: {tracking_df.shape}')
# loading rest of tracking data
for i in range(1, len(file_list)):
  iter_data = pd.read_csv(file_list[i], iterator=True, chunksize=1000)
  df_filtered = pd.concat([
    chunk[
        (chunk['event'].notna()) &
        (chunk['event'] != 'huddle_break_offense') &
        (chunk['event'] != 'huddle_start_offense') &
        (chunk['frameType'] != 'AFTER_SNAP')
    ] for chunk in iter_data])
  tracking_df = pd.concat([tracking_df,df_filtered],ignore_index=True)
  print(f' - Tracking Week {i + 1} loaded: {tracking_df.shape}')


#@title Functions for Data Processing
# function to get list of all features with null values
def null_features(df):
  null_features = []
  rows, columns = df.shape
  print(f'Null Features:')
  for feature in df.columns:
    null_count = df[feature].isnull().sum()
    null_percentage = null_count / rows
    if null_count != 0:
      null_features.append(feature)
      print(f' - {feature}: {null_count} / {rows} = {null_percentage}')
  return null_features

# One-Hot Encodes specific features from inputted DataFrame and returns One-Hot Encoded DataFrame
def one_hot_encode(df, one_hot_feats):
  # Initialize OneHotEncoder
  encoder = OneHotEncoder(sparse_output=False)
  # Fit and transform one_hot_feats
  one_hot_encoded = encoder.fit_transform(df[one_hot_feats])
  # Create DataFrame with encoded columns
  oh_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(one_hot_feats))
  # Concatenate One-Hot columns
  df_encoded = pd.concat([df.drop(one_hot_feats,axis=1), oh_df],axis=1)
  return df_encoded


#@title game_clock_to_seconds(clock)
# converts game clock into seconds, with the max being 15 minutes
def game_clock_to_seconds(clock):
  minutes, seconds = map(int, clock.split(':'))
  return minutes * 60 + seconds


plays_df['gameClockSeconds'] = plays_df['gameClock'].apply(game_clock_to_seconds)
print(plays_df.shape)


plays_df_seconds = plays_df.drop(columns=['gameClock'],axis=1)
print(plays_df_seconds.shape)


#@title label_df(): function to playType and schemeType features
# Play Prediction: playType = {'run', 'pass'} or {0,1} where 0 = run and 1 = pass
# Scheme Prediction: schemeType = {'man', 'zone', 'other'} or {0,1,2} where 0 = man, 1 = zone, 2 = other
def label_df(df):
  playType = []
  schemeType = []
  play_label_mapping = {0:'run', 1:'pass'}
  scheme_label_mapping = {0:'man', 1:'zone', 2:'other'}
  for index, row in df.iterrows(): # labeling each play
    # playType
    if not pd.isna(row['passResult']): # is a pass
      playType.append(1)
    elif not pd.isna(row['rushLocationType']): # is a rush
      playType.append(0)
    else: # neither pass or run
      playType.append(np.nan)
    # schemeType
    if row['pff_manZone'] == 'Man': # is man
      schemeType.append(0)
    elif row['pff_manZone'] == 'Zone': # is Zone
      schemeType.append(1)
    elif row['pff_manZone'] == 'Other':
      schemeType.append(2)
    else: # is NaN
      schemeType.append(np.nan)
  df['playType'] = playType # adding feature
  df['schemeType'] = schemeType # adding feature
  print(df.shape)
  print(f'Play Label Mapping: {play_label_mapping}\nValues: {set(df["playType"].dropna())}')
  print(f'Scheme Label Mapping: {scheme_label_mapping}\nValues: {set(df["schemeType"].dropna())}')
  df.head()
  return df, play_label_mapping, scheme_label_mapping


plays_df_labeled, play_label_mapping, scheme_label_mapping = label_df(plays_df_seconds)


# dropping features that detail the results of play, playType, and defensive scheme
play_result_features = ['playDescription','passResult','passLength','targetX','targetY','dropbackType','dropbackDistance','passLocationType','timeToThrow','timeInTackleBox','timeToSack','passTippedAtLine','unblockedPressure','qbSpike','qbKneel','qbSneak','rushLocationType','penaltyYards','prePenaltyYardsGained','yardsGained','homeTeamWinProbabilityAdded','visitorTeamWinProbilityAdded','expectedPointsAdded','isDropback','pff_runConceptPrimary','pff_runConceptSecondary','pff_runPassOption','pff_passCoverage','pff_manZone']
# leaving only X features and singular Y feature playType
plays_filtered = plays_df_labeled.drop(columns=play_result_features,axis=1)
print(plays_filtered.shape)


rows, columns = plays_filtered.shape
pass_count = plays_filtered['playType'].value_counts().get(1,0)
run_count = plays_filtered['playType'].value_counts().get(0,0)
man_count = plays_filtered['schemeType'].value_counts().get(0,0)
zone_count = plays_filtered['schemeType'].value_counts().get(1,0)
other_count = plays_filtered['schemeType'].value_counts().get(2,0)

print(f'Play Type:')
print(f' - pass: {pass_count} / {rows} = {pass_count / rows}')
print(f' - run: {run_count} / {rows} = {run_count / rows}')
print(f' - Total: {pass_count + run_count}')
print(f'Scheme Type:')
print(f' - man: {man_count} / {rows} = {man_count / rows}')
print(f' - zone: {zone_count} / {rows} = {zone_count / rows}')
print(f' - other: {other_count} / {rows} = {other_count / rows}')
print(f' - Total: {man_count + zone_count + other_count}')


#@title figuring out which playIds to drop
play_null_feats = null_features(plays_filtered)
null_plays = plays_filtered['playNullifiedByPenalty'].value_counts().get('Y',0)
valid_plays = plays_filtered['playNullifiedByPenalty'].value_counts().get('N',0)
print(f'Number of nullified plays: {null_plays} / {rows} = {null_plays / rows}')
print(f'Number of valid plays: {valid_plays} / {rows} = {null_plays / rows}')
print(f'Total Plays: {null_plays} + {valid_plays} = {null_plays + valid_plays}')


playIds_to_drop = plays_filtered[plays_filtered[play_null_feats].isnull().any(axis=1)]['playId']
plays_cleaned = plays_filtered[~plays_filtered['playId'].isin(playIds_to_drop)].reset_index(drop=True)
print(f'plays_cleaned: {plays_cleaned.shape}')


plays_oh = one_hot_encode(plays_cleaned, ['offenseFormation', 'receiverAlignment'])
print(plays_oh.shape)


tracking_oh = one_hot_encode(tracking_df, ['playDirection','frameType','event']) # exclude club
print(tracking_oh.shape)


#@title Finding Max Number of Pre-Snap events (Frames)
# stores the maximum number of frames seen
max_frames = 0
# go through each play
for index, row in tqdm.tqdm(plays_oh.iterrows(), total=plays_oh.shape[0], desc='Finding F'):
  # get game and play
  game, play = row['gameId'], row['playId']
  # getting play info
  tracking_play = tracking_oh[(tracking_oh['playId'] == play) & (tracking_oh['gameId'] == game)]
  # getting frames
  frames = list(set(tracking_play['frameId'].values.tolist()))
  if len(frames) > max_frames:
    max_frames = len(frames)
print(f'\nMax Number of pre-snap events seen across weeks: {max_frames}')


#@title initializing arrays to store data and setting F
F = max_frames # number of tensor layers
# Xs
FPTs = [] # Field Position Tensors, full field
game_situations = [] # Holds relevant play data for playId
# IDs
tensor_ids = [] # has the {'gameId':gameId,'playId':playId} for each Position Tensor, indexes match
FPTs_m_ids = [] # has the (gameId,playId,nflId,frameId) for each matrix in each FPT, indexes match
# Ys
tensor_labels_play = [] # has the playType associated with each Field Position Tensor, indexes match
tensor_labels_scheme = [] # has the schemeType associated with each Field Position Tensor, indexes match
# features for each Field Postion matrix (fpm) in a FPT
fpm_features = []
# features for each game_situation
game_situation_features = []
# order of players for each fpm that makes up the FPT
FPT_players = [] # {nflId: index, ...}


#@title go through all plays and games and make tensors and format data for training and testing
for index, row in tqdm.tqdm(plays_oh.iterrows(), total=plays_oh.shape[0], desc="Constructing Tensors from Dataset"):
# for index, row in tqdm.tqdm(plays_subset.iterrows(), total=plays_subset.shape[0], desc="Constructing Tensors from Subset"): # for testing
  game, play, play_label, scheme_label = row['gameId'], row['playId'], row['playType'], row['schemeType']
  tensor_ids.append({'gameId':game,'playId':play}) # adding id for current tensor
  tensor_labels_play.append(play_label) # adding playType for current tensor
  tensor_labels_scheme.append(scheme_label) # adding schemeType for current tensor
  week = games_df[games_df['gameId'] == game]['week'].iloc[0]
  home_team, away_team = games_df[games_df['gameId'] == game]['homeTeamAbbr'].iloc[0], games_df[games_df['gameId'] == game]['visitorTeamAbbr'].iloc[0]
  # get offense and defense teams
  offense, defense = row['possessionTeam'], row['defensiveTeam']
  # getting relevant play info for current play
  play_info = [
      'quarter','down','yardsToGo','yardlineNumber','gameClockSeconds','absoluteYardlineNumber','offenseFormation_EMPTY','offenseFormation_I_FORM','offenseFormation_JUMBO','offenseFormation_PISTOL','offenseFormation_SHOTGUN','offenseFormation_SINGLEBACK','offenseFormation_WILDCAT','receiverAlignment_1x0','receiverAlignment_1x1','receiverAlignment_2x0','receiverAlignment_2x1','receiverAlignment_2x2','receiverAlignment_3x0','receiverAlignment_3x1','receiverAlignment_3x2','receiverAlignment_3x3','receiverAlignment_4x1','receiverAlignment_4x2',
  ]
  game_situation = pd.DataFrame([row[play_info].values], columns=play_info)
  home_score, home_win_prob, visitor_score, visitor_win_prob = row['preSnapHomeScore'], row['preSnapHomeTeamWinProbability'], row['preSnapVisitorScore'], row['preSnapVisitorTeamWinProbability']
  # replace preSnapHome[field] and preVisitor[field] with preSnapOffense[field] and preSnapDefense[field]
  if offense == home_team: # offense is home team
    side_info = pd.DataFrame({'preSnapOffenseScore': [home_score],'preSnapDefenseScore': [visitor_score],'preSnapOffenseWinProb': [home_win_prob],'preSnapDefenseWinProb': [visitor_win_prob]})
  else:
    side_info = pd.DataFrame({'preSnapOffenseScore': [visitor_score],'preSnapDefenseScore': [home_score],'preSnapOffenseWinProb': [visitor_win_prob],'preSnapDefenseWinProb': [home_win_prob]})
  game_situation_processed = pd.concat([game_situation, side_info], axis=1)
  processed_feats = game_situation_processed.columns.tolist()
  if len(game_situation_features) == 0:
    game_situation_features = processed_feats
  game_situation_tensor = torch.tensor(game_situation_processed.values, dtype=torch.float32)
  game_situations.append(game_situation_tensor)

  # get play tracking data for all event frames before the snap
  tracking_play = tracking_oh[(tracking_oh['playId'] == play) & (tracking_oh['gameId'] == game)]

  # field info - all 22 players
  tracking_field = tracking_play[tracking_play['displayName'] != 'football']

  # get frameIds
  frames = list(set(tracking_play['frameId']))
  frames.sort() # sort in ascending order

  fpms = [] # field position matrices, each matrix is a tensor
  fpms_ids = [] # field position matrices ids (gameId, playId, nflId, frameId), index matches data in fpms, data is pandas dataframe
  starting_formation = {} # order of nflIds seen in frames[0], used to order and format tensors

  # build tensor
  for f in frames:

    field_f = pd.DataFrame(tracking_field[tracking_field['frameId'] == f].reset_index(drop=True)) # field for current frame

    # add side_[side] feature for frameId data, done for entire field
    side_offense, side_defense = [], []
    for j, player in field_f.iterrows():
      side_offense.append(1 if player['club'] == offense else 0)
      side_defense.append(1 if player['club'] == defense else 0)
    field_f['side_offense'] = side_offense
    field_f['side_defense'] = side_defense

    # features needed to identify unique frame's rows (players)
    frame_id = ['gameId','playId','nflId','frameId']
    # drop the following information from each frame
    tracking_target_feats = ['nflId','x','y','s','a','dis','o','dir']

    # field
    field_frame_id = field_f[frame_id]
    field_f_filtered = field_f[tracking_target_feats]
    if len(starting_formation) == 0:# for first frame
      field_f_sorted = field_f_filtered.sort_values(by=['x','y','s'], na_position='last', ignore_index=True) # sorting in ascending order
      starting_formation = {k:v for v, k in enumerate(field_f_sorted['nflId'])}
      FPT_players.append(starting_formation) # preserving order
    else: # for subsequent frames
      field_f_sorted = field_f_filtered.sort_values(by=['nflId'], key=lambda x: x.map(starting_formation), ignore_index=True)

    field_f_filtered = field_f_filtered.drop(columns=['nflId'],axis=1) # remove nflId
    field_f_sorted = field_f_sorted.drop(columns=['nflId'],axis=1)
    field_f_matrix = field_f_sorted.values # make matrix
    field_f_filtered_features = field_f_filtered.columns.tolist() # store features
    field_f_tensor = torch.tensor(field_f_matrix, dtype=torch.float32) # convert to tensor
    field_f_tensor_padded = nn.functional.pad(field_f_tensor, (7, 8), mode='constant', value=0) # adding zero padding [22,7] -> [22,22]
    fpms.append(field_f_tensor_padded) # adding field position matrix for current frame
    fpms_ids.append(field_frame_id) # adding id info for current frame of field position

    # saving features for matrices
    if len(fpm_features) == 0:
      fpm_features = field_f_filtered_features

  while len(fpms) < F: # add frames till len(fpms) == F
    fpms.append(fpms[-1])
    fpms_ids.append(fpms_ids[-1])

  FPT = torch.stack(fpms, dim=0) # making play player position tensor
  FPTs.append(FPT)
  FPTs_m_ids.append(fpms_ids)


#@title Summarizing Formatted Data
print(f'{len(FPTs)} FPTs of size {FPTs[0].size()}, {len(FPTs_m_ids)} IDs')
print(f'game_siutations {len(game_situations)}, of size {game_situations[0].size()}')
print('-----------------------------------------------------------------------')
print(f'tensor_ids {len(tensor_ids)}:\n - keys: {tensor_ids[0].keys()}')
print(f'FPTs_m_ids[i]: {len(FPTs_m_ids[0])} dfs of shape {FPTs_m_ids[0][0].shape}\n - features: {FPTs_m_ids[0][0].columns.tolist()}\n - vals for first row: {FPTs_m_ids[0][0].iloc[0].tolist()}')
print('-----------------------------------------------------------------------')
print(f'fpm_features: {len(fpm_features)}\n {fpm_features}')
print(f'game_situation_features: {len(game_situation_features)}\n {game_situation_features}')
print(f'FPT_players[0]:\n {FPT_players[0]}')
print('-----------------------------------------------------------------------')
print(f'tensor_labels_play {len(tensor_labels_play)}')
print(f'tensor_labels_scheme {len(tensor_labels_scheme)}')


indices = np.arange(len(tensor_ids)) # will use to reference already made dataset
# splitting into train / test split
train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=77)


#@title CustomDataset() class for just tracking_df data
# allows label and tensor arrays to be loaded into DataLoader based on indices
class CustomDataset(Dataset):
  def __init__(self, tracking_data, play_data, labels):
    self.tracking_data = tracking_data
    self.play_data = play_data
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    field = self.tracking_data[idx].type(torch.float32) # for CNN + RNN
    game_situation = self.play_data[idx].type(torch.float32)
    label = self.labels[idx]
    return field, game_situation, label


#@title making datset of [[FPT, game_situation, label],...] using CustomDataset()
dataset_play = CustomDataset(FPTs, game_situations, tensor_labels_play)
# training dataset
train_dataset_play = Subset(dataset_play, train_indices)
# testing dataset
test_dataset_play = Subset(dataset_play, test_indices)


#@title MixedNet(numChannels, cnnOutChannels, factor, seqLen, additioonalInputSize, numClasses, dropoutRate)
class MixedNet(nn.Module): # CNN (spatial data) => RNN (temporal data)
  def __init__(self, numChannels, cnnOutChannels, additionalInputSize, factor, seqLen,  numClasses=2, dropoutRate=0.3):
    super(MixedNet, self).__init__() # call parent constructor
    # save important features
    self.seq_len = seqLen
    ## PROCESSING FIELD TENSORS
    self.cnn = nn.Sequential( # CNN - spatial modeling: Input = [batch_size, 5, 22, 22] Tensor
        nn.Conv2d(in_channels=numChannels, out_channels=32, kernel_size=(1,5), stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,5), stride=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(1,5), stride=1), # size is now 22 x 10
        nn.BatchNorm2d(num_features=64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=cnnOutChannels, kernel_size=(5,1), stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(5,1), stride=1), # size is now 10 x 10
    )
    self.field_fc = nn.Sequential( # connecting CNN => RNN
        nn.BatchNorm2d(num_features=cnnOutChannels),
        nn.Flatten(),
        nn.Linear(in_features=cnnOutChannels * 10 * 10, out_features=cnnOutChannels),
        nn.ReLU(),
    )
    # RNN - temporal modeling
    rnn_input_size = cnnOutChannels // seqLen
    rnn_hidden_size = int(cnnOutChannels * factor)
    self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_size, batch_first=True)
    ## PROCESSING GAME SITUATION
    self.game_nn = nn.Sequential(
        nn.Linear(in_features=additionalInputSize, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=rnn_hidden_size),
        nn.ReLU(),
    )
    # merge RNN + GAME_NN
    self.merged_fc = nn.Sequential(
        nn.Linear(in_features=(2*rnn_hidden_size), out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(),
        nn.Dropout(p=dropoutRate),
        nn.Linear(in_features=128, out_features=numClasses), # should be 2
    )
    # initialize softmax for classification
    self.logSoftmax = nn.LogSoftmax(dim=1)

  def forward(self, field_input, game_input):
    ## PROCESS FIELD SPATIAL AND TIME DATA
    field_input = self.cnn(field_input) # CNN
    field_input = self.field_fc(field_input) # preprare for RNN
    batch_size = field_input.size(0)
    input_size = field_input.size(1) // self.seq_len
    field_input = field_input.view(batch_size, self.seq_len, input_size)
    _, (hidden, _) = self.rnn(field_input) # RNN, hidden: [num_layers, batchsize, hidden_size]
    hidden_state = hidden[-1].squeeze(0) # process for final fc layer
    ## PROCESS GAME SITUATION DATA
    game_input = self.game_nn(game_input).squeeze(1) # process for final fc layer
    # combine RNN and additional network
    combined_features = torch.cat((hidden_state, game_input), dim=1)
    x = self.merged_fc(combined_features)
    # classify
    output = self.logSoftmax(x)
    return output


model = MixedNet(5, 64, len(game_situation_features), 2, 8, 2, 0.5)


#@title train_k_fold(model, dataset, k_folds, epochs, batch_size, lr) - FIELD AND GAME DATA
def train_k_fold(model, dataset, k_folds=5, epochs=50, batch_size=100, lr=0.01):
  best_model = None
  best_f1 = 0
  kfold = KFold(n_splits=k_folds, shuffle=True)
  # Metric storage
  fold_metrics = []
  # training in each fold
  for fold, (train_ids, val_ids) in enumerate(tqdm.tqdm(kfold.split(dataset), total=k_folds, desc='K-Fold Cross Validation')):
    print(f'\nFold {fold + 1}/{k_folds}')
    # subset data into train and val
    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
    # initialize model, loss, and optimizer
    current_model = deepcopy(model) # brand new model for each fold
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Training
    for epoch in range(epochs):
      current_model.train()
      epoch_loss = 0
      with tqdm.tqdm(train_loader, desc=f'Training Fold {fold + 1}, Epoch {epoch+1}/{epochs}', ncols=100, unit='batch', leave=True) as t:
        for field_inputs, game_inputs, targets in t:
          field_inputs, game_inputs, targets = field_inputs.float(), game_inputs.float(), targets.long()
          outputs = current_model(field_inputs, game_inputs).squeeze()
          loss = criterion(outputs, targets)
          optimizer.zero_grad() # optimizing
          loss.backward()
          optimizer.step()
          epoch_loss += loss.item() # calculating loss
          t.set_postfix({'Batch Loss': loss.item()})
      print(f'\nEpoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}')
    # Validation
    current_model.eval()
    val_preds = []
    val_targets = []
    val_loss = 0
    with tqdm.tqdm(val_loader, desc=f'Validating Fold {fold + 1}') as t:
      for field_inputs, game_inputs, targets in t:
        with torch.no_grad():
          field_inputs, game_inputs, targets = field_inputs.float(), game_inputs.float(), targets.long()
          outputs = current_model(field_inputs, game_inputs).squeeze()
          loss = criterion(outputs, targets)
          val_loss += loss.item()
          preds = torch.argmax(outputs, dim=1)
          val_preds.extend(preds.tolist())
          val_targets.extend(targets.tolist())
    # Compute metrics
    acc = accuracy_score(val_targets, val_preds)
    precision = precision_score(val_targets, val_preds)
    recall = recall_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds)
    # display and store
    print(f'\nValidation Metrics for Fold {fold + 1}')
    print(f'  Accuarcy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
    fold_metrics.append({'accuracy':acc, 'precision':precision, 'recall':recall, 'f1_score':f1})
    #saving best model
    if f1 > best_f1:
      best_f1 = f1
      best_model = deepcopy(current_model)
  # Averaging out metrics across folds
  avg_metrics = {
      'accuracy': sum(m['accuracy'] for m in fold_metrics) / k_folds,
      'precision': sum(m['precision'] for m in fold_metrics) / k_folds,
      'recall': sum(m['recall'] for m in fold_metrics) / k_folds,
      'f1_score': sum(m['f1_score'] for m in fold_metrics) / k_folds,
  }
  print(f'\nAverage Metrics Across Folds:')
  print(f'  Accuarcy: {avg_metrics["accuracy"]:.4f}, Precision: {avg_metrics["precision"]:.4f}, Recall: {avg_metrics["recall"]:.4f}, F1-Score: {avg_metrics["f1_score"]:.4f}')
  return best_model, avg_metrics


trained_model, train_avg_metrics = train_k_fold(model, train_dataset_play, k_folds=5, epochs=5, batch_size=1000, lr=0.01)


#@title evaluate(model, dataset, k_folds, epochs, batch_size, lr) TRACKING AND GAME SITUAITON DATA
def evaluate(model, eval_dataset, epochs=5, batch_size=100):
  # load dataset
  eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
  # initialize model, loss, and optimizer
  criterion = nn.CrossEntropyLoss()
  # Validation
  model.eval()
  val_preds = []
  val_targets = []
  val_loss = 0
  with tqdm.tqdm(eval_loader, desc=f'Evaluating on Test Set') as t:
    for field_inputs, game_inputs, targets in t:
      with torch.no_grad():
        field_inputs, game_inputs, targets = field_inputs.float(), game_inputs.float(), targets.long()
        outputs = model(field_inputs, game_inputs).squeeze()
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        preds = torch.argmax(outputs, dim=1) # converts logits to binary prediction
        val_preds.extend(preds.tolist())
        val_targets.extend(targets.tolist())
    # Compute metrics
    acc = accuracy_score(val_targets, val_preds)
    precision = precision_score(val_targets, val_preds)
    recall = recall_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds)
  # Recording metrics
  eval_metrics = {'accuracy': acc,'precision': precision,'recall': recall,'f1_score': f1}
  print(f'Evalution Metrics:')
  print(f'  Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}')
  return eval_metrics


eval_metrics = evaluate(trained_model, test_dataset_play, epochs=5, batch_size=64)

