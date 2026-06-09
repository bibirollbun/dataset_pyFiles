VERSION = "k134"

# ローカル用
#PATH = '../ignore_dir/input/cmi-detect-behavior-with-sensor-data/'
#PATH_OUTPUT = 'ignore_dir/'

# Kaggle用
PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/'
PATH_OUTPUT = ''



# general library
import datetime
import numpy as np
import os
import pandas as pd
import polars as pl
import random
from tqdm.notebook import tqdm

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# ML argorithm
import lightgbm as lgb

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset, random_split

# Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
#import japanize_matplotlib

# warnings
import warnings
warnings.filterwarnings('ignore')

# Import evaluation API
import kaggle_evaluation.cmi_inference_server


start = datetime.datetime.now()


# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Target
BFRB_GESTURE = [
	'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch', 'Eyebrow - pull hair', 
	'Eyelash - pull hair', 'Neck - pinch skin', 'Neck - scratch', 'Cheek - pinch skin',
]
NON_BFRB_GESTURE = [
	'Drink from bottle/cup', 'Glasses on/off', 'Pull air toward your face', 'Pinch knee/leg skin', 'Scratch knee/leg skin', 
	'Write name on leg', 'Text on phone', 'Feel around in tray and pull out an object', 'Write name in air', 'Wave hello', 
]
GESTURE = BFRB_GESTURE + NON_BFRB_GESTURE
print(len(BFRB_GESTURE), len(NON_BFRB_GESTURE), len(GESTURE))


# Feature
FEATURES_IMU = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
FEATURES_THM = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
FEATURES = FEATURES_IMU # + FEATURES_THM
print(f"Number of explain variables: {len(FEATURES)}")


VAL_SIZE = 0.2 # 訓練データと検証データを分割するときの割合
BATCH_SIZE = 64 # 64 # バッチサイズ
LR = 0.001 # 学習率
WEIGHT_DECAY = 1e-7 # 重み減衰
NUM_EPOCHS = 150 #80 # エポック数

TRIM_SEQ_LEN = 103 # 128 # シーケンス長


def show_df(df, n_head=3):
    print(df.shape)
    display(df.head(n_head))


# To calculate the competion score
def convert_gesture_num_bin(lis_gesture_num):
	"""0-7: True(BFRB), 8-17: False(Non-BFRB)"""
	return [True if gn < len(BFRB_GESTURE) else False for gn in lis_gesture_num]

def convert_gesture_num_macro(lis_gesture_num):
	"""0-7: そのまま, 8-17: -1(Non-BFRB)"""	
	return [gn if gn < len(BFRB_GESTURE) else -1 for gn in lis_gesture_num]

def calculate_scores_bin(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, zero_division=0)
	recall = recall_score(y_true, y_pred, zero_division=0)
	f1 = f1_score(y_true, y_pred, pos_label=True, zero_division=0, average='binary')
	return accuracy, precision, recall, f1

def calculate_scores_macro(y_true, y_pred):
	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred, zero_division=0, average='macro')
	recall = recall_score(y_true, y_pred, zero_division=0, average='macro')
	f1 = f1_score(y_true, y_pred, zero_division=0, average='macro')
	return accuracy, precision, recall, f1

def get_scores(y_true, y_pred):
	y_true_bin = convert_gesture_num_bin(y_true)
	y_pred_bin = convert_gesture_num_bin(y_pred)
	acc_bin, prc_bin, rec_bin, f1_bin = calculate_scores_bin(y_true_bin, y_pred_bin)

	y_true_macro = convert_gesture_num_macro(y_true)
	y_pred_macro = convert_gesture_num_macro(y_pred)
	acc_macro, prc_macro, rec_macro, f1_macro = calculate_scores_macro(y_true_macro, y_pred_macro)
	
	final_score = (f1_bin + f1_macro) * 0.5
	#print(f"Competition Score: {compe_score}")
	return [acc_bin, acc_macro, prc_bin, prc_macro, rec_bin, rec_macro, f1_bin, f1_macro, final_score]


SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    #np.random.seed(seed)
    np.random.RandomState(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    torch.backends.cudnn.daterministic = True
    torch.use_deterministic_algorithms = True
    # torch.backends.cudnn.benchmark = False
seed_everything(SEED)


train_df = pd.read_csv(f"{PATH}/train.csv")
test_df = pd.read_csv(f"{PATH}/test.csv")
train_demographics_df = pd.read_csv(f"{PATH}/train_demographics.csv")
test_demographics_df = pd.read_csv(f"{PATH}/test_demographics.csv")


show_df(train_df)
show_df(test_df)


#train_df[:1000].to_csv('../ignore_dir/preprocess/train_df_1000.csv', index=False)


def filling(df):
	"""
	fill NA
	"""
	return df.ffill().bfill().fillna(0)


def gesture2numeric(df):
	"""
	ジェスチャー列の値を数値に変換し、列追加する。Convert gesture labels to numeric values and add a new column.

	Parameters:
		df (pd.DataFrame): DataFrame containing gesture labels.

	Returns:
		df (pd.DataFrame): DataFrame with an additional column for numeric gesture labels.
		dic_gesture (dict): Dictionary mapping gesture labels to numeric values.	  
	"""
	dic_gesture = {gesture: idx for idx, gesture in enumerate(GESTURE)}
	df['gesture_numeric'] = df['gesture'].map(dic_gesture)

	return df, dic_gesture

def gesture2ohe(df):
	"""
	ジェスチャー列の値をワンホットラベルに変換し、列追加する。

	Parameters:
		df (pd.DataFrame): Input DataFrame.

	Returns:
		pd.DataFrame: DataFrame with additional one-hot encoded columns.
	"""
	df_ohe = pd.get_dummies(df['gesture'], dtype=int) 
	return pd.concat([df, df_ohe], axis=1)

def standardize(df, features, fit=False, scaler=None):
	"""
	標準化を行う関数:
		各特徴量の平均を0、標準偏差を1に変換

	Parameters:
		df (pd.DataFrame): 入力データフレーム
		features (list): 標準化する特徴量のリスト

	Returns:
		pd.DataFrame: 標準化されたデータフレーム
		scaler (StandardScaler): 標準化のためのスケーラーオブジェクト
	"""
	if fit:
		scaler = StandardScaler()
		scaler.fit(df[features])
	df[features] = scaler.transform(df[features])
	return df, scaler


def data_shaping(df, y_boolean):
	"""
	データの整形を行う関数:
		subjectとsequence_idでグループ化
		各グループの特徴量FEATURESを抽出し、numpy配列に変換（X_ret）
		各グループのgesture_numericとgestureを抽出
		gesture_numericが一意でない場合は-1を設定し、gestureを'error'とする(y_ret, ystr_ret)
		各グループのsubjectとsequence_idをキーとしてmeta_retに格納

	Parameters:
		df (pd.DataFrame): 入力データフレーム
		y_boolean (bool): ターゲット変数を含むかどうかのフラグ

	Returns:
		X_ret (list): 各グループの特徴量を格納したリスト
			リストの要素はnumpy配列(n_feature(in_channel), sequence_length)
		ret (list): 各グループのgesture_numericを格納したリスト
			グループ内に複数のgesture_numが存在する場合は-1を設定
		ystr_ret (list)各グループのgestureを格納したリスト
			グループ内に複数のgestureが存在する場合は'error'を設定
		meta_ret (list): 各グループのsubjectとsequence_idを格納したリスト

	"""
	X_ret = []
	ynum_ret = []
	yohe_ret = []
	ystr_ret = []
	meta_ret =[]

	df_gb = df.groupby(['subject', 'sequence_id'])
	for keys, data in tqdm(df_gb):
		data_ft = np.array(data[FEATURES].values, dtype=np.float32).reshape(len(FEATURES), -1) # reshape to (n_features, sequence length)
		X_ret.append(data_ft)
		meta_ret.append(keys)
		
		if y_boolean:
			gesture_num = data['gesture_numeric']
			gesture_ohe = data[GESTURE]
			gesture_str = data['gesture']
			if gesture_num.nunique() == 1:
				target_num = gesture_num.iloc[0]
				target_ohe = gesture_ohe.iloc[0].values.tolist()
				target_str = gesture_str.iloc[0]
			else:
				target_num = -1
				target_ohe = [-1] * len(GESTURE)  # ワンホットエンコーディングのためのリスト
				target_str = 'error'
			ynum_ret.append(target_num)
			yohe_ret.append(target_ohe)
			ystr_ret.append(target_str)
		
	return X_ret, ynum_ret, yohe_ret, ystr_ret, meta_ret


def trim_sequence(X):
	"""
	シーケンス長を揃える関数

	Parameters:
		X(list): 説明変数のリスト
	 		各要素はシーケンス長が可変のnumpy配列

	Returns:
		X_ret (list): 各要素のシーケンス長を揃えたnumpy配列のリスト
	"""
	X_ret = []
	for x in tqdm(X):
		# シーケンス長がTRIM_SEQ_LEN以下の場合はパディングを行い、TRIM_SEQ_LENを超える場合は切り捨て
		if x.shape[1] <= TRIM_SEQ_LEN:
			pad = np.zeros((len(FEATURES), TRIM_SEQ_LEN - x.shape[1]), dtype=np.float32)
			x_padded = np.concatenate([x, pad], axis=1)
			if x_padded.shape[1] != TRIM_SEQ_LEN:
				print(f"Warning: うまくパディングできてないかも→{x_padded.shape[1]}")
		else:
			x_padded = x[:, :TRIM_SEQ_LEN]
		X_ret.append(x_padded)
	return X_ret


# コピー
train_pp = train_df.copy()
test_pp = test_df.copy()
# 欠損値補完
train_pp = filling(train_pp)
test_pp = filling(test_pp)


# ワンホットエンコーディング
train_pp = gesture2ohe(train_pp) # 訓練のみ

# ラベルエンコーディング
train_pp, dic_gesture = gesture2numeric(train_pp)


# 標準化
display(train_pp[FEATURES].describe())
train_pp, scaler = standardize(train_pp, FEATURES, fit=True, scaler=None)
display(train_pp[FEATURES].describe())


# データ整形
X_train, ynum_train, yohe_train, ystr_train, meta_train = data_shaping(train_pp, y_boolean=True)
print(f"X_train: {len(X_train)}, ynum_train:{len(ynum_train)}, yohe_train:{len(yohe_train)}, ystr_train: {len(ystr_train)}, meta_tr: {len(meta_train)}")
# シーケンス長を揃える
X_train_trim = trim_sequence(X_train)


# シーケンス長について確認
def check_sequence_length(lis_X):
	min, max = lis_X[0].shape[1], lis_X[0].shape[1]
	for record in tqdm(lis_X):
		if record.shape[1] < min: min = record.shape[1]
		if record.shape[1] > max: max = record.shape[1]
	print(f"sequence length >>> min:{min}, max:{max}")
check_sequence_length(X_train)
check_sequence_length(X_train_trim)


# ターゲットの頻度分布を可視化
count_BFRB = []
count_NON_BFRB = []
count_error = []
n_sample = 0
for g in GESTURE+['error']:
	c = ystr_train.count(g)
	n_sample += c
	if g in BFRB_GESTURE:
		count_BFRB.append(c)
		count_NON_BFRB.append(0)
		count_error.append(0)
	elif g in NON_BFRB_GESTURE:
		count_BFRB.append(0)
		count_NON_BFRB.append(c)
		count_error.append(0)
	else:
		count_BFRB.append(0)
		count_NON_BFRB.append(0)
		count_error.append(c)
plt.figure()
plt.bar(GESTURE+['error'], count_BFRB, label='BFRB')
plt.bar(GESTURE+['error'], count_NON_BFRB, label='Non-BFRB', bottom=count_BFRB)
plt.bar(GESTURE+['error'], count_error, label='error', bottom=np.array(count_BFRB)+np.array(count_NON_BFRB))
plt.xticks(rotation=90)
plt.title(f"Gesture's frequency (samples:{n_sample})")
plt.ylim(0, 800)
plt.ylabel("frequency")
plt.yticks(np.arange(0, 801, 200))
plt.grid(axis='y')
plt.legend()
plt.show()


# BFRBとNON-BFRBの割合を確認
sum_BFRB = sum(count_BFRB)
sum_NON_BFRB = sum(count_NON_BFRB)
sum_error = sum(count_error)

plt.figure()
plt.pie([sum_BFRB, sum_NON_BFRB, sum_error], startangle=90, counterclock=False,
        labels=['BFRB', 'NON-BFRB', 'error'], labeldistance=None,
        autopct="%.1f%%", pctdistance=0.5, )
plt.legend(loc='upper right')
plt.title(f'Percentage of gesture (samples:{sum_BFRB + sum_NON_BFRB})')
plt.show()


# subject列のユニーク要素とその数を取得
unique_subject = train_df['subject'].unique().tolist()
nunique_subject = train_df['subject'].nunique()

# random.sampleを使って、VAL_SIZEの割合でval_subjectを選ぶ。残りをtr_subjectとする。
val_subject = random.sample(unique_subject, int(nunique_subject*VAL_SIZE))
tr_subject = [s for s in unique_subject if s not in val_subject]
print(f"tr subject length: {len(tr_subject)} ex.{tr_subject[:3]} ... {tr_subject[-3:]}")
print(f"val subject length: {len(val_subject)} ex. {val_subject[:3]} ... {val_subject[-3:]}")

# インデックスを取得
idx_tr  = [i for i in range(len(meta_train)) if meta_train[i][0] in tr_subject]
idx_val = [i for i in range(len(meta_train)) if meta_train[i][0] in val_subject]
print(len(idx_tr), len(idx_val))

# インデックスを使ってデータを抽出する関数
def extract(lis, idx):
	return [lis[i] for i in range(len(lis)) if i in idx]

X_tr = extract(X_train_trim, idx_tr)
X_val = extract(X_train_trim, idx_val)
ynum_tr = extract(ynum_train, idx_tr)
ynum_val = extract(ynum_train, idx_val)
yohe_tr = extract(yohe_train, idx_tr)
yohe_val = extract(yohe_train, idx_val)
ystr_tr = extract(ystr_train, idx_tr)
ystr_val = extract(ystr_train, idx_val)
meta_tr = extract(meta_train, idx_tr)
meta_val = extract(meta_train, idx_val)
print(len(X_tr), len(ynum_tr), len(yohe_tr), len(ystr_tr), len(meta_tr))
print(len(X_val), len(ynum_val), len(yohe_val), len(ystr_val), len(meta_val))


print(ystr_tr[0])
print(ynum_tr[0])
print(yohe_tr[0])


# === 目的変数の設定 ========================
y_tr = ynum_tr.copy()
y_val = ynum_val.copy()
# =========================================

# データセットクラスを定義
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in data]
        self.labels = torch.tensor(labels, dtype=torch.long) # 目的変数が数値の場合
        #self.labels = [torch.tensor(l, dtype=torch.float32) for l in labels] # 目的変数がワンホットラベルの場合

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
# データセット作成
tr_dataset = TimeSeriesDataset(X_tr, y_tr)
print(len(tr_dataset))

val_dataset = TimeSeriesDataset(X_val, y_val)
print(len(val_dataset))

# データローダーを作成
tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# データローダーの確認
for inputs, labels in val_loader:
	break
print(type(inputs))
print(len(inputs))
print(inputs[0])
print(inputs[0].shape)


print(type(labels))
print(len(labels))
print(labels[0])
print(labels[0].shape)


in_channel = len(FEATURES) # 入力チャネル数
sequence_length = TRIM_SEQ_LEN # シーケンス長
n_output = len(GESTURE) # 出力クラス数
print(f'in_channel: {in_channel}, sequence_length: {sequence_length}, n_output: {n_output}')


class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 512, kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            # Block 2
            nn.Conv1d(512, 768, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            # Block 3
            nn.Conv1d(768, 1024, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),

            # Block 4
            nn.Conv1d(1024, 1536, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(1536),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),

            # Block 5
            nn.Conv1d(1536, 2048, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.AdaptiveMaxPool1d(1),  # GlobalMaxPooling1D 相当
            nn.Dropout(0.5),
        )

        self.fc_block = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # 入力: [batch_size, channels, sequence_length]
        x = self.conv_block(x)     # → [batch_size, 2048, 1]
        x = x.squeeze(-1)          # → [batch_size, 2048]
        x = self.fc_block(x)       # → [batch_size, num_classes]
        return x


# インスタンス作成
net = CNN(in_channel, n_output).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss() # 交差エントロピー誤差

# 最適化関数と学習率
optimizer = optim.Adam(net.parameters(), lr=LR)


print(net)


# モデルのサマリー表示
summary(net, input_size=(BATCH_SIZE, in_channel, TRIM_SEQ_LEN))


# Early stopping and ReduceLROnPlateau
class EarlyStopping:
    def __init__(self, patience=10, restore_best_weights=True, mode='min'):
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_state_dict = None
        self.counter = 0

    def step(self, score, model):
        if self.mode == 'min':
            improved = score < self.best_score
        elif self.mode == 'max':
            improved = score > self.best_score
        else:
            raise ValueError("Mode must be either 'min' or 'max'")

        if improved:
            self.best_score = score
            self.best_state_dict = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_state_dict)
                return True
        return False

early_stopping = EarlyStopping(patience=10, restore_best_weights=True, mode='max')


# 学習率スケジューラの定義
reduce_lr_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
	optimizer, 
	mode='max', 
	factor=0.7, 
	patience=3, 
	min_lr=1e-7, 
	# verbose=True
)

# Add a logging function to monitor learning rate reductions
def log_reduce_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Learning rate reduced to: {param_group['lr']:.7f}\n")


# # ReduceLROnPlateau without the 'verbose' parameter
# reduce_lr_on_plateau = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     factor=0.7, 
#     patience=3, 
#     min_lr=1e-7  # Remove verbose
# )

# # Add a logging function to monitor learning rate reductions
# def log_reduce_lr(learning_rate):
#     print(f"Learning rate reduced to: {learning_rate}\n")


 # 繰り返し数、
 # 訓練損失、検証損失、
 # 訓練acc, 検証acc, 訓練acc(Binary), 検証acc(Binary), 訓練acc(Macro), 検証acc(Macro), 
 # 訓練f1(Binary), 検証f1(Binary), 訓練f1(Macro), 検証f1(Macro), 訓練最終スコア, 検証最終スコア


history_tr = np.zeros((0, 12))
history_val = np.zeros((0, 12))

for epoch in tqdm(range(NUM_EPOCHS)):
	# ===== 訓練モード =====
	net.train()
	running_loss_tr = 0.0 # 累積損失を格納
	pred_tr = [] # 予測値を格納
	true_tr = [] # 正解ラベルを格納
	for X_tr_batch, y_tr_batch in tr_loader: #tqdm(tr_loader):
		# データをデバイスに転送
		X_tr_batch = X_tr_batch.to(device)
		y_tr_batch = y_tr_batch.to(device)
		# 勾配を初期化
		optimizer.zero_grad()
		# 順伝
		output_batch = net(X_tr_batch)
		# 損失計算
		loss = criterion(output_batch, y_tr_batch)
		# 逆伝播
		loss.backward()
		# パラメータ更新
		optimizer.step()
		# ロスの蓄積
		running_loss_tr += loss.item()
		_, pred_batch = torch.max(output_batch, 1) # 予測値を取得
		pred_tr.extend(pred_batch.cpu().numpy())
		true_tr.extend(y_tr_batch.cpu().numpy())


	# ===== 検証モード =====
	net.eval()
	running_loss_val = 0.0
	pred_val = []
	true_val = []
	with torch.no_grad():
		for X_val_batch, y_val_batch in val_loader: #tqdm(val_loader): 
			# データをデバイスに結合
			X_val_batch = X_val_batch.to(device)
			y_val_batch = y_val_batch.to(device)
			# 順伝播
			output_batch = net(X_val_batch)
			# 損失計算
			loss = criterion(output_batch, y_val_batch)
			# 予測値を計算
			running_loss_val += loss.item()
			_, pred_batch = torch.max(output_batch, 1)
			pred_val.extend(pred_batch.cpu().numpy())
			true_val.extend(y_val_batch.cpu().numpy())

	# 損失+正解率
	loss_tr = running_loss_tr / len(tr_loader)
	loss_val = running_loss_val / len(val_loader)
	accuracy_tr = (np.array(pred_tr)==np.array(true_tr)).mean()
	accuracy_val = (np.array(pred_val)==np.array(true_val)).mean()

	# スコア
	scores_tr = get_scores(true_tr, pred_tr)
	scores_val = get_scores(true_val, pred_val)

	# historyに保存
	h_tr = [epoch+1, loss_tr, accuracy_tr] + scores_tr
	h_val = [epoch+1, loss_val, accuracy_val] + scores_val
	history_tr = np.vstack([history_tr, h_tr])
	history_val = np.vstack([history_val, h_val])

	# 結果の表示
	print(f'Epoch:{h_tr[0]}/{NUM_EPOCHS}')
	print(f'[TRAIN]: loss:{h_tr[1]:.3f}, acc:{h_tr[2]:.2f}, acc_bin:{h_tr[3]:.2f}, acc_macro:{h_tr[4]:.2f}, f1_bin:{h_tr[-3]:.2f}, f1_macro:{h_tr[-2]:.2f}, final_score:{h_tr[-1]:.3f}')
	print(f'[VALID]: loss:{h_val[1]:.3f}, acc:{h_val[2]:.2f}, acc_bin:{h_val[3]:.2f}, acc_macro:{h_val[4]:.2f}, f1_bin:{h_val[-3]:.2f}, f1_macro:{h_val[-2]:.2f}, final_score:{h_val[-1]:.3f}')

	# 学習率の調整
	old_lr = optimizer.param_groups[0]['lr']
	reduce_lr_on_plateau.step(h_val[-1]) # 指標がminかmaxか注意。
	new_lr = optimizer.param_groups[0]['lr']

	if new_lr != old_lr:
		log_reduce_lr(optimizer)

	# Early stopping check
	if early_stopping.step(h_val[-1], net):
		print("Early stopping triggered.")
		break

# Save the model
torch.save(net.state_dict(), f"{PATH_OUTPUT}trained_model_{VERSION}.pth")
print("Training complete.")


# # コンペ指標を別の方法（cmi_2025_metric_copy_for_import.py）で計算する
# from cmi_2025_metric_copy_for_import import CompetitionMetric

# net.eval()  # Set the model to evaluation mode
# y_val_pred_probs = []

# with torch.no_grad():
#     for inputs in val_loader:
#         inputs = inputs[0].to(device)  # Move to the same device as the model
#         outputs = net(inputs)  # Permute if necessary
#         y_val_pred_probs.append(outputs.cpu().numpy())

# y_val_pred_probs = np.concatenate(y_val_pred_probs, axis=0)
# print(y_val_pred_probs.shape)

# y_val_pred = np.argmax(y_val_pred_probs, axis=1)
# print(y_val_pred[:10])  # Display first 10 predictions

# y_val_true = y_val
# print(y_val_true[:10])  # Display first 10 true labels

# val_pred_labels = pd.Series(y_val_pred).map(lambda i: GESTURE[i])
# val_true_labels = pd.Series(y_val_true).map(lambda i: GESTURE[i])

# # Build DataFrames for the metric
# val_submission = pd.DataFrame({'gesture': val_pred_labels})
# val_solution = pd.DataFrame({'gesture': val_true_labels})

# # Run competition metric
# metric = CompetitionMetric()
# score = metric.calculate_hierarchical_f1(val_solution, val_submission)
# print(f"Estimated leaderboard (val) score: {score:.4f}")


def show_learning_curve(history_tr, history_val, ylabel, n_col):
	"""
	学習曲線を描画する関数

	Parameters
	------------------
	history_tr (np.array): 訓練結果が格納されたnumpy配列
	history_val (np.array): 検証結果が格納されたnumpy配列
	ylabel (str): y軸のラベル（タイトルにも使われる）
	n_col (int): データの列番号

	Return
	------------------
	None

	Display
	------------------
	学習曲線を表示
	"""
	ticks_interval = 10 #num_epochs // 10
	num_epochs = len(history_tr)

	plt.figure(figsize=(8, 8))
	plt.plot(history_tr[:, 0], history_tr[:, n_col], label='Train')
	plt.plot(history_val[:, 0], history_val[:, n_col], label='Validation')
	plt.xlabel('Epochs')
	plt.ylabel(ylabel)
	plt.title(f'Learning curve ({ylabel})')
	plt.legend()
	plt.grid()
	plt.show()


show_learning_curve(history_tr, history_val, 'Loss', 1)


show_learning_curve(history_tr, history_val, 'Accuracy(18 classes)', 2)
show_learning_curve(history_tr, history_val, 'Accuracy(Binary)', 3)
show_learning_curve(history_tr, history_val, 'Accuracy(Macro)', 4)

show_learning_curve(history_tr, history_val, 'F1(Binary)', -3)
show_learning_curve(history_tr, history_val, 'F1(Macro)', -2)
show_learning_curve(history_tr, history_val, 'Final Score', -1)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
	try:
		# pandasに変換
		data = sequence.to_pandas()

		# データの前処理
		data = filling(data)
		data, _ = standardize(data, FEATURES, fit=False, scaler=scaler)  # 標準化（訓練時にfitしたスケーラーを使用）
		X_subm, _, _, _, meta_subm = data_shaping(data, y_boolean=False)
		X_subm_trim = trim_sequence(X_subm)

		# データローダーの作成
		subm_dataset = TimeSeriesDataset(X_subm_trim, [0]*len(X_subm_trim))  # ラベルは不要なのでダミーを使用
		subm_loader = DataLoader(subm_dataset, batch_size=BATCH_SIZE, shuffle=False)

		# 予測
		net.eval()
		with torch.no_grad():
			for X_subm_batch, _ in subm_loader:
				X_subm_batch = X_subm_batch.to(device)
				output_batch = net(X_subm_batch)
				_, pred_batch = torch.max(output_batch, 1)
				pred_subm = pred_batch.cpu().numpy()
			pred_gesture = GESTURE[pred_subm[0]]
				
	except Exception as e:
		print(e.__class__.__name__)
		print(e.args)
		print(e)
		print(f"{e.__class__.__name__}: {e}")
		pred_gesture = 'Text on phone'

	print(pred_gesture)
	return pred_gesture



# Launch inference server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            f'{PATH}/test.csv',
            f'{PATH}/test_demographics.csv',
			# '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
			# '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


end = datetime.datetime.now()
print(f"Start time: \t{start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End time: \t{end.strftime('%Y-%m-%d %H:%M:%S')}")




