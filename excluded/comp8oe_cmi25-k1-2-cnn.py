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
from sklearn.metrics import f1_score, classification_report

# ML argorithm
import lightgbm as lgb

# torch
import torch
import torch.nn as nn
import torch.optim as optim
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


def show_df(df, n_head=3):
    print(df.shape)
    display(df.head(n_head))


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
FEATURES_ACC = ['acc_x', 'acc_y', 'acc_z']
FEATURES = FEATURES_ACC
print(len(FEATURES_ACC), len(FEATURES))


# data path
#PATH = '../ignore_dir/input/cmi-detect-behavior-with-sensor-data' 
PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data' # Kaggle用


VAL_SIZE = 0.25 # 訓練データと検証データを分割するときの割合
BATCH_SIZE = 32 # 64 # バッチサイズ
LR = 0.0001 # 学習率
N_HIDDEN = 32
NUM_EPOCHS = 30 #80 # エポック数

MAX_SEQ_LEN = 128 # シーケンス長


# Check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


SEED = 123
def seed_everything(seed=42):
    random.seed(seed)
    #np.random.seed(seed)
    np.random.RandomState(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
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

def gesture2numeric(df, gesture):
	"""
	Convert gesture labels to numeric values.

	Parameters:
		df (pd.DataFrame): DataFrame containing gesture labels.
		gesture (list): List of gesture labels.

	Returns:
		pd.DataFrame: DataFrame with an additional column for numeric gesture labels.
		dict: Dictionary mapping gesture labels to numeric values.	  
	"""
	dic_gesture = {gesture: idx for idx, gesture in enumerate(GESTURE)}
	df['gesture_numeric'] = df['gesture'].map(dic_gesture)

	return df, dic_gesture

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
	y_ret = []
	ystr_ret = []
	meta_ret =[]

	df_gb = df.groupby(['subject', 'sequence_id'])
	for keys, data in tqdm(df_gb):
		data_ft = np.array(data[FEATURES].values, dtype=np.float32).reshape(len(FEATURES), -1) # reshape to (n_features, sequence length)
		X_ret.append(data_ft)
		meta_ret.append(keys)
		
		if y_boolean:
			gesture_num = data['gesture_numeric']
			gesture_str = data['gesture']
			if gesture_num.nunique() == 1:
				target_num = gesture_num.iloc[0]
				target_str = gesture_str.iloc[0]
			else:
				target_num = -1
				target_str = 'error'
			y_ret.append(target_num)
			ystr_ret.append(target_str)
		
	return X_ret, y_ret, ystr_ret, meta_ret


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
		# シーケンス長がMAX_SEQ_LEN以下の場合はパディングを行い、MAX_SEQ_LENを超える場合は切り捨て
		if x.shape[1] <= MAX_SEQ_LEN:
			pad = np.zeros((len(FEATURES), MAX_SEQ_LEN - x.shape[1]), dtype=np.float32)
			x_padded = np.concatenate([x, pad], axis=1)
			if x_padded.shape[1] != MAX_SEQ_LEN:
				print(f"Warning: うまくパディングできてないかも→{x_padded.shape[1]}")
		else:
			x_padded = x[:, :MAX_SEQ_LEN]
		X_ret.append(x_padded)
	return X_ret


# 欠損値補完
train_df = filling(train_df)
# ラベルエンコーディング
train_df, dic_gesture = gesture2numeric(train_df, GESTURE) # 訓練のみ
# データ整形
X_train, y_train, ystr_train, meta_train = data_shaping(train_df, y_boolean=True)
print(f"X_train: {len(X_train)}, y_train: {len(y_train)}, ystr_train: {len(ystr_train)}, meta_tr: {len(meta_train)}")
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
y_tr = extract(y_train, idx_tr)
y_val = extract(y_train, idx_val)
ystr_tr = extract(ystr_train, idx_tr)
ystr_val = extract(ystr_train, idx_val)
meta_tr = extract(meta_train, idx_tr)
meta_val = extract(meta_train, idx_val)
print(len(X_tr), len(y_tr), len(ystr_tr), len(meta_tr))
print(len(X_val), len(y_val), len(ystr_val), len(meta_val))


# データセットクラスを定義
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in data]
        self.labels = torch.tensor(labels, dtype=torch.long)

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


for inputs, labels in val_loader:
	break
print(type(inputs))
print(len(inputs))
print(inputs)
print(inputs[0].shape)


in_channel = len(FEATURES) # 入力チャネル数
sequence_length = MAX_SEQ_LEN # シーケンス長
n_output = len(GESTURE) # 出力クラス数
print(f'in_channel: {in_channel}, sequence_length: {sequence_length}, n_output: {n_output}')


class CNN(nn.Module):
    def __init__(self, in_channels, sequence_length, n_hidden, n_output):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(32*(sequence_length//2), n_hidden)
        self.l2 = nn.Linear(n_hidden, n_output)
        
        self.features = nn.Sequential(
            self.conv1,
            self.relu,
            self.conv2,
            self.relu,
            self.maxpool
        )
        
        self.classifier = nn.Sequential(
            self.l1,
            self.relu,
            self.l2
        )

    def forward(self, x):
        x1 = self.features(x)
        x2 = self.flatten(x1)
        x3 = self.classifier(x2)
        return x3 


# インスタンス作成
net = CNN(in_channel, sequence_length, N_HIDDEN, n_output).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss() # 交差エントロピー誤差

# 最適化関数と学習率
optimizer = optim.Adam(net.parameters(), lr=LR)


history = np.zeros((0, 7)) # (繰り返し数、訓練損失、訓練精度、訓練F1スコア、検証損失、検証精度、検証F1スコア)

for epoch in tqdm(range(NUM_EPOCHS)):
	# ===== 訓練モード =====
	net.train()
	tr_running_loss = 0.0 # 累積損失を格納
	tr_preds = []
	tr_labels = []
	for inputs, labels in tr_loader: #tqdm(tr_loader):
		# データをデバイスに転送
		inputs = inputs.to(device)
		labels = labels.to(device)

		# 勾配を初期化
		optimizer.zero_grad()

		# 順伝播
		outputs = net(inputs)

		# 損失計算
		loss = criterion(outputs, labels)

		# 逆伝播
		loss.backward()

		# パラメータ更新
		optimizer.step()

		# ロスの蓄積
		tr_running_loss += loss.item()
		_, predicted = torch.max(outputs, 1) # 予測値を取得
		tr_preds.extend(predicted.cpu().numpy())
		tr_labels.extend(labels.cpu().numpy())


	# ===== 検証モード =====
	net.eval()
	val_running_loss = 0.0
	val_preds = []
	val_labels = []
	with torch.no_grad():
		for inputs, labels in val_loader: #tqdm(val_loader): 
			# データをデバイスに結合
			inputs = inputs.to(device)
			labels = labels.to(device)

			# 順伝播
			outputs = net(inputs)

			# 損失計算
			loss = criterion(outputs, labels)

			# 予測値を計算
			val_running_loss += loss.item()
			_, predicted = torch.max(outputs, 1)
			val_preds.extend(predicted.cpu().numpy())
			val_labels.extend(labels.cpu().numpy())

	# 損失や評価指標の計算
	tr_loss = tr_running_loss / len(tr_loader)
	val_loss = val_running_loss / len(val_loader)
	tr_accuracy  = (np.array(tr_preds) == np.array(tr_labels)).mean()
	val_accuracy = (np.array(val_preds) == np.array(val_labels)).mean()
	tr_f1_macro  = f1_score(tr_labels, tr_preds, average='macro')
	val_f1_macro = f1_score(val_labels, val_preds, average='macro')
	print(f'Epoch:{epoch+1}/{NUM_EPOCHS}')
	print(f'tr_loss :{tr_loss:.6f}, tr_acc :{tr_accuracy:.6f}, tr_f1_macro :{tr_f1_macro:.6f}')
	print(f'val_loss:{val_loss:.6f}, val_acc:{val_accuracy:.6f}, val_f1_macro:{val_f1_macro:.6f}')

	# historyに保存
	history = np.vstack([history, [epoch+1, tr_loss, tr_accuracy, tr_f1_macro, val_loss, val_accuracy, val_f1_macro]])



def show_learning_curve(history, ylabel, n_col_tr, n_col_val):
	"""
	学習曲線を描画する関数

	Parameters
	------------------
	history: np.array: (繰り返し数, 訓練損失, 訓練精度, 訓練F1, 検証損失, 検証精度, 検証F1)
		訓練結果が格納されたnumpy配列
	ylabel: str
		y軸のラベル（タイトルにも使われる）
	n_col_train: int
		訓練データの列番号
	n_col_val: int
		検証データの列番号

	Return
	------------------
	None

	Display
	------------------
	学習曲線を表示
	"""
	ticks_interval = 10 #num_epochs // 10
	num_epochs = len(history)

	plt.figure(figsize=(8, 8))
	plt.plot(history[:, 0], history[:, n_col_tr], label='Train')
	plt.plot(history[:, 0], history[:, n_col_val], label='Validation')
	plt.xlabel('Epochs')
	plt.ylabel(ylabel)
	plt.title(f'Learning curve ({ylabel})')
	plt.legend()
	plt.grid()
	plt.show()


# 検証損失と精度の確認
print(f'Val:init  - loss:{history[ 0,4]:.5f}, acc:{history[ 0,5]:.5f}, F1:{history[ 0,6]:.5f}') 
print(f'Val:final - loss:{history[-1,4]:.5f}, acc:{history[-1,5]:.5f}, F1:{history[-1,6]:.5f}')

# 学習曲線の表示
show_learning_curve(history, 'loss', 1, 4) # 損失
show_learning_curve(history, 'acc', 2, 5) # 精度
show_learning_curve(history, 'F1_macro', 3, 6) # F1_macro



def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
	try:
		# pandasに変換
		data = sequence.to_pandas()

		# データの前処理
		data = filling(data)
		X_subm, _, _, meta_subm = data_shaping(data, y_boolean=False)
		X_subm_trim = trim_sequence(X_subm)

		# データローダーの作成
		subm_dataset = TimeSeriesDataset(X_subm_trim, [0]*len(X_subm_trim))  # ラベルは不要なのでダミーを使用
		subm_loader = DataLoader(subm_dataset, batch_size=BATCH_SIZE, shuffle=False)

		# 予測
		net.eval()
		with torch.no_grad():
			for inputs, _ in subm_loader:
				inputs = inputs.to(device)
				outputs = net(inputs)
				_, predicted = torch.max(outputs, 1)
				predicted = predicted.cpu().numpy()
			predicted_gesture = GESTURE[predicted[0]]
				
	except:
		predicted_gesture = 'Text on phone'

	print(predicted_gesture)
	return predicted_gesture



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




