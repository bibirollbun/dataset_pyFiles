try:
    import rdkit
    import selfies
except:
    !pip install --quiet rdkit
    !pip install --quiet selfies


import rdkit
import selfies
print('PIP INSTALL OK!!!!')


class CFG:
    regression_dim=256
    
    batch_size=20
    lr=0.001 #005,
    seed=123
    
    num_epoch = 25
    early_stopping_patient=10000

DEVICE = 'cuda' #'cuda'

import numpy as np 
import random 
import torch

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CFG.seed)
print('CFG OK!!!!')


#model
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, cfg=CFG):
        super(Net, self).__init__()
        self.D = nn.Parameter(torch.ones(1))
      
        self.pretrained_model = AutoModel.from_pretrained("ibm/materials.selfies-ted")
        #self.model.resize_token_embeddings(len(tokenizer))

        last_dm = 1024   # self.pretrained_model.config.hidden_size
        self.property = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(last_dm, cfg.regression_dim),
            nn.SiLU(),
            nn.Linear(cfg.regression_dim, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        batch_size = len(input_ids)
        output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #pooled = mean_pooling(embedding, attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
     
        property = self.property(pooled)  # shape: (batch_size, 1)
        property = property.reshape(-1) 
        return property

# data
import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
 
from rdkit import Chem
import selfies as sf


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical SMILES using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


class SmilesDataset(Dataset):
	def __init__(self,
	             df,
	             tokenizer
	             ):
		self.df = df
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		smiles = row['SMILES'].replace("*", "[At]")
		smiles = smiles
		smiles = canonicalize_smiles(smiles)

		selfies = sf.encoder(smiles)
		selfies = selfies.replace("][", "] [")

		target = row['Tg']
		target = torch.tensor(target, dtype=torch.float32)

		return selfies, target

print('HELPER OK!!!!')


#let'd fine tune a model for Tg

from timeit import default_timer as timer
def time_to_str(t, mode='min'):
	if mode=='min':
		t  = int(t/60)
		hr = t//60
		min = t%60
		return '%2d hr %02d min'%(hr,min)

	elif mode=='sec':
		t   = int(t)
		min = t//60
		sec = t%60
		return '%2d min %02d sec'%(min,sec)

	else:
		raise NotImplementedError



#---

cfg=CFG

#modelling
model = Net()
model.to(DEVICE)
print(model.property)


#llrd_parameters = Layer-wise learning rate decay"
 
def param_setting1(model):
	for param in model.pretrained_model.parameters():
		param.requires_grad = False
		return [
			{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": cfg.lr,}
		]

def param_setting2(model):
	return [
	 	{"params": model.pretrained_model.parameters(),
	        "lr": 0.00001, "weight_decay": 0.0},
		{"params": model.property.parameters(),
		   "lr": cfg.lr,  "weight_decay": 0.01},
	]

optimizer = torch.optim.Adam( 
	#param_setting1(model)
	param_setting2(model)

	# filter(lambda p: p.requires_grad, model.parameters()),
	# lr=cfg.lr
)

tokenizer = AutoTokenizer.from_pretrained("ibm/materials.selfies-ted")




#my fold=0
train_id = [10142210, 13838538, 16498242, 30582999, 36217683, 41684723, 49737348, 52181246, 53248904, 63969847, 64606581, 68581387, 70273581, 77456502, 79415769, 81531670, 89077090, 112351140, 112553722, 119301714, 124053674, 131992742, 152467749, 155951033, 175456331, 175513548, 177977769, 180651708, 184043122, 190194752, 196264517, 201251785, 210803302, 215274198, 218059466, 218350404, 219039564, 223791997, 234483624, 234538951, 236163730, 238223157, 243853291, 245779778, 257850952, 265038627, 267049794, 267912832, 270662526, 274614510, 284595179, 286520814, 286659449, 286907970, 287416791, 289972048, 295343824, 296780712, 299349220, 308745808, 313907766, 321955140, 330131913, 351687879, 364021169, 368592179, 379965710, 383339614, 391108347, 392174026, 392422616, 402492102, 403135402, 407061099, 407820973, 427458054, 434463983, 440221059, 440284269, 441800426, 447646076, 447692460, 454723677, 454925520, 455581331, 455760856, 457518738, 476733840, 478096853, 483787501, 484741548, 488617798, 505042674, 506576027, 509241405, 509577164, 510815019, 528293396, 531044594, 541941802, 545546596, 552340314, 552823663, 583338677, 593164560, 608511939, 616177559, 634363683, 640431640, 645076897, 663317313, 666970760, 676741869, 681607778, 682136136, 691290005, 691340990, 692373242, 717229286, 722303181, 723086841, 723637794, 731417662, 737341727, 742571113, 746817611, 749003022, 752379788, 754272312, 754541700, 762944097, 765731942, 776515253, 778024152, 779018837, 783150586, 787518160, 791544306, 792850218, 817179455, 817764960, 819286039, 820510299, 821229257, 822072702, 822972602, 826258963, 841072650, 847044135, 852138859, 865948061, 866951127, 873453731, 875480288, 876238099, 881173814, 892241258, 897919146, 903930104, 913768396, 918157319, 918773447, 925163617, 926945718, 933916766, 936419032, 948650769, 949756192, 950036559, 952776097, 954755494, 954804864, 956860993, 961207214, 966864975, 968949443, 975372229, 985685560, 998917125, 1002421560, 1003531277, 1004048009, 1012389657, 1047211829, 1048177525, 1048219303, 1048356010, 1050100593, 1051120785, 1057142863, 1065659422, 1065918379, 1074560123, 1076149389, 1076737435, 1087567752, 1105455152, 1106240394, 1108992243, 1110669170, 1114536999, 1118322325, 1132763612, 1132889582, 1140194913, 1143212466, 1150491517, 1150855035, 1153274738, 1159981352, 1160783446, 1166293848, 1169677958, 1173471549, 1179190627, 1186143208, 1189744629, 1197069937, 1201052937, 1202816034, 1208726540, 1218861482, 1220689198, 1226489424, 1230182375, 1242989659, 1244604743, 1245052771, 1257932675, 1259746879, 1261888727, 1267767457, 1268591883, 1268903954, 1289453744, 1293239198, 1294607317, 1296755065, 1300127001, 1302563585, 1308684557, 1318838343, 1319410597, 1321934906, 1330783970, 1336532550, 1342666772, 1347977204, 1351618025, 1362985746, 1363453372, 1366538369, 1367879345, 1369478865, 1370311648, 1373519109, 1377024014, 1387608181, 1387610033, 1388694213, 1408675476, 1414225049, 1428459269, 1430519110, 1439560323, 1441221742, 1444841962, 1450460730, 1452233273, 1461036801, 1463175891, 1463811998, 1469266389, 1473745481, 1474070468, 1476896117, 1479037928, 1480445744, 1481083200, 1486442979, 1491854107, 1500711821, 1504708152, 1504763552, 1510186873, 1527614677, 1528389333, 1528836681, 1530902893, 1531901492, 1538292595, 1543966603, 1547263064, 1558525027, 1559096063, 1561282165, 1562112815, 1562477562, 1563279330, 1564587390, 1567869573, 1568726838, 1570200324, 1570654852, 1580898665, 1581749502, 1583167273, 1583954692, 1601240060, 1601398939, 1611681441, 1638121977, 1640191566, 1640937510, 1641687238, 1645801286, 1652549742, 1653539066, 1656589637, 1657554335, 1663220707, 1678768525, 1701758700, 1704241791, 1710475458, 1715886560, 1718419889, 1725259416, 1732679124, 1734361691, 1738001334, 1751469406, 1766008027, 1777622609, 1799502171, 1804735174, 1806857427, 1807042238, 1809093717, 1809467126, 1809480718, 1815220094, 1823366229, 1828453397, 1833216472, 1835545898, 1837801743, 1844857669, 1845526826, 1849381724, 1865362645, 1869415100, 1873866400, 1877161213, 1881954049, 1889222328, 1894241731, 1896863943, 1901480658, 1905216047, 1911963342, 1917841461, 1922704516, 1926476264, 1930079807, 1932564704, 1933529441, 1933712132, 1933820661, 1933889140, 1949514946, 1949823161, 1952404953, 1960180963, 1965690727, 1969083594, 1971749373, 1975447756, 1980536603, 1982249582, 1992987283, 1995858685, 1999832779, 2003251624, 2005402308, 2008537603, 2013412684, 2015550348, 2030717926, 2030734021, 2038784698, 2042324873, 2049185141, 2049722638, 2050550943, 2053842039, 2055219346, 2055325406, 2063026043, 2065153152, 2066262373, 2068351519, 2077226118, 2112160292, 2116365788, 2117950580, 2124040823, 2147435020]
valid_id = [42779237, 48005195, 71102106, 75443529, 133986624, 150383443, 207198573, 233996434, 240232361, 289478370, 293023790, 314916411, 327552819, 368595823, 369610662, 380596855, 384803806, 393047512, 393692859, 404993408, 429074783, 441045416, 462771322, 471856670, 506857730, 513101692, 528781081, 535364463, 548292823, 551862261, 573587884, 576198194, 578229955, 590474607, 623331916, 645691711, 695348815, 709226489, 730567526, 768467775, 790707240, 799217349, 812618019, 832174357, 834642680, 859644583, 867714785, 875491202, 926851553, 953356077, 980661216, 980966583, 1029595953, 1065912423, 1092542084, 1111539989, 1128237994, 1159550508, 1209928353, 1213234132, 1248950707, 1259156024, 1297972519, 1312487500, 1332933153, 1387768222, 1392018017, 1415406733, 1424144967, 1447445383, 1453017209, 1504210048, 1515920130, 1523715614, 1612493686, 1623779443, 1623821086, 1693140082, 1704546800, 1706654876, 1721517021, 1738997335, 1749688285, 1779122339, 1784177503, 1819092687, 1837160865, 1838148916, 1868650247, 1919732272, 1950874064, 1969287325, 2003072511, 2006821254, 2011509747, 2015647676, 2053127764, 2066820207, 2067484299, 2074481050, 2080041266, 2095077077, 2130807414]

kaggle_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train_df = kaggle_df[kaggle_df['id'].isin(train_id)].reset_index(drop=True)
valid_df = kaggle_df[kaggle_df['id'].isin(valid_id)].reset_index(drop=True)

train_dataset = SmilesDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
valid_dataset = SmilesDataset(valid_df, tokenizer)
valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

print('train_dataset', len(train_dataset))
print('valid_dataset', len(valid_dataset))




#start training here !!!
def do_valid(model, valid_loader):
	model.eval()
	valid_mae = 0
	valid_rmse = 0
	with torch.no_grad():
		for iteration, (psmiles, target) in enumerate(valid_loader):
			batch_size = len(psmiles)

			tokenized = tokenizer(
				psmiles,
				padding=True,  # Pad to the longest in batch
				truncation=True,  # Truncate longer sequences to model max length
				return_tensors="pt",  # Return PyTorch tensors
			)
			tokenized = {key: val.to(DEVICE) for key, val in tokenized.items()}
			target = target.to(DEVICE)
			predict = model(
				tokenized['input_ids'],
				tokenized['attention_mask']
			)

			valid_mae += F.l1_loss(predict, target).item() * batch_size
			valid_rmse += F.mse_loss(predict, target).item() * batch_size


	valid_mae = valid_mae / len(valid_loader.dataset)
	valid_rmse = np.sqrt(valid_rmse / len(valid_loader.dataset))
	return valid_mae, valid_rmse


valid_mae, valid_rmse  = 0, 0
best_valid_loss = float('inf')
patience_counter = 0
best_state_dict = None
start_timer = timer()


print('')
print('epoch, batch_loss, valid_mae/rmse')
print('--------------------------------')
for epoch in range(cfg.num_epoch):
	model.train()
	train_loss = 0
	for iteration, (psmiles, target) in enumerate(train_loader):
		tokenized = tokenizer(
			psmiles,
			padding=True,  # Pad to the longest in batch
			truncation=True,  # Truncate longer sequences to model max length
			return_tensors="pt",  # Return PyTorch tensors
		)
		tokenized = {key: val.to(DEVICE) for key, val in tokenized.items()}
		target = target.to(DEVICE)
		with torch.amp.autocast('cuda',dtype=torch.bfloat16):
			predict = model(
				tokenized['input_ids'],
				tokenized['attention_mask']
			)
			loss = F.l1_loss(predict, target)
			batch_loss = loss.item()

		# Backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if epoch % 500 == 0:
		pass
		##torch.save(model.state_dict(),'model.pth')
    
	if epoch% 1 == 0:
		valid_mae, valid_rmse = do_valid(model, valid_loader)
	if epoch % 1 == 0:
		print(
			f'{epoch:3d}',
			f'{batch_loss:7.3f}',
			f'{valid_mae:7.3f}',
			f'{valid_rmse:7.3f}',
			time_to_str(timer() - start_timer))

'''

Sequential(
  (0): Dropout(p=0.1, inplace=False)
  (1): Linear(in_features=1024, out_features=256, bias=True)
  (2): SiLU()
  (3): Linear(in_features=256, out_features=1, bias=True)
)
train_dataset 408
valid_dataset 103

epoch, batch_loss, valid_mae/rmse
--------------------------------
  0  88.323  77.122  99.258  0 hr 00 min
  1  47.373  68.287  89.059  0 hr 00 min
  2  69.122  57.157  75.747  0 hr 01 min
  3  67.202  57.668  73.379  0 hr 01 min
  4  51.414  54.328  68.463  0 hr 01 min
  5  43.547  50.524  63.798  0 hr 02 min
  6  30.723  50.940  63.795  0 hr 02 min
  7  30.898  54.572  68.478  0 hr 02 min
  8  33.197  52.997  69.012  0 hr 03 min
  9  26.839  50.210  63.962  0 hr 03 min
 10  31.376  51.018  65.183  0 hr 04 min
 11  21.063  51.309  65.851  0 hr 04 min
 12  27.142  48.997  63.307  0 hr 04 min
 13  22.973  51.161  64.548  0 hr 05 min
 14  24.209  47.394  60.762  0 hr 05 min
 15  24.239  51.979  65.800  0 hr 05 min
 16  19.110  52.100  64.807  0 hr 06 min
 17  12.643  52.647  65.262  0 hr 06 min
 18  14.968  51.642  64.535  0 hr 07 min
 19  13.476  51.215  63.967  0 hr 07 min
 20  20.374  50.372  62.867  0 hr 07 min
 21  25.880  50.988  64.246  0 hr 08 min
 22  19.245  50.505  62.814  0 hr 08 min
 23  10.950  51.785  64.674  0 hr 08 min
 24  17.176  50.201  62.230  0 hr 09 min

'''

