import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm,trange
import matplotlib.pyplot as plt
import copy

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# train
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
n_train_base = len(df_train)
print(f"base train data: {n_train_base}")

# additional data
df_additional = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")
n_train_additional = len(df_additional)
print(f"additional data: {n_train_additional}")
df_train = pd.concat((df_train,df_additional))

# test
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
n_test = len(df_test)
print(f"test data: {n_test}")

df_train.head()


def preprocess(df_train,df_test,numerical,categorical,target):
    
    # standardize numerical if numerical is not empty
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[numerical]) if numerical else np.empty((len(df_train),0))
    X_test_num = scaler.transform(df_test[numerical]) if numerical else np.empty((len(df_test),0))

    # one hot encode categorical features
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)
    X_train_cat = enc.fit_transform(df_train[categorical])
    X_test_cat  = enc.transform(df_test[categorical])

    X_train = np.hstack((X_train_num,X_train_cat))
    X_test = np.hstack((X_test_num,X_test_cat))

    # label encode target
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    
    Y_train = le_target.fit_transform(df_train[target])
    return X_train,Y_train,X_test,le_target

numerical = []
categorical = ["Soil Type","Crop Type","Temparature","Humidity","Moisture","Nitrogen","Potassium","Phosphorous"]
target = "Fertilizer Name"

X_train,Y_train,X_test,le_target=preprocess(df_train,df_test,numerical,categorical,target)


import torch
import torch.nn as nn
import torch.nn.functional as F


# average precision for a single instance
def apk(predicted,actual,k):
    for i in range(min(k,len(predicted))):
        if predicted[i] == actual:
            return 1.0/(i+1) # return at first occurence
    return 0.0
    
# mean average precision at k 
def mapk(predicted,actual,k):
    return np.mean([apk(pred,act,k) for pred,act in zip(predicted,actual)])

# compute MAP@K from a distribution probabilty over classes
def compute_mapk_from_proba(y_pred_proba,actual,k):
    predicted = torch.topk(y_pred_proba,k,dim=1).indices
    return mapk(predicted,actual,k)


X_train = torch.tensor(X_train).to(torch.float32)
Y_train = torch.tensor(Y_train)
X_test  = torch.tensor(X_test).to(torch.float32)
n_class = len(Y_train.unique())
n_features = X_train.shape[0]

print(f"X_train - {X_train.shape}\nY_train - {Y_train.shape}\nX_test  - {X_test.shape}\nn_class - {n_class}")


@torch.no_grad()
def model_predict(model,X,BATCH_SIZE=256):
    device = next(model.parameters()).device
    n = X.shape[0]
    y_pred = []
    for i in range(0,n,BATCH_SIZE):
        x_batch = X[i:min(i+BATCH_SIZE,n)].to(device)
        y_pred.append(model(x_batch))
    y_pred = torch.vstack(y_pred)
    return y_pred

# evaluation of a model over a given X and Y
def evaluate(model,X,Y_true,criterion,BATCH_SIZE=256):
    Y_pred = model_predict(model,X,BATCH_SIZE)
    Y_true = Y_true.to(Y_pred.device)
    res = criterion(Y_pred,Y_true)
    if torch.is_tensor(res):
        res = res.item()
    return res

# train model and return best model according to early stopping policy
def train_model(model,x_train,y_train,x_valid,y_valid,criterion,training_config):
    device = next(model.parameters()).device
    
    BATCH_SIZE = training_config['BATCH_SIZE']
    N_EPOCHS   = training_config['N_EPOCHS']
    step_eval  = training_config['step_eval']
    patience   = training_config['patience']
    
    # data loader
    dl_train = torch.utils.data.DataLoader(list(zip(x_train,y_train)),BATCH_SIZE,True,drop_last=True)
    
    # optimizer
    optimizer = getattr(torch.optim,training_config['optimizer'])
    optimizer = optimizer(model.parameters(),**training_config['optimizer_args'])

    # scheduler
    scheduler = getattr(torch.optim.lr_scheduler,training_config['scheduler'])
    if CONFIG['scheduler'] == 'OneCycleLR':
        CONFIG['scheduler_args']['total_steps'] = len(dl_train) * N_EPOCHS
    scheduler = scheduler(optimizer,**training_config['scheduler_args'])

    #
    training_results = {
        'loss_batch': [],
        'loss_valid': [evaluate(model,x_valid,y_valid,criterion)],
        'lr': [],
        'steps_per_epoch':len(dl_train)
    }
    
    best_weights = model.state_dict().copy()
    best_eval = training_results['loss_valid'][0]
    best_eval_step = 0
    
    global_step = 0
    # training loop
    for epoch in tqdm(range(N_EPOCHS),leave=True,position=0,smoothing=0.0,desc="Training",ncols=600):
        
        for i,batch in enumerate(tqdm(dl_train,leave=True,smoothing=0.0,position=1,
                                      desc=f'- epoch n°{epoch+1}/{N_EPOCHS}',ncols=600
                                     )):
            model.train()
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device),y_batch.to(device)

            # froward
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # record metrics
            global_step += 1
            
            training_results['loss_batch'].append(loss.item())
            training_results['lr'].append(scheduler.get_last_lr()[0])

            # eval
            if global_step % step_eval == 0:
                model.eval()
                last_eval = evaluate(model,x_valid,y_valid,criterion)
                training_results['loss_valid'].append(last_eval)
                
                if last_eval < best_eval:
                    best_eval = last_eval
                    best_eval_step = global_step
                    best_weights = copy.deepcopy(model.state_dict())
                elif global_step - best_eval_step > patience: # early stop
                    tqdm.write(f"Early stopped at {global_step}")
                    break

        else:
            continue  # only executed if the inner loop did NOT break
        break  # only executed if the inner loop DID break
    model.load_state_dict(best_weights)
    return model,training_results


def reset_weights(model):
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def build_sequential_model(input_dim, layers, output_dim, activation, dropout_rate=0.0,momentum=0.1):
    model_layers = []

    # Input layer
    prev_dim = input_dim

    for layer_dim in layers:
        model_layers.append(nn.Linear(prev_dim, layer_dim))
        model_layers.append(activation())
        model_layers.append(nn.BatchNorm1d(layer_dim, momentum=momentum))
        model_layers.append(nn.Dropout(p=dropout_rate))
        prev_dim = layer_dim

    # Output layer
    model_layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*model_layers)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print("Using device",device)

# Fix seed
torch.manual_seed(314159265358)

# Model definition
model = build_sequential_model(X_train.shape[1],[256],n_class,nn.ReLU,0.4,0.01).to(device)
print(model)

# Training configuration
CONFIG = {
    'BATCH_SIZE' : 256,
    'N_EPOCHS'   : 40,
    'optimizer':'AdamW',
    'optimizer_args':{
        'lr':1e-3, # Note : unused parameter if scheduler is OneCycleLR
        'weight_decay':0
    },
    'scheduler' : 'OneCycleLR', # LambdaLR,OneCycleLR
    'scheduler_args':{
        'max_lr': 2e-3,
        'div_factor': 10.0,
        'final_div_factor':1.0,
        'pct_start':0.1
    },
    'step_eval':  800,
    'patience' : 15000,
}

criterion = nn.CrossEntropyLoss()

# KFold
FOLDS = 10
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

# MAP@3
map_at_3 = lambda y_pred,y_true: compute_mapk_from_proba(y_pred,y_true,3)

# Object to store results
pred_oof  = torch.zeros((len(X_train),n_class))
pred_test = torch.zeros((len(X_test),n_class,FOLDS))
all_training_results = []

# CV
for i, (train_idx, valid_idx) in enumerate(kf.split(X_train, Y_train)):
    print("\n\n" + "#"*20 + f" FOLD {i+1} " + "#"*20)
    
    x_train_fold = X_train[train_idx]
    y_train_fold = Y_train[train_idx]
    x_valid_fold = X_train[valid_idx]
    y_valid_fold = Y_train[valid_idx]

    # reset model before training
    reset_weights(model)
    model,training_results_fold = train_model(model,x_train_fold,y_train_fold,x_valid_fold,y_valid_fold,criterion,CONFIG)

    all_training_results.append(training_results_fold)

    # compute metrics on train and valid
    model.eval()
    pred_train = model_predict(model,x_train_fold).cpu()
    loss_train = criterion(pred_train,y_train_fold).item()
    mapk_train = map_at_3(pred_train,y_train_fold)
    
    pred_valid = model_predict(model,x_valid_fold).cpu()
    loss_valid = criterion(pred_valid,y_valid_fold).item()
    mapk_valid = map_at_3(pred_valid,y_valid_fold)
    pred_oof[valid_idx] = pred_valid

    # test prediction
    pred_test[:,:,i] = model_predict(model,X_test).cpu()
    
    print(f"[train loss {loss_train:.4f}] - [valid loss {loss_valid:.4f}] "
          f"| [train map@3 {mapk_train:.4f}] - [valid map@3 {mapk_valid:.4f}]")
    
# Final RMSE
full_mapk = map_at_3(pred_oof,Y_train)
base_train_mapk = map_at_3(pred_oof[:n_train_base],Y_train[:n_train_base])
add_train_mapk  = map_at_3(pred_oof[n_train_base:],Y_train[n_train_base:])
print(f"\n\nFinal CV full train MAP@3: {full_mapk:.4f}")
print(f"Final CV base train (750k) MAP@3: {base_train_mapk:.4f}")
print(f"Final CV additional (100k) MAP@3: {add_train_mapk:.4f}")

np.savetxt("pred_oof_log_proba.csv" ,pred_oof.numpy(),delimiter=",")
np.savetxt("pred_test_log_proba.csv",pred_test.numpy().reshape(n_test,n_class * FOLDS),delimiter=",")


# smoothing
def moving_average(x,w):
    smoothed_x  = np.convolve(x,np.ones(w),'same')
    smoothed_x /= np.convolve(np.ones(len(x)),np.ones(w),'same')
    return smoothed_x

plt.style.use('default')
plt.subplots(FOLDS, 2,figsize=(10,2.5*FOLDS), gridspec_kw={'width_ratios': [3, 1]})

steps_per_epoch = all_training_results[0]['steps_per_epoch']
for i,t in enumerate(all_training_results):
    # train and valid loss
    plt.subplot(FOLDS,2,2*i+1)
    
    loss_batch = np.array(t['loss_batch'])
    w = max(1,len(loss_batch)//100) # window for moving average smoothing
    loss_batch_smoothed = moving_average(loss_batch,w)
    train_indices = np.arange(1,len(loss_batch)+1)/steps_per_epoch
    plt.plot(train_indices,loss_batch_smoothed,label='loss batch')

    loss_valid = np.array(t['loss_valid'])
    valid_indices = np.arange(len(loss_valid)) * CONFIG['step_eval'] / steps_per_epoch
    plt.plot(valid_indices,loss_valid,label='loss valid')
    
    plt.ylim((1.90,1.945))
    plt.xlabel("epoch")
    plt.grid()
    plt.legend(loc= "upper right")
    
    # learning rate
    plt.subplot(FOLDS,2,2*i+2)
    plt.plot(train_indices,t['lr'],label='lr')
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()


# transform log probabilites to probabilities
# pred_test_softmax = F.softmax(pred_test, dim=1)

# average over folds
avg_test_proba = pred_test.mean(axis=2)

# extract ranking from probability distribution
top3_class_indices = torch.topk(avg_test_proba,3,dim=1).indices

# convert ranking back to labels
top_3_labels = le_target.inverse_transform(top3_class_indices.ravel()).reshape(-1,3)

# save submission as csv
submission = pd.DataFrame({
    'id': df_test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)

