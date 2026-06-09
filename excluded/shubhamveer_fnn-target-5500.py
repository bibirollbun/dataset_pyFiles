# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, TensorDataset


np.random.seed(0)
torch.cuda.manual_seed(10)


train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


train_unchanged = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_unchanged = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


train_data.hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()


# cleaning
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


# numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
# train_data = remove_outliers_iqr(train_data, numerical_cols)


def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 2)

def calculate_bmr(weight_kg, height_cm, age, sex):
    # 0 = male, 1 = female
    if sex == 0:
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age + 5, 2)
    elif sex == 1:
        return round(10 * weight_kg + 6.25 * height_cm - 5 * age - 161, 2)

def max_heart_rate(age, sex):
    return float(round((192 if sex == 0 else 189) - 0.007 * (age ** 2), 0))

def intensity_zone(hr, mhr):
    percent = (hr / mhr) * 100
    if percent < 50:
        return 1
    elif percent < 60:
        return 2
    elif percent < 70:
        return 3
    elif percent < 85:
        return 4
        
    else:
        return 4
    

def estimate_met(intensity):
    return {1: 1.5, 2: 2.5, 3: 4.0, 4: 6.0}.get(intensity, 1.0)

def estimate_body_fat(bmi, age, sex):
    return round(1.20 * bmi + 0.23 * age + 10.8 * sex - 16.2, 2)

def age_group(age):
    if age < 30:
        return '20s'
    elif age < 40:
        return '30s'
    elif age < 50:
        return '40s'
    elif age < 60:
        return '50s'
    else:
        return '60+'

def activity_level(duration):
    if duration < 10:
        return 'very_short'
    elif duration < 20:
        return 'short'
    elif duration < 40:
        return 'moderate'
    elif duration < 60:
        return 'long'
    else:
        return 'very_long'

def calories_burner(duration, heart_rate, weight, age, sex):
    if sex == 0:
        return duration * (0.4472*heart_rate + 0.1263*weight + 0.074*age - 55.0969) / 4.184

    else:
        return duration * (0.6309*heart_rate + 0.1988*weight + 0.2017*age - 20.4022) / 4.184



def preprocess(df):
    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # BMI
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight'], row['Height']), axis=1)
    
    # BMR
    df['BMR'] = df.apply(lambda row: calculate_bmr(row['Weight'], row['Height'], row['Age'], row['Sex']), axis=1)

    # Max Heart Rate
    df['MaxHR'] = df.apply(lambda row: max_heart_rate(row['Age'], row['Sex']), axis=1)

    # Intensity Zone
    df['IntensityLevel'] = df.apply(lambda row: intensity_zone(row['Heart_Rate'], row['MaxHR']), axis=1)

    # adding metabolic_stress
    df['Metabolic_stress'] = df['Duration'] * df['Heart_Rate'] * df['Body_Temp'] / 1000
    
    
    # MET
    df['MET'] = df['IntensityLevel'].apply(estimate_met)

    # calories_burned
    df['calories_burned'] = df.apply(lambda row: calories_burner(row['Duration'], row['Heart_Rate'], row['Weight'], row['Age'], row['Sex']), axis=1)
    
    
    # durationxheart rate
    df['DurationxHeart_Rate'] = (df['Duration']*df['Heart_Rate']).round(2)

    # durationxbodytemp rate
    df['DurationxBody_Temp'] = (df['Duration']*df['Body_Temp']).round(2)
    
    # bodytempxheart rate
    df['Body_TempxHeart_Rate'] = (df['Body_Temp']*df['Heart_Rate']).round(2)

    # Body Fat
    df['BodyFat'] = df.apply(lambda row: estimate_body_fat(row['BMI'], row['Age'], row['Sex']), axis=1)

    # lean_mass
    df['lean_mass'] = df['Weight'] * (1 - df['BodyFat'] / 100)

    # Activity Level (one-hot)
    df['ActivityLevel'] = df['Duration'].apply(activity_level)
    df = pd.get_dummies(df, columns=['ActivityLevel'], drop_first=True, dtype=int)

    # One-hot encode Intensity and MET
    df = pd.get_dummies(df, columns=['IntensityLevel', 'MET'], drop_first=True ,dtype=int)

    return df


# train and test : adding new features
train_data = preprocess(train_data)
test_data = preprocess(test_data)





train_data.columns


train_data = train_data.drop(['id', 'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp'], axis = 1)
test_data = test_data.drop(['id', 'Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate',
       'Body_Temp'], axis = 1)

train_data['Calories'] = train_data.pop('Calories')


X, Y = train_data.drop('Calories', axis=1), train_data[['Calories']]
xtrain, xval, ytrain, yval = train_test_split(X, Y, test_size = 0.3, shuffle=True)


numerical_cols = xtrain.select_dtypes(include=['float64', 'int64']).columns
mktrain = remove_outliers_iqr(pd.concat([xtrain,ytrain], axis=1), numerical_cols)
ytrain = mktrain[['Calories']]
xtrain = mktrain.drop(['Calories'], axis=1)


corr_matrix = train_data.corr()

plt.figure(figsize=(30,30))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.plot()


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

def scaling(scaler, train=None, val=None, test=None, cols=None):
    scal = scaler()
    
    if cols == None:
        cols = list(train.columns)


    scaled_train,scaled_test = train.copy(), test.copy()
    
    scaled_train[cols] = scal.fit_transform(train[cols])
    scaled_test[cols] = scal.transform(test[cols])
    
    if val is not None:
        scaled_val = val.copy()
        scaled_val[cols] = scal.transform(val[cols])
        return scaled_train, scaled_val, scaled_test
    

    return scaled_train, scaled_test

def skew_correct(train, val, test, cols=None):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    train[cols] = pt.fit_transform(train[cols])
    val[cols] = pt.transform(val[cols])
    test[cols] = pt.transform(test[cols])

    return train, val, test
    


one_hot_cols = [col for col in train_data.columns if set(train_data[col]) <= {0, 1}]
continuous_cols = [col for col in train_data.columns if col not in one_hot_cols and col != 'Calories']


xtrain[continuous_cols].skew()


xtrain, xval, test_data = skew_correct(xtrain, xval, test_data, cols = ['MaxHR', 'calories_burned'])
xtrain, xval, test_data = scaling(train=xtrain, val=xval, test=test_data, scaler=RobustScaler, cols = continuous_cols)


xtrain


# def add_noise_auto(train, val=None, test=None, std=0.02, mean=0.0, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     def noise(df):
#         cols = df.select_dtypes(include=['float64']).columns
#         df = df.copy()
#         noise_vals = np.random.normal(mean, std, df[cols].shape)
#         df[cols] = pd.DataFrame(df[cols].values + noise_vals, columns=cols, index=df.index)
#         return df

#     train_noisy = noise(train)
#     val_noisy = noise(val)
#     test_noisy = noise(test)

#     return train_noisy, val_noisy, test_noisy




# xtrain, xval, test_data = add_noise_auto(train=xtrain, val=xval, test=test_data, seed=0)


print(xtrain.shape, xval.shape, test_data.shape)
print(ytrain.shape, yval.shape)


from sklearn.feature_selection import SelectKBest, f_regression

def get_top_k_features(X: pd.DataFrame, y: pd.DataFrame, k: int = 10):
    """
    Selects and returns the top K feature column names based on f_regression scores.

    Args:
        train_dataset (pd.DataFrame): The training dataset (features + target).
        target_column (str): Name of the target/output column.
        k (int): Number of top features to return.

    Returns:
        List[str]: List of top K feature column names.
    """
   # Apply SelectKBest
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(
        X, y)

    # Get mask and selected column names
    mask = selector.get_support()
    top_features = X.columns[mask]

    return list(top_features)



# tensor creating
trainx_array = torch.tensor(xtrain.values, dtype=torch.float32)
testx_array = torch.tensor(xval.values, dtype=torch.float32)

trainy_array = torch.tensor(ytrain.values, dtype=torch.float32)
testy_array = torch.tensor(yval.values, dtype=torch.float32)
print("tensor created")

# tensor dataset
tensor_train = TensorDataset(trainx_array, trainy_array)
tensor_test  = TensorDataset(testx_array, testy_array)
print('dataset created')


#loaders 

trainloader = torch.utils.data.DataLoader(tensor_train, batch_size=65536,num_workers=4, pin_memory=True, shuffle=True)
testloader = torch.utils.data.DataLoader(tensor_test, batch_size=65536,num_workers=4, pin_memory=True,  shuffle=False)


import torch.nn.init as init

class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 512),
            nn.BatchNorm1d(512, momentum=0.5),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(512,256),
            nn.BatchNorm1d(256, momentum=0.5),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.1),


            nn.Linear(256,1)
        )

    def forward(self, x):
        return self.net(x)


class RMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSLELoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Ensures float dtype and positivity
        y_pred = torch.clamp(y_pred.float(), min=0)
        y_true = torch.clamp(y_true.float(), min=0)
        
        # log1p = log(1 + x), more stable for small values
        log_pred = torch.log1p(y_pred + self.eps)
        log_true = torch.log1p(y_true + self.eps)
        
        return torch.sqrt(self.mse(log_pred, log_true) + self.eps)


def init_nn(m, alpha=1.0):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_( m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)  # gamma = 1
        init.constant_(m.bias, 0) 


fn = FNN().to(device).apply(lambda m: init_nn(m, alpha=0.1))


import copy
opt = optim.AdamW(fn.parameters(),lr=3e-1, weight_decay=0.7, eps=1e-8, amsgrad=True)
# opt = optim.Adam(fn.parameters(),lr=1e-1, amsgrad=False)
# opt = optim.SGD(fn.parameters(),weight_decay=1e-2, lr=1e-2, momentum=0.9)
loss_fn= RMSLELoss()



def train_and_save_best(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device : torch.device,
    pat : int=5,
    save_path: str = "best_model.pt"
):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    model.to(device)
    best_val_loss = float('inf')
    model_state_dict_best = model.state_dict()  # Safe fallback
    optim_best = optimizer.state_dict()

    
    train_loss_list = []
    test_loss_list = []
    pat_counter = 0

    for epoch in range(1, epochs+1):
        # ---- Training ----
        model.train()
        train_running_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
    
            bs = xb.size(0)
            train_running_loss += loss.item() * bs
            train_n += bs

        
        train_loss = train_running_loss / train_n
        train_loss_list.append(train_loss)
            
        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                batch_loss = loss_fn(preds, yb).item()
                val_loss += batch_loss * xb.size(0)
                n += xb.size(0)
        val_loss /= n
        scheduler.step(val_loss)
        test_loss_list.append(val_loss)

        
        print(f"Epoch {epoch:3d} â€” Val Loss: {val_loss:.6f}, Train Loss: {train_loss:.6f}")

        # ---- Check for improvement ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            pat_counter = 0
            save_path = f"best_model{best_val_loss:.4f}.pt"
            model_state_dict_best = copy.deepcopy(model.state_dict())
            optim_best = copy.deepcopy(optimizer.state_dict())
            
            print(f"  ðŸŽ‰ New best model saved (loss {best_val_loss:.6f})")

        else:
            pat_counter+=1
            if pat_counter >= pat:
                print(f"early stopping at epochs{epoch}")
                break
                

    # ---- After training: load best weights ----

    if model_state_dict_best is not None:
        torch.save({
            'model_state_dict': model_state_dict_best,
            'optimizer_state_dict': optim_best,
        }, f'checkpoint_optim{round(best_val_loss, 6)}.pth')

    
    model.load_state_dict(model_state_dict_best)
    print(f"Best model (val loss {best_val_loss:.6f}) loaded from {save_path}")

    plt.plot(train_loss_list, label='train_loss')
    plt.plot(test_loss_list, label='test_loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return model,train_loss_list,test_loss_list



model,train_list,test_list = train_and_save_best(fn, trainloader, testloader, opt, loss_fn, epochs=50,device=device,pat=5)


final_test_sub = torch.tensor(test_data.values, dtype=torch.float32).to(device)

model.eval()


with torch.no_grad():
    pred_final = model(final_test_sub).to(device)

pred_final
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

cal = pred_final.to('cpu').numpy().squeeze()

df = pd.DataFrame({
    'id': test['id'],
    'Calories': cal
})

df['Calories'] = df['Calories'].map(lambda x: f"{x:.3f}")

df.to_csv('submission.csv', index=False)
df




