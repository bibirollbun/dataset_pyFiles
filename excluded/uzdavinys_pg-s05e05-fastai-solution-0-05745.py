# notebook params
TRANSFORM_TARGET = True
ROUND_PREDICTIONS = True

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are ava# taken from
# https://www.kaggle.com/code/jhoward/why-you-should-use-a-framework
from pathlib import Path
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    !pip install -Uqq fastai
    
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plots for statistical analysis
import matplotlib.pyplot as plt # for data visualization
from sklearn.metrics import mean_squared_log_error # model validation metric

# define default colors for plots in notebook
from matplotlib import cycler
from matplotlib.colors import LinearSegmentedColormap
colors = ["#068D9D", "#53599A", "#607BB0", "#6D9DC5", "#77BECF", "#80DED9", "#AEECEF"]

plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=cycler('color', colors))

# fast ai libraries
from fastai.tabular.all import *

# for measuring model's performance
from sklearn import metrics

# constants
TARGET_NAME = "Calories"
SEED = 42
BATCH_SIZE = 10240
MAX_EPOCHS = 200
NO_MODELS = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


def set_seed_value(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Random SEED set to {seed}.")

set_seed_value(SEED)


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv', index_col='id')
# original dataset
df_orig = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv")
df_orig = df_orig.rename(columns={"Gender": "Sex"})


# transform target
if TRANSFORM_TARGET:
    df_train[TARGET_NAME] = np.log1p(df_train[TARGET_NAME])
    df_orig[TARGET_NAME]= np.log1p(df_orig[TARGET_NAME])


def add_features(df):
    """
    How new features were created can be found in this notebook:
    https://www.kaggle.com/code/uzdavinys/pg-s05e05-create-new-features#7.3-most-important-features
    """
    # BMI (Body Mass Index)
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # Resting Metabolic Rate (RMR)
    # reference https://www.calculator.net/bmr-calculator.html
    # 0 - female, 1 - men 
    
    # Mifflin-St Jeor Equation
    df["rmr_miff"] = 10 * df["Weight"] + 6.25 * df["Height"] - 5 * df["Age"]    
    cond = df["Sex"] == 'female'
    df.loc[cond, "rmr_miff"] = df.loc[cond, "rmr_miff"] - 161
    cond = df["Sex"] == 'male'
    df.loc[cond, "rmr_miff"] = df.loc[cond, "rmr_miff"] + 5
    
    # Revised Harris-Benedict Equation
    cond = df["Sex"] == 'female'
    df.loc[cond, "rmr_harris"] = 9.247 * df.loc[cond, "Weight"] + 4.799 * df.loc[cond, "Height"] - 5.677 * df.loc[cond, "Age"] + 88.362
    cond = df["Sex"] == 'male'
    df.loc[cond, "rmr_harris"] = 13.397 * df.loc[cond, "Weight"] + 3.098 * df.loc[cond, "Height"] - 4.330 * df.loc[cond, "Age"] + 447.593

    # Heart Rate Reserve
    df['max_HR'] = 220 - df['Age']

    # Workout Intensity
    df['workout_intensity'] = df['Heart_Rate'] / df['max_HR']

    # Exercise Energy Expenditure Rate
    df['expenditure_rate'] =  df['Heart_Rate'] * df['Weight']  / df['Duration']

    # Age Group
    df['age_range'] = pd.cut(
        df['Age'],
        bins=[17, 30, 45, 60, float('inf')],
        labels=['Young', 'Middle_aged', 'Older', 'Senior']
    )

    # Weight Status Category
    df['bmi_category'] = pd.cut(
        df['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 39.9, float('inf')],
        labels=['Underweight', 'Healthy weight', 'Overweight', 'Obese', 'Severely obese']
    )

    del df["max_HR"]

    # new features from notebook
    df["Duration_x_Heart_Rate"] = df["Duration"] * df["Heart_Rate"]
    df["Duration_o_Heart_Rate"] = df["Duration"] / df["Heart_Rate"]
    df["Duration_x_Body_Temp"] = df["Duration"] * df["Body_Temp"]
    df["Body_Temp_x_Heart_Rate"] = df["Body_Temp"] * df["Heart_Rate"]
    df["Age_x_Duration"] = df["Age"] * df["Duration"]
    df["Age_x_Heart_Rate"] = df["Age"] * df["Heart_Rate"]
    df["Age_x_Body_Temp"] = df["Age"] * df["Body_Temp"]
    df["Height_x_Weight"] = df["Height"] * df["Weight"]
    
    return df


df_train = add_features(df_train)
df_test = add_features(df_test)
df_orig = add_features(df_orig)


def convert_2_dataloader(df, cat_names, cont_names, target_names, seed):
    """
    Custom function to convert pandas DataFrame to loader obbject
    """
    # convert pandas DataFrame to fastai DataLoader object
    # code snippet taken from
    # https://docs.fast.ai/tabular.learner.html
    splits = RandomSplitter(valid_pct=0.2, seed = SEED)(df)

    procs = [Categorify, FillMissing, Normalize]

    # tabular object (only categorical features)
    to = TabularPandas(df[cat_names + cont_names + target_names],
                       procs = procs,
                       cat_names = cat_names,
                       cont_names = cont_names,
                       y_names = target_names,
                       splits = splits)

    # create dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dls = to.dataloaders(BATCH_SIZE, device = device)
    
    return dls


# column names for categorical and continous features

# original features
# cat_names = ["Sex"]
# cont_names = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

# all features (version 1)
cat_names = ["Sex", 'age_range', 'bmi_category']
cont_names = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
              'BMI', 'rmr_miff', 'rmr_harris', 'workout_intensity', 'expenditure_rate']

# add new features from https://www.kaggle.com/code/jimgruman/food-glorious-food
# notebook
cont_names += ['Duration_x_Heart_Rate', 'Duration_o_Heart_Rate',
               'Duration_x_Body_Temp', 'Body_Temp_x_Heart_Rate', 'Age_x_Duration',
               'Age_x_Heart_Rate', 'Age_x_Body_Temp', 'Height_x_Weight']

# best features
# cat_names = ["Sex"]
# cont_names = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp',
#               'rmr_miff', 'rmr_harris', 'workout_intensity', 'expenditure_rate']


# dataloader for training model
dls = convert_2_dataloader(df_train, cat_names, cont_names, target_names = [TARGET_NAME], seed = SEED)


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):

        actual = torch.clamp(actual, min=1e-6)
        pred = torch.clamp(pred, min=1e-6)
    
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


# create very small learner
learner = tabular_learner(
    dls, layers = [256, 128, 64, 32],
    n_out = 1,
    metrics = [mae, rmse],
    # loss_func= RMSLELoss(),
    cbs=[
        ReduceLROnPlateau(monitor='valid_loss', min_delta=0.5, patience=10), # for reducing learning rate
        EarlyStoppingCallback(monitor='valid_loss', min_delta=0, patience=10) # early stopping
    ]
)

learner.summary()


learner.lr_find(suggest_funcs=(slide, valley))


learner.fit(MAX_EPOCHS, lr = 0.002)


# get few random predictions
learner.show_results()


# empty DataFrame to store predictions from different models
df_preds_valid = pd.DataFrame()
preds, y = learner.get_preds()
df_preds_valid.loc[:, [TARGET_NAME]] = y.numpy()
df_preds_valid.loc[:, f"{TARGET_NAME}_pred"] = preds.numpy()

# just in case fill negative values with zeros
print(f"{len(df_preds_valid.loc[df_preds_valid[f'{TARGET_NAME}_pred']<0])} negative predictions found.")

# clip predictions
mn = df_train.Calories.min()
mx = df_train.Calories.max()
df_preds_valid[f"{TARGET_NAME}_pred"] = np.clip(df_preds_valid[f"{TARGET_NAME}_pred"], a_min = mn, a_max = mx)
    
preds = df_preds_valid[f"{TARGET_NAME}_pred"].values

if TRANSFORM_TARGET:
    y = np.expm1(y)
    preds = np.expm1(preds)
    if ROUND_PREDICTIONS:
        y = y.numpy().reshape(-1).round(2)
        preds = preds.round(2)
else:
    if ROUND_PREDICTIONS:
        df_preds_valid = df_preds_valid.round(2)

# return 10 random predictions
df_preds_valid.sample(10)


rmsle_val = np.sqrt(mean_squared_log_error(y, preds))
print(f"Validation root mean squared logarithmic error regression loss: {rmsle_val:.6f}")


# original sample
dl = learner.dls.test_dl(df_orig)
preds_orig, y = learner.get_preds(dl=dl)

preds_orig = preds_orig.numpy().reshape(-1)

mn = df_train.Calories.min()
mx = df_train.Calories.max()
preds_orig =  np.clip(preds_orig, mn, mx)

if TRANSFORM_TARGET:
    y = np.expm1(y)
    preds_orig = np.expm1(preds_orig)
    
if ROUND_PREDICTIONS:
    y = y.numpy().reshape(-1).round(2)
    preds_orig = preds_orig.round(2)


rmsle_orig = np.sqrt(mean_squared_log_error(y, preds_orig))
print(f"Root mean squared logarithmic error regression loss on original sample: {rmsle_orig:.6f}")


def ensemble(dls, learner_cnt=5, learn_rate=0.02, layers=[256, 128, 64, 32]):
    """
    Helper function to create multiple models
    """
    learners = list()
    for _ in range(learner_cnt):
        print(f"Learner {_} started training.")
        learner = tabular_learner(dls,
                                  layers = layers,
                                  n_out = 1,
                                  metrics = [mae, rmse],
                                  # loss_func= RMSLELoss(),
                                  cbs=[ReduceLROnPlateau(monitor='valid_loss', min_delta=0.5, patience=10), # for reducing learning rate
                                       EarlyStoppingCallback(monitor='valid_loss', min_delta=0, patience=10)]) # early stopping
        with learner.no_bar(), learner.no_logging():
            learner.fit(MAX_EPOCHS, lr = learn_rate)
            
        learners.append(learner)
        print(f"Learner {_} trained.")
    return learners


%%time
learners_0 = ensemble(dls, learner_cnt=NO_MODELS, learn_rate=0.002, layers=[256, 128, 64, 32])


# empty DataFrame to store predictions from different models
df_valid = pd.DataFrame()

for i, _l in enumerate(learners_0):
    preds, y = _l.get_preds()
    # load real values only first time
    if i == 0:
        df_valid.loc[:, [TARGET_NAME]] = y.numpy().reshape(-1)
    df_valid.loc[:, f"pred_{i}"] = preds.numpy().reshape(-1)
    
# get average predictions for each target
df_valid.loc[:, "pred_mean"] = df_valid.iloc[:, 1:].mean(axis = 1)
df_valid.head()


# just in case fill negative values with zeros
print(f'{len(df_valid.loc[df_valid["pred_mean"]<1])} negative predictions found.')

# clip predictions
mn = df_train.Calories.min()
mx = df_train.Calories.max()
df_valid["pred_mean"] = np.clip(df_valid["pred_mean"], a_min = mn, a_max = mx)

preds = df_valid["pred_mean"].values

if TRANSFORM_TARGET:
    y = np.expm1(y)
    preds = np.expm1(preds)
    if ROUND_PREDICTIONS:
        y = y.numpy().reshape(-1).round(2)
        preds = preds.round(2)
else:
    if ROUND_PREDICTIONS:
        df_valid = df_valid.round(2)

# return 10 random predictions
df_valid.sample(10)


rmsle_val = np.sqrt(mean_squared_log_error(y, preds))
print(f"Validation root mean squared logarithmic error regression loss: {rmsle_val:.6f}")


# # create test dataset dataloader object
# # and make predictions from single model
# dl = learner.dls.test_dl(df_test)
# preds_test, _ = learner.get_preds(dl=dl)

# mn = df_train.Calories.min()
# mx = df_train.Calories.max()

# preds_test = preds_test.numpy().reshape(-1)
# if TRANSFORM_TARGET:
#     preds_test =  np.clip(preds_test, mn, mx)
#     preds_test = np.expm1(preds_test)
#     df_sub['Calories'] = preds_test
# else:
#     df_sub['Calories'] =  np.clip(preds_test, mn, mx)

# if ROUND_PREDICTIONS:
#     df_sub['Calories'] = df_sub['Calories'].apply(lambda x: round(x, 2))

# df_sub.to_csv("submission.csv")
# df_sub.head(10)


# empty DataFrame to store predictions from different models
df_preds = pd.DataFrame()

for i, _l in enumerate(learners_0):
    dl = learner.dls.test_dl(df_test)
    preds, _ = _l.get_preds(dl=dl)
    df_preds.loc[:, f"pred_{i}"] = preds.numpy().reshape(-1)
    
# get average predictions for each target
df_preds.loc[:, "pred_mean"] = df_preds.iloc[:, 1:].mean(axis = 1)

# get random predictions
df_preds.sample(5)


mn = df_train.Calories.min()
mx = df_train.Calories.max()

preds_test = df_preds['pred_mean'].values
if TRANSFORM_TARGET:
    preds_test = np.clip(preds_test, mn, mx)
    preds_test = np.expm1(preds_test)
    df_sub['Calories'] = preds_test
else:
    df_sub['Calories'] =  np.clip(preds_test, mn, mx)

if ROUND_PREDICTIONS:
    df_sub['Calories'] = df_sub['Calories'].apply(lambda x: round(x, 2))
df_sub.to_csv('submission.csv')
df_sub.head(10)




