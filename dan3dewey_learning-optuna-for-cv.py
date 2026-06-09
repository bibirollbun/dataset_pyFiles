# The usual things
import numpy as np
import matplotlib.pyplot as plt
# Set matplotlib defaults
plt.rcdefaults()
import pandas as pd

##import seaborn as sns  # for heat map

from scipy.stats import chi2_contingency
from scipy.signal import savgol_filter

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression

import xgboost as xgb
##from xgboost import XGBRegressor
##from xgboost import XGBClassifier

from catboost import CatBoostRegressor

import optuna



import warnings
warnings.simplefilter('ignore')


def confusion_dots_roc(y01, yprob, title, roc_label, plots_prefix="", plotsize=(10, 4)):
    # Make a "confusion dots" plot and show an ROC curve.
    #   y01 is the binary truth.
    #   yprob is the predicted proba from the model.
    # Copied and modified from roc_plots.py at "MOA peel the onion" notebook.

    y_thresh = 0.5  # used 
    plot_alpha = 0.35 #0.20
       
    ysframe = pd.DataFrame([y01, yprob], index=['y', 'y_prob']).transpose()
    # Add a blurred y column
    ysframe['y (blurred)'] = ysframe['y'] + 0.1 * np.random.randn(len(ysframe))

    # Plot the real y (blurred) vs the predicted probability
    # Note the flipped ylim values.
    ysframe.plot.scatter('y_prob', 'y (blurred)', figsize=plotsize,
                         s=2, xlim=(0.0, 1.0), ylim=(1.8, -0.8), alpha=plot_alpha)
    # show the "correct" locations on the plot
    plt.plot([0.0, y_thresh], [0.0, 0.0], '-',
        color='green', linewidth=3)
    plt.plot([y_thresh, y_thresh], [0.0, 1.0], '-',
        color='gray', linewidth=2)
    plt.plot([y_thresh, 1.0], [1.0, 1.0], '-',
        color='green', linewidth=3)
    plt.title("Confusion-dots Plot: " + title, fontsize=16)
    # some labels
    ythr2 = y_thresh/2.0
    plt.text(ythr2 - 0.03, 1.52, "FN", fontsize=16, color='red')
    plt.text(ythr2 + 0.5 - 0.03, 1.52, "TP", fontsize=16, color='green')
    plt.text(ythr2 - 0.03, -0.50, "TN", fontsize=16, color='green')
    plt.text(ythr2 + 0.5 - 0.03, -0.50, "FP", fontsize=16, color='red')

    if len(plots_prefix) > 0: plt.savefig(plots_prefix+"_dots.png")
    plt.show()

    # Make the ROC plot
    fpr, tpr, thresholds = metrics.roc_curve(y01, yprob)
    roc_auc = metrics.auc(fpr, tpr)
    thisfig = plt.figure(figsize=(3,3))
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name=roc_label)
    display.plot()
    plt.plot([0.,1.],[0.,1.],c='gray',alpha=0.3)
    if len(plots_prefix) > 0: plt.savefig(plots_prefix+"_roc.png")
    plt.show()


# Make simple derivative
def dseries_dt(input_series):
    values = np.array(input_series)
    dvals_dt = values.copy()
    dvals_dt[1:-1] = (values[2:] - values[0:-2])/2.0
    # the end points
    dvals_dt[0] = (values[1] - values[0])
    dvals_dt[-1] = (values[-1] - values[-2])
    return dvals_dt


train = pd.read_csv("../input/playground-series-s5e3/train.csv").fillna(method='ffill')
test = pd.read_csv("../input/playground-series-s5e3/test.csv").fillna(method='ffill')

# Fix the incorrect day-of-year values in train
train['day'] = (train.index % 365) + 1
if False:
    plt.figure(figsize=(8,3))
    plt.scatter(train['id'],train['day'],s=3)
    plt.show()

# Adjust some features in train and test
for the_df in [train,test]:
    # Remove the large offset pressure
    the_df['pressure'] -= 990.0
    # Create a deltatemp feature -- No, not so useful
    ##the_df['deltatemp'] = the_df['maxtemp'] - the_df['mintemp']
    ##
    # Add day_sin and day_cos, shifted so sin is centered on high rain section
    the_df['day_sin'] = np.sin( 2.0*np.pi*(the_df['day'] - 0.5 - 15.0)/365.0 )
    the_df['day_cos'] = np.cos( 2.0*np.pi*(the_df['day'] - 0.5 - 15.0)/365.0 )
    # Add delta_dewpoint
    the_df['delta_dewpoint'] = the_df['temparature'] - the_df['dewpoint']
    # Combinations suggested by:
    #  https://www.kaggle.com/code/berkereryilmaz/rainfall-prediction-0-90-with-catboost
    the_df['cloud*humid/press'] = np.log(the_df['cloud'] * the_df['humidity'] /
                                                train['pressure'])
    the_df['dew*humid/sun'] = np.log((1.0+the_df['dewpoint']) * (0.01+the_df['humidity']) /
                                                (0.01+the_df['sunshine']))
    # Suggested by:
    #  https://www.kaggle.com/competitions/playground-series-s5e3/discussion/568647
    ##the_df['delta_wind'] = abs(the_df['winddirection'] - the_df['winddirection'].shift(1))
    ##the_df.loc[the_df.index[0], 'delta_wind'] = 0.0
    # Replace tempArature with the average of the three temps
    ##the_df['temparature'] = (the_df['temparature'] + the_df['maxtemp'] + the_df['mintemp'])/3.0
    
# Keep tempArature, remove min and max temps
train = train.drop(columns=['maxtemp','mintemp'])
test = test.drop(columns=['maxtemp','mintemp'])


# Add a yearly-averaged probability of rain as a function of day
# Create a rolling smooth of binary rainfall, 3-passes of uniform ~ 2nd degree
wind_width = 2  # 2 or 4 width in days
train['rainsmooth'] = train['rainfall'].rolling(
                            wind_width+1, center=True, min_periods=1
                            ).mean().fillna(method='bfill').rolling(
                            wind_width+1, center=True, min_periods=1
                            ).mean().fillna(method='bfill').rolling(
                            wind_width+1, center=True, min_periods=1
                            ).mean().fillna(method='bfill')
# Average over years
prob_rain_by_day = train[['day','rainsmooth']].groupby('day').agg(np.mean)['rainsmooth']

# Add a prob_rain feature as a function of the day
train['prob_rain'] = np.array(prob_rain_by_day[train['day']])
test['prob_rain'] = np.array(prob_rain_by_day[test['day']])
# Add its derivative    
train['dprob_dt'] = 3.0*dseries_dt(train['prob_rain'])
test['dprob_dt'] = 3.0*dseries_dt(test['prob_rain'])

# Plot the smoothed and averaged prob of rainfall, etc.
plt.figure(figsize=(10,4))
plt.scatter(train['day'],train['rainsmooth'],s=2,c='orange')
plt.scatter(train['day'],train['prob_rain'],s=2,c='blue')
plt.scatter(train['day'],train['dprob_dt'],s=2,c='red')
plt.scatter(train['day'],0.2*train['day_sin'],s=1,c='lightblue')
plt.scatter(train['day'],0.2*train['day_cos'],s=1,c='lightblue')
plt.plot([1,365],[0.0,0.0,],c='gray')
plt.xlabel("Day of Year")
plt.title("6-year Average Probability of Rain (blue), dProb/dt (red), smoothed data (orange)")
plt.savefig("ave_rain_vs_day.png")
plt.show()

# Drop the rainsmooth column since it's not the target nor a predictor.
train = train.drop(columns=['rainsmooth'])

# Add a non-feature rain_clr to color-code scatter plots
train['rain_clr'] = '#8888FF'  # rain color
train.loc[train['rainfall'] < 0.5, 'rain_clr'] = 'orange'  # not rain color

# Add a non-feature column to do stratification by rainfall and quadrant of year
train['stratify'] = 4*train['rainfall'] + 2*(train['day_sin'] > 0.0) + 1*(train['day_cos'] > 0.0)
##train['stratify'].value_counts()

##print(train.info())
##print(train.describe())
train.tail(5)


# Set these variables for modeling use below
target_name = 'rainfall'
# All the possible features
features = ([col for col in train.columns.to_list() if
                    col not in ['id',target_name,'rain_clr','stratify']])
# Drop some low importance features?
features.remove('day')  # Not needed given day_sin, day_cos and prob_rain
features.remove('day_sin')
#
features.remove('dprob_dt')
features.remove('winddirection')


# Show a color-coded scatter matrix of the features
print(features)
pd.plotting.scatter_matrix(train[features], alpha=0.2, figsize=(12,12), c=train['rain_clr'])
plt.savefig("features_scatter_matrix.png")
plt.show()


# Look at (some of) the features' and target's Time Series and Autocorrelation
# ['pressure', 'temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine',
#  'winddirection', 'windspeed', 'prob_rain', 'dprob_dt', 'rainfall']

# Include a simulated rainfall series based on prob_rain
sim_rainfall = []
for this_p in train['prob_rain']:
    sim_rainfall.append(np.random.choice([0,1], p=[1-this_p,this_p]))
train['sim_rainfall'] = sim_rainfall

for this_col in ['cloud','sunshine','humidity','delta_dewpoint','rainfall','sim_rainfall']:
    if True:
        plt.figure(figsize=(10,2.5))
        plt.scatter(train['day'],train[this_col],s=3)
        plt.xlim(0.,370.0)
        ##plt.ylim(-0.25,0.25)
        plt.ylabel(this_col)
        plt.title(this_col+": Time Series vs day")
        plt.show()
    # Autocorrelation
    automax = 0.35
    ##if this_col in ['humidity', 'cloud', 'sunshine', 'windspeed', 'rainfall', 'sim_rainfall']:
    ##    automax = 0.35
    plt.figure(figsize=(10,2.5))
    pd.plotting.autocorrelation_plot(train[this_col])
    plt.xlim(0.,100.0)
    plt.ylim(-0.20,automax)
    plt.title(this_col+"'s Autocorrelation")
    plt.show()

# Drop the simulated rainfall
train = train.drop(columns=['sim_rainfall'])


# Get and look at the autocorrelation for the original data
RainML = pd.read_csv("../input/rainfall-prediction-using-machine-learning/Rainfall.csv"
                            ).fillna(method='ffill')
RainML['rainfall'] = 1.0*(RainML['rainfall'] == "yes")
##RainML.columns


for this_col in []:  #['sunshine','rainfall']:
    if True:
        plt.figure(figsize=(10,2.5))
        plt.scatter(RainML.index,RainML[this_col],s=3)
        plt.scatter(train['day'],train['prob_rain'],s=2,c='orange',alpha=0.3)
        plt.xlim(0.,)
        plt.ylabel(this_col)
        plt.title("RainML "+this_col+": Time Series vs Index")
        plt.show()
    # Autocorrelation
    automax = 0.35
    plt.figure(figsize=(10,2.5))
    pd.plotting.autocorrelation_plot(RainML[this_col])
    plt.xlim(0.,100.0)
    plt.ylim(-0.20,automax)
    plt.title("RainML "+this_col+"'s Autocorrelation")
    plt.show()


# AUC from simple probability
if False:
    confusion_dots_roc(train['rainfall'], train['prob_rain'], 
                   "6-year Prob of Rain gives AUC={:.4f}".format(
                       metrics.roc_auc_score(train['rainfall'], train['prob_rain'])),
                   "Prob Rain", plots_prefix="", plotsize=(10,3.5))


# Number of folds
FOLDS = 7

# Parameters not tuned
sample_by = 'tree'  # 'tree' or 'level'
other_params = {
            'objective' : 'binary:logistic',
            'eval_metric' : "auc",
            'random_state': 9,
            'n_estimators' : 1000,
            ##'learning_rate' : 0.04,
            'scale_pos_weight' : 1,
            'early_stopping_rounds' : 30,
            'enable_categorical' : True,
            'max_cat_threshold' : 10, # max number of categories considered when partitioned:
            'max_cat_to_onehot' : 4,  # ge this: partition into children nodes, else onehots
            'alpha': 0.096,
            'gamma' : 1.5,
            'device' : "cpu"
    }

def objective(trial, use_params={}):
    # Added use_param as a way to rerun it for a given parameter set and get oof predictions.
    # Variables that are assumed to be in the environment (a bit sloppy):
    #   FOLDS, ntrials, other_params, sample_by, train, target_name, test
    # This CV code is based on https://www.kaggle.com/code/dan3dewey/cibmtr-simple-lr-coxph-xgb-models
    # Get the parameters to use
    if use_params == {}:
        trial_num = trial.number
        opt_params = {
            'max_depth': trial.suggest_categorical('max_depth', [2, 3, 4, 5, 6, 7]),
            'colsample_by'+sample_by: trial.suggest_uniform(
                            'colsample_by'+sample_by, 0.35, 0.75),
            'subsample': trial.suggest_uniform('subsample', 0.45, 0.85),
            'learning_rate': trial.suggest_loguniform(
                            'learning_rate', 0.02, 0.07),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 90),
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        }
    else:
        trial_num = -1
        opt_params = use_params
    # Randomize the splits for each trial based on digits in continuous parameters:
    kf_state =  1*(int(1e5*opt_params['colsample_by'+sample_by]) % 10)  # 5th decimal digit
    kf_state += 10*(int(1e5*opt_params['subsample']) % 10)
    kf_state += 100*(int(1e5*opt_params['learning_rate']) % 10)
    kf_state += 1000*(int(1e5*opt_params['lambda']) % 10)
    # Do stratified CV with the trial parameters
    kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=kf_state)
    # Save the predictions for oof and test
    oof_xgb = np.zeros(len(train))
    test_xgb = np.zeros(len(test))  # test predictions
    ave_iterations = 0
    for i, (train_index, valid_index) in enumerate(
                                    kf.split(train, train['stratify'])):
        # The values for this fold
        x_train = train.loc[train_index,features].copy()
        y_train = train.loc[train_index,target_name]
        x_valid = train.loc[valid_index,features].copy()
        y_valid = train.loc[valid_index,target_name]
        x_test = test[features].copy()

        # Combine the parameters
        params = other_params | opt_params

        # Set up the model
        model_xgb = xgb.XGBClassifier(**params)  
        # Fit the model
        model_xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=None)
        
        ave_iterations += model_xgb.best_iteration + 1 # best, early stopped, iteration
        # Set the oof predictions
        oof_xgb[valid_index] = model_xgb.predict_proba(x_valid)[:,1]
        if use_params != {}:
            # Update the test predictions, they will be returned
            test_xgb += model_xgb.predict_proba(test[features])[:,1] / kf.n_splits

    auc = metrics.roc_auc_score(train[target_name], oof_xgb)
    # Squish the 0-to-some-value range so that low values don't dominate the plot scale
    if auc < 0.85:
        auc = 0.80 + 0.05*auc/0.85  # 0 to 0.85 maps to 0.80 to 0.85
    if (trial_num < 0) or (ntrials < 51):  # ntrials need not be defined when use_params is given
        print("{:3}:  Ave iters per fold = {:.0f}   oof AUC = {:.4f}   kf_state = {}".format(
                    trial_num, ave_iterations/FOLDS, auc, kf_state))
    # Return the adjusted oof auc, optionally with the oof and test predictions
    if use_params == {}:
        return auc
    else:
        feat_import = model_xgb.feature_importances_
        return auc, oof_xgb, test_xgb, feat_import


# Do the study
# . . . . . . Or not: will skip to "Predictions using ..." and use previous hyper-parameters
do_study = True

if do_study:
    ntrials = 1000

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # From: ERROR WARNING INFO DEBUG
    study = optuna.create_study(direction='maximize')  # want maximum AUC
    study.optimize(objective, n_trials=ntrials)


# Show various results from study
if do_study:
    print('\nNumber of finished trials:', len(study.trials))
    # Best Optuna parameters
    best_trial_params = study.best_trial.params
    print("Best trial was {} with oof AUC = {:.4f} and params:".format(
            study.best_trial.number, study.best_value))
    print(best_trial_params)

    # All results are in a dataframe
    study_data = study.trials_dataframe().sort_values(by='value',ascending=False).reset_index(drop=True)
    # Show just the number, value and param values
    study_data.iloc[0:5, [0,1]+list(range(5,5+len(best_trial_params)))]

    # Added "matplotlib" into the following visualization routines, e.g.,
    #   optuna.visualization.matplotlib.plot_optimization_history(study)
    # Can save them with the usual
    #   plt.savefig("optim_history.png")
    # After Optuna plots, go back to matplotlib defaults
    #   plt.rcdefaults()
    
    #plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig("optim_history.png")

    #plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig("optim_parallel_coords.png")

    # plot_slice: shows the evolution of the search.
    # Shows where in hyperparameter space the search went and which parts of the space were explored more.
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig("optim_slices.png")

    #plot_contour: plots parameter interactions on an interactive chart. You can choose which hyperparameters you would like to explore.
    if False:   # Show the continuous params
        optuna.visualization.matplotlib.plot_contour(study, params=[
                            'colsample_by'+sample_by,
                            'subsample',
                            'learning_rate',
                            'lambda'])
        plt.savefig("optim_contours.png")

    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig("optim_importances.png")

    #Visualize empirical distribution function
    ##optuna.visualization.matplotlib.plot_edf(study)
    ##plt.savefig("optim_distribution.png")
    # Show pdf instead of cdf (using optuna plot formatting)
    plt.figure(figsize=(6,3))
    trial_aucs = np.clip(study_data['value'],0.860,1.0)
    plt.hist(trial_aucs,bins=np.linspace(0.86,0.91,51))
    plt.xlim(0.86,0.91)
    plt.title("Histogram of the Trials' AUC values")
    plt.savefig("hist_of_trial_aucs.png")
    plt.show()

    # Back to matplotlib defaults
    plt.rcdefaults()


# Instead of using the single best set of parameters,
# get a set of trials' params to define models for fitting and averaging
# (Can get the params for any trial using: study.trials[number].params )
use_previous = False   # True = overrides do_study and uses previous.

if do_study and not use_previous:
    # Get the params from a set of the high-scoring trials
    num_offset = 20  # How far back from the maximum to start
    num_to_ave = 20  # How many trials to average
    spacing = 9      # use every nth
    param_set = []
    for num_trial in study_data.loc[range(
                num_offset, num_offset + num_to_ave*spacing, spacing),'number']:
        param_set.append(study.trials[num_trial].params)
else:
    # Can use previously determined parameters, these are from v32:
    param_set = [
        {'max_depth': 2, 'colsample_bytree': 0.6084211873448573, 'subsample': 0.5040493896117686, 'learning_rate': 0.06640837807118771, 'min_child_weight': 6, 'lambda': 0.02792293543884954},
        {'max_depth': 2, 'colsample_bytree': 0.6226149873893421, 'subsample': 0.567151319869339, 'learning_rate': 0.06562419545556779, 'min_child_weight': 8, 'lambda': 0.03979809539717699},
        {'max_depth': 2, 'colsample_bytree': 0.6353114751915732, 'subsample': 0.5294242995889109, 'learning_rate': 0.06843113188639983, 'min_child_weight': 9, 'lambda': 0.03438287630409512},
        {'max_depth': 2, 'colsample_bytree': 0.6084052712482292, 'subsample': 0.511244555659431, 'learning_rate': 0.06661018640891653, 'min_child_weight': 7, 'lambda': 0.027400675426968696},
        {'max_depth': 2, 'colsample_bytree': 0.6272816170107905, 'subsample': 0.5478824833925634, 'learning_rate': 0.06990660684975804, 'min_child_weight': 12, 'lambda': 0.044099431403728},
        {'max_depth': 2, 'colsample_bytree': 0.6003256041321838, 'subsample': 0.4987833043404876, 'learning_rate': 0.06992369680032576, 'min_child_weight': 5, 'lambda': 0.006626604962285037},
        {'max_depth': 2, 'colsample_bytree': 0.6163098539543657, 'subsample': 0.5241110149336041, 'learning_rate': 0.06690612802193746, 'min_child_weight': 13, 'lambda': 0.06242670238143977},
        {'max_depth': 2, 'colsample_bytree': 0.6222799589205465, 'subsample': 0.4816917479727495, 'learning_rate': 0.06706162440901946, 'min_child_weight': 16, 'lambda': 0.03860385596881546},
        {'max_depth': 2, 'colsample_bytree': 0.6392908090388127, 'subsample': 0.520541886181286, 'learning_rate': 0.0637847159166569, 'min_child_weight': 8, 'lambda': 0.0360413497749475},
        {'max_depth': 2, 'colsample_bytree': 0.6522418536384017, 'subsample': 0.5428086492199251, 'learning_rate': 0.06995286569894069, 'min_child_weight': 16, 'lambda': 0.038318128849167216}
        ]


# Get and average the oof and test predictions for the parameter set:
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
feature_importance = np.zeros(len(features))

num_to_ave = len(param_set)
for these_params in param_set:
    print("Params:",these_params)
    # Use the Optuna objective function to fit the model and return results
    trial_auc, trial_oof, trial_test, feat_import = objective(
                                            None, use_params=these_params)
    oof_preds += trial_oof / num_to_ave
    test_preds += trial_test / num_to_ave
    feature_importance += feat_import / num_to_ave


# Show AUC for the oof predictions
oof_auc = metrics.roc_auc_score(train['rainfall'], oof_preds)
confusion_dots_roc(train['rainfall'], oof_preds, "  OOF Predictions    AUC = {:.4f}".format(
                    oof_auc), "XGB Clf", plotsize=(10,3.5),
                  plots_prefix="confuse_dots_train.png")


# Autocorrelation of oof predictions
if True:
    automax = 0.35
    plt.figure(figsize=(10,2.5))
    pd.plotting.autocorrelation_plot(oof_preds)
    plt.xlim(0.,100.0)
    plt.ylim(-0.20,automax)
    plt.title("oof_preds"+"'s Autocorrelation")
    plt.show()
if False:
    automax = 1.0
    plt.figure(figsize=(10,2.5))
    pd.plotting.autocorrelation_plot(train['prob_rain'])
    plt.xlim(0.,100.0)
    plt.ylim(-0.20,automax)
    plt.title("Probability of Rain (6-yr ave)"+"'s Autocorrelation")
    plt.show()


# List the features and show importances
print("Features used:\n", features)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(8, 12))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
##plt.xlim(0.0,0.10)  
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  # Flip features for better readability
plt.savefig("XGB_feature_importances.png")
plt.show()


# Very simple prediction for test
##test['rainfall'] = test['prob_rain']

# Prediction from the model
test['rainfall'] = test_preds

plt.figure(figsize=(8,2.5))
plt.hist(test['rainfall'],bins=100)
plt.title("Histogram of Test Predictions")
plt.xlim(0,1)
plt.ylabel("Number")
plt.savefig("test_preds_hist.png")
plt.show()



# Submit the prediction
test[['id','rainfall']].to_csv("submission.csv", header=True, 
                        index=False, na_rep='', float_format='%.3f')

##!head submission.csv


# Vers  oof AUC  Notes
# v16    8966
# v17    8980
# v18    8969
# v19    8978
# v20    8967
# v21    8983
# v22    8968
# v23    8965
# v24    8961
# v25    8936
# v26    8982
# v27     "     Same as v26, by using v26 hyperparameters so that study() doesn't have to run.
# v28
# v29    9031   Use different k-fold random states for each trial (based on sample-by-tree's 4th,5th digits.)
# v30    9013   Use different k-fold random states for each trial (based on 2 params' 4th,5th digits.)
# v31 *  9012   Use different k-fold random states for each trial (based on 4 params' 5th digits.)
# v32    9026   Add 2 engineered features and remove dprob_dt, winddirection. Keep v31 random k-fold scheme. 
# v33    9003   Use averaged temps for tempArature.
#        9027   Back to simple tempArature (v32) but reduce smoothing value of prob_rain from 4 to 2.
# v34 *  9038   Included delta_dewpoint feature.
#        9033   Do v34 but change bytree to bylevel; keep by tree.
# v35    9041   Do v34, bytree, final prediction: skip 3, average next 20.

#     * = competition submit

