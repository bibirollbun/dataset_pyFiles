%%capture
!pip install autogluon.tabular scikit-learn==1.5.2
#!pip install "autogluon.tabular[tabpfn]==1.4.0"


import os
import glob

import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')


# LOAD DATA

train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv').drop(columns = 'id')
test_df  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv').drop(columns = 'id')
 
print(f'Train : {train_df.shape[0]}')
print(f'Test  : {test_df.shape[0]}')


# UNLOAD ALL AUTOGLUON OOF

def load_and_concat_csv(path, id_col=None, pred_col="prediction", prefix=""):
    """
    Load semua CSV dari folder, urutkan namanya, ambil kolom prediksi,
    simpan kolom id kalau ada, dan gabungkan jadi 1 DataFrame.
    
    Parameters:
        path (str): folder path
        id_col (str or None): nama kolom id (None kalau tidak ada)
        pred_col (str): nama kolom prediksi
        prefix (str): prefix tambahan untuk nama kolom (default "")
    """
    all_files = sorted(glob.glob(os.path.join(path, "*.csv")))
    
    id_df = None
    dfs = {}
    
    for f in all_files:
        base = os.path.splitext(os.path.basename(f))[0]
        name = prefix + base  # tambahin prefix di sini
        df = pd.read_csv(f)
        
        # Simpan id hanya dari file pertama kalau ada
        if id_col is not None and id_df is None:
            id_df = df[[id_col]]
        
        # Ambil kolom prediksi
        dfs[name] = df[pred_col]
        
        print(f"âœ… Successfully load {name}")
    
    # Gabungkan semua prediksi jadi kolom
    pred_df = pd.concat(dfs, axis=1)
    
    # Kalau ada id, gabungkan dengan id di paling kiri
    if id_df is not None:
        combined_df = pd.concat([id_df, pred_df], axis=1)
    else:
        combined_df = pred_df
    
    print(f"\nðŸŽ‰ Semua file dari '{path}' selesai digabung!")
    print(f"Total file digabung: {len(all_files)}")
    return combined_df
    
# LOAD OOF DATA
oof_path = "/kaggle/input/autogluon-oof/AutoGluon OOF/OOF"
oof_df = load_and_concat_csv(oof_path, id_col=None, pred_col="oof_prediction")

# LOAD TEST DATA
test_path = "/kaggle/input/autogluon-oof/AutoGluon OOF/Test"
submission_df = load_and_concat_csv(test_path, id_col="id", pred_col="accident_risk", prefix = "oof_")


# MERGE LOADED FILE WITH REAL DATA

merge_train = pd.concat((train_df, oof_df),axis = 1)
merge_test  = pd.concat((test_df, submission_df), axis = 1).drop(columns = 'id')

merge_train.shape, merge_test.shape


display(merge_train)
display(merge_test)


# CHECK TARGET DISTRIBUTION

sns.histplot(train_df['accident_risk'], kde = True, bins = 50)


# CHECK CORRELATION BETWEEN OOF

# GET ALL OOF AUTOGLUON COLUMNS
autogluon_features = merge_train.iloc[:, 13:].columns

# HEATMAP CORRELATION
plt.figure(figsize = (16, 9))
corr_matrix = merge_train[autogluon_features].corr(method = 'spearman')
sns.heatmap(corr_matrix, annot = False, cmap = 'coolwarm', fmt = '.2f')


# DROP USELESS FEATURES

merge_train = merge_train.drop(columns = ['oof_autogluon6_cv055930_lb05546'])
merge_test  = merge_test.drop(columns  = ['oof_autogluon6_cv055930_lb05546'])

print(f'Useless Features Dropped! âœ…')


# FEATURE ENGINEERING 

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

train = clip(f)(merge_train)
test = clip(f)(merge_test)

merge_train['score'] = train
merge_test['score']  = test

print("Added New Feature âœ…")

merge_train


# TARGET ENCODER

#FEATURES = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 'weather', 'road_signs_present',	'public_road', 'time_of_day', 'holiday', 'school_season', 'num_reported_accidents', 'score']

# DO ENCODING FOR EACH X COLUMN
#for col in FEATURES:
#    mean_encode = orig_df.groupby(col)['accident_risk'].mean()
#    std_encode  = orig_df.groupby(col)['accident_risk'].std()

#    # APPLY MEAN ENCODE
#    train_df[f'mean_{col}'] = train_df[col].map(mean_encode)
#    test_df[f'mean_{col}']  = test_df[col].map(mean_encode)

#train_df


# AUTOGLUON

# DEFINE AUTOGLUON
predictor = TabularPredictor(label = 'accident_risk',
                             problem_type = 'regression',
                             eval_metric = 'rmse')

# TRAIN AUTOGLUON
predictor.fit(merge_train,
              presets = 'best_quality',
              time_limit = 3600 * 11,
              auto_stack = True,
              #num_bag_folds = 7,
              #num_bag_sets = 3,
              num_cpus = 4,
              verbosity = 1,
              #ag_args_fit={'num_gpus': 1}
             )


# COMPARE MODELS
predictor.leaderboard()


%%time
# CHECK FEATURE IMPORTANCES 

importance_df = predictor.feature_importance(merge_train[:100])

importance_df.style.background_gradient(subset=['importance', 'stddev'], cmap='Blues')


# PLOT FEATURE IMPORTANCE

imp = importance_df['importance'].sort_values(ascending=True)

plt.figure(figsize=(6, 8))
imp.plot(kind='barh', color='steelblue')
plt.title('Feature Importance (AutoGluon)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()


# CHECKING BEST MODEL 

best_model = predictor.model_best

print(f'Best Model : {best_model}')


%%time
# CHECK SUBMISSION

# TEST DATA PREDICTION
y_test = predictor.predict(merge_test)

submission = pd.read_csv(r'/kaggle/input/playground-series-s5e10/sample_submission.csv')

submission['accident_risk'] = y_test

submission


# GET OOF (OUT-OF-FOLD) PREDICTION

# GET OOF
oof_predictions = predictor.predict_oof()

# CONVERT TO DATAFRAME
y_pred = oof_predictions.to_frame(name = 'oof_prediction')  # ---> RETURN DATAFRAME
oof_df = pd.DataFrame(y_pred)

oof_df



# SAVE SUBMISSION
submission.to_csv(r'autogluon15.csv', index = False)
oof_df.to_csv(r'oof_autogluon15.csv', index = False)


# EVALUATION

rmse = mean_squared_error(train_df['accident_risk'], oof_predictions, squared = False)
r2   = r2_score(train_df['accident_risk'], oof_predictions)

print(f'RMSE : {rmse}')
print(f'R2   : {r2}"')


# RESIDUAL PLOT

y_true = train_df['accident_risk']

plt.figure(figsize = (12, 5))

# ACTUAL VS PREDICTED DATA
plt.subplot(1, 2, 1)
plt.scatter(x = y_true, y = y_pred, alpha = 0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs Predicted Data')

# RESIDUAL PLOT
residual = y_true - y_pred['oof_prediction'].values

plt.subplot(1, 2, 2)
plt.scatter(x = y_pred, y = residual, alpha = 0.6)
plt.axhline(y=0, color='r', linestyle='--')  # garis nol sebagai referensi
plt.xlabel("Predicted values")
plt.ylabel("Residuals (y_true - y_pred)")
plt.title("Residual Plot")

plt.show()


# DISTRIBUTION COMPARISON

plt.figure(figsize = (12, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(merge_train['accident_risk'], label = 'True Label', fill = False)
sns.kdeplot(oof_predictions, label = 'Predicted Label (OOF)', fill = False)
plt.title('True vs Predicted Accident Risk Distribution')
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(y_test, label = 'Test Acccident Risk', fill = False)
plt.title('Test Accident Risk Distribution')
plt.legend()

plt.show()

