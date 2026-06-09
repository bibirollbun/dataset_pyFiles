import polars as pl, pandas as pd, numpy as np
import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
train=pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")


"""
Hierarchical macro F1 metric for the CMI 2025 Challenge.

This script defines a single entry point `score(solution, submission, row_id_column_name)`
that the Kaggle metrics orchestrator will call.
It performs validation on submission IDs and computes a combined binary & multiclass F1 score.
"""





class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in trjay and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )

        return 0.5 * f1_binary + 0.5 * f1_macro


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str
) -> float:
    """
    Compute hierarchical macro F1 for the CMI 2025 challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'sequence_id': unique identifier for each sequence
      - 'gesture': one of the eight target gestures or "Non-Target"

    This metric averages:
    1. Binary F1 on SequenceType (Target vs Non-Target)
    2. Macro F1 on gesture (mapping non-targets to "Non-Target")

    Raises ParticipantVisibleError for invalid submissions,
    including invalid SequenceType or gesture values.


    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'gesture': ['Eyebrow - pull hair']*4})
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Forehead - pull hairline']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.5
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Text on phone']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    >>> score(solution, solution, row_id_column_name=row_id_column_name)
    1.0
    """
    # Validate required columns
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")

    metric = CompetitionMetric()
    return metric.calculate_hierarchical_f1(solution, submission)


train=pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_demo=pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")



def create_advanced_imu_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates advanced statistical and phase-approximated features 
    from IMU data ONLY.
    """
    imu_cols = [col for col in df.columns if 'acc_' in col or 'rot_' in col]
    aggs = []
    
    # 1. Whole-sequence statistical features
    for col in imu_cols:
        aggs.extend([
            pl.mean(col).alias(f'{col}_mean'),
            pl.std(col).alias(f'{col}_std'),
            pl.max(col).alias(f'{col}_max'),
            pl.min(col).alias(f'{col}_min'),
            pl.quantile(col, 0.25).alias(f'{col}_q25'),
            pl.quantile(col, 0.75).alias(f'{col}_q75'),
        ])

    # 2. Difference features
    for col in imu_cols:
        aggs.extend([
            (pl.col(col).diff().fill_null(0)).mean().alias(f'{col}_diff_mean'),
            (pl.col(col).diff().fill_null(0)).std().alias(f'{col}_diff_std'),
        ])
        
    # 3. Phase-approximated features
    for part_name, part_expr in [
        ('first_30pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.3),
        ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
        ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
    ]:
        for col in imu_cols:
            aggs.extend([
                (pl.when(part_expr).then(pl.col(col))).mean().alias(f'{col}_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(col))).std().alias(f'{col}_std_{part_name}'),
            ])

    feature_df = df.group_by('sequence_id').agg(aggs).fill_null(0)
    return feature_df


train_features=create_advanced_imu_features(train)

targets=train.group_by('sequence_id').agg(
    pl.first('gesture'),
    pl.first('subject'),
    pl.first('sequence_type')
)

train_features_full = train_features.join(targets, on='sequence_id',how='left')
train_features_full=train_features_full.join(train_demo, on='subject',how='left')

train2 = train_features_full.to_pandas()



import matplotlib.pyplot as plt
import seaborn as sns

# plt.scatterplot(dat)
# sns.countplot(data=train2,x='gesture')
sns.histplot(data=train2, x='age',bins=7)
plt.show()


from tqdm import tqdm
# lgb hyperparameters
lgb_params = {"device":'gpu',                  # enables GPU
    "tree_method":'gpu_hist',       # optional, for XGBoost-style syntax
    'objective': 'multiclass',
    'class_weight': 'balanced',
    'n_estimators': 10_000,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
    'learning_rate': 0.0682511123410753,
    'num_leaves': 285,
    'max_depth': 11,
    'reg_alpha': 0.015067299011315672,
    'reg_lambda': 0.408041751307428,
    'colsample_bytree': 0.4028382627354783,
    'subsample': 0.6873886039232031
}

# Make X, y and group sets
X = train2.drop(columns=['sequence_id', 'gesture', 'subject', 'sequence_type'])
y = train2['gesture']
groups = train2['subject']

# Model training setting
k=5
gkf = GroupKFold(n_splits=k)


oof_predictions = np.zeros(X.shape[0]).astype(int)
models=[]
scores=[]

folds = list(gkf.split(X, y, groups=groups))  # Ensure it's a fixed iterable

# Start training (CV)
t = tqdm(range(len(folds)), desc="Training folds")
for i in t:

    train_idx, valid_idx = folds[i]
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]

    le=LabelEncoder()
    y_train_enc=le.fit_transform(y_train)
    y_valid_enc=le.transform(y_valid)


    model=lgb.LGBMClassifier(**lgb_params)
    # Training
    model.fit(X_train, y_train_enc,eval_set=[(X_valid, y_valid_enc)], callbacks=[lgb.early_stopping(100, verbose=False)])
    # Prediction
    y_pred=model.predict(X_valid)

    # OOF prediction 
    oof_predictions[valid_idx]=y_pred
    
    # Validation scores 
    y_true=pd.DataFrame({'id':range(len(y.loc[valid_idx])),'gesture':y.loc[valid_idx]})
    y_pred=pd.DataFrame({'id':range(len(y_pred)),'gesture':le.inverse_transform(oof_predictions[valid_idx])})

    performance_score = score(y_true, y_pred,row_id_column_name='id')

    # Append models and scores
    models.append(model)
    scores.append(performance_score)
    
    t.set_postfix({'Fold': i+1, 'Score': round(performance_score, 4)})

    
# create dataframes of true and predicted labels
y_true_total=pd.DataFrame({'id':range(len(y)),'gesture':y})
y_pred_total=pd.DataFrame({'id':range(len(y)),'gesture':le.inverse_transform(oof_predictions)})

    # Calculate score
performance_score_overall = score(y_true, y_pred,row_id_column_name='id')
print(f"OOF Score: {performance_score_overall} ")   
 


# for now choose the last trained model and label encoder in CV
import kaggle_evaluation.cmi_inference_server
feature_list=X.columns.tolist()
le=LabelEncoder()
le.fit(y)

# --- 3. Define Predict Function ---
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Receives a single test sequence and predicts the gesture using ensemble of models.
    """
    feature_df = create_advanced_imu_features(sequence)
    
    
    subjects=sequence.group_by('sequence_id').agg(pl.first('subject'))
    feature_df=feature_df.join(subjects, on='sequence_id',how='left')
    feature_df_with_dem=feature_df.join(demographics, on='subject',how='left')
   
    X_test = feature_df_with_dem[feature_list]
    
    # Predict class probabilities with all models
    probs = [model.predict_proba(X_test) for model in models]

    # Average predictions
    avg_probs = sum(probs) / len(probs)

    # Get class index with highest average probability
    pred_idx = avg_probs.argmax(axis=1)[0]

    # Decode back to gesture string
    pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label


# --- 4. Start the Inference Server ---
print("Starting inference server...")
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # This branch runs when you submit the notebook
    inference_server.serve()
else:
    # This branch runs for local testing in the notebook editor
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

