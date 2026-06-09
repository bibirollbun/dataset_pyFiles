import pandas as pd
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


BATCH_SIZE = 128
N_FOLDS = 3


df_train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df_test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


bge_model = SentenceTransformer(
    model_name_or_path="/kaggle/input/baai/transformers/bge-small-en-v1.5/1",
    device="cuda",
)
gemma_model = SentenceTransformer(
    model_name_or_path="/kaggle/input/gemma-3/transformers/gemma-3-270m-it/1",
    device="cuda",
)


def calc_emb(df, embedding_model):
    body_emb = embedding_model.encode(
        sentences=df["body"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    pos1_emb = embedding_model.encode(
        sentences=df["positive_example_1"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    pos2_emb = embedding_model.encode(
        sentences=df["positive_example_2"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    neg1_emb = embedding_model.encode(
        sentences=df["negative_example_1"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    neg2_emb = embedding_model.encode(
        sentences=df["negative_example_2"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        device="cuda",
        normalize_embeddings=True,
    )
    return body_emb, pos1_emb, pos2_emb, neg1_emb, neg2_emb

def calc_score(body_emb, pos1_emb, pos2_emb, neg1_emb, neg2_emb):
    body_pos1_sim = (body_emb * pos1_emb).sum(dim=1)
    body_pos2_sim = (body_emb * pos2_emb).sum(dim=1)
    body_neg1_sim = (body_emb * neg1_emb).sum(dim=1)
    body_neg2_sim = (body_emb * neg2_emb).sum(dim=1)
    sim_score = body_pos1_sim + body_pos2_sim - (body_neg1_sim + body_neg2_sim)
    return sim_score

def calc_emb_score(df, embedding_model):
    body_emb, pos1_emb, pos2_emb, neg1_emb, neg2_emb = calc_emb(df, embedding_model)
    sim_score = calc_score(body_emb, pos1_emb, pos2_emb, neg1_emb, neg2_emb)
    return sim_score


def train_lgbm_cv(
    df_X_train,
    sr_y_train,
    df_X_test,
    # sr_y_test,
    params,
    cv,
    feat_cols,
    enable_log = True,
):

    metrics = []
    df_importance_list = []
    model_list = []
    df_pred_test_list = []

    for i_fold, (train_idx, val_idx) in enumerate(cv.split(df_X_train, sr_y_train)):
        print(f'Fold {i_fold}')
        df_X_train_train = df_X_train.iloc[train_idx]
        df_X_train_val = df_X_train.iloc[val_idx]
        sr_y_train_train = sr_y_train.iloc[train_idx]
        sr_y_train_val = sr_y_train.iloc[val_idx]
        evals = {}

        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.record_evaluation(evals)
        ]
        if enable_log:
            callbacks.append(lgb.log_evaluation(period=100))

        lgb_train = lgb.Dataset(df_X_train_train[feat_cols], sr_y_train_train, feature_name=feat_cols)
        lgb_eval = lgb.Dataset(df_X_train_val[feat_cols], sr_y_train_val, feature_name=feat_cols, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            callbacks=callbacks
        )

        preds_train = model.predict(df_X_train_train[feat_cols].values)
        preds_val = model.predict(df_X_train_val[feat_cols].values)
        preds_test = model.predict(df_X_test[feat_cols].values)

        df_pred_test = pd.DataFrame({'pred_prob': preds_test}, index=df_X_test.index)
        df_pred_test.index.name = df_X_test.index.name
        df_pred_test['fold_index'] = i_fold
        df_pred_test_list.append(df_pred_test)

        auc_score_train = roc_auc_score(sr_y_train_train, preds_train)
        auc_score_val = roc_auc_score(sr_y_train_val, preds_val)

        metrics.append({
            'auc_score_train': auc_score_train,
            'auc_score_val': auc_score_val,
            'fold_id': i_fold
        })

        df_importance = pd.Series(
            data=model.feature_importance(),
            index=feat_cols
        ).sort_values(ascending=False)
        df_importance_list.append(df_importance)

        model_list.append(model)

    return model_list, pd.DataFrame(metrics), df_importance_list, pd.concat(df_pred_test_list)



body_emb_train_bge, pos1_emb_train_bge, pos2_emb_train_bge, neg1_emb_train_bge, neg2_emb_train_bge = calc_emb(df_train, bge_model)
body_emb_test_bge, pos1_emb_test_bge, pos2_emb_test_bge, neg1_emb_test_bge, neg2_emb_test_bge = calc_emb(df_test, bge_model)

body_emb_train_gemma, pos1_emb_train_gemma, pos2_emb_train_gemma, neg1_emb_train_gemma, neg2_emb_train_gemma = calc_emb(df_train, gemma_model)
body_emb_test_gemma, pos1_emb_test_gemma, pos2_emb_test_gemma, neg1_emb_test_gemma, neg2_emb_test_gemma = calc_emb(df_test, gemma_model)

rule_le = LabelEncoder()
rule_le.fit(df_train['rule'].tolist() + df_test['rule'].tolist())
df_train['rule_label'] = rule_le.transform(df_train['rule'])
df_test['rule_label'] = rule_le.transform(df_test['rule'])

BGE_DIM = body_emb_train_bge.shape[-1]
GEMMA_DIM = body_emb_train_gemma.shape[-1]
BGE_EMB_COLS = [f'bge_emb_{i:04d}' for i in range(BGE_DIM)]
GEMMA_EMB_COLS = [f'gemma_emb_{i:04d}' for i in range(GEMMA_DIM)]
FEAT_COLS = BGE_EMB_COLS + GEMMA_EMB_COLS

df_train = df_train.set_index('row_id')
df_test = df_test.set_index('row_id')

df_body_emb_train_bge = pd.DataFrame(body_emb_train_bge.cpu().numpy(), index=df_train.index, columns=BGE_EMB_COLS)
df_body_emb_train_bge.index.name = 'row_id'
df_pos1_emb_train_bge = pd.DataFrame(pos1_emb_train_bge.cpu().numpy(), index=df_train.index, columns=BGE_EMB_COLS)
df_pos1_emb_train_bge.index.name = 'row_id'
df_pos2_emb_train_bge = pd.DataFrame(pos2_emb_train_bge.cpu().numpy(), index=df_train.index, columns=BGE_EMB_COLS)
df_pos2_emb_train_bge.index.name = 'row_id'
df_neg1_emb_train_bge = pd.DataFrame(neg1_emb_train_bge.cpu().numpy(), index=df_train.index, columns=BGE_EMB_COLS)
df_neg1_emb_train_bge.index.name = 'row_id'
df_neg2_emb_train_bge = pd.DataFrame(neg2_emb_train_bge.cpu().numpy(), index=df_train.index, columns=BGE_EMB_COLS)
df_neg2_emb_train_bge.index.name = 'row_id'

df_body_emb_test_bge = pd.DataFrame(body_emb_test_bge.cpu().numpy(), index=df_test.index, columns=BGE_EMB_COLS)
df_body_emb_test_bge.index.name = 'row_id'
df_pos1_emb_test_bge = pd.DataFrame(pos1_emb_test_bge.cpu().numpy(), index=df_test.index, columns=BGE_EMB_COLS)
df_pos1_emb_test_bge.index.name = 'row_id'
df_pos2_emb_test_bge = pd.DataFrame(pos2_emb_test_bge.cpu().numpy(), index=df_test.index, columns=BGE_EMB_COLS)
df_pos2_emb_test_bge.index.name = 'row_id'
df_neg1_emb_test_bge = pd.DataFrame(neg1_emb_test_bge.cpu().numpy(), index=df_test.index, columns=BGE_EMB_COLS)
df_neg1_emb_test_bge.index.name = 'row_id'
df_neg2_emb_test_bge = pd.DataFrame(neg2_emb_test_bge.cpu().numpy(), index=df_test.index, columns=BGE_EMB_COLS)
df_neg2_emb_test_bge.index.name = 'row_id'

df_body_emb_train_gemma = pd.DataFrame(body_emb_train_gemma.cpu().numpy(), index=df_train.index, columns=GEMMA_EMB_COLS)
df_body_emb_train_gemma.index.name = 'row_id'
df_body_emb_train_gemma['rule_violation'] = df_train['rule_violation']
df_body_emb_train_gemma['rule_label'] = df_train['rule_label']
df_pos1_emb_train_gemma = pd.DataFrame(pos1_emb_train_gemma.cpu().numpy(), index=df_train.index, columns=GEMMA_EMB_COLS)
df_pos1_emb_train_gemma.index.name = 'row_id'
df_pos1_emb_train_gemma['rule_violation'] = 1
df_pos1_emb_train_gemma['rule_label'] = df_train['rule_label']
df_pos2_emb_train_gemma = pd.DataFrame(pos2_emb_train_gemma.cpu().numpy(), index=df_train.index, columns=GEMMA_EMB_COLS)
df_pos2_emb_train_gemma.index.name = 'row_id'
df_pos2_emb_train_gemma['rule_violation'] = 1
df_pos2_emb_train_gemma['rule_label'] = df_train['rule_label']
df_neg1_emb_train_gemma = pd.DataFrame(neg1_emb_train_gemma.cpu().numpy(), index=df_train.index, columns=GEMMA_EMB_COLS)
df_neg1_emb_train_gemma.index.name = 'row_id'
df_neg1_emb_train_gemma['rule_violation'] = 0
df_neg1_emb_train_gemma['rule_label'] = df_train['rule_label']
df_neg2_emb_train_gemma = pd.DataFrame(neg2_emb_train_gemma.cpu().numpy(), index=df_train.index, columns=GEMMA_EMB_COLS)
df_neg2_emb_train_gemma.index.name = 'row_id'
df_neg2_emb_train_gemma['rule_violation'] = 0
df_neg2_emb_train_gemma['rule_label'] = df_train['rule_label']

df_body_emb_test_gemma = pd.DataFrame(body_emb_test_gemma.cpu().numpy(), index=df_test.index, columns=GEMMA_EMB_COLS)
df_body_emb_test_gemma.index.name = 'row_id'
df_body_emb_test_gemma['rule_violation'] = None
df_body_emb_test_gemma['rule_label'] = df_test['rule_label']
df_pos1_emb_test_gemma = pd.DataFrame(pos1_emb_test_gemma.cpu().numpy(), index=df_test.index, columns=GEMMA_EMB_COLS)
df_pos1_emb_test_gemma.index.name = 'row_id'
df_pos1_emb_test_gemma['rule_violation'] = 1
df_pos1_emb_test_gemma['rule_label'] = df_test['rule_label']
df_pos2_emb_test_gemma = pd.DataFrame(pos2_emb_test_gemma.cpu().numpy(), index=df_test.index, columns=GEMMA_EMB_COLS)
df_pos2_emb_test_gemma.index.name = 'row_id'
df_pos2_emb_test_gemma['rule_violation'] = 1
df_pos2_emb_test_gemma['rule_label'] = df_test['rule_label']
df_neg1_emb_test_gemma = pd.DataFrame(neg1_emb_test_gemma.cpu().numpy(), index=df_test.index, columns=GEMMA_EMB_COLS)
df_neg1_emb_test_gemma.index.name = 'row_id'
df_neg1_emb_test_gemma['rule_violation'] = 0
df_neg1_emb_test_gemma['rule_label'] = df_test['rule_label']
df_neg2_emb_test_gemma = pd.DataFrame(neg2_emb_test_gemma.cpu().numpy(), index=df_test.index, columns=GEMMA_EMB_COLS)
df_neg2_emb_test_gemma.index.name = 'row_id'
df_neg2_emb_test_gemma['rule_violation'] = 0
df_neg2_emb_test_gemma['rule_label'] = df_test['rule_label']

df_train_bge_lgbm = pd.concat([
    df_body_emb_train_bge,
    df_pos1_emb_train_bge, df_pos2_emb_train_bge,
    df_neg1_emb_train_bge, df_neg2_emb_train_bge,
    df_pos1_emb_test_bge, df_pos2_emb_test_bge,
    df_neg1_emb_test_bge, df_neg2_emb_test_bge,
])
df_train_gemma_lgbm = pd.concat([
    df_body_emb_train_gemma,
    df_pos1_emb_train_gemma, df_pos2_emb_train_gemma,
    df_neg1_emb_train_gemma, df_neg2_emb_train_gemma,
    df_pos1_emb_test_gemma, df_pos2_emb_test_gemma,
    df_neg1_emb_test_gemma, df_neg2_emb_test_gemma,
])
df_train_lgbm = pd.concat([df_train_bge_lgbm, df_train_gemma_lgbm], axis=1)
df_test_lgbm = pd.concat([df_body_emb_test_bge, df_body_emb_test_gemma], axis=1)
assert not df_train_lgbm.isnull().any().any()


lgbm_params = {
    ### fixed parmeter ###
    "objective": "binary",
    "metric": "binary_logloss",
    "num_rounds": 3000,
    "device": "cpu",
    "verbose": -1,
    "seed": 42,
    ### tuning parameter ###
    "learning_rate": 0.02,
    "max_depth": 6,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "feature_fraction_bynode": 0.8,
    # "bagging_fraction": 0.8,
    # "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.3,
    "extra_trees": True,
    "max_bin": 128,
}

df_pred_test_rule_list = []

for rule_label, df_train_lgbm_rule in df_train_lgbm.groupby('rule_label'):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    df_test_lgbm_rule = df_test_lgbm[df_test_lgbm['rule_label'] == rule_label]
    model, df_metrics, df_importance_list, df_pred_test_rule = train_lgbm_cv(
        df_X_train=df_train_lgbm_rule[FEAT_COLS],
        sr_y_train=df_train_lgbm_rule['rule_violation'],
        df_X_test=df_test_lgbm_rule[FEAT_COLS],
        params=lgbm_params,
        cv=cv,
        feat_cols=FEAT_COLS,
    )
    print(rule_label, df_metrics)
    df_pred_test_rule = df_pred_test_rule.groupby('row_id')[['pred_prob']].mean()
    df_pred_test_rule_list.append(df_pred_test_rule)

df_pred_test = pd.concat(df_pred_test_rule_list).rename(columns={'pred_prob': 'rule_violation'})


df_pred_test


df_pred_test.to_csv('submission.csv')




