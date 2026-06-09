import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from itertools import combinations

SKIP_VALIDATION = False

def numeric_id_from_folder(aid: str) -> str:
    if isinstance(aid, str) and aid.startswith("article_"):
        return aid.replace("article_", "")
    return aid

def load_test_predictions():
    deberta_dir = Path("/kaggle/input/5-head-deberta-ensemble-ii-0-917-lb/hf_run_deberta_ensemble_test_oof/")
    ettin_dir = Path("/kaggle/input/5-head-ettin-ensemble-lb-0-927/hf_run_ettin_ensemble_400m/")
    ettin2_dir = Path("/kaggle/input/6-head-ettin-encoder-ensemble/hf_run_ettin_ensemble_400m/")
    roberta_dir = Path("/kaggle/input/5-head-roberta-ensemble/hf_run_roberta_ensemble_test_oof/")
    deberta_test = pd.read_csv(deberta_dir / "test_oof_aggregated.csv")
    ettin_test = pd.read_csv(ettin_dir / "test_oof_aggregated.csv")
    ettin2_test = pd.read_csv(ettin2_dir / "test_oof_aggregated.csv")
    roberta_test = pd.read_csv(roberta_dir / "test_oof_aggregated.csv")
    return deberta_test, ettin_test, ettin2_test, roberta_test

def load_validation_predictions():
    deberta_dir = Path("/kaggle/input/5-head-deberta-ensemble-ii-0-917-lb/hf_run_deberta_ensemble_test_oof/")
    ettin_dir = Path("/kaggle/input/5-head-ettin-ensemble-lb-0-927/hf_run_ettin_ensemble_400m/")
    ettin2_dir = Path("/kaggle/input/6-head-ettin-encoder-ensemble/hf_run_ettin_ensemble_400m/")
    roberta_dir = Path("/kaggle/input/5-head-roberta-ensemble/hf_run_roberta_ensemble_test_oof/")
    deberta_val = pd.read_csv(deberta_dir / "oof_ensemble_average.csv")
    ettin_val = pd.read_csv(ettin_dir / "oof_ensemble_average.csv")
    ettin2_val = pd.read_csv(ettin2_dir / "oof_ensemble_average.csv")
    roberta_val = pd.read_csv(roberta_dir / "oof_ensemble_average.csv")
    return deberta_val, ettin_val, ettin2_val, roberta_val

def average_within_model(df, model_name):
    averaged = df.groupby(['id', 'file_idx'])['prob_real'].mean().reset_index()
    return averaged

def geometric_power_average(probs_list, power=2.0):
    clipped_probs = [np.clip(p, 1e-8, 1 - 1e-8) for p in probs_list]
    log_sum = sum(power * np.log(p) for p in clipped_probs)
    result = np.exp(log_sum / (len(probs_list) * power))
    return result

def calibrate_predictions(val_probs, val_labels, test_probs, method='isotonic'):
    val_probs = np.asarray(val_probs).reshape(-1, 1)
    test_probs = np.asarray(test_probs).reshape(-1, 1)
    val_labels = np.asarray(val_labels).ravel()
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_probs.ravel(), val_labels)
        calibrated = calibrator.predict(test_probs.ravel())
    elif method == 'platt':
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(val_probs, val_labels)
        calibrated = calibrator.predict_proba(test_probs)[:, 1]
    else:
        raise ValueError("Unsupported calibration method")
    return np.clip(calibrated, 1e-12, 1 - 1e-12)

def create_all_ensembles(deberta_df, ettin_df, ettin2_df, roberta_df):
    merge_cols = ['id', 'file_idx']
    combined = deberta_df.merge(ettin_df, on=merge_cols, suffixes=('_deberta', '_ettin'), how='inner')
    combined = combined.merge(ettin2_df, on=merge_cols, how='inner')
    combined.rename(columns={'prob_real': 'prob_real_ettin2'}, inplace=True)
    combined = combined.merge(roberta_df, on=merge_cols, how='inner')
    combined.rename(columns={'prob_real': 'prob_real_roberta'}, inplace=True)
    prob_cols = ['prob_real_deberta', 'prob_real_ettin', 'prob_real_ettin2', 'prob_real_roberta']
    combined['prob_real_simple_avg'] = combined[prob_cols].mean(axis=1)
    for col in prob_cols:
        combined[f'rank_{col}'] = combined[col].rank(method='average')
    rank_cols = [f'rank_{c}' for c in prob_cols]
    combined['avg_rank'] = combined[rank_cols].mean(axis=1)
    min_rank = combined['avg_rank'].min()
    max_rank = combined['avg_rank'].max()
    if max_rank - min_rank > 0:
        combined['prob_real_rank_avg'] = (combined['avg_rank'] - min_rank) / (max_rank - min_rank)
    else:
        combined['prob_real_rank_avg'] = combined['avg_rank']
    powers = [1.5, 2.0, 3.0]
    for power in powers:
        combined[f'prob_real_geo_power_{power}'] = combined[prob_cols].apply(
            lambda row: geometric_power_average(row.values, power), axis=1
        )
    return combined

def calibrate_ensemble_predictions(combined_df):
    if SKIP_VALIDATION:
        print("SKIP_VALIDATION is True -> skipping calibration")
        return combined_df
    try:
        deberta_val, ettin_val, ettin2_val, roberta_val = load_validation_predictions()
    except Exception as ex:
        print("Warning: validation files not found or unreadable:", ex)
        return combined_df
    d_avg = average_within_model(deberta_val, "DeBERTa_val")
    e_avg = average_within_model(ettin_val, "Ettin1_val")
    e2_avg = average_within_model(ettin2_val, "Ettin2_val")
    r_avg = average_within_model(roberta_val, "RoBERTa_val")
    val_combined = d_avg.merge(e_avg, on=['id','file_idx'], suffixes=('_deberta','_ettin'))
    val_combined = val_combined.merge(e2_avg, on=['id','file_idx'])
    val_combined.rename(columns={'prob_real': 'prob_real_ettin2'}, inplace=True)
    val_combined = val_combined.merge(r_avg, on=['id','file_idx'])
    val_combined.rename(columns={'prob_real': 'prob_real_roberta'}, inplace=True)
    prob_cols_val = ['prob_real_deberta', 'prob_real_ettin', 'prob_real_ettin2', 'prob_real_roberta']
    val_combined['prob_real_simple_avg'] = val_combined[prob_cols_val].mean(axis=1)
    if 'label' in deberta_val.columns:
        val_labels_df = deberta_val[['id','file_idx','label']].drop_duplicates()
        val_combined = val_combined.merge(val_labels_df, on=['id','file_idx'])
        combined_df['prob_real_simple_avg_calibrated'] = calibrate_predictions(
            val_combined['prob_real_simple_avg'].values,
            val_combined['label'].values,
            combined_df['prob_real_simple_avg'].values,
            method='isotonic'
        )
        if 'prob_real_rank_avg' in combined_df.columns:
            for col in prob_cols_val:
                val_combined[f'rank_{col}'] = val_combined[col].rank(method='average')
            val_combined['avg_rank'] = val_combined[[f'rank_{c}' for c in prob_cols_val]].mean(axis=1)
            min_rank = val_combined['avg_rank'].min()
            max_rank = val_combined['avg_rank'].max()
            if max_rank - min_rank > 0:
                val_combined['prob_real_rank_avg'] = (val_combined['avg_rank'] - min_rank) / (max_rank - min_rank)
            else:
                val_combined['prob_real_rank_avg'] = val_combined['avg_rank']
            combined_df['prob_real_rank_avg_calibrated'] = calibrate_predictions(
                val_combined['prob_real_rank_avg'].values,
                val_combined['label'].values,
                combined_df['prob_real_rank_avg'].values,
                method='isotonic'
            )
    else:
        print("Validation loaded but no 'label' column found; skipping calibration steps that require labels")
    return combined_df

def create_submission(ensemble_df, prob_column, output_path):
    submission_rows = []
    for article_id, group in ensemble_df.groupby('id'):
        best_idx = group[prob_column].idxmax()
        best_file = group.loc[best_idx]
        submission_rows.append({
            'id': int(numeric_id_from_folder(article_id)),
            'real_text_id': int(best_file['file_idx'])
        })
    submission_df = pd.DataFrame(submission_rows)
    submission_df = submission_df.sort_values('id')
    submission_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
    return submission_df

def compute_weights_from_scores(lb_scores: dict, method: str = "proportional"):
    names = list(lb_scores.keys())
    vals = np.array([lb_scores[n] for n in names], dtype=float)
    if method == "proportional":
        w = vals / vals.sum()
    elif method == "softmax":
        ex = np.exp(vals - vals.max())
        w = ex / ex.sum()
    elif method == "inverse_error":
        eps = 1e-12
        errs = 1.0 - vals
        inv = 1.0 / (errs + eps)
        w = inv / inv.sum()
    else:
        raise ValueError("method must be 'proportional','softmax' or 'inverse_error'")
    return dict(zip(names, w))

def apply_weighted_ensemble(df, model_prob_cols, weights):
    w = np.asarray(weights, dtype=float)
    probs = df[model_prob_cols].values
    result = probs.dot(w)
    return np.clip(result, 1e-12, 1 - 1e-12)

def learn_weights_via_val(val_probs_df, val_labels, model_prob_cols, bounds=(0.0, 1.0)):
    X = val_probs_df[model_prob_cols].values
    y = np.asarray(val_labels).ravel()
    n_models = X.shape[1]
    def obj(w):
        preds = np.clip(X.dot(w), 1e-12, 1 - 1e-12)
        return log_loss(y, preds)
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bnds = tuple([bounds] * n_models)
    x0 = np.ones(n_models) / n_models
    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        print("Warning: SLSQP did not converge:", res.message)
    w_opt = np.clip(res.x, 0.0, None)
    if w_opt.sum() <= 0:
        w_opt = np.ones(n_models) / n_models
    else:
        w_opt = w_opt / w_opt.sum()
    return w_opt

def analyze_ensemble(ensemble_df):
    prob_cols = [c for c in ensemble_df.columns if c.startswith('prob_real_')]
    for col in prob_cols:
        ensemble_df[f'pred_{col}'] = (ensemble_df[col] > 0.5).astype(int)
    model_cols = [c for c in prob_cols]
    pairwise = {}
    for a, b in combinations(model_cols, 2):
        agreement = (ensemble_df[f'pred_{a}'] == ensemble_df[f'pred_{b}']).mean()
        pairwise[f"{a} vs {b}"] = agreement
    for k, v in pairwise.items():
        print(f"{k}: {v:.4f}")
    return ensemble_df

def main():
    deberta_test, ettin_test, ettin2_test, roberta_test = load_test_predictions()
    deberta_avg = average_within_model(deberta_test, "DeBERTa")
    ettin_avg = average_within_model(ettin_test, "Ettin1")
    ettin2_avg = average_within_model(ettin2_test, "Ettin2")
    roberta_avg = average_within_model(roberta_test, "RoBERTa")
    ensemble_df = create_all_ensembles(deberta_avg, ettin_avg, ettin2_avg, roberta_avg)
    ensemble_df = calibrate_ensemble_predictions(ensemble_df)
    ensemble_df = analyze_ensemble(ensemble_df)
    ensemble_df.to_csv("ensemble_all_methods_detailed.csv", index=False)
    lb_scores = {
        'ettin2': 0.93568,
        'roberta': 0.92116,
        'deberta': 0.91701,
        'ettin1': 0.92738
    }
    model_prob_cols = ['prob_real_deberta','prob_real_ettin','prob_real_ettin2','prob_real_roberta']
    name_to_col = {
        'deberta': 'prob_real_deberta',
        'ettin1': 'prob_real_ettin',
        'ettin2': 'prob_real_ettin2',
        'roberta': 'prob_real_roberta'
    }
    lb_in_order = {k: lb_scores[k] for k in ['deberta','ettin1','ettin2','roberta']}
    weights_prop_named = compute_weights_from_scores(lb_in_order, method='proportional')
    weights_soft_named = compute_weights_from_scores(lb_in_order, method='softmax')
    weights_inv_named = compute_weights_from_scores(lb_in_order, method='inverse_error')
    def named_to_array(named_dict, model_cols, name_map):
        short_to_col = {k: name_map[k] for k in name_map}
        col_to_short = {v:k for k,v in short_to_col.items()}
        return np.array([named_dict[col_to_short[c]] for c in model_cols], dtype=float)
    weights_prop = named_to_array(weights_prop_named, model_prob_cols, name_to_col)
    weights_soft = named_to_array(weights_soft_named, model_prob_cols, name_to_col)
    weights_inv = named_to_array(weights_inv_named, model_prob_cols, name_to_col)
    ensemble_df['prob_weighted_proportional'] = apply_weighted_ensemble(ensemble_df, model_prob_cols, weights_prop)
    create_submission(ensemble_df, 'prob_weighted_proportional', 'submission_weighted_proportional.csv')
    ensemble_df['prob_weighted_softmax'] = apply_weighted_ensemble(ensemble_df, model_prob_cols, weights_soft)
    create_submission(ensemble_df, 'prob_weighted_softmax', 'submission_weighted_softmax.csv')
    ensemble_df['prob_weighted_inverse_error'] = apply_weighted_ensemble(ensemble_df, model_prob_cols, weights_inv)
    create_submission(ensemble_df, 'prob_weighted_inverse_error', 'submission_weighted_inverse_error.csv')
    if not SKIP_VALIDATION:
        try:
            deberta_val, ettin_val, ettin2_val, roberta_val = load_validation_predictions()
            if 'label' in deberta_val.columns:
                d_avg = average_within_model(deberta_val, "DeBERTa_val")
                e_avg = average_within_model(ettin_val, "Ettin1_val")
                e2_avg = average_within_model(ettin2_val, "Ettin2_val")
                r_avg = average_within_model(roberta_val, "RoBERTa_val")
                val_comb = d_avg.merge(e_avg, on=['id','file_idx'], suffixes=('_deberta','_ettin'))
                val_comb = val_comb.merge(e2_avg, on=['id','file_idx'])
                val_comb.rename(columns={'prob_real': 'prob_real_ettin2'}, inplace=True)
                val_comb = val_comb.merge(r_avg, on=['id','file_idx'])
                val_comb.rename(columns={'prob_real': 'prob_real_roberta'}, inplace=True)
                val_prob_cols = ['prob_real_deberta','prob_real_ettin','prob_real_ettin2','prob_real_roberta']
                val_labels = val_comb['label'].values
                w_opt = learn_weights_via_val(val_comb, val_labels, val_prob_cols)
                ensemble_df['prob_weighted_opt_val'] = apply_weighted_ensemble(ensemble_df, model_prob_cols, w_opt)
                create_submission(ensemble_df, 'prob_weighted_opt_val', 'submission_weighted_opt_val.csv')
            else:
                print("Validation loaded but 'label' column not found; skipping optimized weight learning.")
        except Exception as ex:
            print("Skipping optimized weight learning due to error:", ex)
    common_exports = [
        ('prob_real_simple_avg','submission_simple_average.csv'),
        ('prob_real_rank_avg','submission_rank_average.csv'),
        ('prob_real_geo_power_2.0','submission_geometric_power_2.csv'),
        ('prob_real_geo_power_3.0','submission_geometric_power_3.csv')
    ]
    for col, fname in common_exports:
        if col in ensemble_df.columns:
            create_submission(ensemble_df, col, fname)
    print("Done")

if __name__ == "__main__":
    main()











