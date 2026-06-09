import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier

import torch

import gc


import sentence_transformers
print(sentence_transformers.__version__)





torch.cuda.is_available()


for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {device_name}")


TRAIN_DATASET_PATH = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
TEST_DATASET_PATH = "/kaggle/input/jigsaw-agile-community-rules/test.csv"


train_df = pd.read_csv(TRAIN_DATASET_PATH)
test_df = pd.read_csv(TEST_DATASET_PATH)


train_df.shape, test_df.shape


# Using only query prefix as recommended by e5 team for embeddings to be used for classification tasks
PREFIX_Q = "query: "
# PREFIX_D = "passage: "



bodies_train = PREFIX_Q + train_df["body"].to_numpy()
rules_train = PREFIX_Q + 'rule: ' + train_df["rule"].to_numpy()
subrs_train = PREFIX_Q + 'subreddit: r/' + train_df["subreddit"].to_numpy()
pos1s_train = PREFIX_Q + train_df["positive_example_1"].to_numpy()
pos2s_train = PREFIX_Q + train_df["positive_example_2"].to_numpy()
neg1s_train = PREFIX_Q + train_df["negative_example_1"].to_numpy()
neg2s_train = PREFIX_Q + train_df["negative_example_2"].to_numpy()

y_rule_violation_train = train_df["rule_violation"].to_numpy()


subrs_and_rule_train = subrs_train + ". " + rules_train


subrs_and_rule_train[0]


bodies_test = PREFIX_Q + test_df["body"].to_numpy()
rules_test = PREFIX_Q + 'rule: ' + test_df["rule"].to_numpy()
subrs_test = PREFIX_Q + 'subreddit: r/' + test_df["subreddit"].to_numpy()
pos1s_test = PREFIX_Q + test_df["positive_example_1"].to_numpy()
pos2s_test = PREFIX_Q + test_df["positive_example_2"].to_numpy()
neg1s_test = PREFIX_Q + test_df["negative_example_1"].to_numpy()
neg2s_test = PREFIX_Q + test_df["negative_example_2"].to_numpy()




subrs_and_rule_test = subrs_test + ". " + rules_test


subrs_and_rule_test[0]


MODEL_URIS = [
    "/kaggle/input/lailax-model-202510021249-e5-base-mrr-at-10-x/other/latest/2",
    "/kaggle/input/lailax-model-202510021318-e5-base-ndcg-at-10-x/other/latest/2",
    "/kaggle/input/lailax-model-202510021333-e5-base-map-x/other/latest/2",
]


accumulated_probs = None
n_classifiers = len(MODEL_URIS)
EMB_DIM = 768


for idx, model_uri in enumerate(MODEL_URIS):
    model = SentenceTransformer(model_uri, device="cuda:0")
    print("model uri=", model_uri)
    print(model)

    bodies_train_emb = model.encode(bodies_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    rules_train_emb = model.encode(rules_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    subrs_train_emb = model.encode(subrs_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    subrs_and_rule_train_emb = model.encode(subrs_and_rule_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    pos1s_train_emb = model.encode(pos1s_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    pos2s_train_emb = model.encode(pos2s_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    neg1s_train_emb = model.encode(neg1s_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    neg2s_train_emb = model.encode(neg2s_train, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]

    bodies_test_emb = model.encode(bodies_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    rules_test_emb = model.encode(rules_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    subrs_test_emb = model.encode(subrs_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    subrs_and_rule_test_emb = model.encode(subrs_and_rule_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    pos1s_test_emb = model.encode(pos1s_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    pos2s_test_emb = model.encode(pos2s_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    neg1s_test_emb = model.encode(neg1s_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]
    neg2s_test_emb = model.encode(neg2s_test, normalize_embeddings=True, show_progress_bar=True)[:, :EMB_DIM]

    X = []
    Y = []
    for i in range(len(bodies_train_emb)):
        X.append(np.concatenate((subrs_train_emb[i], rules_train_emb[i], bodies_train_emb[i], np.dot(subrs_and_rule_train_emb[i], bodies_train_emb[i])), axis=None))
        Y.append(y_rule_violation_train[i])
        # ---
        X.append(np.concatenate((subrs_train_emb[i], rules_train_emb[i], pos1s_train_emb[i], np.dot(subrs_and_rule_train_emb[i], pos1s_train_emb[i])), axis=None))
        Y.append(1)
    
        X.append(np.concatenate((subrs_train_emb[i], rules_train_emb[i], pos2s_train_emb[i], np.dot(subrs_and_rule_train_emb[i], pos2s_train_emb[i])), axis=None))
        Y.append(1)
        # ---
        X.append(np.concatenate((subrs_train_emb[i], rules_train_emb[i], neg1s_train_emb[i], np.dot(subrs_and_rule_train_emb[i], neg1s_train_emb[i])), axis=None))
        Y.append(0)
    
        X.append(np.concatenate((subrs_train_emb[i], rules_train_emb[i], neg2s_train_emb[i], np.dot(subrs_and_rule_train_emb[i], neg2s_train_emb[i])), axis=None))
        Y.append(0)

    X_test = []
    for i in range(len(bodies_test_emb)):
        # the examples for which we need to predicts
        X_test.append(np.concatenate((subrs_test_emb[i], rules_test_emb[i], bodies_test_emb[i], np.dot(subrs_and_rule_test_emb[i], bodies_test_emb[i])), axis=None))
        
        # collecting additional training examples:
        X.append(np.concatenate((subrs_test_emb[i], rules_test_emb[i], pos1s_test_emb[i], np.dot(subrs_and_rule_test_emb[i], pos1s_test_emb[i])), axis=None))
        Y.append(1)
    
        X.append(np.concatenate((subrs_test_emb[i], rules_test_emb[i], pos2s_test_emb[i], np.dot(subrs_and_rule_test_emb[i], pos2s_test_emb[i])), axis=None))
        Y.append(1)
        # ---
        X.append(np.concatenate((subrs_test_emb[i], rules_test_emb[i], neg1s_test_emb[i], np.dot(subrs_and_rule_test_emb[i], neg1s_test_emb[i])), axis=None))
        Y.append(0)
    
        X.append(np.concatenate((subrs_test_emb[i], rules_test_emb[i], neg2s_test_emb[i], np.dot(subrs_and_rule_test_emb[i], neg2s_test_emb[i])), axis=None))
        Y.append(0)

    print("GPU mem before releasing model")
    print(torch.cuda.list_gpu_processes(0)) 
    print(torch.cuda.list_gpu_processes(1)) 
    
    del model
    
    gc.collect() 
    torch.cuda.empty_cache()
    
    print("GPU mem after releasing model")
    print(torch.cuda.list_gpu_processes(0)) 
    print(torch.cuda.list_gpu_processes(1)) 

    print("building the ensemble")
    
    xtrees = ExtraTreesClassifier(
        n_estimators=1500,  # Slightly increase
        max_depth=None,
        min_samples_split=5,  # Reduce for more splits
        min_samples_leaf=2,   # Reduce for more granular leaves
        max_features='sqrt',
        bootstrap=False,
        # class_weight='balanced',  # Try balanced if classes are imbalanced
        random_state=42,
        verbose=True,
        n_jobs=-1,
    )

    xtrees_regularized = ExtraTreesClassifier(
        n_estimators=1500,
        max_depth=None,           # Keep unlimited
        min_samples_split=50,     # Much higher (was 5)
        min_samples_leaf=25,      # Much higher (was 2)
        max_features='sqrt',
        bootstrap=False,
        random_state=123,         # Different seed!
        verbose=1,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingClassifier(
        max_iter=1000,              # Like n_estimators
        learning_rate=0.05,
        max_depth=9,                # Can go deeper than regular GBC
        min_samples_leaf=5,
        max_leaf_nodes=127,         # Controls tree complexity
        l2_regularization=1.0,
        max_bins=255,               # Default, controls binning
        # class_weight='balanced',  # If needed
        random_state=42,
        verbose=1,
    )

    lgb = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=9,              # Can go deeper efficiently
        num_leaves=127,           # 2^7 - 1, controls complexity
        min_child_samples=5,      # Similar to min_samples_leaf
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        # is_unbalance=True,      # If imbalanced
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=1500,
        max_depth=None,           # Match ET's depth
        min_samples_split=5,      # Same as ET
        min_samples_leaf=2,       # Same as ET
        max_features='sqrt',      # Same as ET
        bootstrap=True,           # Key RF difference from ET
        max_samples=0.8,          # Subsample for more diversity
        # class_weight='balanced',
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    
    
    ensemble = VotingClassifier(
        [
            ('xtrees', xtrees),
            ('xtrees_regularized', xtrees_regularized),
            ('hgb', hgb),
            ('lgb', lgb),
            ('rf', rf),
            ('log_reg', LogisticRegression(max_iter=1000, C=1.0, n_jobs=1))
        ], 
        voting='soft', 
        n_jobs=1,
        # weights=[2., 1.5, 1., 1., 1., 0.5],
        weights=[1, 1, 1, 1, 1, 1],
        verbose=True
    ) # n_jobs=1 Sequential across estimators


    print("Start training the ensemble...")

    ensemble.fit(X, Y)

    print("training done.")

    print("releasing mem deleting training examples after fitting")
    # Clear intermediate variables
    del X, Y
    gc.collect()
    print("Clear X, Y intermediate variables done.")

    probs_list = []
    
    for name, estimator in ensemble.named_estimators_.items():
        probs = estimator.predict_proba(X_test)
        probs_list.append(probs)
        
    avg_probs = np.mean(probs_list, axis=0)

    # Accumulate probabilities
    if accumulated_probs is None:
        accumulated_probs = avg_probs
    else:
        accumulated_probs += avg_probs
    
    # Free memory by deleting the model
    del ensemble
    gc.collect()
    print("Clear ensemble intermediate variable done.")

    print(f"Voting classifier ensemble {idx+1}/{n_classifiers} complete")




avg_probs_acc = accumulated_probs / n_classifiers

y_pred_proba = avg_probs_acc[:, 1]


y_pred_proba[0:100]


row_id_test = test_df["row_id"].to_numpy()


# the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': row_id_test,
    'rule_violation': y_pred_proba
})

# Display the DataFrame to verify
print("Submission DataFrame:")
print(submission_df.head())

# Save to CSV with header (index=False to avoid row numbers)
submission_df.to_csv('submission.csv', index=False)


import datetime

x = datetime.datetime.now()
print(x)
print("~ fin ~")




