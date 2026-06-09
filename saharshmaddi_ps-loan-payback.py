!pip install feature-engine


import jax
import jax.numpy as jnp
from jax import jit
import flax
import flax.linen as nn
import optax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import feature_engine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import sklearn.ensemble as ens
import xgboost as xgb
import lightgbm as lgbm
import catboost as cat

import optuna


print("Jax Version: ", jax.__version__ )
print("Flax Version: ", flax.__version__)


import os
import warnings

warnings.filterwarnings("ignore")

class CFG:
    ROOT = "/kaggle/input/playground-series-s5e11/"
    train_path = os.path.join(ROOT + "train.csv")
    test_path = os.path.join(ROOT + "test.csv")
    sample_sub_path = os.path.join(ROOT + "sample_submission.csv")

    n_splits = 5
    batch_size = 32
    regularize = 1e-3

    seed = 0
    target = "loan_paid_back"
    device = jax.devices("gpu")[0]
    cores = os.cpu_count()

print(f"Number of CPU Cores (if using cpu): {CFG.cores}, Device in Use: {CFG.device}")


data = pd.read_csv(CFG.train_path)
train = data.drop(columns = ["id"])
train.head(10)


train.info()


train.describe()
num_cols = train.describe().columns.to_list()
cat_cols = [cname for cname in train.columns if cname not in num_cols]
print(f"Num Cols; {num_cols}\nCat Cols: {cat_cols}")


fig, axes = plt.subplots(ncols = 3, nrows = 2, figsize = (10,5))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    if col == CFG.target:
        continue
    sns.histplot(x = train[col], ax = ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(ncols = 3, nrows = 2, figsize = (10,5))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    if col == CFG.target:
        continue
    sns.boxplot(y = train[col], ax = ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()


#Apply VST
train[num_cols].skew()


right_skew = ["annual_income", "debt_to_income_ratio", "loan_amount"]
left_skew = ["credit_score"]
from feature_engine.transformation import LogTransformer, BoxCoxTransformer
log_t = LogTransformer(variables = right_skew)
train_log = log_t.fit_transform(train[right_skew])
fig, axes = plt.subplots(ncols = 2, nrows = 3, figsize = (10, 10))
for i, col in enumerate(right_skew):
    sns.histplot(x = train[col], ax = axes[i, 0])
    axes[i, 0].set_title(col)
    sns.histplot(x = train_log[col], ax = axes[i, 1])
    axes[i, 1].set_title(f"Log Transformation of {col}")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize = (10, 3))
box_cox = BoxCoxTransformer(variables = left_skew)
box_train = box_cox.fit_transform(train[left_skew])
for i, col in enumerate(left_skew):
    sns.histplot(x = train[col], ax = axes[0])
    axes[0].set_title(col)
    sns.histplot(x = box_train[col], ax = axes[1])
    axes[1].set_title(f"Box Cox Transformation of {col}")
plt.tight_layout()
plt.show()


corr = train[num_cols].corr()
plt.figure(figsize = (15,10))
sns.heatmap(corr, annot = True, cmap = "twilight")
plt.show()


vst_train = train[["interest_rate", "loan_amount", "credit_score"]].join(
    train_log[["annual_income", "debt_to_income_ratio"]]
)
vst_train = vst_train.join(train[[CFG.target]])
vst_corr = vst_train.corr()
plt.figure(figsize = (15, 10))
sns.heatmap(vst_corr, annot = True, cmap = "twilight")
plt.show()


train


y_train = train[CFG.target]
x_train = train.drop(columns = [CFG.target])


num_cols.remove(CFG.target)
num_cols


def ordinal(df):
    encoded_cols = []
    unique_maps = []
    for c in cat_cols:
        unique_vals, inv = np.unique(df[c].values, return_inverse = True)
        encoded_cols.append(inv)
        unique_maps.append(unique_vals)
    
    encoded = np.vstack(encoded_cols).T
    return encoded

# Standard scaling formula for ref : z = (x-mu)/sigma

@jit
def scale(x):
    mu = jnp.mean(x)
    sigma = jnp.std(x)
    z = (x-mu)/sigma
    return z

#place numerical values onto current device
x_device = jax.device_put(jnp.array(x_train[num_cols].values), device = CFG.device)
encoded = ordinal(x_train[cat_cols])
jencoded = jax.device_put(jnp.array(encoded), device = CFG.device)
scaled = scale(x_device)

jtrain = jnp.hstack([jencoded, scaled])


jtrain


print(jtrain.device)


x_train = np.array(jtrain)


base_models = {
    "xgb":xgb.XGBClassifier(tree_method = "gpu_hist", device = "cuda", random_state = CFG.seed),
    "lgbm":lgbm.LGBMClassifier(device = "gpu", random_state = CFG.seed),
    "cat":cat.CatBoostClassifier(task_type = "GPU", verbose = 0, random_state = CFG.seed),
    "hgbm":ens.HistGradientBoostingClassifier(random_state = CFG.seed)
}

kf = KFold(n_splits = CFG.n_splits, shuffle = True, random_state = CFG.seed)


#def objective_xgb(trial):
 #   params = {
#        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#        "max_depth": trial.suggest_int("max_depth", 3, 10),
#        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
 #       "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
  #  }
   # model = xgb.XGBClassifier(**params, tree_method="gpu_hist", device = "cuda", random_state=42)
    #scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
#    return np.mean(scores)

#study_xgb = optuna.create_study(study_name="xgb_tuning", direction="maximize")
#study_xgb.optimize(objective_xgb, n_trials=25)
#best_xgb_params = study_xgb.best_params
#print(best_xgb_params)


best_xgb_params = {'n_estimators': 846, 'max_depth': 4, 'learning_rate': 0.1139271851977041, 'subsample': 0.7684214324389558, 'colsample_bytree': 0.5090853732843084}


#silence lgbm and catboost
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTGBM_VERBOSE"] = "0"
os.environ["CATBOOST_LOGGING_LEVEL"] = "Silent"


#def objective_cat(trial):
 #   params = {
  #      "iterations": trial.suggest_int("iterations", 100, 1000),
   #     "depth": trial.suggest_int("depth", 3, 10),
    #    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
     #   "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
      #  "random_strength": trial.suggest_float("random_strength", 0.0, 1.0)
#    }
 #   model = cat.CatBoostClassifier(
  #      **params,
   #     loss_function="Logloss",
    #    eval_metric="Accuracy",
     #   task_type="GPU",
      #  random_seed=0,
       # verbose=False
#    )
 #   scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="accuracy", n_jobs=1) # cant handle -1
  #  return np.mean(scores)

#study_cat = optuna.create_study(study_name="cat_tuning", direction="maximize")
#study_cat.optimize(objective_cat, n_trials=25)
#best_cat_params = study_cat.best_params
#print(best_cat_params)


best_cat_params = {'iterations': 894, 'depth': 3, 'learning_rate': 0.27815739813841206, 'l2_leaf_reg': 4.113675174074722, 'random_strength': 0.059895754974824224}


#def objective_lgbm(trial):
 #   params = {
  #      "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
   #     "max_depth": trial.suggest_int("max_depth", 3, 10),
    #    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
     #   "subsample": trial.suggest_float("subsample", 0.5, 1.0),
      #  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
#    }
 #   model = lgbm.LGBMClassifier(**params, device = "gpu", verbose=-1, random_state=0)
  #  scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
   # return np.mean(scores)

#study_lgbm = optuna.create_study(study_name="lgbm_tuning", direction="maximize")
#study_lgbm.optimize(objective_lgbm, n_trials=25)
#best_lgbm_params = study_lgbm.best_params
#print(best_lgbm_params)



best_lgbm_params = {'n_estimators': 577, 'max_depth': 4, 'learning_rate': 0.14012215109890813, 'subsample': 0.6999544133631017, 'colsample_bytree': 0.7919978289279921}


#def objective_hgb(trial):
 #   params = {
 #       "max_iter": trial.suggest_int("max_iter", 100, 1000),
 #       "max_depth": trial.suggest_int("max_depth", 3, 10),
 #       "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
 #       "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
 #       "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
 #       "max_bins": trial.suggest_int("max_bins", 100, 255)
 #   }
#
 #   model = ens.HistGradientBoostingClassifier(
  #      **params, random_state=42
   # )
#    scores = cross_val_score(model, x_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
 #   return np.mean(scores)

#study_hgb = optuna.create_study(study_name = "hgbc_tuning", direction="maximize")
#study_hgb.optimize(objective_hgb, n_trials=25)
#best_hgb_params = study_hgb.best_params
#print(best_hgb_params)


best_hgbm_params = {'max_iter': 991, 'max_depth': 5, 'learning_rate': 0.2757259010379551, 'min_samples_leaf': 50, 'l2_regularization': 0.2770923689818961, 'max_bins': 255}


def svote_preds(base_models, x_train, y_train, x_test): # soft voting
    preds = []
    for name, model in base_models.items():
        model = clone(model)
        model.fit(x_train, y_train)
        preds.append(model.predict_proba(x_test)[:, 1])

    final_pred = np.mean(preds, axis = 0)
    return final_pred


stack_kf = KFold(n_splits = CFG.n_splits, shuffle = True, random_state = CFG.seed+1) # different splits
tuned_bmodels = {
    "xgb":xgb.XGBClassifier(**best_xgb_params, tree_method = "gpu_hist", device = "cuda", random_state = CFG.seed),
    "lgbm":lgbm.LGBMClassifier(**best_lgbm_params, random_state = CFG.seed),
    "cat":cat.CatBoostClassifier(**best_cat_params, task_type = "GPU", verbose = 0, random_state = CFG.seed),
    "hgbm":ens.HistGradientBoostingClassifier(**best_hgbm_params, random_state = CFG.seed)
}


vote_kf = KFold(n_splits = CFG.n_splits, shuffle = True, random_state = CFG.seed)
baseline_scores = []

for fold, (tr_idx,val_idx) in enumerate(vote_kf.split(x_train, y_train)):
    print(f"\n--- BASELINE FOLD {fold+1} ---")
    x_tr, x_val = x_train[tr_idx], x_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    val_preds = svote_preds(tuned_bmodels, x_tr, y_tr, x_val)
    auc = roc_auc_score(y_val, val_preds)
    print(f"Fold AUC: {auc:.6f}")
    baseline_scores.append(auc)

baseline_mean = np.mean(baseline_scores)
baseline_std = np.std(baseline_scores)

print("Baseline Soft Vote")
print("MEAN AUC", baseline_mean)
print("STD AUC", baseline_std)


tdata = pd.read_csv(CFG.test_path)
test = tdata.drop(columns = ["id"])
tcat_cols = [cname for cname in test.columns if test[cname].dtype == "object"]
tnum_cols = [cname for cname in test.columns if cname not in tcat_cols]
#use jax to speed up preprocessing process
test_device = jax.device_put(jnp.array(test[tnum_cols].values), device = CFG.device)
test_encoded = ordinal(test[tcat_cols])
tencoded = jax.device_put(jnp.array(test_encoded), device = CFG.device)
tscaled = scale(test_device)
jtest = jnp.hstack([tencoded, tscaled])
test = np.array(jtest)
x_test = test


print(f"y_train shape: {y_train.shape}")
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")


n_models = len(base_models)
oof_preds = np.zeros((len(x_train), n_models), dtype = np.float32)
test_preds = np.zeros((len(x_test), n_models), dtype = np.float32)

for midx, (name, model) in enumerate(base_models.items()):
    print(f"Generating preds for {name}")
    for fold, (tr_idx, val_idx) in enumerate(stack_kf.split(x_train, y_train)):
        x_tr, x_val = x_train[tr_idx], x_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        m = clone(model)
        m.fit(x_tr, y_tr)

        oof_preds[val_idx, midx] = m.predict_proba(x_val)[:, 1]
        test_preds[:, midx] += m.predict_proba(x_test)[:, 1]/CFG.n_splits 

print("Recieved all predictions from base models")


from sklearn.model_selection import train_test_split as tts
x_mtr, x_mval, y_mtr, y_mval = tts(oof_preds, y_train, test_size=0.2,random_state = CFG.seed)


x_mtrain = jax.device_put(jnp.array(x_mtr,  dtype=jnp.float32), device = CFG.device)
y_mtrain = jax.device_put(jnp.array(y_mtr,  dtype=jnp.float32), device = CFG.device)

x_mval   = jax.device_put(jnp.array(x_mval, dtype=jnp.float32), device = CFG.device)
y_mval   = jax.device_put(jnp.array(y_mval, dtype=jnp.float32), device = CFG.device)


print(x_mtrain.shape, y_mtrain.shape, x_mval.shape, y_mval.shape)


from dataclasses import field
class meta_model(nn.Module):
    hidden_dims: list = field(default_factory=lambda: [32, 32, 64])
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, training = True):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.BatchNorm(use_running_average = not training)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate = self.dropout_rate, deterministic = not training)(x)

        x = nn.Dense(1)(x)
        return x.squeeze() # logits


def loss_fn(logits, y):
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()

def compute_auc(logits, y):
    probs = jax.nn.sigmoid(logits)
    return roc_auc_score(np.array(y), np.array(probs))




from flax.training import train_state
class State(train_state.TrainState):
    batch_stats: dict


def create_state(rng, x_sample):
    model = meta_model()
    variables = model.init(rng,x_sample,training=True)
    tx = optax.adam(learning_rate=1e-3)

    return State.create(
        apply_fn = model.apply,
        params = variables["params"],
        tx=tx,
        batch_stats = variables["batch_stats"]
    )


@jit
def train_step(state, x, y, rng):
    def forward(params):
        logits, updates = state.apply_fn(
            {"params":params, "batch_stats":state.batch_stats},
            x,
            training=True,
            rngs = {"dropout":rng},
            mutable=["batch_stats"]
        )
        loss = loss_fn(logits,y)
        return loss,(logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(forward, has_aux = True)(state.params)
    state = state.apply_gradients(grads = grads)
    state = state.replace(batch_stats = updates["batch_stats"])
    return state, loss, logits


def batch_loader(x, y, batch_size):
    n = len(x)
    idx = np.random.permutation(n)
    for i in range(0,n,batch_size):
        batch_idx = idx[i : i+batch_size]
        yield x[batch_idx], y[batch_idx]


def train_model(x_train, y_train, x_val, y_val, max_epochs = 50, batch_size = 64, patience = 8):
    rng = jax.random.PRNGKey(CFG.seed)
    state = create_state(rng, x_train[:1])
    best_auc = 0.0
    patience_counter = 0
    for epoch in range(max_epochs):
        for xb, yb in batch_loader(x_train, y_train, batch_size):
            rng, subrng = jax.random.split(rng)
            state, loss, _ = train_step(state, xb, yb, subrng)

        val_logits = state.apply_fn(
            {"params":state.params, "batch_stats":state.batch_stats},
            x_val,
            training = False
        )
        val_auc = compute_auc(val_logits, y_val)
        print(f"Epoch {epoch+1:03d} | Val AUC {val_auc:.6f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = state
            patience_counter = 0
        else:
            patience_counter+=1

        if patience_counter >= patience:
            print("Early Stopping Triggered")
            break

    return best_state


def predict_meta(state, x):
    logits = state.apply_fn(
        {"params":state.params, "batch_stats":state.batch_stats},
        x,
        training = False
    )
    return jax.nn.sigmoid(logits)


state = train_model(
    x_mtrain, y_mtrain, x_mval, y_mval, max_epochs = 50, batch_size = 64, patience=8
)

val_preds = predict_meta(state, x_mval)
stack_auc = roc_auc_score(y_mval, np.array(val_preds))
print("\n=========== FINAL COMPARISON ===========")
print("BASELINE AUC :", baseline_mean)
print("STACK AUC    :", stack_auc)
print("IMPROVEMENT  :", stack_auc - baseline_mean)



# dang thats rough


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(oof_preds, y_train)
lr_oof_preds = lr.predict_proba(oof_preds)[:,1]
lr_auc = roc_auc_score(y_train, lr_oof_preds)
lr_test_preds = lr.predict_proba(test_preds)[:,1]
print("\n=========== Very FINAL COMPARISON ===========")
print("BASELINE AUC :", baseline_mean)
print("STACK AUC    :", stack_auc)
print("LR AUC:", lr_auc)
print("IMPROVEMENT  :", stack_auc - baseline_mean)


x_jtest = jax.device_put(jnp.array(test_preds, dtype = jnp.float32), device = CFG.device)
meta_test_preds = predict_meta(state, x_jtest)
meta_test_preds = np.array(meta_test_preds)


vote_test_preds = svote_preds(tuned_bmodels, x_train, y_train, x_test)


# Weighted avg of voting and stacking
final_test_preds = 0.6 * vote_test_preds + 0.4 * meta_test_preds


submission = pd.DataFrame({
    "id" : tdata["id"],
    "loan_paid_back":final_test_preds
})
submission = submission.to_csv("submission.csv", index = False)

