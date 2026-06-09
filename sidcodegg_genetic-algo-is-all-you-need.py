import pandas as pd
import numpy as np
import operator
import random
from sklearn.metrics import accuracy_score
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


train.info()


test.info()


num_cols = train.select_dtypes(include='number').columns.drop('id')
cat_cols = train.select_dtypes(include='object').columns.drop('Personality')


for col in num_cols:
    train[f'{col}_missing'] = train[col].isna()
    test[f'{col}_missing'] = test[col].isna()
    median = train[col].median()
    train[col] = train[col].fillna(median)
    test[col] = test[col].fillna(median)

for col in cat_cols:
    train[f'{col}_missing'] = train[col].isna()
    test[f'{col}_missing'] = test[col].isna()
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    test[col] = test[col].fillna(mode)
    all_cats = pd.Categorical(pd.concat([train[col], test[col]])).categories
    train[col] = pd.Categorical(train[col], categories=all_cats)
    test[col] = pd.Categorical(test[col], categories=all_cats)




train.info()


test.info()


train_y = (train["Personality"] == "Extrovert").to_numpy()

features = [c for c in train.columns
            if c not in ("id", "Personality")
            and not c.endswith("_missing")]

extro_markers = {}
intro_markers = {}

for col in features:
    if pd.api.types.is_numeric_dtype(train[col]):
        thresh = train[col].median()
        mean_extro = train.loc[train_y, col].mean()
        mean_intro = train.loc[~train_y, col].mean()

        if mean_extro > mean_intro:
            extro_markers[col] = ('>', thresh)
            intro_markers[col] = ('<', thresh)
        else:
            extro_markers[col] = ('<', thresh)
            intro_markers[col] = ('>', thresh)

    else:
        mode_extro = train.loc[train_y, col].mode()[0]
        mode_intro = train.loc[~train_y, col].mode()[0]
        extro_markers[col] = mode_extro
        intro_markers[col] = mode_intro


def pretty_print_markers(markers, title):
    print(f"\n{title} Markers:")
    for k, v in markers.items():
        print(f"  {k}: {v}")

pretty_print_markers(extro_markers, "Extrovert")
pretty_print_markers(intro_markers, "Introvert")


def build_matrix(df, markers):
    mat = np.zeros((len(df), len(markers)), dtype=int)
    for i, col in enumerate(markers):
        rule = markers[col]
        if isinstance(rule, tuple):
            op_symbol, thresh = rule
            if op_symbol == ">":
                mat[:, i] = (df[col] > thresh).astype(int)
            else:
                mat[:, i] = (df[col] < thresh).astype(int)
        else:
            mat[:, i] = (df[col] == rule).astype(int)
    return mat

train_extro = build_matrix(train, extro_markers)
train_intro = build_matrix(train, intro_markers)
test_extro = build_matrix(test, extro_markers)
test_intro = build_matrix(test, intro_markers)


M = len(features)
POP, GENS, MUT = 30, 20, 0.1

pop = [np.random.randint(0, 5, size=M) for _ in range(POP)]
best_weights = None
best_acc = 0.0

for gen in range(GENS):
    weights_matrix = np.vstack(pop)  
    extro_scores = train_extro.dot(weights_matrix.T)
    intro_scores = train_intro.dot(weights_matrix.T)    
    final_scores = extro_scores - intro_scores
    preds = (final_scores >= 0)
    accs = (preds == train_y[:, None]).mean(axis=0)
    for acc, w in zip(accs, pop):
        if acc > best_acc:
            best_acc, best_weights = acc, w.copy()
    ranked = sorted(zip(accs, pop), key=lambda x: x[0], reverse=True)
    pop = [w for _, w in ranked[: POP // 2]]

    while len(pop) < POP:
        p1, p2 = random.sample(pop, 2)
        cut = random.randint(1, M - 1)
        child = np.concatenate([p1[:cut], p2[cut:]])
        if random.random() < MUT:
            idx = random.randrange(M)
            child[idx] = max(0, child[idx] + random.randint(-2, 2))
        pop.append(child)

print("\nBest weights:", best_weights)
print("Training accuracy:", best_acc)





s_extro = test_extro.dot(best_weights)
s_intro = test_intro.dot(best_weights)
final_test_scores = s_extro - s_intro
pred_labels = np.where(final_test_scores >= 0, "Extrovert", "Introvert")

submission = pd.DataFrame({
    "id": test["id"],
    "Personality": pred_labels
})
submission.to_csv("submission.csv", index=False)
print("\nSubmission head:")
print(submission.head())

