import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, mean_absolute_error

# ====================================================
# Load Data
# ====================================================
train = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv',
                    parse_dates=['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate'])
test = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/test.csv',
                   parse_dates=['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate'])

# ====================================================
# Cleaning function
# ====================================================
def clean_ExtendedQuantity(o):
    if type(o) != str:
        return np.nan
    if '\n' in o:
        return 3
    o = o.replace(',', '')
    while len(o) > 0 and o[-1] == '_':
        o = o[:-1]
    if len(o) >= 5 and o[-3:] == ' EA':
        o = o[-5:-3]
    try:
        return float(o)
    except ValueError as e:
        print(e)

for df in [train, test]:
    df['ExtendedQuantity'] = df['ExtendedQuantity'].apply(clean_ExtendedQuantity)
    df['ItemDescription'] = df['ItemDescription'].str.replace('\n', '')
    df['ItemDescription'] = df['ItemDescription'].str.strip()
train['QtyShipped'] = train['QtyShipped'].apply(clean_ExtendedQuantity)

# ====================================================
# Subset Data
# ====================================================
train_c = train[train.id < 53422]
train_r = train_c.dropna(subset=['QtyShipped'])

# ====================================================
# Classification Model: Predict MasterItemNo
# ====================================================
model_c = make_pipeline(OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10000),
                        DecisionTreeClassifier(
                                               criterion='entropy',
                                               splitter='best'))
features_c = ['ItemDescription']
oof_c = cross_val_predict(model_c, train[features_c], train.MasterItemNo)
accuracy = accuracy_score(train.MasterItemNo, oof_c)
precision, recall, f1, _ = precision_recall_fscore_support(train.MasterItemNo, oof_c, average='macro', zero_division=0.0)
print(f"# {accuracy=:.5f}   {f1=:.5f}   {precision=:.5f}   {recall=:.5f}")

# ====================================================
# Regression Model: Predict QtyShipped
# ====================================================
model_r = make_pipeline(SimpleImputer(),
                        DecisionTreeRegressor(
                                              min_samples_leaf=4,
                                              min_samples_split=6,
                                              splitter='best'))

features_r = ['ExtendedQuantity']
oof_r = cross_val_predict(model_r, train_r[features_r], train_r.QtyShipped)
mae = mean_absolute_error(train_r.QtyShipped, oof_r)
norm_mae = mae / (train_r.QtyShipped.max() - train_r.QtyShipped.min())
reg_score = 1 - norm_mae.clip(0, 1)
print(f"# {mae=:.1f}   {norm_mae=:.5f}   {reg_score=:.5f}")

# ====================================================
# Final CV Score
# ====================================================
final_cv_score = 0.25 * accuracy + 0.25 * f1 + 0.5 * reg_score
print(f"# Final CV Score: {final_cv_score:.5f}")

# ====================================================
# Train Final Models and Submission
# ====================================================
model_c.fit(train_c[features_c], train_c.MasterItemNo)
model_r.fit(train_r[features_r], train_r.QtyShipped)


sub = pd.DataFrame({
    'id': test.id,
    'MasterItemNo': model_c.predict(test[features_c]),
    'QtyShipped': model_r.predict(test[features_r])
})

sub.to_csv('/kaggle/working/submission.csv', index=False)
print("# Saved final submission.csv")
sub.head()




