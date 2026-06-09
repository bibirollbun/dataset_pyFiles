import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support, classification_report, mean_absolute_error


train = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv',
                    # index_col='id',
                    parse_dates=['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate'],
                    date_format={'CONSTRUCTION_START_DATE': '%m/%d/%Y %H:%M',
                                 'SUBSTANTIAL_COMPLETION_DATE': '%m/%d/%Y %H:%M',
                                 'invoiceDate': 'mixed'})
test = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/test.csv',
                   # index_col='id',
                   parse_dates=['CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE', 'invoiceDate'],
                   date_format={'CONSTRUCTION_START_DATE': '%m/%d/%Y %H:%M',
                                'SUBSTANTIAL_COMPLETION_DATE': '%m/%d/%Y %H:%M',
                                'invoiceDate': '%m/%d/%Y'})
train#.iloc[:,10:]


def clean_ExtendedQuantity(o):
    if type(o) != str:
        return np.nan
    if '\n' in o:
        return 3
    o = o.replace(',', '')
    while len(o) > 0 and o[-1] == '-':
        o = o[:-1]
    if len(o) >= 5 and o[-3:] == ' EA':
        o = o[-5:-3]
    try:
        return float(o)
    except ValueError as e:
        print(e)

for df in [train, test]:
    df['ExtendedQuantity'] = df['ExtendedQuantity'].apply(clean_ExtendedQuantity)
    df['ItemDescription'] = df['ItemDescription'].str.replace('\n', ' ')
    df['ItemDescription'] = df['ItemDescription'].str.strip()
train['QtyShipped'] = train['QtyShipped'].apply(clean_ExtendedQuantity)


print(set(train.columns) - set(test.columns))

both = pd.concat([train, test])

df = []
for col in test.columns:
    missing_train = train[col].isna().mean().round(3) * 100
    missing_test = test[col].isna().mean().round(3) * 100
    unique_train = len(set(train[col].fillna(99999999999)))
    unique_test = len(set(test[col].fillna(99999999999)))
    unique_both = len(set(both[col].fillna(99999999999)))
    df.append((col, train[col].dtype, missing_train, missing_test, unique_train, unique_test, unique_both, unique_train + unique_test - unique_both))
df = pd.DataFrame(df, columns=['column', 'dtype', 'percent_missing_train', 'percent_missing_test', 'unique_train', 'unique_test', 'unique_both', 'overlap'])
df['percent_new'] = (1 - df.overlap / df.unique_test).round(3) * 100
df


trs = train.groupby(train.id // 1000).size()
tes = test.groupby(test.id // 1000).size()
plt.barh(trs.index, trs, label='train')
plt.barh(tes.index, tes, left=trs[tes.index], label='test')
plt.gca().invert_yaxis()
plt.legend(loc='lower left')
plt.xlabel('sample count')
plt.ylabel('id // 1000')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title("This isn't a random split!")
plt.show()


pd.DataFrame({'train': train.select_dtypes('float').mean(), 'test': test.select_dtypes('float').mean()})


for f in ['ExtendedPrice', 'ExtendedQuantity']:
    plt.figure(figsize=(12, 4))
    plt.hist(train[f].dropna(), density=True, alpha=0.5,
             label=f'train between {train[f].min():.1e} and {train[f].max():.1e}',
             color='b', bins=50)
    plt.title(f"{f} histogram")
    plt.xlabel(f)
    plt.ylabel('count')
    plt.legend()
    plt.show()
    plt.figure(figsize=(12, 4))
    plt.hist(test[f].dropna(), density=True, alpha=0.5,
             label=f'test between {test[f].min():.1e} and {test[f].max():.1e}',
             color='g', bins=50)
    plt.title(f"{f} histogram")
    plt.xlabel(f)
    plt.ylabel('count')
    plt.legend()
    plt.show()


plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.scatter(list(train.QtyShipped), train.index, s=1, c='c')
plt.axhline(10736, color='gray')
plt.xlabel('target QtyShipped')
plt.ylabel('train.index')
plt.title('The five clusters of training data')

plt.subplot(2, 3, 2)
plt.scatter(train.id, train.index, s=1)
plt.axhline(10736, color='gray') # index
plt.axvline(13422, color='gray') # id
plt.xlabel('train.id')
plt.ylabel('train.index')
plt.title('ids < 13422 are shuffled in train,\n ≥ 13422 are not')

plt.subplot(2, 3, 3)
plt.scatter(list(train.SUBSTANTIAL_COMPLETION_DATE), train.index, s=1, c='c')
plt.axhline(10736, color='gray')
plt.xlabel('train.SUBSTANTIAL_COMPLETION_DATE')
plt.ylabel('train.index')
plt.title('completion date distribution\nchanges at id=13422')

plt.subplot(2, 3, 5)
plt.scatter(test.id, test.index, s=1)
plt.xlabel('test.id')
plt.ylabel('test.index')
plt.title('ids < 13422 are shuffled in train,\n ≥ 13422 are not')

plt.subplot(2, 3, 6)
plt.scatter(list(test.SUBSTANTIAL_COMPLETION_DATE), test.index, s=1, c='c')
plt.xlabel('test.SUBSTANTIAL_COMPLETION_DATE')
plt.ylabel('test.index')
plt.title('completion date distribution\nchanges at id=13422')

plt.tight_layout()
plt.show()


# Keep only training samples with known target and with id < 13422
train_c = train[train.id < 13422]
train_r = train_c.dropna(subset=['QtyShipped'])



plt.figure(figsize=(12, 3))
plt.title('Histogram of regression targets')
plt.hist(train_r.QtyShipped[train_r.id<13422], bins=100)
plt.xlabel('QtyShipped')
plt.ylabel('count')
plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(train.ExtendedQuantity, train.QtyShipped, s=1, c=train.id<13422, cmap='bwr')
plt.gca().set_aspect('equal')
plt.xlabel('ExtendedQuantity')
plt.ylabel('target QtyShipped')
plt.title('A predictor for QtyShipped (whole train dataset)')
legend_elements = [Line2D([0], [0], color='b', lw=0, marker='o', label='id ≥ 13422'),
                   Line2D([0], [0], color='r', lw=0, marker='o', label='id < 13422')]
plt.legend(handles=legend_elements)

plt.subplot(1, 2, 2)
plt.scatter(train_r.ExtendedQuantity, train_r.QtyShipped, s=1, c='r')
plt.gca().set_aspect('equal')
plt.xlabel('ExtendedQuantity')
plt.ylabel('target QtyShipped')
plt.title('A predictor for QtyShipped (subset of train)')

plt.show()


vc = train.MasterItemNo.value_counts()
plt.figure(figsize=(12, 3))
plt.bar(np.arange(len(vc)), vc, width=1)
# plt.xticks(np.arange(len(vc)), vc.index)
plt.title('Histogram of MasterItemNo')
plt.xlabel('MasterItemNo index')
plt.ylabel('count')
plt.show()


model_c = make_pipeline(OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=10000), DecisionTreeClassifier())
features_c = ['ItemDescription']
oof_c = cross_val_predict(model_c, train[features_c], train.MasterItemNo)
accuracy = accuracy_score(train.MasterItemNo, oof_c)
precision, recall, f1, _ = precision_recall_fscore_support(train.MasterItemNo, oof_c, average='macro', zero_division=0.0)
print(f"# {accuracy=:.5f}   {f1=:.5f}   {precision=:.5f}   {recall=:.5f}")
# print(classification_report(train.MasterItemNo, oof_c))
# acc=0.82979   f1=0.42904   precision=0.43951   recall=0.42365


# model_r = DummyRegressor(strategy='median')
model_r = make_pipeline(SimpleImputer(), DecisionTreeRegressor())
features_r = ['ExtendedQuantity']
oof_r = cross_val_predict(model_r, train_r[features_r], train_r.QtyShipped)
mae = mean_absolute_error(train_r.QtyShipped, oof_r)
norm_mae = mae / (train_r.QtyShipped.max() - train_r.QtyShipped.min())
reg_score = 1 - norm_mae.clip(0, 1)
print(f"# {mae=:.1f}   {norm_mae=:.5f}   {reg_score=:.5f}")
# mae=34.7   norm_mae=0.00895   reg_score=0.99105


plt.figure(figsize=(10, 6))
plt.title('Univariate regression for QtyShipped')
plt.scatter(train_r.ExtendedQuantity, oof_r, s=1, color='g')
plt.gca().set_aspect('equal')
plt.xlabel('ExtendedQuantity')
plt.ylabel('QtyShipped predicted')
plt.show()


print(f"# Final cv score: {0.25 * accuracy + 0.25 * f1 + 0.5 * reg_score:.5f}")
# Final cv score: 0.81024


# Refit both models to the full training dataset
model_c.fit(train_c[features_c], train_c.MasterItemNo)
model_r.fit(train_r[features_r], train_r.QtyShipped)

# Predict the test targets
sub = pd.DataFrame({'id': test.id,
                    'MasterItemNo': model_c.predict(test[features_c]),
                    'QtyShipped': model_r.predict(test[features_r])})
sub.to_csv('submission.csv', index=False)
display(sub)
!head submission.csv




