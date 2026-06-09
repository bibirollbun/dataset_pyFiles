from xgboost import train as xgb_train, DMatrix
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy.random as rnd
import numpy as np
from scipy.stats import pearsonr
import gc
import polars as pl


RANDOM_SEED = 1
NUM_SPLITS = 5
label_name = 'label'
early_stopping_iterations = 10


CAT_SELECTED_FEATS = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
                      'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21',
                      'X22', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X33', 'X35', 'X36', 'X37', 'X38', 'X39',
                      'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54',
                      'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68',
                      'X69', 'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X83',
                      'X84', 'X87', 'X88', 'X90', 'X91', 'X92', 'X93', 'X94', 'X96', 'X98', 'X99', 'X100', 'X101',
                      'X102', 'X103', 'X105', 'X106', 'X107', 'X108', 'X109', 'X111', 'X112', 'X113', 'X114', 'X115',
                      'X117', 'X118', 'X119', 'X121', 'X123', 'X124', 'X125', 'X126', 'X127', 'X129', 'X130', 'X131',
                      'X133', 'X135', 'X136', 'X137', 'X141', 'X142', 'X143', 'X144', 'X145', 'X147', 'X148', 'X149',
                      'X150', 'X151', 'X153', 'X154', 'X155', 'X156', 'X157', 'X159', 'X160', 'X161', 'X162', 'X163',
                      'X165', 'X166', 'X167', 'X169', 'X171', 'X172', 'X173', 'X174', 'X175', 'X177', 'X178', 'X180',
                      'X183', 'X184', 'X185', 'X186', 'X187', 'X188', 'X189', 'X190', 'X191', 'X192', 'X193', 'X194',
                      'X195', 'X196', 'X197', 'X198', 'X199', 'X200', 'X201', 'X202', 'X203', 'X204', 'X205', 'X206',
                      'X207', 'X208', 'X209', 'X210', 'X211', 'X212', 'X213', 'X214', 'X215', 'X216', 'X217', 'X218',
                      'X219', 'X220', 'X221', 'X222', 'X223', 'X224', 'X225', 'X226', 'X227', 'X228', 'X229', 'X230',
                      'X231', 'X232', 'X233', 'X234', 'X235', 'X236', 'X237', 'X238', 'X239', 'X240', 'X241', 'X242',
                      'X243', 'X244', 'X245', 'X246', 'X247', 'X248', 'X249', 'X250', 'X251', 'X252', 'X253', 'X254',
                      'X255', 'X256', 'X257', 'X258', 'X259', 'X260', 'X261', 'X262', 'X263', 'X265', 'X266', 'X267',
                      'X268', 'X269', 'X270', 'X271', 'X273', 'X274', 'X275', 'X278', 'X279', 'X280', 'X282', 'X286',
                      'X288', 'X291', 'X292', 'X293', 'X299', 'X301', 'X302', 'X304', 'X305', 'X306', 'X307', 'X308',
                      'X309', 'X310', 'X311', 'X312', 'X313', 'X314', 'X315', 'X316', 'X317', 'X318', 'X319', 'X320',
                      'X321', 'X322', 'X323', 'X324', 'X325', 'X326', 'X327', 'X328', 'X329', 'X330', 'X334', 'X336',
                      'X339', 'X340', 'X346', 'X347', 'X348', 'X349', 'X350', 'X352', 'X353', 'X354', 'X355', 'X356',
                      'X358', 'X359', 'X360', 'X361', 'X362', 'X364', 'X365', 'X366', 'X367', 'X368', 'X370', 'X371',
                      'X372', 'X373', 'X374', 'X376', 'X377', 'X378', 'X379', 'X382', 'X384', 'X385', 'X386', 'X388',
                      'X389', 'X390', 'X391', 'X392', 'X394', 'X395', 'X396', 'X397', 'X398', 'X400', 'X401', 'X402',
                      'X403', 'X404', 'X406', 'X407', 'X408', 'X409', 'X410', 'X412', 'X413', 'X414', 'X415', 'X416',
                      'X418', 'X420', 'X421', 'X422', 'X424', 'X426', 'X427', 'X430', 'X431', 'X432', 'X433', 'X434',
                      'X435', 'X436', 'X437', 'X438', 'X439', 'X440', 'X441', 'X442', 'X443', 'X444', 'X445', 'X446',
                      'X447', 'X448', 'X449', 'X451', 'X452', 'X453', 'X454', 'X455', 'X456', 'X457', 'X458', 'X459',
                      'X460', 'X461', 'X462', 'X463', 'X465', 'X467', 'X468', 'X469', 'X470', 'X471', 'X472', 'X473',
                      'X474', 'X475', 'X476', 'X477', 'X478', 'X479', 'X480', 'X481', 'X482', 'X483', 'X484', 'X485',
                      'X486', 'X487', 'X488', 'X489', 'X490', 'X491', 'X492', 'X493', 'X494', 'X495', 'X496', 'X497',
                      'X498', 'X499', 'X500', 'X501', 'X502', 'X503', 'X504', 'X505', 'X506', 'X507', 'X508', 'X509',
                      'X510', 'X511', 'X512', 'X513', 'X514', 'X515', 'X516', 'X517', 'X518', 'X519', 'X520', 'X521',
                      'X522', 'X523', 'X524', 'X525', 'X526', 'X527', 'X528', 'X529', 'X531', 'X534', 'X535', 'X536',
                      'X537', 'X539', 'X540', 'X541', 'X542', 'X543', 'X544', 'X545', 'X546', 'X547', 'X548', 'X549',
                      'X550', 'X551', 'X552', 'X553', 'X554', 'X555', 'X556', 'X557', 'X558', 'X559', 'X560', 'X561',
                      'X562', 'X563', 'X564', 'X565', 'X566', 'X567', 'X568', 'X569', 'X570', 'X571', 'X572', 'X573',
                      'X574', 'X576', 'X578', 'X579', 'X580', 'X581', 'X582', 'X583', 'X584', 'X586', 'X587', 'X588',
                      'X589', 'X590', 'X591', 'X592', 'X593', 'X594', 'X595', 'X596', 'X597', 'X599', 'X600', 'X601',
                      'X602', 'X603', 'X604', 'X605', 'X606', 'X607', 'X608', 'X609', 'X610', 'X611', 'X613', 'X614',
                      'X615', 'X616', 'X617', 'X618', 'X619', 'X620', 'X621', 'X622', 'X623', 'X624', 'X625', 'X626',
                      'X627', 'X628', 'X629', 'X630', 'X631', 'X632', 'X633', 'X634', 'X635', 'X636', 'X637', 'X638',
                      'X639', 'X640', 'X641', 'X642', 'X643', 'X644', 'X645', 'X646', 'X647', 'X648', 'X649', 'X650',
                      'X651', 'X652', 'X653', 'X654', 'X655', 'X656', 'X657', 'X658', 'X660', 'X661', 'X663', 'X664',
                      'X665', 'X666', 'X667', 'X668', 'X669', 'X670', 'X671', 'X672', 'X673', 'X674', 'X675', 'X676',
                      'X678', 'X679', 'X680', 'X681', 'X682', 'X684', 'X686', 'X687', 'X688', 'X690', 'X691', 'X693',
                      'X695', 'X696', 'X718', 'X719', 'X720', 'X721', 'X722', 'X723', 'X724', 'X725', 'X726', 'X727',
                      'X728', 'X729', 'X730', 'X731', 'X732', 'X733', 'X734', 'X735', 'X736', 'X737', 'X738', 'X739',
                      'X740', 'X741', 'X742', 'X743', 'X744', 'X745', 'X746', 'X747', 'X748', 'X749', 'X750', 'X751',
                      'X752', 'X753', 'X754', 'X755', 'X756', 'X757', 'X758', 'X759', 'X760', 'X761', 'X762', 'X763',
                      'X764', 'X765', 'X766', 'X767', 'X768', 'X769', 'X770', 'X771', 'X772', 'X773', 'X774', 'X775',
                      'X776', 'X777', 'X778', 'X779', 'X780', 'X781', 'X782', 'X783', 'X784', 'X785', 'X786', 'X788',
                      'X789', 'X790', 'X791', 'X792', 'X793', 'X794', 'X795', 'X796', 'X797', 'X798', 'X799', 'X800',
                      'X801', 'X802', 'X803', 'X804', 'X805', 'X806', 'X807', 'X808', 'X809', 'X810', 'X811', 'X812',
                      'X813', 'X814', 'X815', 'X816', 'X817', 'X818', 'X819', 'X820', 'X821', 'X822', 'X823', 'X824',
                      'X825', 'X826', 'X827', 'X828', 'X829', 'X830', 'X831', 'X832', 'X833', 'X834', 'X835', 'X836',
                      'X837', 'X838', 'X839', 'X840', 'X842', 'X843', 'X844', 'X845', 'X846', 'X847', 'X848', 'X849',
                      'X850', 'X851', 'X852', 'X853', 'X855', 'X856', 'X857', 'X859', 'X860', 'X861', 'X862', 'X863',
                      'X866', 'X874', 'X875', 'X876', 'X877', 'X878', 'X879', 'X880', 'X882', 'X884', 'X885', 'X886',
                      'order_book_imbalance', 'executed_trade_imbalance', 'buy_sell_ratio', 'buy_contribution',
                      'sell_contribution', 'relative_bid_strength', 'relative_ask_strength']


def get_competition_like_splits(data:pd.DataFrame, label_name:str='label', num_splits:int=5, random_seed:int=1):
    """Creates shuffled timeseries splits (shuffling is performed after creating the splits)"""
    X = data.copy()
    y = X.pop(label_name)
    splitter = TimeSeriesSplit(num_splits)
    splits = splitter.split(X,y)
    return [(rnd.default_rng(seed=random_seed).permutation(split[0]), split[1]) for split in splits]


def safe_div(numerator, denominator):
    """
    Divides the numerator by the denominator and provides a result. If the denominator is 0, the output will
    be replaced with NaN to avoid division errors.

    :param numerator: The number to be divided.
    :param denominator: The number by which numerator is divided. Zero values are replaced with NaN.
    :return: The result of the division or NaN if the denominator is 0.
    """
    return numerator / denominator.replace(0, np.nan)


def engineer_features(data:pd.DataFrame) -> pd.DataFrame:
    """Basic feature engineering using the named columns"""
    # 1. Order Book Imbalance (OBI)
    data["order_book_imbalance"] = (data["bid_qty"] - data["ask_qty"]) / (
        data["bid_qty"] + data["ask_qty"]
    )

    # 2. Executed Trade Imbalance (ETI)
    data["executed_trade_imbalance"] = (data["buy_qty"] - data["sell_qty"]) / (
        data["buy_qty"] + data["sell_qty"]
    )

    # 3. Market Buy/Sell Ratio
    data["buy_sell_ratio"] = safe_div(data["buy_qty"], data["sell_qty"])

    # 4. Volume Contribution Ratios
    data["buy_contribution"] = safe_div(data["buy_qty"], data["volume"])
    data["sell_contribution"] = safe_div(data["sell_qty"], data["volume"])

    # 5. Relative Bid/Ask Strength
    data["relative_bid_strength"] = safe_div(data["bid_qty"], data["sell_qty"])
    data["relative_ask_strength"] = safe_div(data["ask_qty"], data["buy_qty"])
    return data


def pearson_eval(preds, dmatrix):
    """XGBoost compatible pearson R"""
    y_true = dmatrix.get_label()
    return 'pearson', float(pearsonr(y_true, preds)[0])


def drop_high_corr_columns(
    train_df: pd.DataFrame, threshold: float = 0.96, label_name:str=label_name
):
    y = train_df.pop(label_name)
    # Convert pandas DataFrames to Polars
    train_pl = pl.from_pandas(train_df)

    # Compute correlation matrix on train data
    corr_df = train_pl.corr()

    # Extract column names and correlation matrix as NumPy array
    columns = corr_df.columns
    corr_np = corr_df.to_numpy()

    # Find columns to drop based on upper triangle and threshold
    upper = np.triu(np.ones(corr_np.shape), k=1)
    to_drop = [
        columns[j]
        for i in range(corr_np.shape[0])
        for j in range(corr_np.shape[1])
        if upper[i, j] and corr_np[i, j] > threshold
    ]

    to_drop = list(set(to_drop))

    # Drop columns from train and test Polars DataFrames
    train_clean = train_pl.drop(to_drop)
    print(f'Dropped {len(to_drop)} features')

    # Convert back to pandas and return
    return pd.concat([train_clean.to_pandas(), y], axis=1)


data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')


data


train_features = CAT_SELECTED_FEATS + [label_name]


data = engineer_features(data)[train_features]


data = drop_high_corr_columns(data)


data


val_features = list(data.columns)
val_features.remove(label_name)


splits = get_competition_like_splits(data)


param = {"objective": "reg:squarederror",
        "tree_method": 'hist'}


scores = []
train_scores = []
models = []
num_fold = 0
# Loop over CV Splits
for train_idx, test_idx in splits:
    # Get Train and Test Data via Split Index
    X_train = data.iloc[train_idx].replace([np.inf, -np.inf], np.nan).copy()
    y_train = X_train.pop(label_name)
    X_test = data.iloc[test_idx].replace([np.inf, -np.inf], np.nan).copy()
    y_test = X_test.pop(label_name)

    # Create XGBoost Matrices for training
    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    # Prepare Evaluation Splits
    evals = [(dtrain, "train"), (dtest, "eval")]

    # Train Model with Early Stopping
    booster = xgb_train(
            params=param,
            dtrain=dtrain,
            num_boost_round=200000,
            evals=evals,
            custom_metric=pearson_eval,
            maximize=True,
            early_stopping_rounds=early_stopping_iterations,
            verbose_eval=False,
    )
    # Get predictions
    preds = booster.predict(dtest)
    train_preds = booster.predict(dtrain)
    # Calculate Train/Val Pearson Scores
    train_score = pearsonr(train_preds, y_train)[0]
    score = pearsonr(preds, y_test)[0]
    # Save Scores and Model
    scores.append(score)
    train_scores.append(train_score)
    models.append(booster)
    # Print Fold Results 
    print(50*'=')
    print(f'Results of CV Fold {num_fold}')
    print(f'Train Pearson: {train_score:.5f}\nVal Pearson: {score:.5f}')
    print(50*'=')
print(f'Mean Train PearsonR: {np.mean(train_scores):.5f}\nMean Val PearsonR: {np.mean(scores):.5f}')


del data # delete training data to free up memory space


gc.collect()


test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


test_data = engineer_features(test_data)[val_features]


test_dmatrix = DMatrix(test_data)


final_predictions = []
for model in models: 
    final_predictions.append(model.predict(test_dmatrix))


sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


len(sample_submission)


final_prediction = np.mean(final_predictions, axis=0)


sample_submission['prediction'] = final_prediction


sample_submission


sample_submission.to_csv('submission.csv', index=False)

