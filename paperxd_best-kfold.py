%load_ext cudf.pandas
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from cuml.preprocessing import TargetEncoder
from xgboost import XGBRegressor
import xgboost as xgb
import gc
import rmm
rmm.reinitialize(pool_allocator = False)


def label_encode(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    df['Podcast_Name'] = df.Podcast_Name.map(podc_dict)
    df['Genre'] = df.Genre.map(genr_dict)
    df['Publication_Day'] = df.Publication_Day.map(week_dict)
    df['Publication_Time'] = df.Publication_Time.map(time_dict)
    df['Episode_Sentiment'] = df.Episode_Sentiment.map(sent_dict)
    df['Episode_Title'] = df.Episode_Title.str.split(' ').str.get(1)
    return df


new_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
old_train = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv').drop_duplicates().dropna(subset = ['Listening_Time_minutes'])
pt1 = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
train = pd.concat([new_train, old_train], axis = 0).drop(columns = ['id']).reset_index(drop = True)
target = train.Listening_Time_minutes
p1 = train.drop(columns = ['Listening_Time_minutes'])
p1 = label_encode(p1)
pt1 = pt1.drop(columns = ['id'])
pt1 = label_encode(pt1)
int_columns = ['Episode_Length_minutes']
del new_train, old_train
gc.collect()


params = {'max_depth': 14, 'learning_rate': 0.011015629090199364, 'subsample': 0.9116787269528593, 'colsample_bytree': 0.7827627872771932, 'colsample_bylevel': 0.6648599558828336, 'min_child_weight': 11, 'gamma': 5.2709692650185715, 'reg_lambda': 0.012221079454198924, 'reg_alpha': 8.45962277411465, 'objective': 'reg:squarederror', 'device': 'cuda', 'verbosity': 2, 'random_state': 42}
kf = KFold(n_splits = 7, shuffle = True, random_state = 69)
oof = np.zeros(len(p1))
preds = np.zeros(len(pt1))
i = 1
for train_idx, test_idx in kf.split(p1):
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    x_train_list = []
    x_test_list = []
    test_list = []
    for i in range(2,7):
        add_data = pd.read_csv(f'/kaggle/input/ultimate-dataset/podcast_new_features{i}.csv')
        add_x_train = add_data.iloc[train_idx]
        add_x_test = add_data.iloc[test_idx]
        del add_data
        gc.collect()
        add_test = pd.read_csv(f'/kaggle/input/ultimate-dataset/podcast_test_new_features{i}.csv')

        for col in add_x_train.columns:
            encoder = TargetEncoder()
            x_train_col = pd.Series(encoder.fit_transform(add_x_train[col], y_train).astype('float32'), name = col)
            x_test_col = pd.Series(encoder.transform(add_x_test[col]).astype('float32'), name = col)
            test_col = pd.Series(encoder.transform(add_test[col]).astype('float32'), name = col)
            
            x_train_list.append(x_train_col)
            x_test_list.append(x_test_col)
            test_list.append(test_col)

            encoder = None
            del encoder, x_train_col, x_test_col, test_col
            gc.collect()

        del add_x_train, add_x_test, add_test
        gc.collect()
        print(i)

    x_train_comb = pd.concat(x_train_list, axis = 1)
    del x_train_list
    gc.collect()
    x_test_comb = pd.concat(x_test_list, axis = 1)
    del x_test_list
    gc.collect()
    test_comb = pd.concat(test_list, axis = 1)
    del test_list
    gc.collect()
    print('Combinations Done')

    p1_train, p1_test = p1.iloc[train_idx], p1.iloc[test_idx]
    p1_train_int, p1_train_enc = p1_train[int_columns], p1_train.drop(columns = int_columns)
    p1_test_int, p1_test_enc = p1_test[int_columns], p1_test.drop(columns = int_columns)
    test_int, test_enc = pt1[int_columns], pt1.drop(columns = int_columns)

    p1_train_list = []
    p1_test_list = []
    pt1_list = []
    for col in p1_train_enc.columns:
        encoder = TargetEncoder()
        p1_train_col = pd.Series(encoder.fit_transform(p1_train_enc[col], y_train).astype('float32'), name = col)
        p1_test_col = pd.Series(encoder.transform(p1_test_enc[col]).astype('float32'), name = col)
        pt1_col = pd.Series(encoder.transform(test_enc[col]).astype('float32'), name = col)

        p1_train_list.append(p1_train_col)
        p1_test_list.append(p1_test_col)
        pt1_list.append(pt1_col)

        encoder = None
        del encoder, p1_train_col, p1_test_col, pt1_col
        gc.collect()

    p1_train_enc = pd.concat(p1_train_list, axis = 1)
    p1_test_enc = pd.concat(p1_test_list, axis = 1)
    test_enc = pd.concat(pt1_list, axis = 1)
    del p1_train_list, p1_test_list, pt1_list
    gc.collect()

    p1_train = pd.concat([p1_train_int.reset_index(drop = True), p1_train_enc], axis = 1)
    p1_test = pd.concat([p1_test_int.reset_index(drop = True), p1_test_enc], axis = 1)
    pt1_new = pd.concat([test_int, test_enc], axis = 1)

    del p1_train_int, p1_train_enc, p1_test_int, p1_test_enc, test_int, test_enc
    gc.collect()

    x_train = pd.concat([p1_train, x_train_comb], axis = 1)
    del p1_train, x_train_comb
    gc.collect()
    x_test = pd.concat([p1_test, x_test_comb], axis = 1)
    del p1_test, x_test_comb
    gc.collect()
    test = pd.concat([pt1_new, test_comb], axis = 1)
    del pt1_new, test_comb
    gc.collect()
    
    dtrain = xgb.DMatrix(x_train.values, label = y_train.values)
    del x_train, y_train
    gc.collect()
    dvalid  = xgb.DMatrix(x_test.values, label  = y_test.values)
    del x_test, y_test
    gc.collect()
    dtest = xgb.DMatrix(test.values)
    del test
    gc.collect()
    print('Starting to fit')

    booster = xgb.train(params, dtrain, num_boost_round = 10000, early_stopping_rounds = 100, evals = [(dtrain, 'train'), (dvalid, 'eval')], verbose_eval = 100)

    oof[test_idx] = booster.predict(dvalid)
    
    preds += booster.predict(dtest) / kf.n_splits
    print(i)
    i += 1
    booster = None

    del dtrain, dvalid, dtest, booster
    gc.collect()
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sub.Listening_Time_minutes = preds
sub.to_csv('submission.csv', index = False)
print(np.sqrt(mean_squared_error(target, oof)))

