def feature_eng(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    df['Episode_Title'] = df['Episode_Title'].str[8:].astype('category')

    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)

    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    return df

def target_encode(train, valid, test, col, target='Listening_Time_minutes', kfold=5, smooth=20, agg='mean'):
    train['kfold'] = ((train.index) % kfold)
    col_name = '_'.join(col)
    train[f'TE_{agg.upper()}_' + col_name] = 0.
    for i in range(kfold):
        df_tmp = train[train['kfold'] != i]
        if agg == 'mean': mn = train[target].mean()
        elif agg == 'median': mn = train[target].median()
        elif agg == 'min': mn = train[target].min()
        elif agg == 'max': mn = train[target].max()
        elif agg == 'nunique': mn = 0
        df_tmp = df_tmp[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
        df_tmp.columns = col + [agg, 'count']
        if agg == 'nunique':
            df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
        else:
            df_tmp['TE_tmp'] = ((df_tmp[agg] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
        df_tmp_m = train[col + ['kfold', f'TE_{agg.upper()}_' + col_name]].merge(df_tmp, how='left', left_on=col, right_on=col)
        df_tmp_m.loc[df_tmp_m['kfold'] == i, f'TE_{agg.upper()}_' + col_name] = df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_tmp']
        train[f'TE_{agg.upper()}_' + col_name] = df_tmp_m[f'TE_{agg.upper()}_' + col_name].fillna(mn).values

    df_tmp = train[col + [target]].groupby(col).agg([agg, 'count']).reset_index()
    if agg == 'mean': mn = train[target].mean()
    elif agg == 'median': mn = train[target].median()
    elif agg == 'min': mn = train[target].min()
    elif agg == 'max': mn = train[target].max()
    elif agg == 'nunique': mn = 0
    df_tmp.columns = col + [agg, 'count']
    if agg == 'nunique':
        df_tmp['TE_tmp'] = df_tmp[agg] / df_tmp['count']
    else:
        df_tmp['TE_tmp'] = ((df_tmp[agg] * df_tmp['count']) + (mn * smooth)) / (df_tmp['count'] + smooth)
    df_tmp_m = valid[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    valid[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    valid[f'TE_{agg.upper()}_' + col_name] = valid[f'TE_{agg.upper()}_' + col_name].astype('float32')

    df_tmp_m = test[col].merge(df_tmp, how='left', left_on=col, right_on=col)
    test[f'TE_{agg.upper()}_' + col_name] = df_tmp_m['TE_tmp'].fillna(mn).values
    test[f'TE_{agg.upper()}_' + col_name] = test[f'TE_{agg.upper()}_' + col_name].astype('float32')

    train = train.drop('kfold', axis=1)
    train[f'TE_{agg.upper()}_' + col_name] = train[f'TE_{agg.upper()}_' + col_name].astype('float32')

    return (train, valid, test)

def count_encode(train, valid, test, col):
    counts = train[col].value_counts()

    train[f'CE_{col}'] = train[col].map(counts)
    valid[f'CE_{col}'] = valid[col].map(counts).fillna(0)
    test[f'CE_{col}'] = test[col].map(counts).fillna(0)
    return (train, valid, test)

train = pd.read_csv('/content/kaggle/train.csv')

X = train.drop(['Listening_Time_minutes', 'id'], axis=1)
y = train['Listening_Time_minutes']

X_test = pd.read_csv('/content/kaggle/test.csv')
X_test = X_test.drop('id', axis=1)

X = feature_eng(X)
X_test = feature_eng(X_test)

for i in tqdm(range(len(added_cols))):
    cols = added_cols[i]
    name = ''
    for col in cols:
        name += col + '_'
    name = name[:-1]

    X[name] = ''
    X_test[name] = ''
    for col in cols:
        X[name] += X[col].astype(str) + '_'
        X_test[name] += X_test[col].astype(str) + '_'
    X[name] = X[name].str[:-1]
    X_test[name] = X_test[name].str[:-1]

    combined = pd.concat([X[name], X_test[name]], copy=False)
    combined, _ = combined.factorize()
    X[name] = combined[:len(X)]
    X_test[name] = combined[len(X):]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    new_col_mean = np.zeros(len(X))
    new_col_median = np.zeros(len(X))
    new_col_nunique = np.zeros(len(X))
    new_col_count = np.zeros(len(X))

    for train_idx, val_idx in kf.split(np.zeros(len(X))):
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val = X.loc[val_idx]

        X_train, X_val, X_test = target_encode(pd.concat([X_train, y_train], axis=1), X_val, X_test, [name], smooth=10, agg='mean')
        X_train, X_val, X_test = target_encode(X_train, X_val, X_test, [name], smooth=0, agg='median')
        X_train, X_val, X_test = target_encode(X_train, X_val, X_test, [name], smooth=0, agg='nunique')
        X_train = X_train.drop('Listening_Time_minutes', axis=1)
        X_train, X_val, X_test = count_encode(X_train, X_val, X_test, name)

        new_col_mean[val_idx] = X_val['TE_MEAN_' + name]
        new_col_median[val_idx] = X_val['TE_MEDIAN_' + name]
        new_col_nunique[val_idx] = X_val['TE_NUNIQUE_' + name]
        new_col_count[val_idx] = X_val['CE_' + name]

        del X_train, y_train, X_val
        gc.collect()

    X['TE_MEAN_' + name] = new_col_mean
    X['TE_MEDIAN_' + name] = new_col_median
    X['TE_NUNIQUE_' + name] = new_col_nunique
    X['CE_' + name] = new_col_count

    del new_col_mean, new_col_median, new_col_nunique, new_col_count, combined
    gc.collect()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
pred = np.zeros(len(X_test))

for i, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(X)))):
    print(f'Fold {i + 1}:')

    X_train_fold = X.loc[train_idx]
    y_train_fold = y.loc[train_idx]
    X_val_fold = X.loc[val_idx]
    y_val_fold = y.loc[val_idx]

    param_grid = {'colsample_bytree': 0.5937663157166836,
                  'subsample': 0.9760914347380399,
                  'learning_rate': 0.013811801433276795,
                  'reg_lambda': 1.5713105422653082,
                  'reg_alpha': 3.997449754566204,
                  'max_depth': 10,
                  'gamma': 0.08875673638989912}
    model = XGBRegressor(**param_grid, n_estimators=10000, enable_categorical=True, eval_metric='rmse', early_stopping_rounds=300, random_state=42, device='cuda')
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=100)

    oof[val_idx] = model.predict(X_val_fold)
    pred += model.predict(X_test)

    score = root_mean_squared_error(model.predict(X_val_fold), y_val_fold)
    print(f'Fold {i + 1} RMSE: {score}')

    del model, X_train_fold, y_train_fold, X_val_fold, y_val_fold
    gc.collect()

pred /= 5

