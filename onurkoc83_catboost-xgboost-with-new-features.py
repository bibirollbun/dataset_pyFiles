import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train['Sex'] = train.Sex.map({'male':1,'female':0})
test['Sex'] = test.Sex.map({'male':1,'female':0})
train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = train.drop_duplicates(subset=train.columns, keep='first').reset_index(drop=True)
train = train.groupby(by=cols)['Calories'].min().reset_index()
train.shape


# Her bir Duration değeri için yeni feature'lar oluşturalım
# Önce benzersiz Duration değerlerini bulalım
unique_durations_train = train['Duration'].unique()
unique_durations_test = test['Duration'].unique()

# Train için her bir Duration değeri için yeni feature'lar oluşturalım
for duration in unique_durations_train:
    # Yeni sütun isimleri oluşturalım
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col = f'Body_Temp_Duration_{int(duration)}'
    
    # Yeni sütunları oluşturalım
    # Eğer Duration değeri belirli bir değere eşitse Heart_Rate ve Body_Temp değerlerini al, değilse 0 yap
    train[heart_rate_col] = np.where(train['Duration'] == duration, train['Heart_Rate'], 0)
    train[body_temp_col] = np.where(train['Duration'] == duration, train['Body_Temp'], 0)

# Test için her bir Duration değeri için yeni feature'lar oluşturalım
for duration in unique_durations_test:
    # Yeni sütun isimleri oluşturalım
    heart_rate_col = f'Heart_Rate_Duration_{int(duration)}'
    body_temp_col = f'Body_Temp_Duration_{int(duration)}'
    
    # Yeni sütunları oluşturalım
    # Eğer Duration değeri belirli bir değere eşitse Heart_Rate ve Body_Temp değerlerini al, değilse 0 yap
    test[heart_rate_col] = np.where(test['Duration'] == duration, test['Heart_Rate'], 0)
    test[body_temp_col] = np.where(test['Duration'] == duration, test['Body_Temp'], 0)



# Her bir Age değeri için yeni feature'lar oluşturalım
# Önce benzersiz Age değerlerini bulalım
unique_ages_train = train['Age'].unique()
unique_ages_test = test['Age'].unique()

# Train için her bir Age değeri için yeni feature'lar oluşturalım
for age in unique_ages_train:
    # Yeni sütun isimleri oluşturalım
    heart_rate_col = f'Heart_Rate_Age_{int(age)}'
    body_temp_col = f'Body_Temp_Age_{int(age)}'
    
    # Yeni sütunları oluşturalım
    # Eğer Age değeri belirli bir değere eşitse Heart_Rate ve Body_Temp değerlerini al, değilse 0 yap
    train[heart_rate_col] = np.where(train['Age'] == age, train['Heart_Rate'], 0)
    train[body_temp_col] = np.where(train['Age'] == age, train['Body_Temp'], 0)

# Test için her bir Age değeri için yeni feature'lar oluşturalım
for age in unique_ages_test:
    # Yeni sütun isimleri oluşturalım
    heart_rate_col = f'Heart_Rate_Age_{int(age)}'
    body_temp_col = f'Body_Temp_Age_{int(age)}'
    
    # Yeni sütunları oluşturalım
    # Eğer Age değeri belirli bir değere eşitse Heart_Rate ve Body_Temp değerlerini al, değilse 0 yap
    test[heart_rate_col] = np.where(test['Age'] == age, test['Heart_Rate'], 0)
    test[body_temp_col] = np.where(test['Age'] == age, test['Body_Temp'], 0)


def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    # İki liste arasındaki ikili çarpımlar
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

list1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list2 = ['Sex']

train = add_feature_cross_terms(train, list1, list2)
test = add_feature_cross_terms(test, list1, list2)


# Sex değerlerini tersine çevirme (0 -> 1, 1 -> 0)
train['Sex_Reversed'] = 1 - train['Sex']
test['Sex_Reversed'] = 1 - test['Sex']

def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    # İki liste arasındaki ikili çarpımlar
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

list1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list2 = ['Sex', 'Sex_Reversed']

train = add_feature_cross_terms(train, list1, list2)
test = add_feature_cross_terms(test, list1, list2)
train.drop(columns=['Sex_Reversed'],inplace=True)
test.drop(columns=['Sex_Reversed'],inplace=True)



train.columns


def add_categorical_aggregations(df):
    # Kategorik sütunlar
    categorical_cols = ['Sex']
    # Sayısal sütunlar
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']
    
    # Tüm kategorik sütun kombinasyonları için döngü
    for i in range(1, len(categorical_cols) + 1):
        if i == 1:  # Tek kategorik sütun
            for cat_col in categorical_cols:
                # Her sayısal sütun için agregasyonlar
                aggs = df.groupby(cat_col).agg({
                    num_col: ['min', 'max'] for num_col in numerical_cols
                })
                
                # Sütun isimlerini düzleştir
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                
                # Ana veri çerçevesiyle birleştir
                df = df.merge(aggs, on=cat_col, how='left')
        
        elif i == 2:  # İkili kategorik sütun kombinasyonları
            for j in range(len(categorical_cols)):
                for k in range(j+1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    
                    # Her sayısal sütun için agregasyonlar
                    aggs = df.groupby([cat_col1, cat_col2]).agg({
                        num_col: ['min', 'max'] for num_col in numerical_cols
                    })
                    
                    # Sütun isimlerini düzleştir
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    
                    # Ana veri çerçevesiyle birleştir
                    df = df.merge(aggs, on=[cat_col1, cat_col2], how='left')
        
        elif i == 3:  # Üçlü kategorik sütun kombinasyonu
            # Tüm kategorik sütunlar için agregasyonlar
            aggs = df.groupby(categorical_cols).agg({
                num_col: ['min', 'max'] for num_col in numerical_cols
            })
            
            # Sütun isimlerini düzleştir
            aggs.columns = [f"all_cat_{num_col}_{agg}" for num_col, agg in aggs.columns]
            
            # Ana veri çerçevesiyle birleştir
            df = df.merge(aggs, on=categorical_cols, how='left')
    
    return df

# Kullanım örneği:
train = add_categorical_aggregations(train)
test = add_categorical_aggregations(test)


train.columns



cat_features = ['Sex']
for col in cat_features:
    train[col] = train[col].astype('int32').astype('category')
    test[col] = test[col].astype('int32').astype('category')




train.shape


# Sütun sırası aynı mı kontrol edelim
columns_match = train.columns.equals(test.columns.append(pd.Index(['Calories'])))
print(f"Sütun sırası aynı mı: {columns_match}")

# Eğer sütun sırası aynı değilse, düzeltelim
if not columns_match:
    # Calories sütununu düşürelim
    train_without_calories = train.drop(columns=['Calories'])
    
    # Test sütunlarının sırasını alıp train'e uygulayalım
    common_columns = [col for col in test.columns if col in train_without_calories.columns]
    
    # Yeni sırayla train ve test dataframe'lerini yeniden oluşturalım
    train_without_calories = train_without_calories[common_columns]
    test = test[common_columns]
    
    # Calories'i tekrar ekleyelim
    train = pd.concat([train_without_calories, train['Calories']], axis=1)
    
    print("Sütun sırası düzeltildi.")

# Tekrar kontrol edelim
train_without_calories = train.drop(columns=['Calories'])
columns_match_after_drop = train_without_calories.columns.equals(test.columns)
print(f"Calories düştükten sonra sütun sırası aynı mı: {columns_match_after_drop}")


train.shape


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer

# RMSLE metriği için fonksiyon
def rmsle(y_true, y_pred):
    # Negatif değerleri 0'a eşitle
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# X ve y oluştur
X = train.drop(['Calories'], axis=1)

# Hedef değişken olarak Calories'i kullanacağız
y = np.log1p(train['Calories'])

# Duration'ı kategorik hale getir (stratified fold için)
n_bins = 10
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
duration_binned = discretizer.fit_transform(train[['Duration']]).astype(int).flatten()

params = {    
    'iterations': 3000,
    'learning_rate': 0.02,
    'depth': 12,
    #'bootstrap_type':'Bernoulli',
    #'grow_policy':'Lossguide',
    #'boosting_type':'Plain',
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'cat_features': cat_features,
    'verbose': 100,
    'task_type': 'GPU',  # GPU kullanımını etkinleştir
    #'devices': '0',      # Kullanılacak GPU cihazı (0, 1, vs.)

}

# Duration'a göre Stratified KFold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
test_predictions = np.zeros(len(test))
oof_predictions = np.zeros(len(train))  # Out-of-fold tahminleri için dizi

for fold, (train_idx, val_idx) in enumerate(skf.split(X, duration_binned)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(
      **params
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        #cat_features=cat_features,
        use_best_model=True,
        verbose=1000
    )
    
    # Modelin tahminleri (log1p dönüşümü yapılmış Calories değerleri)
    val_pred = model.predict(X_val)
    
    # Tahminleri orijinal ölçeğe geri dönüştür
    val_pred_original = np.expm1(val_pred)
    
    # OOF tahminlerini kaydet
    oof_predictions[val_idx] = val_pred_original
    
    # Gerçek Calories değerleri
    y_val_calories = train.iloc[val_idx]['Calories']
    
    # RMSLE hesapla
    fold_score = rmsle(y_val_calories, val_pred_original)
    scores.append(fold_score)
    
    print(f"Fold {fold+1} - RMSLE: {fold_score:.5f}")
    
    # Test tahmini
    test_fold_preds = np.expm1(model.predict(test))
    test_predictions += test_fold_preds / skf.n_splits

# OOF tahminlerini kaydet
#np.save('oof_predictions_cat.npy', oof_predictions)

# Ortalama ve standart sapma skorları
mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"\nOrtalama RMSLE: {mean_score:.5f}")
print(f"Standart Sapma RMSLE: {std_score:.5f}")



sub_df=pd.DataFrame()

sub_df['id']=submission['id']
sub_df['Calories_cat']=np.clip(test_predictions, 1, 314)


#submission['Calories'] = np.clip(test_predictions, 1, 314)
#submission.to_csv('submission_cat.csv', index=False)


sub_df['Calories_cat'].median()


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
train['Sex'] = train.Sex.map({'male':1,'female':0})
test['Sex'] = test.Sex.map({'male':1,'female':0})
train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)
cols = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train = train.drop_duplicates(subset=train.columns, keep='first').reset_index(drop=True)
train = train.groupby(by=cols)['Calories'].max().reset_index()
train.shape


# Feature Engineering
# Reverse Sex values (0 -> 1, 1 -> 0) to create interaction features for both genders
train['Sex_Reversed'] = 1 - train['Sex']
test['Sex_Reversed'] = 1 - test['Sex']

def add_feature_cross_terms(df, list1, list2):
    df_new = df.copy()
    # Create cross-product terms between two feature lists
    for feature1 in list1:
        for feature2 in list2:
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

# Define features for interaction
list1 = ['Duration', 'Heart_Rate', 'Body_Temp']
list2 = ['Sex', 'Sex_Reversed']

# Apply feature interactions to both train and test sets
train = add_feature_cross_terms(train, list1, list2)
test = add_feature_cross_terms(test, list1, list2)

# Remove temporary features after creating interactions
train.drop(columns=['Sex_Reversed'],inplace=True)
test.drop(columns=['Sex_Reversed'],inplace=True)



def add_categorical_aggregations(df):
    # Kategorik sütunlar
    categorical_cols = ['Sex']
    # Sayısal sütunlar
    numerical_cols = ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']
    
    # Tüm kategorik sütun kombinasyonları için döngü
    for i in range(1, len(categorical_cols) + 1):
        if i == 1:  # Tek kategorik sütun
            for cat_col in categorical_cols:
                # Her sayısal sütun için agregasyonlar
                aggs = df.groupby(cat_col).agg({
                    num_col: ['min', 'max'] for num_col in numerical_cols
                })
                
                # Sütun isimlerini düzleştir
                aggs.columns = [f"{cat_col}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                
                # Ana veri çerçevesiyle birleştir
                df = df.merge(aggs, on=cat_col, how='left')
        
        elif i == 2:  # İkili kategorik sütun kombinasyonları
            for j in range(len(categorical_cols)):
                for k in range(j+1, len(categorical_cols)):
                    cat_col1 = categorical_cols[j]
                    cat_col2 = categorical_cols[k]
                    
                    # Her sayısal sütun için agregasyonlar
                    aggs = df.groupby([cat_col1, cat_col2]).agg({
                        num_col: ['min', 'max'] for num_col in numerical_cols
                    })
                    
                    # Sütun isimlerini düzleştir
                    aggs.columns = [f"{cat_col1}_{cat_col2}_{num_col}_{agg}" for num_col, agg in aggs.columns]
                    
                    # Ana veri çerçevesiyle birleştir
                    df = df.merge(aggs, on=[cat_col1, cat_col2], how='left')
        
        elif i == 3:  # Üçlü kategorik sütun kombinasyonu
            # Tüm kategorik sütunlar için agregasyonlar
            aggs = df.groupby(categorical_cols).agg({
                num_col: ['min', 'max'] for num_col in numerical_cols
            })
            
            # Sütun isimlerini düzleştir
            aggs.columns = [f"all_cat_{num_col}_{agg}" for num_col, agg in aggs.columns]
            
            # Ana veri çerçevesiyle birleştir
            df = df.merge(aggs, on=categorical_cols, how='left')
    
    return df

# Kullanım örneği:
train = add_categorical_aggregations(train)
test = add_categorical_aggregations(test)


train.drop(columns=['Sex'],inplace=True)
test.drop(columns=['Sex'],inplace=True)



# Sütun sırası aynı mı kontrol edelim
columns_match = train.columns.equals(test.columns.append(pd.Index(['Calories'])))
print(f"Sütun sırası aynı mı: {columns_match}")

# Eğer sütun sırası aynı değilse, düzeltelim
if not columns_match:
    # Calories sütununu düşürelim
    train_without_calories = train.drop(columns=['Calories'])
    
    # Test sütunlarının sırasını alıp train'e uygulayalım
    common_columns = [col for col in test.columns if col in train_without_calories.columns]
    
    # Yeni sırayla train ve test dataframe'lerini yeniden oluşturalım
    train_without_calories = train_without_calories[common_columns]
    test = test[common_columns]
    
    # Calories'i tekrar ekleyelim
    train = pd.concat([train_without_calories, train['Calories']], axis=1)
    
    print("Sütun sırası düzeltildi.")

# Tekrar kontrol edelim
train_without_calories = train.drop(columns=['Calories'])
columns_match_after_drop = train_without_calories.columns.equals(test.columns)
print(f"Calories düştükten sonra sütun sırası aynı mı: {columns_match_after_drop}")


train.shape


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# RMSLE metriği için fonksiyon
def rmsle(y_true, y_pred):
    # Negatif değerleri 0'a eşitle
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# X ve y oluştur
X = train.drop(['Calories'], axis=1)
num = 4
# Hedef değişken olarak Calories'i kullanacağız
y = np.log1p(train['Calories'])

params = {    
    'max_depth': 9,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'gamma': 0.01,
    'max_delta_step': 2,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'random_state': 42,
    'early_stopping_rounds': 100,
    'tree_method': 'gpu_hist'  # GPU hızlandırma için gpu_hist kullanıyoruz
}

# KFold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
test_predictions = np.zeros(len(test))
oof_predictions = np.zeros(len(X))  # Out-of-fold tahminleri için dizi

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Modelin tahminleri (log1p dönüşümü yapılmış Calories değerleri)
    val_pred = model.predict(X_val)
    
    # Tahminleri orijinal ölçeğe geri dönüştür
    val_pred_original = np.expm1(val_pred)
    
    # OOF tahminlerini kaydet
    oof_predictions[val_idx] = val_pred_original
    
    # Gerçek Calories değerleri
    y_val_calories = train.iloc[val_idx]['Calories']
    
    # RMSLE hesapla
    fold_score = rmsle(y_val_calories, val_pred_original)
    scores.append(fold_score)
    
    print(f"Fold {fold+1} - RMSLE: {fold_score:.5f}")
    
    # Test tahmini
    test_fold_preds = np.expm1(model.predict(test))
    test_predictions += test_fold_preds / kf.n_splits

# OOF tahminlerini .npy dosyası olarak kaydet
#np.save('oof_predictions_xgb.npy', oof_predictions)

# Ortalama ve standart sapma skorları
mean_score = np.mean(scores)
std_score = np.std(scores)
print(f"\nOrtalama RMSLE: {mean_score:.5f}")
print(f"Standart Sapma RMSLE: {std_score:.5f}")


sub_df['Calories_xgb']=np.clip(test_predictions, 1, 314)

#submission['Calories'] = np.clip(test_predictions, 1, 314)
#submission.to_csv('submission_xgb.csv', index=False)


sub_df['Calories_xgb'].median()


sub_df['Calories'] = (sub_df['Calories_cat']+sub_df['Calories_xgb'])/2


sub_df[['id','Calories']].to_csv('submission.csv', index=False)

