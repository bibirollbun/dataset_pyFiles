import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from numpy import NaN
import sys

def to_lower(s):
    if type(s) == str:
        return s.lower()
    else:
        return " "

def compare_df(df1, df2):
    ret_list = []
    df_list_1 = list(df1)
    df_list_2 = list(df2)
    for i in range(len(df_list_1)):
        if df_list_1[i] != df_list_2[i]:
            ret_list.append(df_list_1[i])
    return ret_list
    

def make_feature(df):
    
    mlb = MultiLabelBinarizer()
    lbl = LabelEncoder()
    
    # Удалим бесполезные колонки
    df.drop(columns=['Unnamed: 0', 'ФИО'], inplace=True)
    
    # Колонку предмет преобразуем в 0 и 1 по предметам (у нас всего два предмета)
    df = pd.get_dummies(df, columns=['предмет'])
    #df['предмет_informatika'] = df['предмет_informatika'].astype(int)
    #df['предмет_matematika'] = df['предмет_matematika'].astype(int)
    # Добавим колонку с общим количеством направлений 
    df['tutor_head_tags_len'] = df['tutor_head_tags'].apply(lambda x: len(literal_eval(x)))
    # Разобъём по направлениям
    df['tutor_head_tags'] = df['tutor_head_tags'].apply(lambda x: list(literal_eval(x)))
    tutor_head = mlb.fit_transform(df['tutor_head_tags'])
    tutor_df = pd.DataFrame(tutor_head, columns=mlb.classes_)
    df = pd.concat([df, tutor_df], axis=1)
    # Уберём все NAN из рейтинга
    df['tutor_rating'] = df['tutor_rating'].fillna(0)
    # Посчитаем опыт
    df['experience'] = df['experience'].str.replace(r"[^\d\.]", "", regex=True).astype('float64')
    df['experience'] = df['experience'].fillna(df['experience'].mean())
    # Посчитаем категории
    df['cat_num'] = df['categories'].apply(lambda x: len(literal_eval(x)))
    df['categories'] = df['categories'].apply(lambda x: list(literal_eval(x)))
    cat_encoded = mlb.fit_transform(df['categories'])
    cat_df = pd.DataFrame(cat_encoded, columns=mlb.classes_)
    df = pd.concat([df, cat_df], axis=1)
    # Посчитаем количество образований
    df['Education_1'] = df['Education_1'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['Education_2'] = df['Education_2'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['Education_3'] = df['Education_3'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['Education_4'] = df['Education_4'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['Education_5'] = df['Education_5'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['Education_6'] = df['Education_6'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['edu_num'] = df[['Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5', 'Education_6']].sum(axis=1)
    df.drop(columns=['Education_1', 'Education_2', 'Education_3', 'Education_4', 'Education_5', 'Education_6'], inplace=True)
    # Посчитаем учёные степени
    df['sci_1'] = df['Ученая степень 1'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['sci_2'] = df['Ученая степень 2'].apply(lambda s: 0 if pd.isnull(s) or pd.isna(s) else 1)
    df['sci_num'] = df[['sci_1', 'sci_2']].sum(axis=1)
    df['Ученая степень 1'] = df['Ученая степень 1'].apply(lambda s: to_lower(s))
    #df['sci_names'] = df['Ученая степень 1'].apply(lambda s: list(literal_eval(s)))
    #print(df['Ученая степень 1'].value_counts())
    lbl.fit(df['Ученая степень 1'])
    df['Ученая степень 1'] = lbl.transform(df['Ученая степень 1'])
    
    df.drop(columns=['sci_1','sci_2', 'Desc_Education_6', 'Ученое звание 2'], inplace=True)
    # Возьмём только числовые колонки
    numberic_cols = df.select_dtypes(include='number').columns
    df = df[numberic_cols]
    
    return df

train = pd.read_excel("/kaggle/input/salary/train.xlsx")
test = pd.read_excel("/kaggle/input/salary/test.xlsx")
#sample_submit = pd.read_csv("//sample_submit.csv")

print(train.info())

print("Train size: (%s,%s)" % (train.shape[0], train.shape[1]))
print("Test size: (%s,%s)" % (test.shape[0], test.shape[1]))
#print("Sample submit size: (%s,%s)" % (sample_submit.shape[0],sample_submit.shape[1]))
train = make_feature(train)
test = make_feature(test)

test_cols = list(test)
train_cols = list(train.drop(columns=['mean_price']))

for t in train_cols:
    if not t in test_cols:
        print(t)
        test.insert(train.columns.get_loc(t), t, 0)

for t in test_cols:
    if not t in train_cols:
        train.insert(test.columns.get_loc(t), t, 0)

clean_train = train.drop(columns=['mean_price'])
replace_list = compare_df(test, clean_train)
if len(replace_list) > 0:
    print(replace_list)
    df_train = clean_train[replace_list]
    df_test = test[replace_list]
    clean_train.drop(columns=replace_list, inplace=True)
    test.drop(columns=replace_list, inplace=True)
    clean_train = pd.concat([clean_train, df_train, train['mean_price']], axis=1)
    test = pd.concat([test, df_test], axis=1)
    rep_list = compare_df(test, clean_train)
    if len(rep_list):
        print(rep_list)
    else:
        print('OK')
    train = clean_train
    
        #print("%s. %s" % (i, test_cols[i]))

#sys.exit(0)

print(train.info())

X = train.drop(columns=['mean_price'])
Y = train['mean_price']

print("Total X size: %s" % len(X))
print("Total Y size: %s" % len(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
print("X train size: %s" % len(X_train))
print("Y train size: %s" % len(y_train))

#print("Train size: x=(%s,%s) y=(%s,%s)"%(X_train.shape[0], X_train.shape[1], y_train.shape[0], y_train.shape[1]))
#print("Test size: x=(%s,%s) y=(%s,%s)"%(X_test.shape[0], X_test.shape[1], y_test.shape[0], y_test.shape[1]))

lig_reg = LinearRegression()
lig_reg.fit(X_train, y_train)

y_pred = lig_reg.predict(X_test)
print('MAE: %s' % mean_absolute_error(y_test, y_pred))
print('MSE: %s' % mean_squared_error(y_test, y_pred))
print('R2 score: %s' % r2_score(y_test, y_pred))
scores = cross_val_score(lig_reg, X, Y, cv=5, scoring='neg_mean_squared_error')
print("Final scores: %s mean: %s" % (scores, scores.mean()))
y_test = list(y_test)
y_pred = list(y_pred)
plt.plot(y_test, color='blue')
plt.plot(y_pred, color='red')
plt.show()



print(test.info())
y_pred = lig_reg.predict(test)
#print(y_pred)

result_df = pd.DataFrame(y_pred, columns=["mean_price"])
print(result_df)
#result_df.to_csv('result.csv', index=True, index_label='index', sep=',')

