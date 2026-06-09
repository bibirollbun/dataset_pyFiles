import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score,f1_score,classification_report, precision_score, recall_score, log_loss, roc_auc_score, confusion_matrix
from collections import defaultdict, Counter
import time

#XGBoost
import xgboost as xgb
from xgboost import XGBClassifier,  plot_importance



train_csv_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_csv_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original_csv_df = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


train_df = train_csv_df.copy()
test_df = test_csv_df.copy()
original_df = original_csv_df.copy()

train_df = train_df.drop(columns=['id'])


# 定数フィーチャの追加。モデルがうまく動くように「土台」として、すべての行で値が同じ列を特徴追加する
train_df['const_feat'] = 1
test_df['const_feat'] = 1
original_df['const_feat'] = 1


#外部データ追加
original_copy = original_df.copy()

for _ in range(6):
    original_df = pd.concat([original_df, original_copy], axis=0, ignore_index=True)
    
#確認用表示
print(len(train_df))
print(len(original_df))


#ラベルエンコーダでカテゴリ列を整数変換
le_Fertilizer = LabelEncoder()
le_Fertilizer.fit(train_df['Fertilizer Name'])
train_df['Fertilizer Name'] = le_Fertilizer.fit_transform(train_df['Fertilizer Name'])

original_df['Fertilizer Name'] = le_Fertilizer.fit_transform(original_df['Fertilizer Name'])

#確認用表示
for i, class_label in enumerate(le_Fertilizer.classes_):
    print(f"{i}:{class_label}")


# ワンホットエンコーディングでカテゴリをバイナリベクトルへ変換
train_df = pd.get_dummies(
    train_df, 
    columns=['Soil Type','Crop Type'], 
    drop_first=False
)
test_df = pd.get_dummies(
    test_df, 
    columns=['Soil Type','Crop Type'], 
    drop_first=False
)
original_df = pd.get_dummies(
    original_df, 
    columns=['Soil Type','Crop Type'], 
    drop_first=False
)


#int64とboolをint8へ変換しメモリ使用量削減する
for col in test_df.select_dtypes(include=['int64','bool']).columns:
    if col != 'id':
        test_df[col] = test_df[col].astype('int8')
        train_df[col] = train_df[col].astype('int8')
        original_df[col] = original_df[col].astype('int8')

train_df['Fertilizer Name'] = train_df['Fertilizer Name'].astype('int8')
original_df['Fertilizer Name'] = original_df['Fertilizer Name'].astype('int8')

#id列はint8では繰り上がってしまうため、int32に変換
test_df['id'] = test_df['id'].astype('int32')


def map_at_3(y_true, y_pred_proba, k=3):
    """
    モデルが予測した上位3つの候補の中に正解があるか評価する指標。
    ・正解が1位ならスコアは1
    ・2位なら0.5
    ・3位なら約0.33
    上記ポイントが与えられ、全サンプルで平均した値がスコアになり、
    「正解がどれくらい上位にきているか」を評価しています。
    """

    map_score = 0.0
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    
    for i in range(len(y_true)):
        top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]#予測されたラベル上位3つ
        
        if y_true[i] in top_k_preds:
            rank = np.where(top_k_preds == y_true[i])[0][0] + 1#正解ラベル
            map_score += 1.0 / rank #正解ラベルが予測3位までに存在したら順によってスコアをつける
            
    return map_score / len(y_true)


#処理時間
start = time.time()


#特徴量と目的変数
X = train_df.drop(columns=['Fertilizer Name']) 
y = train_df['Fertilizer Name'] 
print(sorted(y.unique()))


#各クロスバリエーションの結果格納先
all_y_valid = []
all_y_pred = []
all_y_pred_proba = []
fold_aucs = []


#クラス数を定義
num_classes = train_df['Fertilizer Name'].nunique()


#手動パラメータ
params = {
    # 'tree_method': 'hist',# CPU
    'n_estimators': 20000,
    'objective': 'multi:softprob',#多クラス取り扱い
    'random_state': 10,
    'verbosity': 0,
    'eval_metric': 'mlogloss',#多クラスの対数損失
    'booster': 'gbtree',#勾配ブースティング木設定
    'n_jobs': -1,
    'learning_rate': 0.01,
    'lambda': 0.057,
    'alpha': 5.62,
    'colsample_bytree': 0.26, 
    'subsample': 0.83,
    'max_depth': 16,
    'min_child_weight': 3,
    'tree_method': 'gpu_hist',# GPU使用（学習）
    'predictor': 'gpu_predictor'  # GPU（推論）
}


#クロスバリエーション
kf = StratifiedKFold(
    n_splits=3,
    shuffle=True, #データをシャッフルして分割
    random_state=1
)
models = []  # 各foldのモデルを保存
all_f1_scores = []
class_scores = defaultdict(list)

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]


    # originalデータをtrainに追加
    X_train = pd.concat([X_train, original_df.drop(columns=['Fertilizer Name'])], axis=0)
    y_train = pd.concat([y_train, original_df['Fertilizer Name']], axis=0)
    

    model = XGBClassifier(
        **params,
        use_label_encoder=False,
        early_stopping_rounds=200,
        num_class=num_classes
    )
    
    model.fit(
            X_train, 
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=0,
            # verbose=100,  # 100ステップごとにログ表示
                    )
    models.append(model)
    
    #予測
    y_pred = model.predict(X_valid)
    y_pred_labels = y_pred
    y_pred_proba = model.predict_proba(X_valid)#各クラスの確率
    
    auc = roc_auc_score(y_valid, y_pred_proba, multi_class='ovr')


    # 評価レポート
    report = classification_report(
        y_valid, 
        y_pred_labels, 
        labels=range(len(le_Fertilizer.classes_)),#常に全クラスを報告対象に含める
        target_names=le_Fertilizer.classes_, 
        output_dict=True,
        zero_division=0  # ゼロ割回避
    )

    
    macro_f1 = f1_score(y_valid, y_pred_labels, average='macro')
    all_f1_scores.append(macro_f1)

    for cls in le_Fertilizer.classes_:
        class_scores[cls].append(report[cls]['f1-score'])

    fold_map = map_at_3(y_valid, y_pred_proba)
    print(f"Fold {fold+1} MAP@3: {fold_map:.5f}")
    
    all_y_valid.extend(y_valid)
    all_y_pred.extend(y_pred_labels)
    all_y_pred_proba.extend(y_pred_proba)
    fold_aucs.append(auc)


# 結果表示
print('\n==== クラス別平均 F1 ====')
print('label,平均F1,バラつき')
for cls in le_Fertilizer.classes_:
    scores = class_scores[cls]
    print(f'{cls},{np.mean(scores):.3f},±{np.std(scores):.3f}')

print('\n==== 平均スコア ====')
print(f'平均AUC-ROC: {np.mean(fold_aucs):.4f}（標準偏差: {np.std(fold_aucs):.4f}）')

print()
precision = precision_score(all_y_valid, all_y_pred, average='macro')
print(f'Precision: {precision:.4f}')

recall = recall_score(all_y_valid, all_y_pred, average='macro')
print(f'Recall: {recall:.4f}')

accuracy = accuracy_score(all_y_valid, all_y_pred)
print(f'Accuracy: {accuracy:.4f}')

#確率付きの予測と実際のラベルの差」を評価する指標
log_loss_value = log_loss(all_y_valid, all_y_pred_proba)
print(f'Log Loss: {log_loss_value:.4f}')

auc = roc_auc_score(all_y_valid, all_y_pred_proba, multi_class='ovr')
print(f'AUC-ROC: {auc:.4f}')

print()
print(f'処理時間: {time.time() - start:.2f} 秒')

#ラベルクラスの数値
label_mapping = dict(zip(
                le_Fertilizer.classes_, 
                le_Fertilizer.transform(le_Fertilizer.classes_)
))

# 確認
print(label_mapping)

# 最終 fold（もしくは各foldで）の混同行列
cm = confusion_matrix(all_y_valid, all_y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=le_Fertilizer.classes_,
            yticklabels=le_Fertilizer.classes_
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Fold {})'.format(fold + 1))
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 予測
X_test = test_df.drop(columns=['id'])
test_preds = test_preds = [model.predict_proba(X_test) for model in models]# 予測確率を取得
avg_test_proba = np.mean(test_preds, axis=0)

# 上位3つのインデックス（高い順）
top3_indices = np.argsort(avg_test_proba, axis=1)[:, ::-1][:, :3]

top3_labels = [
    le_Fertilizer.inverse_transform(row)  
    for row in top3_indices
]
# print(top3_labels)
# スペース区切りで連結
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})

print('submission.csv')
# 保存
submission.to_csv('submission.csv', index=False)

