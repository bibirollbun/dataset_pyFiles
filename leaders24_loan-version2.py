import pandas as pd

# 读取数据集
df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))


# 划分特征变量和目标变量
train =df.drop(['id', 'loan_paid_back'], axis=1)
y =df['loan_paid_back']


from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 对 object 类型数据进行编码
categorical_cols = train.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
df = encoder.fit(train[categorical_cols])




numeric_cols=train.select_dtypes(exclude=['object']).columns


train_encoded=pd.DataFrame(
    encoder.transform(train[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=train.index
)



test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


test_encoder=pd.DataFrame(
    encoder.transform(test_df[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols),
    index=test_df.index
)


train=pd.concat([train[numeric_cols],train_encoded],axis=1)
test=pd.concat([test_df[numeric_cols],test_encoder],axis=1)


train


test


train.isna().sum()
test.isna().sum()


# 对特征变量进行标准化
scaler = StandardScaler()
train[numeric_cols]=scaler.fit_transform(train[numeric_cols])
test[numeric_cols]=scaler.transform(test[numeric_cols])


train


# 划分训练集和测试集
X=train
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y)


# 构建逻辑回归模型
model=LogisticRegression(
    max_iter=5000,
    penalty='l2',
    C=1e-3,
)
model.fit(X_train,y_train)


# 预测概率
y_predict=model.predict_proba(X_val)[:,1]


# 将预测概率转换为0,1
thresold=0.8
y_pre=(y_predict>=thresold).astype(int)


from sklearn.metrics import roc_curve,auc,RocCurveDisplay
import matplotlib.pyplot as plt


fpr,tpr,thresholds=roc_curve(y_val,y_pre)
roc_auc=auc(fpr,tpr)


plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


model.fit(X,y)


y_prob=model.predict_proba(test)[:,1]
prediction=(y_prob>=thresold).astype(int)
test_id=test_df['id']
submission=pd.DataFrame({
    'id':test_id,
    'loan_paid_back':prediction
})
submission.to_csv('/kaggle/working/sub.csv',index=False)


submission.head()




