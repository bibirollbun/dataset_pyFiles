#Q. [마케팅] 자동차 시장 세분화
# 자동차 회사는 새로운 전략을 수립하기 위해 4개의 시장으로 세분화했습니다.
# 기존 고객 분류 자료를 바탕으로 신규 고객이 어떤 분류에 속할지 예측해주세요!
# 예측할 값(y): "Segmentation" (1,2,3,4)
# 평가: Macro f1-score
# data: train.csv, test.csv
# 제출 형식:

# ID,Segmentation
# 458989,1
# 458994,2
# 459000,3
# 459003,4

# 데이터 불러오기
import pandas as pd
train = pd.read_csv("../input/big-data-analytics-certification-kr-2022/train.csv")
test = pd.read_csv("../input/big-data-analytics-certification-kr-2022/test.csv")
print(train.shape, test.shape)
print(train.head())



#결측치 확인
test_ID = test['ID'] 
target = train['Segmentation']
train = train.drop(['ID','Segmentation'],axis=1)
test = test.drop('ID',axis=1)


from sklearn.preprocessing import LabelEncoder
cols = train.select_dtypes(include='object').columns

for col in cols:
	le = LabelEncoder()
	train[col] = le.fit_transform(train[col])
	test[col] = le.transform(test[col])

#print(test.head())
#X_tr, y_tr
#X_val, y_val


from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(train, target, test_size=0.1, random_state=42)
#print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0, max_depth=10, n_estimators=700)
model.fit(X_tr,y_tr)
pred = model.predict(X_val)



#Macro f1-score
from sklearn.metrics import f1_score
print('f1_score:', f1_score(y_val,pred,average='macro'))

from sklearn.model_selection import cross_val_score
score = cross_val_score(model, train, target, scoring='f1_macro', cv=5)
#print(scores)
print('cross_val_score:', score.mean())


pred = model.predict(test)
#print(pred)

submit = pd.DataFrame({
	'ID': test_ID,
	'Segmentation' : pred
})

submit.to_csv("submission.csv", index=False)


#0.5030609450063005 max_depth=13, n_estimators=700
#0.5117853671702189 max_depth=10, n_estimators=700
#0.5089662290918752 max_depth=10, n_estimators=2000
#0.508149330968678 max_depth=8, n_estimators=700
#0.5149440475768575 max_depth=10, n_estimators=700 test_size=0.2
#0.5301949277687051 max_depth=10, n_estimators=700 test_size=0.1
#0.5037405025549284 max_depth=25, n_estimators=2000
#0.5037405025549284 max_depth=25, n_estimators=2000 test_size=0.2

#-----------------------------------
# test_size=0.3, max_depth=9, n_estimators=1500
# f1_score: 0.5096778620926838
# cross_val_score: 0.5296543627887574
# test_score : 0.32475 

# test_size=0.1, max_depth=10, n_estimators=700
# f1_score: 0.5301949277687051
# cross_val_score: 0.5264570246438967



