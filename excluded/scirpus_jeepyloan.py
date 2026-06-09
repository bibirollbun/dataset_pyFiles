import os
import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.metrics import roc_auc_score
from scipy.special import expit


envpath = "/kaggle/input/playground-series-s5e11/"
if(not os.path.isfile(envpath+"train.csv")):
    envpath = "./"
print(envpath)


train = pd.read_csv(envpath+'train.csv',index_col="id")
test = pd.read_csv(envpath+'test.csv',index_col="id")



x = train.pop("grade_subgrade")
train.insert(10,"grade_subgrade_n", x.str[1])
train.insert(10,"grade_subgrade_a", x.str[0])
x = test.pop("grade_subgrade")
test.insert(10,"grade_subgrade_n", x.str[1])
test.insert(10,"grade_subgrade_a", x.str[0])


train.columns


test.columns


cats = []
for c in test.columns:
    if(len(train[c].unique())<30):
        cats.append(c)

cats


numerics = list(set(test.columns).difference(set(cats)))
numerics


cb = CatBoostEncoder()
cbtrain = cb.fit_transform(train.loc[:,cats],train.loan_paid_back)
cbtest = cb.transform(test.loc[:,cats])
cbtrain = pd.concat([cbtrain,train.loc[:,numerics]],axis=1)
cbtrain["loan_paid_back"] = train.loan_paid_back.values
cbtest = pd.concat([cbtest,test.loc[:,numerics]],axis=1)



cbtrain.head()


cbtest.head()


cbtrain[["loan_amount",	"credit_score",	"annual_income"]] = np.log(cbtrain[["loan_amount",	"credit_score",	"annual_income"]])
cbtest[["loan_amount",	"credit_score",	"annual_income"]] = np.log(cbtest[["loan_amount",	"credit_score",	"annual_income"]])


def gpI(data):
    o = expit(  1.378933 +
                0.602692*np.tanh((((data["employment_status"]) / (((-1.000000 -  1.0 ) + data["grade_subgrade_a"] * 2.0)) -  1.0 ) + data["marital_status"] * ( 9.0 ) / (data["employment_status"] * 2.0) * 2.0 / 2.0)/2.0  * 2.0 * 2.0 * -1.) +
                0.719304*np.tanh((((data["debt_to_income_ratio"] * (((data["debt_to_income_ratio"] - data["employment_status"] * 2.0) + data["loan_amount"] * data["grade_subgrade_n"] * data["interest_rate"])/2.0  + data["loan_amount"] * 2.0 * 2.0 * 2.0)/2.0  + data["loan_amount"]) + data["loan_amount"] * (data["debt_to_income_ratio"] - data["employment_status"] * 2.0))/2.0 ) / ((0.000000 - data["employment_status"] * 2.0))) +
                0.998406*np.tanh((data["debt_to_income_ratio"] - ( 6.0  - data["employment_status"] * ((data["debt_to_income_ratio"] - ( 6.0  - data["employment_status"] * (data["employment_status"] + (data["credit_score"] - data["debt_to_income_ratio"]))) * -1.) * -1. + (data["employment_status"] * (data["employment_status"] + (data["credit_score"] - data["debt_to_income_ratio"])) - data["debt_to_income_ratio"]))) * -1.) * -1.) +
                0.998016*np.tanh((data["grade_subgrade_a"] - (((-1.000000) / (data["grade_subgrade_a"] * -1. * data["grade_subgrade_a"] * 2.0) - (data["grade_subgrade_a"] +  0.301936  * -1.)) - ((data["debt_to_income_ratio"]) / (data["grade_subgrade_a"] / 2.0) - ((data["debt_to_income_ratio"]) / (data["grade_subgrade_n"] / 2.0)) / (data["employment_status"] / 2.0))))) +
                0.998410*np.tanh(((-1.000000) / ((data["employment_status"]) / (data["gender"])) + (((-1.000000) / (data["employment_status"]) + (((data["debt_to_income_ratio"] * -1. * data["grade_subgrade_a"] + data["education_level"]) - data["marital_status"]) + ((-1.000000 + (data["employment_status"]) / (data["gender"])) + (data["employment_status"]) / (data["gender"])))) + data["employment_status"]))) +
                0.998872*np.tanh(data["debt_to_income_ratio"] * (data["debt_to_income_ratio"]) / (data["employment_status"]) * ((((data["debt_to_income_ratio"]) / (data["employment_status"]) + data["employment_status"]) + data["employment_status"])) / (data["employment_status"]) * 2.0 * (data["debt_to_income_ratio"]) / (data["employment_status"]) * ((data["debt_to_income_ratio"] + data["debt_to_income_ratio"]) + data["debt_to_income_ratio"]) * -1. / 2.0) +
                0.998882*np.tanh((((data["marital_status"]) / (data["employment_status"]) - data["grade_subgrade_a"]) + (((data["grade_subgrade_a"]) / (data["employment_status"])) / (data["employment_status"]) * (data["grade_subgrade_a"]) / (data["employment_status"]) * ((data["grade_subgrade_a"] - data["marital_status"]) + ((data["debt_to_income_ratio"]) / (data["marital_status"]) / 2.0 / 2.0 * -1.) / (data["marital_status"]))/2.0  / 2.0) / (data["debt_to_income_ratio"]))/2.0 ) +
                0.999286*np.tanh((((data["credit_score"] * -1. - data["loan_amount"] / 2.0 / 2.0) / 2.0 / 2.0) / (data["employment_status"]) / 2.0 / 2.0 * -1.) / ((data["loan_amount"] - (data["loan_purpose"]) / ((data["grade_subgrade_a"] - data["credit_score"] * -1. / 2.0 / 2.0 / 2.0 * -1.)) / 2.0)) / 2.0) +
                0.999992*np.tanh((data["grade_subgrade_n"]) / ((((data["debt_to_income_ratio"] - ((data["interest_rate"] - data["debt_to_income_ratio"] * data["debt_to_income_ratio"] *  10.0  * data["debt_to_income_ratio"] *  10.0  *  10.0 ) - (data["interest_rate"]) / (data["debt_to_income_ratio"] *  10.0  * data["marital_status"]))) * 2.0 + data["marital_status"])/2.0 ) / (data["marital_status"]))) +
                0.995594*np.tanh((1.000000) / (1.000000 * (data["credit_score"] + -1.000000)/2.0  / 2.0 * (((( 0.860426 ) / ((-1.000000 + data["employment_status"])) + -1.000000)/2.0 ) / ((data["credit_score"] + (data["credit_score"] + data["loan_amount"] / 2.0)/2.0 )/2.0  * 2.0) * -1. - data["annual_income"]) * data["grade_subgrade_a"] * data["grade_subgrade_a"])) +
                0.998972*np.tanh((data["debt_to_income_ratio"] * -1.) / (((0.000000 + (data["debt_to_income_ratio"] + ((data["loan_amount"]) / ((data["debt_to_income_ratio"]) / (data["employment_status"] / 2.0))) / ((data["debt_to_income_ratio"]) / ((data["employment_status"] + data["employment_status"]))))/2.0 )) / ((data["debt_to_income_ratio"]) / (data["employment_status"] / 2.0)) * 2.0 / 2.0 / 2.0) / 2.0) +
                0.999398*np.tanh(((-1.000000 + data["employment_status"])/2.0 ) / (((-1.000000 + (data["grade_subgrade_a"] + data["employment_status"])/2.0 )/2.0  + data["interest_rate"] * (data["debt_to_income_ratio"] * data["interest_rate"]) / (data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["loan_amount"]))/2.0 ) * -1.) +
                0.999200*np.tanh((data["grade_subgrade_a"]) / ((data["interest_rate"] - ( 9.0  - ((data["annual_income"] / 2.0) / (data["interest_rate"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * -1.) + data["grade_subgrade_a"])/2.0 ) * 2.0) * 2.0)) +
                0.995948*np.tanh((data["loan_purpose"] * -1.) / (data["interest_rate"] * (((data["gender"]) / (data["education_level"] * data["education_level"]) / 2.0) / (data["education_level"] * (data["loan_purpose"] / 2.0) / (data["grade_subgrade_a"]))) / (data["loan_purpose"] * (((data["debt_to_income_ratio"] * 2.0) / (data["grade_subgrade_a"])) / (data["grade_subgrade_a"])) / (data["grade_subgrade_a"])) / 2.0) * -1.) +
                0.998800*np.tanh((((data["gender"] - data["grade_subgrade_a"])) / (data["employment_status"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * 2.0 * -1. * 2.0 * -1.)) / (data["employment_status"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["grade_subgrade_a"] * data["annual_income"] * 2.0 * 2.0) * -1.) +
                0.883800*np.tanh(((data["education_level"] + (data["debt_to_income_ratio"] / 2.0) / ((data["marital_status"] * 2.0) / ((data["grade_subgrade_a"] - (data["education_level"] + (data["debt_to_income_ratio"]) / (data["employment_status"] * 2.0 * 2.0 * -1.)))) / 2.0 * -1.)) + data["marital_status"] * 2.0 * -1. / 2.0)/2.0  * 2.0 * 2.0 * 2.0))
             
    return o


def gpII(data):
    o = expit(  1.378933 +
                0.032600*np.tanh((((6.469239) - 1.000000) - (np.where(((6.469239) - (data["credit_score"] - ((6.362907)) / (data["credit_score"]))) <= data["employment_status"], (data["credit_score"] - (np.where((6.536688) <= ((6.362907)) / (data["employment_status"]), (0.801641), 0 ) - data["employment_status"])), 0 ) - (0.801641))) * -1.) +
                0.043436*np.tanh(np.where((6.626687) > data["credit_score"], (1.000000 - (data["debt_to_income_ratio"]) / (((data["credit_score"] - (6.626687) * ((6.626687) * ((6.626687)) / (data["credit_score"])) / (data["credit_score"] * 2.0))) / ((6.416667) * (6.416667) * ((6.416667)) / (data["credit_score"]))) * 2.0), 0 )) +
                0.050604*np.tanh((((6.626687) - (data["credit_score"]) / (data["employment_status"]))) / ((((1.000000 - data["grade_subgrade_a"] * -1.) / 2.0 + (data["grade_subgrade_a"] - (0.075000) * -1.) * -1.)/2.0  + np.where(data["employment_status"] <= (1.000000 - data["grade_subgrade_a"] * -1.) / 2.0, (6.626687), 0 ) * 2.0)/2.0 )) +
                0.999618*np.tanh(data["debt_to_income_ratio"] * 2.0 * data["debt_to_income_ratio"] * ((((data["debt_to_income_ratio"] + (((data["debt_to_income_ratio"] * 2.0 + (((6.550354)) / (data["employment_status"])) / (data["grade_subgrade_a"]))/2.0 ) / (data["employment_status"] * data["grade_subgrade_a"])) / (data["grade_subgrade_a"]))/2.0 ) / (((0.044500) + data["grade_subgrade_a"]))) / (data["grade_subgrade_a"])) / (data["grade_subgrade_a"]) * -1. * 2.0) +
                0.252060*np.tanh((data["debt_to_income_ratio"] * -1.) / (data["grade_subgrade_a"] * data["grade_subgrade_a"] * ((data["grade_subgrade_a"] + data["employment_status"]) * data["grade_subgrade_a"] + data["grade_subgrade_a"]) * data["grade_subgrade_a"] * data["employment_status"] * (data["grade_subgrade_a"] + data["grade_subgrade_a"]) * data["grade_subgrade_a"] * data["employment_status"] * data["employment_status"] * data["grade_subgrade_a"])) +
                0.068400*np.tanh((((0.091500) * -1.) / ((0.111500) * (1.000000 + ((np.where(data["employment_status"] <= data["employment_status"], (1.000000 + (data["employment_status"] + data["grade_subgrade_a"])/2.0 )/2.0 , 0 ) + data["employment_status"])/2.0  + data["grade_subgrade_a"])/2.0 )/2.0 ) + data["employment_status"]) * 2.0 * 2.0 * 2.0 * 2.0) +
                0.001978*np.tanh((data["employment_status"] - ( 2.0  - ((0.172000) + (data["grade_subgrade_a"] - ((0.058000) - ((0.172000) + data["grade_subgrade_a"] * (data["employment_status"] - ( 1.0  - ((0.172000) + (0.058000) * -1.)/2.0  * 2.0)) * 2.0 * 2.0 * 2.0)/2.0  * 2.0)))/2.0  * 2.0)) * 2.0 * 2.0) +
                0.016800*np.tanh((0.000000 - (np.where(data["employment_status"] <= (0.683061), (6.558901), 0 ) + (np.where(data["employment_status"] <= ((0.847057) + (0.140500)), (((0.847057) - (data["grade_subgrade_a"] * ((0.847057) + (6.542468)) - (0.847057))) + (6.558901)), 0 ) + (0.000000 - ((data["employment_status"] + data["grade_subgrade_a"]) + data["grade_subgrade_a"]))))/2.0 )) +
                0.627600*np.tanh((data["employment_status"] - (((np.where((((6.646964)) / ((((data["credit_score"]) / ((data["employment_status"] + (data["employment_status"] + (data["employment_status"] + (data["employment_status"] + (6.646964)))))) - data["debt_to_income_ratio"]) + data["credit_score"])) - (0.052500)) > data["employment_status"], (0.052500), 0 )) / ((0.052500))) / (data["employment_status"])) / (data["employment_status"]))) +
                0.739640*np.tanh(((data["employment_status"]) / (((0.683061)) / (data["grade_subgrade_a"])) + (((data["debt_to_income_ratio"]) / (data["employment_status"])) / ((0.111500))) / (data["grade_subgrade_a"]) * ((((0.058000) + ((0.683061)) / (data["grade_subgrade_a"]))/2.0  - (((0.683061)) / (data["employment_status"])) / (data["employment_status"])) - ((0.683061)) / (data["employment_status"])) / 2.0)) +
                0.486524*np.tanh(np.where(data["grade_subgrade_a"] > data["employment_status"], (data["grade_subgrade_a"] - ((0.481627)) / (data["employment_status"])) / 2.0 * ((6.513969) / 2.0 + data["grade_subgrade_a"] * -1. * 2.0) * (((6.513969) / 2.0 + data["employment_status"]) + data["grade_subgrade_a"] * -1. * 2.0), 0 )) +
                0.801416*np.tanh(((0.481627) * 2.0 - data["grade_subgrade_a"]) * 2.0 * 2.0 * 2.0 * ((0.481627) * 2.0 - np.where((0.481627) * 2.0 > data["employment_status"], np.where(((0.481627) * 2.0 - data["grade_subgrade_a"]) > (0.084500), (((0.481627) * 2.0 - data["employment_status"]) + (0.183500)), 0 ) * 2.0, 0 ) * 2.0)) +
                0.999896*np.tanh(np.where((data["debt_to_income_ratio"] * -1. + (data["debt_to_income_ratio"] * -1. + data["employment_status"] / 2.0)) <= (0.078500), (((data["debt_to_income_ratio"] + (((0.078500) * -1. + data["grade_subgrade_a"] / 2.0) + (0.078500) * -1.) * -1.) + (0.183500))/2.0 ) / (((0.095500) * -1. / 2.0 + data["employment_status"]) * -1. / 2.0), 0 )) +
                0.376344*np.tanh((((0.710850)) / ((data["employment_status"] + (data["debt_to_income_ratio"]) / ((0.481627) / 2.0))) + np.where(data["employment_status"] <= (0.120500), (data["debt_to_income_ratio"] - (((data["debt_to_income_ratio"]) / ((0.071500)) + (data["debt_to_income_ratio"]) / ((data["employment_status"]) / ((data["debt_to_income_ratio"]) / ((0.071500))))) - ((0.183500) - (6.513969)))) / 2.0, 0 ))/2.0  / 2.0) +
                0.999196*np.tanh(((0.183500) + (data["credit_score"] - ((6.613370) + np.where((((0.183500) + (data["credit_score"] - (6.613370))) * 2.0) / ((((0.183500) + (data["credit_score"] - (6.613370))) + (0.111500) * -1.)) <= (-1.000000 + -1.000000) * 2.0, (0.183500), 0 ))))) +
                0.999932*np.tanh(np.where(data["grade_subgrade_a"] > data["employment_status"], (6.508017) * 2.0 * 2.0 * -1. * (data["debt_to_income_ratio"]) / (((((0.130000)) / (( 0.785185 ) / (data["employment_status"])) + np.where(data["debt_to_income_ratio"] / 2.0 <= data["employment_status"], (data["grade_subgrade_a"] + data["grade_subgrade_a"]), 0 ))) / (data["debt_to_income_ratio"])), 0 ) / 2.0 / 2.0))
    return o


o1 = gpI(cbtrain)
roc_auc_score(cbtrain.loan_paid_back,o1)


o2 = gpII(cbtrain)
roc_auc_score(cbtrain.loan_paid_back,o2)


roc_auc_score(cbtrain.loan_paid_back,(o1+o2)/2)


cbtest["loan_paid_back"] = (gpI(cbtest)+gpII(cbtest))/2.
cbtest[["loan_paid_back"]].to_csv("jeepyloan.csv")

