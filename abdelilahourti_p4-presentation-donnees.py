import pandas as pd


data=pd.read_csv('/kaggle/input/home-credit-default-risk/HomeCredit_columns_description.csv', encoding='ISO-8859-1')
#L'option encoding='ISO-8859-1' est utilisée dans Pandas pour lire des fichiers CSV contenant des caractères spéciaux qui ne sont pas compatibles avec l'encodage par défaut UTF-8


pd.set_option('display.max_rows', None)
display(data)


xtrain=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
xtrain.head(10)


print("Training data shape",xtrain.shape)


xtrain.info()


xtrain.describe()


##### application_test


xtest=pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv',low_memory=False)
xtest.head(10)


print("Testing data shape",xtest.shape)


xtest.info()


xtest.describe()


xbureau=pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv',low_memory=False)
xbureau.head(10)


print("bureau data shape",xbureau.shape)


xbureau.info()


xbureau.describe()


xbureau_balance=pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv', low_memory=False)
xbureau_balance.head(10)


print("bureau_balance data shape",xbureau_balance.shape)


xbureau_balance.info()


xbureau_balance.describe()


xprevious_application=pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv', low_memory=False)
xprevious_application.head(10)


print("previous_application data shape",xprevious_application.shape)


xprevious_application.info()


xprevious_application.describe()


lance=pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', low_memory=False)
xPOS_CASH_balance.head(10)


print("POS_CASH_balance data shape",xPOS_CASH_balance.shape)


xPOS_CASH_balance.info()


xPOS_CASH_balance.describe()


xcredit_card_balance=pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', low_memory=False)
xcredit_card_balance.head(10)


print("credit_card_balance data shape",xbureau_balance.shape)


xcredit_card_balance.info()


xcredit_card_balance.describe()


xinstallments_payments=pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv', low_memory=False)
xinstallments_payments.head(10)


print("installments_payments data shape",xinstallments_payments.shape)


xinstallments_payments.info()


xinstallments_payments.describe()




