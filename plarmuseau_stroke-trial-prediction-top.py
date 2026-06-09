# Importing neccesary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def describepd(data):
    output=[]
    for li in data.columns:
        aantal=len(data)
        vul=len(data[li].dropna())
        vtyp=data[li].dtypes
        uniek=len(data[li].unique())
        if uniek==aantal:
            veldindex='indexfield'
        elif uniek==1:
            veldindex='constant'
        else:
            veldindex=''
        output.append([li,vtyp,np.round(100-vul/aantal*100),np.round(uniek/aantal*100),uniek,veldindex] )
    return pd.DataFrame(output,columns=['label','dtype','%empty','%uniek','aantaluniek','keyparam'])


# Loading the train and test set
train_data = pd.read_csv('/kaggle/input/stroke-trial-prediction/Train1.csv',encoding='ISO_8859_1').set_index('ID')
test_data = pd.read_csv('/kaggle/input/stroke-trial-prediction/Test1.csv',encoding='ISO_8859_1').set_index('ID')


# Printing train data info
describepd(train_data.reset_index())


# Printing test data info
describepd(test_data.reset_index())


# Listing out numeric data type
discreate_feat = []
continuous_feat = []
for feat in train_data.select_dtypes(exclude='O'):
    if train_data[feat].nunique() <= 10:
        discreate_feat.append(feat)
    else:
        continuous_feat.append(feat)

print('Number of Discrete Features:',len(discreate_feat))
print('Number of Continuous Features:',len(continuous_feat))        


for feat in discreate_feat:
    val_count = pd.DataFrame(train_data[feat].value_counts())
    val_count['Count%'] = val_count['count']/train_data.shape[0]*100
    print('='*25)
    print(feat,f'[{train_data[feat].nunique()}]')
    print('-'*25)
    print(val_count)


# Listing out object/categorical data type
object_feat = []
categorical_feat = []
for feat in train_data.select_dtypes(include='O'):
    if train_data[feat].nunique() <= 36:
        categorical_feat.append(feat)
    else:
        object_feat.append(feat)

print('Number of Categorical Features:',len(categorical_feat))
print('Number of Object Features:',len(object_feat))  


# Removing extra features from training data
ls = list(set(train_data.columns).difference(test_data.columns))
ls.remove('DIED') # Excluded the target feature from dropping
ls.append('FLASTD') # All values are NULL

train_data.drop(ls,axis=1,inplace=True)
print('List of Dropped Features: ',ls)

# Function for updatating feature list
def update_feature(features,drop_feat):
    return [feat for feat in features if feat not in drop_feat]

continuous_feat = update_feature(continuous_feat,ls)
discreate_feat = update_feature(discreate_feat,ls)
categorical_feat = update_feature(categorical_feat,ls)
object_feat = update_feature(object_feat,ls)


# fixing null values and dropping unneccesary features
train_data.drop(object_feat,axis=1,inplace=True)
train_data[continuous_feat] = train_data[continuous_feat].fillna(0)
train_data[categorical_feat] = train_data[categorical_feat].fillna('U') # U for Unknown

test_data.drop('FLASTD',axis=1,inplace=True) # No value
test_data.drop(object_feat,axis=1,inplace=True)
test_data[continuous_feat] = test_data[continuous_feat].fillna(0)
test_data[categorical_feat] = test_data[categorical_feat].fillna('U')


# Performing Standardization on continuous features
scaler = StandardScaler()
train_data[continuous_feat] = scaler.fit_transform(train_data[continuous_feat])
test_data[continuous_feat] = scaler.transform(test_data[continuous_feat])

# Performing Label Encoding on categorical features
enc = LabelEncoder()
for feat in categorical_feat:
    train_data[feat] = enc.fit_transform(train_data[feat])
    test_data[feat] = enc.transform(test_data[feat])


train_data


!pip install implicit


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix,vstack

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


#train2=train[['categorie','text']].groupby('categorie')['text'].apply(','.join).reset_index()
def ALSwrap(dfr,dfe,ycol,rangev,ranget):

    from sklearn.metrics import average_precision_score,mean_squared_error
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import laplacian_kernel,cosine_similarity
    import implicit
    dfr=dfr.fillna(0)
    dfe=dfe.fillna(0)
    print('start',dfr.shape,dfe.shape)
    totaal=csr_matrix(pd.concat([dfr.drop(ycol,axis=1),dfe],axis=0).values)
    labels=dfr[ycol]    
    print(totaal.shape)
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split( totaal[:len(dfr),:70] , labels, test_size=0.1, random_state=42)
    print(X_train.shape,X_test.shape)


    result=[]
    stap=int((ranget-rangev-1)/5)
    for comp in range(rangev,ranget,stap):
        model = implicit.als.AlternatingLeastSquares(factors=comp, regularization=0.01, iterations=20)
        model.fit(vstack([X_train,X_test]))
    
        users=model.user_factors
        #prods=model.item_factors
    
        lr=LogisticRegression(n_jobs=-1)
        lr.fit(users[:X_train.shape[0]],y_train)
        
        y_pred=lr.predict(users[X_train.shape[0]:])#X_test.dot(prods))
        score=lr.score(users[X_train.shape[0]:],y_test)
        print(comp,'scor lr',score)
        result.append([comp,score])
        #print par of confusion matrix
        
        if True:
            accuracy = confusion_matrix(y_test, y_pred)
            import seaborn as sn
            import matplotlib.pyplot as plt
            sn.set(font_scale=1.4) # for label size
            sn.heatmap(pd.DataFrame(accuracy[:10,:10]), annot=True, annot_kws={"size": 8}) # font size
            plt.show()
    
    pd.DataFrame(result).plot(0,1)

    model = implicit.als.AlternatingLeastSquares(factors=comp, regularization=0.01, iterations=20)
  
    model.fit(totaal)
    userst=model.user_factors
    lr.fit(userst[:len(dfr)],labels)
    print(userst.shape,totaal.shape,dfr.shape)
    y_pred=lr.predict(userst[len(dfr):])
    pd.DataFrame({
        'ID': dfe.index,
        'PatientDied':['Y' if i==1 else 'N' for i in y_pred]}).to_csv('submisssion_stroke_trial.csv',index=False)

    return
    def modelpredi(ui,ai):
        user_factors = model.user_factors[ui]
        item_factors = model.item_factors[ai]
    
        # Compute the predicted score
        score = user_factors.dot(item_factors)
        return score

ALSwrap(train_data,test_data,'DIED',5,55)


#test=test_data[]
#y_pred=lr.predict(test.dot(prods))


