__author__ = "Ivar Vargas Belizario"
__copyright__ = "Copyright 2025"
__credits__ = ["Ivar Vargas Belizario"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ivar Vargas Belizario"
__email__ = "ivargasbelizario@gmail.com"
__status__ = "development"


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def readData():
    df_train = pd.read_csv('/kaggle/input/shelter-animal-outcomes/train.csv.gz')
    df_test = pd.read_csv('/kaggle/input/shelter-animal-outcomes/test.csv.gz')
    df_submission = pd.read_csv('/kaggle/input/shelter-animal-outcomes/sample_submission.csv.gz')
    return df_train, df_test, df_submission

def calc_age(x):
    x = str(x)
    if x == 'nan': return 0
    age = int(x.split()[0])
    if x.find('year') > -1: return age*365 
    if x.find('month')> -1: return age*30
    if x.find('week')> -1: return age*7
    if x.find('day')> -1: return age
    else: return 0


def transformX(df, atrsel, isTrain=False):
    # df = df.dropna(how="all")
    # df = df.dropna()
    
    # clean
    df.SexuponOutcome = df.SexuponOutcome.fillna("Unknown")
    df.AgeuponOutcome = df.AgeuponOutcome.fillna("0 days")
    
    # check nan values
    nan_count = df.isnull().sum()
    print("nan_count ", isTrain, nan_count)

    
    
    y = []
    if isTrain:
        df['OutcomeType'].replace(['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'],
                        [0, 1, 2, 3, 4], inplace=True)

        y = df['OutcomeType']

    # convert to days
    df['AgeInDays'] = df.AgeuponOutcome.apply(calc_age)
        
    # convert to from str to datetime
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df['weekday'] = df['DateTime'].apply(lambda x: x.weekday())
    df['month'] = df['DateTime'].apply(lambda x: x.month)
    df['is_multi_colors'] = df['Color'].apply(lambda x : 1 if '/' in x else 0)

    df['is_mix'] = 0
    df.loc[df['Breed'].str.contains('Mix'), 'is_mix'] = 1
    
    
    df['IsFemale'] = 0
    df.loc[df['SexuponOutcome'].str.contains('Female'), 'IsFemale'] = 1
    df['IsMale'] = 0
    df.loc[df['SexuponOutcome'].str.contains('Male'), 'IsFemale'] = 1
    
    
    # Extract purebred feature
    df['Purebred'] = 1
    mixed_entries = (df['Breed'].str.contains('\/')) | (df['Breed'].str.contains('Mix'))
    df.loc[mixed_entries, 'Purebred'] = 0


    # Extract aggressive breeds
    aggro_breeds = 'Staffordshire|Pit|Doberman|Chow|Rottweiler|German Shepherd|American Bulldog|Mastiff|Bullmastiff|Husky|Malamute|Akita|Boxer'
    df['AggroBreed'] = 0
    df.loc[df['Breed'].str.contains(aggro_breeds), 'AggroBreed'] = 1


    # Extract Hypoallergenic breeds
    hypoallergenic_breeds = 'Affenpinscher|Afghan|Hairless|Barbet|Bedlington|Bichon|Bolognese|Crested|Schnauzer|Water Spaniel|Kerry|Maltese|Poodle|Portuguese Water|Yorkshire'
    df['HypoallergenicBreed'] = 0
    df.loc[df['Breed'].str.contains(hypoallergenic_breeds), 'HypoallergenicBreed'] = 1

    df['Hour'] = df['DateTime'].dt.round("h").dt.hour
    df['weekend'] = df['DateTime'].dt.weekday.isin([5, 6]).astype(int)

    
    # df['Breed'] = df['Breed'].apply(lambda x: 'Shorthaired' if 'Shorthair' in x else ('Medium Haired' if 'Medium Hair' in x else ('Longhaired' if 'Longhair' in x else 'Other')))

    df.loc[(df['Name'].isnull(), 'HasName')] = 0 
    df.loc[(df['Name'].notnull(), 'HasName')] = 1 

    
    # update the olders
    LE = LabelEncoder()
    df['AnimalType'] = LE.fit_transform(df['AnimalType'])
    LE = LabelEncoder()
    df['Breed'] = LE.fit_transform(df['Breed'])
    LE = LabelEncoder()
    df['Color'] = LE.fit_transform(df['Color'])
    LE = LabelEncoder()
    df['SexuponOutcome'] = LE.fit_transform(df['SexuponOutcome'])
    LE = LabelEncoder()
    df['Breed'] = LE.fit_transform(df['Breed'])

    return df[atrsel], y

if __name__ == '__main__':
    
    df_train, df_test, df_submission = readData()
    print("df_train.columns", df_train.columns)
        

    # set features 
    atrsel = [
                # old features
                "AnimalType",'Breed', 'Color','SexuponOutcome',
                # new features
                 "AgeInDays", 'weekday', 'month', 'HasName', 'is_multi_colors', 'is_mix',
                'Purebred', 'AggroBreed', 'HypoallergenicBreed', 'Hour','weekend','IsFemale','IsMale'
             ]
    
    # data pre-processing and transformation
    print("df_train", df_train)
    X, y = transformX(df_train, atrsel, isTrain=True)
    X_test, _ = transformX(df_test, atrsel)
    print("X, y", X, y)
        
    # split data
    X_train, X_val, y_train, y_val = train_test_split(
                X, y, stratify=y, test_size=0.1, random_state=42)
    
    
    # training model
    clf = RandomForestClassifier(n_estimators=50, random_state=0)
    clf = clf.fit(X_train,y_train)
    y_val_predic =  clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_predic)
    print("Accuracy:", acc)
    
    # prediction in test
    y_tes_predic = clf.predict(X_test)
    print("y_tes_predic", y_tes_predic)
    
    y_tes_predic_proba = clf.predict_proba(X_test) # => [[0.1,0.2,0.2,0.2,0.2],[0.1,0.2,0.2,0.2,0.2]]
    df_submission.iloc[:,1:] = y_tes_predic_proba
    # save in .csv
    df_submission.to_csv('mysubmission.csv', index=False)
    print(df_submission)

