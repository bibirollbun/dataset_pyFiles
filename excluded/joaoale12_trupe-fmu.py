import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

class Preprocessor:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        self.cat_cols = ['CryoSleep', 'VIP']
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fill_missing_values(self):
        for col in self.num_cols:
            median = self.train[col].median()
            self.train[col] = self.train[col].fillna(median)
            self.test[col] = self.test[col].fillna(median)

        for col in self.cat_cols + ['HomePlanet', 'Destination']:
            mode = self.train[col].mode()[0]
            self.train[col] = self.train[col].fillna(mode)
            self.test[col] = self.test[col].fillna(mode)

        for df in [self.train, self.test]:
            df['Cabin'] = df['Cabin'].fillna('G/0/S')
            df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
            df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce').fillna(0)
            df['TotalSpent'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
            df['IsRich'] = (df['TotalSpent'] > 2000).astype(int)
            df['AgeGroup'] = pd.cut(df['Age'], bins=[-1, 12, 18, 30, 50, 100], labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior'])

        self.train = self.train.infer_objects()
        self.test = self.test.infer_objects()

    def encode_features(self):
        cat_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeGroup']
        
        self.train[cat_features] = self.train[cat_features].astype(str)
        self.test[cat_features] = self.test[cat_features].astype(str)

        self.ohe.fit(self.train[cat_features])

        train_cat = pd.DataFrame(
            self.ohe.transform(self.train[cat_features]),
            columns=self.ohe.get_feature_names_out(cat_features)
        )
        test_cat = pd.DataFrame(
            self.ohe.transform(self.test[cat_features]),
            columns=self.ohe.get_feature_names_out(cat_features)
        )

        train_cat.reset_index(drop=True, inplace=True)
        test_cat.reset_index(drop=True, inplace=True)
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

        num_features = self.num_cols + ['CabinNum', 'TotalSpent', 'IsRich']
        train_final = pd.concat([self.train[num_features], train_cat], axis=1)
        test_final = pd.concat([self.test[num_features], test_cat], axis=1)

        return train_final, test_final

class Model:
    def __init__(self):
        self.clf = RandomForestClassifier(random_state=42)

    def train(self, X, y):
        param_dist = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        rs = RandomizedSearchCV(
            self.clf,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            verbose=0,
            n_jobs=-1,
            random_state=42
        )
        rs.fit(X, y)
        self.clf = rs.best_estimator_

    def predict(self, X):
        return self.clf.predict(X)

class Submission:
    def __init__(self, test_df, predictions, filename='submission.csv'):
        self.test_df = test_df
        self.predictions = predictions
        self.filename = filename

    def save(self):
        submission_df = pd.DataFrame({
            'PassengerId': self.test_df['PassengerId'],
            'Transported': self.predictions
        })
        submission_df['Transported'] = submission_df['Transported'].astype(bool)
        submission_df.to_csv(self.filename, index=False)

if __name__ == '__main__':
    data = DataLoader("/kaggle/input/space-ship/train.csv", "/kaggle/input/space-ship/test.csv")

    train_df, val_df = train_test_split(data.train, test_size=0.2, stratify=data.train['Transported'], random_state=42)

    prep = Preprocessor(train_df.copy(), val_df.copy())
    prep.fill_missing_values()
    X_train, X_val = prep.encode_features()

    y_train = train_df['Transported'].astype(bool)
    y_val = val_df['Transported'].astype(bool)

    model = Model()
    model.train(X_train, y_train)
    val_predictions = model.predict(X_val)

    acc = accuracy_score(y_val, val_predictions)
    print(f"Acurácia na validação: {acc:.4f}")

    full_prep = Preprocessor(data.train.copy(), data.test.copy())
    full_prep.fill_missing_values()
    X_full_train, X_test = full_prep.encode_features()
    y_full = data.train['Transported'].astype(bool)

    model_full = Model()
    model_full.train(X_full_train, y_full)
    predictions = model_full.predict(X_test)

    submission = Submission(data.test, predictions)
    submission.save()
    print("Submissão salva como 'submission.csv'.")


