# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/rec-sys-master-fds-2025/train.csv')
train.head()


test = pd.read_csv('/kaggle/input/rec-sys-master-fds-2025/kaggle_baseline.csv')


def compute_rmse(y_pred, y_true):
    """ Compute Root Mean Squared Error. """
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))

def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
    
    return precision_score

def recall(recommended_items, relevant_items):  
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]
    
    return recall_score

def AP(recommended_items, relevant_items):
   
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    ap_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return ap_score


def evaluate(estimate_f,data_train,data_test):
    """ RMSE-based predictive performance evaluation with pandas. """
    ids_to_estimate = zip(data_test.user_id, data_test.movie_id)
    estimated = np.array([estimate_f(u,i) if u in data_train.user_id else 3 for (u,i) in ids_to_estimate ])
    real = data_test.rating.values
    
    return compute_rmse(estimated, real)


def evaluate_algorithm_top(test, recommender_object, at=25, thr_relevant = 4):
    
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0
    
    num_eval = 0


    for user_id in tqdm(test.user_id.unique()):

        relevant_items = test[(test.user_id==user_id )&( test.rating>=thr_relevant)].movie_id.values
        
        if len(relevant_items)>0:
            
            recommended_items = recommender_object.predict_top(user_id, at=at)
            num_eval+=1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += AP(recommended_items, relevant_items)
            
    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval
    
    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, MAP)) 


from scipy import sparse
from scipy.linalg import sqrtm

class RecSys_mf():
    """ Collaborative filtering using SVD. """
    
    def __init__(self, num_components=10):
        """ Constructor """
        self.num_components=num_components
        
        
        
    def fit(self,df_train):
        """ We decompose the R matrix into to submatrices using the training data """
        
        self.train = df_train
        self.urm = pd.pivot_table(df_train[['user_id','movie_id','rating']],columns='movie_id',index='user_id',values='rating')
        
        # We create a dictionary where we will store the user_id and movie_id which correspond 
        # to each index in the Rating matrix
        
        user_index = np.arange(len(self.urm.index))
        self.users = dict(zip(user_index,self.urm.index ))
        self.users_id2index = dict(zip(self.urm.index,user_index)) 
        
        movie_index = np.arange(len(self.urm.columns))
        self.movies = dict(zip(movie_index,self.urm.columns )) 
        self.movies_id2index= dict(zip(self.urm.columns, movie_index))
        self.movies_index2id= dict(zip(movie_index,self.urm.columns))
        self.movie_id2title = dict(df_train.groupby(by=['movie_id','title']).count().index)
        
        self.pop_items = self.train.groupby('movie_id').count()[['rating']]

        train_matrix = np.array(self.urm)
        # we mask those nan value to fill with the mean 
        mask = np.isnan(train_matrix)
        masked_arr = np.ma.masked_array(train_matrix, mask)
        item_means = np.mean(masked_arr, axis=0)

        # nan entries will replaced by the average rating for each item
        #train_matrix = masked_arr.filled(item_means)
        train_matrix = masked_arr.filled(0)
        x = np.tile(item_means, (train_matrix.shape[0],1))         

        # we remove the per item average from all entries.
        # the above mentioned nan entries will be essentially zero now
        train_matrix = train_matrix - x
        U, s, V = np.linalg.svd(train_matrix, full_matrices=False)

        # reconstruct rating matix
        S = np.diag(s[0:self.num_components])
        U = U[:,0:self.num_components]
        V = V[0:self.num_components,:]
        S_root = sqrtm(S)

        USk=np.dot(U,S_root)
        SkV=np.dot(S_root,V)
        Y_hat = np.dot(USk, SkV)
        self.Y_hat = Y_hat + x
        
    def predict_score(self, user_id, movie_id):
        
        if movie_id in self.movies_id2index:
            return self.Y_hat[self.users_id2index[user_id],self.movies_id2index[movie_id]]
        else: # in case it is a new movie 
            return 0

        
    def predict_top(self, user_id, at=5, filter_pop = 100, remove_seen=True):
        '''Given a user_id predict its top AT items'''
        seen_items = self.train[self.train.user_id==user_id].movie_id.values
        unseen_items = set(self.train.movie_id.values) - set(seen_items)
        # filter the non popular items
        #unseen_items = [item for item  in set(self.pop_items[self.pop_items.rating>filter_pop].index) if item in unseen_items]
        predictions = [(item_id,self.predict_score(user_id,item_id)) for item_id in unseen_items]

        sorted_predictions = sorted(predictions, key=lambda x: x[1],reverse = True)[:at]
        return [i[0] for i in sorted_predictions]


train.loc[train.rating<4,'rating']=0
train.loc[train.rating>=4,'rating']=1
train


rec_object = RecSys_mf()
rec_object.fit(train)
rec_object.predict_score(user_id=1,movie_id=2)


rec_object = RecSys_mf()
rec_object.fit(train)
rec_object.predict_score(user_id=1,movie_id=2)


import csv
from tqdm import tqdm
# open the file in the write mode
with open('submission.csv', 'w',encoding='UTF8') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(['user_id', 'prediction'])
    for user_id in tqdm(test.user_id.unique()):
        relevant_items = rec_object.predict_top(user_id, at=25)
        list_relevants = ' '.join([str(elem) for elem in relevant_items])
        writer.writerow([str(user_id),list_relevants])




