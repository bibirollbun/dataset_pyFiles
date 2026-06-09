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


from fastai.collab import *
from fastai.tabular.all import *
set_seed(42)


path = Path('../input/fmi-su-recommender-system-hw-22022')


path


ratings = pd.read_csv(path/'ratings_train.csv')
ratings = ratings.drop(ratings.columns[0], axis=1)
ratings.head()


books = pd.read_csv(path/'/kaggle/input/fmi-su-recommender-system-hw-22022/books.csv')
books.head()


ratings = ratings.merge(books, on='Book_Id')
ratings.head()


dls = CollabDataLoaders.from_df(ratings, item_name='Title', bs=64)
dls.show_batch()


n_users  = len(dls.classes['User_Id'])
n_books = len(dls.classes['Title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
book_factors = torch.randn(n_books, n_factors)


class T(Module):
    def __init__(self): 
        self.a = torch.ones(3)

L(T().parameters())


class T(Module):
    def __init__(self): 
        self.a = nn.Parameter(torch.ones(3))

L(T().parameters())


# to view docs without source code
?nn.Parameter
# or
nn.Parameter?


# to view docs WITH source code
??nn.Parameter
# or
nn.Parameter??


class T(Module):
    def __init__(self): 
        # NOTE! bias=False. Try bias=True and see what happens
        self.a = nn.Linear(1, 3, bias=False)
        # NOTE! Try bias=True and see what happens
#         self.a = nn.Linear(1, 3, bias=True)

t = T()
L(t.parameters())


type(t.a.weight)


def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


class DotProductBias(Module):
    def __init__(self, n_users, n_books, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.book_factors = create_params([n_books, n_factors])
        self.book_bias = create_params([n_books])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        books = self.book_factors[x[:,1]]
        res = (users*books).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.book_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range)


n_factors = 3 # before 50
# n_factors = 3  # Try this
model = DotProductBias(n_users, n_books, n_factors=n_factors)
# model = DotProductBias(n_users, n_books, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)


model.book_bias


book_bias = learn.model.book_bias.squeeze()
idxs = book_bias.argsort()[:5]
[dls.classes['Title'][i] for i in idxs]


idxs = book_bias.argsort(descending=True)[:5]
[dls.classes['Title'][i] for i in idxs]


g = ratings.groupby('Title')['Rating'].count()
top_books = g.sort_values(ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['Title'].o2i[m] for m in top_books])
book_w = learn.model.book_factors[top_idxs].cpu().detach()
book_pca = book_w.pca(3)
fac0,fac1,fac2 = book_pca.t()
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
for i, x, y in zip(top_books[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


# Very interesting!
# Try to set n_factors = 3 and rerun the above cells. 


learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))


# Remember to look for a good lr
learn.lr_find()


learn.fit_one_cycle(1, 5e-3, wd=0.1)


learn.model


book_bias = learn.model.i_bias.weight.squeeze()
idxs = book_bias.argsort(descending=True)[:5]
[dls.classes['Title'][i] for i in idxs]


collab_learner??


EmbeddingDotBias??


book_factors = learn.model.i_weight.weight
idx = dls.classes['Title'].o2i['The Mummies of Urumchi']
distances = nn.CosineSimilarity(dim=1)(book_factors, book_factors[idx][None])
idx = distances.argsort(descending=True)[1]
dls.classes['Title'][idx]


embs = get_emb_sz(dls)
embs


class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)


model = CollabNN(*embs)


learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(1, 5e-3, wd=0.01)


learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(1, 5e-3, wd=0.1)


@delegates(TabularModel)
class EmbeddingNN(TabularModel):
    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs, layers=layers, n_cont=0, out_sz=1, **kwargs)


tst_data = pd.read_csv(path/'ratings_to_predict.csv')

# Set the DataFrame's index to the 'Id' column
tst_data.reset_index(inplace=True)
tst_data.rename(columns={'index': 'Id'}, inplace=True)

tst_data['User_Id'] = tst_data['User_Id'].astype(int)
tst_data['Book_Id'] = tst_data['Book_Id'].astype(str)

# Load book data
books_data = pd.read_csv(path / 'books.csv')
books_data['Book_Id'] = books_data['Book_Id'].astype(str)

# Merge data
tst_data = tst_data.merge(books_data[['Book_Id', 'Title']], on='Book_Id', how='left')

# Creating a Data Loader and Getting Predictions
tst_dl = dls.test_dl(tst_data, bs=64)
tst_preds, _ = learn.get_preds(dl=tst_dl)

# The DataFrame to submit
submission_df = tst_data[['Id']].copy()
submission_df['Rating'] = tst_preds.numpy()
submission_df['Rating'] = submission_df['Rating'].clip(0, 10.5)

# Save to CSV file
submission_df.to_csv('final_submission.csv', index=False, float_format='%.8f')




