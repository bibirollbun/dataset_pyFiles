from fastai.collab import *
from fastai.tabular.all import *
set_seed(42)


path = untar_data(URLs.ML_100k)


path


ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,
                      names=['user','movie','rating','timestamp'])
ratings.head()


movies = pd.read_csv(path/'u.item',  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
movies.head()


ratings = ratings.merge(movies)
ratings.head()


dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()


n_users  = len(dls.classes['user'])
n_movies = len(dls.classes['title'])
n_factors = 5

user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)


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
    def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors[x[:,0]]
        movies = self.movie_factors[x[:,1]]
        res = (users*movies).sum(dim=1)
        res += self.user_bias[x[:,0]] + self.movie_bias[x[:,1]]
        return sigmoid_range(res, *self.y_range)


n_factors = 50
# n_factors = 3  # Try this
model = DotProductBias(n_users, n_movies, n_factors=n_factors)
# model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)


model.movie_bias


movie_bias = learn.model.movie_bias.squeeze()
idxs = movie_bias.argsort()[:5]
[dls.classes['title'][i] for i in idxs]


idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]


g = ratings.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['title'].o2i[m] for m in top_movies])
movie_w = learn.model.movie_factors[top_idxs].cpu().detach()
movie_pca = movie_w.pca(3)
fac0,fac1,fac2 = movie_pca.t()
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x,y,i, color=np.random.rand(3)*0.7, fontsize=11)
plt.show()


# Very interesting!
# Try to set n_factors = 3 and rerun the above cells. 


learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))


# Remember to look for a good lr
learn.lr_find()


learn.fit_one_cycle(5, 5e-3, wd=0.1)


learn.model


movie_bias = learn.model.i_bias.weight.squeeze()
idxs = movie_bias.argsort(descending=True)[:5]
[dls.classes['title'][i] for i in idxs]


collab_learner??


EmbeddingDotBias??


movie_factors = learn.model.i_weight.weight
idx = dls.classes['title'].o2i['Silence of the Lambs, The (1991)']
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1]
dls.classes['title'][idx]


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
learn.fit_one_cycle(5, 5e-3, wd=0.01)


learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)


@delegates(TabularModel)
class EmbeddingNN(TabularModel):
    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs, layers=layers, n_cont=0, out_sz=1, **kwargs)







