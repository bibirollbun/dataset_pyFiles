import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from ipywidgets import *


data_path = "/kaggle/input/try-calculate-exchange-rate-rub-rmb/modified_data.csv"  
raw_data = pd.read_csv(data_path)

raw_data['timestamp'] = pd.to_datetime(raw_data['ftimestamp'])
raw_data = raw_data.drop(columns=['ftimestamp'])
plt.rcParams["figure.figsize"] = (15,3)

plt.plot(raw_data['timestamp'],raw_data['target'])
plt.grid()

plt.show()


data_df = raw_data.dropna()

plt.rcParams["figure.figsize"] = (15,3)

plt.plot(data_df['timestamp'],data_df['target'])
plt.grid()

plt.show()


class Network(object):

    def __init__(self, trainLen=2000, testLen=2000, initLen=100, predictLen=50) :
        self.initLen = initLen
        self.trainLen = trainLen
        self.testLen = testLen
        self.predictLen = predictLen
        self.data = data_df['target'].values
        self.inSize = self.outSize = 1 #Input/Output dimensions
        self.resSize = 300 #Reservoir size (prediction)
        #self.resSize = 1000 #Reservoir size (generation)
        self.fstep = 3 #
        self.a = 0.3 #Leak rate alpha
        self.spectral_radius = 1.25 #Spectral raidus
        self.input_scaling = 1. #Input scaling
        self.reg =  1e-5 #None #Regularization factor - if None,
        #we'd use pseudo-inverse rather than ridge regression

        self.mode = 'prediction'
        #self.mode = 'generative'

        #Change the seed, reservoir performances should be averaged accross
        #at least 20 random instances (with the same set of parameters)
        seed = 37

        #set_seed(seed)

nw = Network()


nw.data = data_df['target'].values
def plot_figure(f) :
    plt.figure(0).clear()
    plt.plot(nw.data[0:int(f)])
    #plt.ylim([-1.1,1.1])
    plt.title('A sample of input data')

interact(plot_figure, f=FloatSlider(value=2000,min=100,max=len(data_df['target']),step=400,continuous_update=True,description='time steps'))


def initialization(nw) :

    #Weights
    nw.Win = (np.random.rand(nw.resSize,1+nw.inSize)-0.5) * nw.input_scaling
    nw.W = np.random.rand(nw.resSize,nw.resSize)-0.5

    #Matrices
    #Allocated memory for the design (collected states) matrix
    nw.X = np.zeros((1+nw.inSize+nw.resSize,nw.trainLen-nw.initLen))
    #Set the corresponding target matrix directly
    nw.Ytarget = nw.data[None,nw.initLen+1:nw.trainLen+1]

    #Run the reservoir with the data and collect X
    nw.x = np.zeros((nw.resSize,1))

    return(nw)


def compute_spectral_radius(nw):
    print('Computing spectral radius...',end=" ")
    rhoW = max(abs(linalg.eig(nw.W)[0]))
    print('Done.')
    nw.W *= nw.spectral_radius / rhoW

    return(nw)


def learning_phase(nw) :
    for t in range(nw.trainLen):
        #Input data
        nw.u = nw.data[t]
        nw.x = (1-nw.a)*nw.x + nw.a*np.tanh( np.dot(nw.Win, np.vstack((1,nw.u)) ) + np.dot( nw.W, nw.x ) )
        #After the initialization, we start modifying X
        if t >= nw.initLen:
            nw.X[:,t-nw.initLen] = np.vstack((1,nw.u,nw.x))[:,0]

    return(nw)

def train_output(nw) :
    nw.X_T = nw.X.T
    if nw.reg is not None:
        # Ridge regression (linear regression with regularization)
        nw.Wout = np.dot(np.dot(nw.Ytarget,nw.X_T), linalg.inv(np.dot(nw.X,nw.X_T) + \
            nw.reg*np.eye(1+nw.inSize+nw.resSize) ) )
    else:
        # Pseudo-inverse
        nw.Wout = np.dot(nw.Ytarget, linalg.pinv(nw.X) )

    return(nw)


def test(nw) :
    #Run the trained ESN in a generative mode. no need to initialize here,
    #because x is initialized with training data and we continue from there.
    nw.Y = np.zeros((nw.outSize,nw.testLen))
    nw.u = nw.data[nw.trainLen]
    for t in range(nw.testLen):
        nw.x = (1-nw.a)*nw.x + nw.a*np.tanh( np.dot(nw.Win, np.vstack((1,nw.u)) ) + np.dot(nw.W,nw.x ) )
        nw.y = np.dot(nw.Wout, np.vstack((1,nw.u,nw.x)) )
        nw.Y[:,t] = nw.y
        if nw.mode == 'generative':
            #Generative mode:
            nw.u = nw.y
        elif nw.mode == 'prediction':
            #Predictive mode:
            nw.u = nw.data[nw.trainLen+t+1]
        else:
            raise(Exception, "ERROR: 'mode' was not set correctly.")

    return(nw)

def forecast(nw) :
    nw.Ytarget = nw.data[None,(nw.initLen+1):-1]
    nw.X = np.zeros((1+nw.inSize+nw.resSize,nw.Ytarget.shape[1]))
    nw.X_T = nw.X.T

    for t in range(nw.Ytarget.shape[1]):
        #Input data
        nw.u = nw.data[t]
        nw.x = (1-nw.a)*nw.x + nw.a*np.tanh( np.dot(nw.Win, np.vstack((1,nw.u)) ) + np.dot( nw.W, nw.x ) )
        #After the initialization, we start modifying X
        if t >= nw.initLen:
            nw.X[:,t-nw.initLen] = np.vstack((1,nw.u,nw.x))[:,0]
    
    if nw.reg is not None:
        # Ridge regression (linear regression with regularization)
        nw.Wout = np.dot(np.dot(nw.Ytarget,nw.X_T), linalg.inv(np.dot(nw.X,nw.X_T) + \
            nw.reg*np.eye(1+nw.inSize+nw.resSize) ) )
    else:
        # Pseudo-inverse
        nw.Wout = np.dot(nw.Ytarget, linalg.pinv(nw.X) )
    
    nw.Yf = np.zeros((nw.outSize,nw.predictLen))
    nw.u = nw.data[-1]
    for t in range(nw.predictLen):
        nw.x = (1-nw.a)*nw.x + nw.a*np.tanh( np.dot(nw.Win, np.vstack((1,nw.u)) ) + np.dot(nw.W,nw.x ) )
        nw.y = np.dot(nw.Wout, np.vstack((1,nw.u,nw.x)) )
        nw.Yf[:,t] = nw.y
        if nw.mode == 'generative':
            #Generative mode:
            nw.u = nw.y
        elif nw.mode == 'prediction':
            #Predictive mode:
            nw.u = nw.data[(-nw.fstep*(t+1)-1):(-nw.fstep*t-1)].mean()
        else:
            raise(Exception, "ERROR: 'mode' was not set correctly.")

    return(nw)

def compute_error(nw) :
    # Computing MSE for the first errorLen iterations
    errorLen = 50
    mse = sum( np.square( nw.data[nw.trainLen+1:nw.trainLen+errorLen+1] - nw.Y[0,0:errorLen] ) ) / errorLen
    print('MSE = ' + str( mse ))
    return(nw)

def compute_network(nw) :
    nw = initialization(nw)
    nw = compute_spectral_radius(nw)
    nw = learning_phase(nw)
    nw = train_output(nw)
    nw = test(nw)
    nw = compute_error(nw)
    nw = forecast(nw)
    return(nw)


select_mode = ToggleButtons(description='Mode:',
    options=['prediction', 'generative'])
var1 = FloatSlider(value=300, min=0, max=1000, step=1, description='resSize')
var2 = FloatSlider(value=100, min=0, max=2000, step=1, description='initLen')
var3 = FloatSlider(value=1000, min=0, max=3000, step=100, description='trainLen')
var4 = FloatSlider(value=1000, min=500, max=3000, step=100, description='testLen')
var5 = FloatSlider(value=1.25, min=0, max=10, step=0.05, description='spectral radius')
var6 = FloatSlider(value=0.3, min=0, max=1, step=0.01, description='leak rate')
#valid = Button(description='Validate')

def record_values(_) :
    #clear_output()
    nw.mode=select_mode.value
    nw.resSize=int(var1.value)
    nw.initLen=int(var2.value)
    nw.trainLen=int(var3.value)
    nw.testLen=int(var4.value)
    nw.spectral_radius=float(var5.value)
    nw.a=float(var6.value)
    print("InitLen:", nw.initLen, "TrainLen:", nw.trainLen, "TestLen:", nw.testLen)
    print("ResSize:", nw.resSize, "Spectral Radius:", nw.spectral_radius, "Leak Rate:", nw.a)
    compute_network(nw)
    return(nw)

display(select_mode)
display(var1)
display(var2)
display(var3)
display(var4)
display(var5)
display(var6)
#display(valid)

#valid.on_click(record_values)



record_values(_)


var7 = FloatSlider(value=100,min=10,max=len(nw.Y.T[0:nw.testLen]),step=100,description='time steps')
#valid = Button(description='Validate')

def trace_graph1(_) :
    #clear_output()
    f=int(var7.value)
    plt.figure(1).clear()
    plt.plot( range(0,nw.trainLen), nw.data[0:nw.trainLen], 'r' )
    plt.plot( range(nw.trainLen+1,nw.trainLen+f+1), nw.data[nw.trainLen+1:nw.trainLen+f+1], 'g' )
    plt.plot(range(len(nw.data)+1,len(nw.data)+1+nw.predictLen), nw.Yf.T[0:nw.predictLen], 'm' )
    plt.plot( range(nw.trainLen+1,nw.trainLen+f+1), nw.Y.T[0:f], 'b' )
    plt.title('Target and generated signals $y(n)$ starting at $n=0$')
    if nw.mode == 'generative':
        plt.legend(['Target signal', 'Free-running predicted signal'])
    elif nw.mode == 'prediction':
        plt.legend(['Train signal','Target signal', 'Predicted signal'])

#valid.on_click(trace_graph1)

display(var7)
#display(valid)


trace_graph1(_)


#var8 = FloatSlider(value=2000,min=10,max=nw.testLen,step=10,description='time steps')
var9 = FloatSlider(value=0.2,min=0.001,max=10,step=0.001,description='amplitude')
#valid = Button(description='Validate')

def trace_graph2(_) :
    #clear_output()
    f=int(var7.value)
    amp=float(var9.value)
    plt.figure(2).clear()
    plt.ylim([-amp,amp])
    plt.plot(nw.data[nw.trainLen+1:nw.trainLen+f+1]-nw.Y[0][0:f], 'g' )
    print(nw.Y[0].shape)
    plt.title('Target and predicted signal difference through time')
    if nw.mode == 'generative':
        plt.legend(['Target signal', 'Free-running predicted signal'])
    elif nw.mode == 'prediction':
        plt.legend(['Target signal', 'Predicted signal'])

#valid.on_click(trace_graph2)

#display(var8)
display(var9)
#display(valid)


trace_graph2(_)


def trace_graph_generative(_) :
    f=nw.testLen
    plt.figure(1).clear()
    plt.plot( range(0,nw.trainLen), nw.data[0:nw.trainLen], 'o' )
    #plt.plot( range(nw.trainLen+1,nw.trainLen+f+1), nw.data[nw.trainLen+1:nw.trainLen+f+1], 'g' )
    plt.plot( range(0,len(nw.data)), nw.data, 'g' )
    plt.plot(range(len(nw.data)+1,len(nw.data)+1+nw.predictLen), nw.Yf.T[0:nw.predictLen], 'm' )
    plt.plot( range(nw.trainLen+1,nw.trainLen+f+1), nw.Y.T[0:f], 'b' )
    plt.legend(['Train signal', 'Real Signal', 'Predicted signal','Check the model signal'])
    plt.grid()
    plt.show()



    submission = []    
    data2competitions = raw_data[raw_data['item_id'] == 5] 
    data2competitions = data2competitions.reset_index()
    data2competitions = data2competitions.dropna()
    #data2competitions['target'] = data2competitions['target'].fillna(0)

    nw.mode='prediction'
    #nw.mode='generative'
    nw.data = data2competitions['target'].values
    nw.resSize=300
    nw.initLen=50
    nw.testLen=50
    nw.predictLen=50
    nw.trainLen=len(data2competitions) - 1*nw.initLen - 1*nw.testLen
    nw.spectral_radius = 1.11
    nw.a = 0.3

    data = data2competitions

    compute_network(nw)
    
    trace_graph_generative(_)
    submission = np.append(submission,nw.Yf.T[0:nw.predictLen])


# example - how mean of forecast period in the past works

y_forecastPoint = np.zeros((nw.outSize,nw.predictLen))
x_forecastPoint = np.zeros((nw.outSize,nw.predictLen))

for t in range(nw.predictLen):
    y_forecastPoint[:,t] = nw.data[(-nw.fstep*(t+1)-1):(-nw.fstep*t-1)].mean()
    x_forecastPoint[:,t] = len(nw.data) - nw.fstep*t 
plt.plot(range(len(nw.data)),nw.data, x_forecastPoint.T, y_forecastPoint.T,'o')
plt.plot(range(len(nw.data)+1,len(nw.data)+1+nw.predictLen), nw.Yf.T[0:nw.predictLen], 'm' )
plt.show()


submission = []
for item_id in range(1,6):
    data2competitions = raw_data[raw_data['item_id'] == item_id] 
    data2competitions = data2competitions.reset_index()
    #data2competitions['target'] = data2competitions['target'].fillna(0)
    data2competitions = data2competitions.dropna()

    nw.mode='prediction'
    #nw.mode='generative'
    nw.data = data2competitions['target'].values
    nw.resSize=300
    nw.initLen=50
    nw.testLen=50
    nw.predictLen=50
    nw.fstep = 5
    nw.trainLen=len(data2competitions) - 1*nw.initLen - 1*nw.testLen
    nw.spectral_radius = 1.11
    nw.a = 0.3

    data = data2competitions

    compute_network(nw)
    
    trace_graph_generative(_)
    submission = np.append(submission,nw.Yf.T[0:nw.predictLen])


submission_df = pd.DataFrame({
    'item_id': range(len(submission)),
    'target': submission  
})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)


submission_df




