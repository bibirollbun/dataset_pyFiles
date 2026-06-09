# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Utilities
def onehotEncoder(Y, ny):
    return np.eye(ny)[Y]


def Sigmoid(X):
    return 1 / (1 + np.exp(-X))


def Softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)

def ReLU(X):
    return np.maximum(0, X)

def ReLU_backward(A):
    return (A > 0).astype(float)



def augment_image(img_flat, p_flip=0.5, max_shift=2):
    img_2d = img_flat.reshape(28, 28) 

    if np.random.rand() < p_flip:
        img_2d = np.fliplr(img_2d)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    shift_x = np.random.randint(-max_shift, max_shift + 1)

    if shift_y != 0:
        img_2d = np.roll(img_2d, shift_y, axis=0)
    if shift_x != 0:
        img_2d = np.roll(img_2d, shift_x, axis=1)

    if shift_y > 0:       # 向下平移, 顶部加0
        img_2d[:shift_y, :] = 0.0
    elif shift_y < 0:     # 向上平移, 底部也要0
        img_2d[shift_y:, :] = 0.0
        
    if shift_x > 0:       #算法跟上面接近
        img_2d[:, :shift_x] = 0.0
    elif shift_x < 0:     
        img_2d[:, shift_x:] = 0.0
            
    return img_2d.reshape(784)  #打回全连接能用的


def augment_batch(X_batch, p_flip=0.5, max_shift=2):
    return np.apply_along_axis(
        lambda img: augment_image(img, p_flip, max_shift), 
        axis=1, 
        arr=X_batch
    )


def initWeights(M):
    l = len(M)
    W = []
    B = []

    for i in range(1, l):
        n_in = M[i-1]
        W.append(np.random.randn(n_in, M[i]) * np.sqrt(1.0 / n_in))
        B.append(np.zeros([1, M[i]]))

    return W, B


def networkForward(X, W, B):
    l = len(W)
    A = [None for i in range(l + 1)]
    A[0] = X
    A_prev = X
    
    for i in range(l - 1):
        Z = A_prev @ W[i] + B[i]
        A_prev = ReLU(Z)  #Sigmoid也太慢了
        A[i + 1] = A_prev
        
    Z_out = A_prev @ W[l - 1] + B[l - 1]
    A[l] = Softmax(Z_out)  
    
    return A

# --------------------------

# Backward propagation
def networkBackward(Y, A, W, lambd):
    l = len(W)
    n = Y.shape[0]
    
    dW = [None for i in range(l)]
    dB = [None for i in range(l)]
    # 先算偏z
    dZ = A[l] - Y
    dW[l-1] = (A[l-1].T @ dZ) / n + (lambd / n) * W[l-1]
    dB[l-1] = np.sum(dZ, axis=0, keepdims=True) / n
    
    for i in range(l - 1, 0, -1):
        dA = dZ @ W[i].T 
        dZ = dA * ReLU_backward(A[i]) 
        dW[i-1] = (A[i-1].T @ dZ) / n + (lambd / n) * W[i-1]
        dB[i-1] = np.sum(dZ, axis=0, keepdims=True) / n
    return dW, dB


# --------------------------

# Update weights by gradient descent
def updateWeights(W, B, dW, dB, lr):
    l = len(W)
    for i in range(l):
        W[i] = W[i] - lr * dW[i]
        B[i] = B[i] - lr * dB[i]
    return W, B

# Compute regularized cost function
def cost(A_l, Y, W, lambd): 
    n = Y.shape[0]
    cross_entropy_cost = -np.sum(Y * np.log(A_l + 1e-9)) / n
    l2_cost = 0
    for w in W:
        l2_cost += np.sum(np.square(w))
    l2_cost = (lambd / (2 * n)) * l2_cost  #太容易过拟合了
    c = cross_entropy_cost + l2_cost 
    
    return c


def train(X, Y, X_val, Y_val, M, lr=0.1, iterations=3000, patience=20, lambd=0.0, batch_size=128):
    costs = []
    val_costs = []
    
    W, B = initWeights(M)

    best_val_cost = np.inf
    patience_counter = 0
    
    best_W = [w.copy() for w in W]
    best_B = [b.copy() for b in B]

    n = X.shape[0]
    for i in range(iterations):

        idx = np.random.choice(n, batch_size, replace=False)
        X_batch = X[idx]
        Y_batch = Y[idx]
        X_batch_aug = augment_batch(X_batch, p_flip=0.5, max_shift=2)
        A = networkForward(X_batch_aug, W, B)
        c = cost(A[-1], Y_batch, W, lambd)
        dW, dB = networkBackward(Y_batch, A, W, lambd)
        W, B = updateWeights(W, B, dW, dB, lr)

        if i % 100 == 0:
            print(f"Cost after iteration {i}: {c:.6f}")
            costs.append(c)
            A_val = networkForward(X_val, W, B)
            val_cost = cost(A_val[-1], Y_val, W, lambd)
            val_costs.append(val_cost)
            print(f"{val_cost:.6f}")
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                best_W = [w.copy() for w in W]
                best_B = [b.copy() for b in B]
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
    return best_W, best_B, costs, val_costs



def predict(X, W, B, Y):
    Y_out = np.zeros([X.shape[0], Y.shape[1]])
    A = networkForward(X, W, B)
    idx = np.argmax(A[-1], axis=1)
    Y_out[range(Y.shape[0]), idx] = 1
    return Y_out


def test(Y, X, W, B):
    Y_out = predict(X, W, B, Y)
    acc = np.sum(Y_out * Y) / Y.shape[0]
    print(f"Accuracy is: {acc*100:.2f}%")
    return acc


def output(X, W, B):
    A = networkForward(X, W, B)
    Y_hat = np.expand_dims(np.argmax(A[-1], axis=1), axis=1)
    idx = np.expand_dims(np.arange(Y_hat.shape[0]), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), 
               header="Index,ID", comments='', delimiter=',', fmt='%d')


iterations = 1000 ###### Training loops
lr = 0.1 ###### Learning rate
patience = 20
validation_split = 0.2  #验证集防过拟合
lambd = 0.01       
M = [784, 256, 128, 10] #大力出奇迹
batch_size = 128 

X_all = np.load("train_data.npy")
X_all = X_all / 255.0

Y_all_labels = np.load("train_label.npy")
Y_all_onehot = onehotEncoder(Y_all_labels, 10)

n_samples = X_all.shape[0]

indices = np.random.permutation(n_samples)
X_all_shuffled = X_all[indices]
Y_all_shuffled = Y_all_onehot[indices]

split_idx = int(n_samples * (1 - validation_split))
X_train = X_all_shuffled[:split_idx]
Y_train = Y_all_shuffled[:split_idx]

X_val = X_all_shuffled[split_idx:]
Y_val = Y_all_shuffled[split_idx:]


W, B, costs, val_costs = train(X_train, Y_train, X_val, Y_val, M, lr, iterations, patience, lambd,batch_size)
plt.figure()
plt.plot(range(len(costs)), costs)
plt.show()
print("验证集上测试看是否过拟合：")
test(Y_val, X_val, W, B)

X_test = np.load("test_data.npy")
X_test = X_test / 255.0
output(X_test, W, B)



