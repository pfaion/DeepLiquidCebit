
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import numpy.linalg
%matplotlib inline

# %%

init_len = 50
train_len = 500

step_size = 0.5
end_time = (train_len + init_len + 1)*step_size
times = np.arange(0, end_time, step_size)

signal = np.sin(times)
train_signal = signal[:train_len]
target_signal = signal[init_len+1:train_len+1]
plt.plot(times, signal, '.-')
# %%

in_size = 1
out_size = 1
res_size = 10
a = 0.3

# %%

W_in = np.random.rand(res_size, 1+in_size) - 0.5
W = np.random.rand(res_size, res_size) - 0.5

eigvals, eigvevs = np.linalg.eig(W)
spec_rad = np.abs(eigvals).max()
W *= 1.0/spec_rad

# %%

X = np.zeros((1+in_size+res_size, train_len - init_len))

x = np.zeros(res_size)
for t, s in enumerate(train_signal):
    u = np.array([1, s])
    x = (1-a)*x + a*np.tanh(W_in @ u + W @ x)
    if t > init_len:
        X[:, t - init_len] = np.hstack((u, x))
        
# %%
Y_t = target_signal
reg = 1e-8
X_T = X.transpose()
W_out = Y_t @ X_T @ np.linalg.inv(X @ X_T + reg*np.eye(1+in_size+res_size))


# %%
test_len = 1000
Y = np.zeros((out_size, test_len))
s = 0.0
x = np.zeros(res_size)


for t in range(test_len):
    u = np.array([1, s])
    x = (1-a)*x + a*np.tanh(W_in @ u + W @ x)
    y = W_out @ np.hstack((u, x))
    Y[:,t] = y
    s = y

plt.plot(Y.T, '.-')
    
# %%

plt.figure(figsize=(12,8))
plt.plot(X[2:].T);
    

plt.matshow(W, cmap=plt.cm.gray)


plt.matshow(np.matrix(W_out), cmap=plt.cm.gray)

