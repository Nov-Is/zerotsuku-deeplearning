import numpy as np

# 4.2.1 2乗和誤差
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t) **2)

# 試してみる
# [2]を正解とする
t = [0,0,1,0,0,0,0,0,0,0]

# 例1:[2]の確率が最も高い場合(0, 6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
sum_squared_error(np.array(y), np.array(t))

# 例2:[7]の確率が最も高い場合(0, 6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
sum_squared_error(np.array(y), np.array(t))

# 4.2.2 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 計算してみる
t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t))
# np.float64(0.510825457099338)

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
cross_entropy_error(np.array(y), np.array(t))
# np.float64(2.302584092994546)

# 4.2.3 ミニバッチ学習
# MNISTデータセットを読み込む
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784) => (枚数, 解像度)
print(t_train.shape) # (60000, 10) => (枚数, one_hotになった正解ラベル)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 例
np.random.choice(60000, 10)

# 4.2.4 [バッチ対応版]交差エントロピー誤差の実装
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# one_hotではない場合
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 4.3 数値微分
# 4.3.1 微分
# 悪い実装例
def numerical_diff(f, x):
    h = 1e-50
    return (f(x+h) - f(x)) / h

# 1e-50ではfloat32型では正しく表現できない
# 10^-4程度であれば良い結果が得られる

# 数値微分では誤差が生じる
# 中心差分で計算すると良い

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

# 4.3.2 数値微分の例
def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1) # 0から20まで0.1刻みのx配列
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.savefig("/app/out_plot/num_diff.png")

numerical_diff(function_1, 5)
# 0.1999999999990898

numerical_diff(function_1, 10)
# 0.2999999999986347

# 4.3.3 偏微分
def function_2(x):
    return x[0]**2 + x[1]**2
    # または
    # return np.sum(x**2)

# Q1 x0 = 3, x1 = 4
def function_tmp1(x0):
    return x0*x0 + 4.0 ** 2.0

numerical_diff(function_tmp1, 3.0)

# Q2 x0 = 3, x1 = 4
def function_tmp2(x1):
    return 3.0 ** 2.0 + x1*x1

numerical_diff(function_tmp2, 4.0)

# 4.4 勾配
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad =np.zeros_like(x) # xと同じ形状の配列を作成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
    
    return grad

numerical_gradient(function_2, np.array([3.0, 4.0]))
# array([6., 8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))
# array([0., 4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))

# 4.4.1 勾配法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    # init_xは初期値, lrはlearning rate(学習率), step_numは繰り返し回数
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

# Q
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([-6.11110793e-10,  8.14814391e-10])

# 学習率が大きすぎる例
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
# array([ 2.34235971e+12, -3.96091057e+12])

# 学習率が小さすぎる例
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
# array([ 2.34235971e+12, -3.96091057e+12])

#4.4.2 ニューラルネットワークに対する勾配
import sys, os
sys.path.append(os.pardir)
sys.path.append("deep-learning-from-scratch")
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # ガウス分布で初期化
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

# 試しに計算
net = simpleNet()
print(net.W) # 重みパラメータ
x = np.array([0.6, 0.9]) # 入力値
p = net.predict(x)
print(p)
np.argmax(p) # 最大値のインデックス

t = np.array([0, 0, 1]) # 正解ラベル
net.loss(x, t)

def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(dW)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

# 4.5 学習アルゴリズムの実装
## 前提
### ニューラルネットワークは、適応可能な重みとバイアスがあり、この重みとバイアスを訓練データに適応するように調整することを学習と呼ぶ。 ニューラルネットワークの学習は次の4つの手順で行う(勾配降下法の中の確率的勾配降下法:SGD)

### ステップ1(ミニバッチ)
#### 訓練データの中からランダムに一部のデータを選び出す。その選ばれたデータをミニバッチと言い、ここでは、そのミニバッチの損失関数の値を減らすことを目的とする。

### ステップ2(勾配の算出)
#### ミニバッチの損失関数を減らすために、各重みパラメータの勾配を求める。勾配は、損失関数の値を最も減らす方向を示す。

### ステップ3(パラメータの更新)
#### 重みパラメータの勾配方向に微少量だけ更新

### ステップ4(繰り返す)
#### ステップ1から3を繰り返す

# 4.5.1 2層のニューラルネットワークのクラス
import sys, os
sys.path.append(os.pardir)
sys.path.append("deep-learning-from-scratch")
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
    
        return cross_entropy_error(y, t)
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

# 例
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
net.params['W1'].shape #(784, 100)
net.params['b1'].shape #(100,)
net.params['W2'].shape #(100, 10)
net.params['b2'].shape #(10,)

# 推論処理は下記で実行
x = np.random.rand(100, 784) #ダミーの入力データ
y = net.predict(x)

# grads変数に勾配情報を格納
x = np.random.rand(100, 784) #ダミーの入力データ
t = np.random.rand(100, 10) #ダミーの入力データ

grads = net.numerical_gradient(x, t)

grads['W1'].shape #(784, 100)
grads['b1'].shape #(100,)
grads['W2'].shape #(100, 10)
grads['b2'].shape #(10,)


# 4.5.2 ミニバッチ学習の実装
import numpy as np
from dataset.mnist import load_mnist

import sys, os
sys.path.append(os.pardir)
sys.path.append("deep-learning-from-scratch")
from ch04.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版!

    # パラメータの更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))
