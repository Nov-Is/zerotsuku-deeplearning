class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y # xとyをひっくり返す
        dy = dout * self.x

        return dx, dy

# りんごの買い物の実装をしてみる
# 順伝播の実装
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# 逆伝播の実装
# backward
# 乗算レイヤ
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)

# 加算レイヤ
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

# 乗算レイヤと加算レイヤを使って、りんご2個とみかん3個の買い物
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num) #(1)
orange_price = mul_orange_layer.forward(orange, orange_num) #(2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price) #(3)
price = mul_tax_layer.forward(all_price, tax) #(4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) #(4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #(3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price) #(2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #(1)

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax)

# 5.5 活性化間数レイヤの実装
# 5.5.1 ReLUレイヤ
class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# 5.5.2 Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

# 5.6 Affine / Softmaxレイヤの実装
# 5.6.1 Affineレイヤ
# 5.6.2 バッチ版Affineレイヤ
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])

X_dot_W
X_dot_W + B

dY = np.array([[1, 2, 3], [4, 5, 6]])
dY

dB = np.sum(dY, axis=0)
dB

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 行列で入力を変換するだけ
        # X (2,3) · W (3,4) + b (4,) = Y (2,4)
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        # dout (2,4) · W.T (4,3) = dx (2,3)
        # 前のレイヤーに渡す勾配。W.T で転置するのは形を合わせるため。
        self.dW = np.dot(self.x.T, dout)
        # x.T (3,2) · dout (2,4) = dW (3,4)
        # W を更新するために使う。W と同じ形になる。
        self.db = np.sum(dout, axis=0)
        # バッチ分を足し合わせる

        return dx
    
        # dx 前のレイヤーに渡す(伝播を続ける)
        # dW Wの更新に使う
        # db bの更新に使う
        # dWとdbはクラスに保存しておき、重みの更新に使用する


# 5.6.3 Softmax-with-Lossレイヤ
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmaxの出力
        self.t = None # 教師データ(one-hot vector)

    def forward(self, x, t):
        self.t = t # 正解ラベル(one-hot)を保存
        self.y = softmax(x) # スコア->確率に変換
        self.loss = cross_entropy_error(self.y, self.t) # 損失を計算

        # x(スコア) -> softmax -> y(確率) -> cross entropy -> L(損失)
        # 例: [2.1, 0.3, 1.5]  →  [0.6, 0.1, 0.3]  →  0.51

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        # dout=1
        # 最終層なので、上流からくる勾配は「損失Lそのもの」=1

        # y-tなのは？
        # softmax + cross entropyを合わせて微分すると複雑な式が綺麗に消える
        # dL/dx = y - t

        # /batch_size
        # バッチないの各サンプルの損失を平均にするための正規化

        return dx

# 5.7 誤差逆伝播法の実装
# 5.7.1 ニューラルネットワークの学習の全体図

# 前提
# ニューラルネットワークは、適応可能な重みとバイアスがあり、
# この重みとバイアスを訓練データに適応するように調整することを「学習」と呼ぶ。

# ステップ1 ミニバッチ
# 訓練データの中からランダムに一部のデータを選び出す

# ステップ2 勾配の算出
# 各重みパラメータに関する損失関数の勾配を求める
# => 誤差逆伝播法

# ステップ3 パラメータの更新
# 重みパラメータを勾配方向に微小量だけ更新する

# ステップ4 繰り返す
# ステップ1, ステップ2, ステップ3を繰り返す

# 5.7.2 誤差逆伝播法に対応したニューラルネットワークの実装
import sys, os
sys.path.append(os.pardir)
sys.path.append("deep-learning-from-scratch")
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    # predict / loss 順伝播
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 誤差逆伝播法
    # numerical_gradient: 数値微分(勾配確認としてgradientの結果が正しいか検証に使われる)
    # gradient: 誤差逆伝播法(実際に使われるのはこっち)
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads
    
    def gradient(self, x, t):
        # forward
        # 1. まず順伝播して中間値をレイヤ内に保存
        self.loss(x, t)

        # backward
        # 2. 最終層から逆順に backward を呼ぶ
        dout = 1
        dout = self.lastLayer.backward(dout) # SoftmaxWithLoss の逆伝播

        layers = list(self.layers.values())
        layers.reverse() # Affine2 → Relu1 → Affine1 の逆順
        for layer in layers:
            dout = layer.backward(dout)
        
        # 設定
        # 3. Affine レイヤが backward 中に計算した dW, db を取り出す
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads


# 5.7.3 誤差逆伝播法の勾配確認
import sys, os
sys.path.append(os.pardir)
sys.path.append("deep-learning-from-scratch")
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backdrop = network.gradient(x_batch, t_batch)

# 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backdrop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))


# 5.7.4 誤差逆伝播法を使った学習
import sys, os
sys.path.append(os.pardir)
sys.path.append("deep-learning-from-scratch")
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000 # 学習の繰り返し回数
train_size = x_train.shape[0]
batch_size = 100 # 1回の学習に使うサンプル数
learning_rate = 0.1 # 学習率

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 1. ランダムにミニバッチを選ぶ
    # ここではイテレーションごとに、trainデータセットから100件ランダム抽出
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 誤差逆伝播法によって勾配を求める
    grad = network.gradient(x_batch, t_batch)

    # 更新
    # パラメータ = パラメータ - 学習率 × 勾配
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train) # 訓練データ全体
        test_acc = network.accuracy(x_test, t_test) # テストデータ全体
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
