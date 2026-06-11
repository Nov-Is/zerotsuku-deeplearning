# 3.2.2
# xには実数しか入力できず、numpy配列を渡せない
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# numpy配列対応した形
def step_function(x):
    y = x > 0
    return y.astype(int)

# 仕組み
import numpy as np
x = np.array([-1.0, 1.0, 2.0])
x

y = x > 0 # xが0より大きいものをTrue, 0以下をFalseに変換
y = y.astype(int) # その後、intに変換して、(0, 1, 1)にする

# 3.2.3 ステップ関数のグラフ
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.savefig("/app/out_plot/step_function_plot.png")

# 3.2.4 シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.savefig("/app/out_plot/sigmoid_plot.png")

# 3.2.7 ReLU関数
def relu(x):
    return np.maximum(0, x)

# 3.3.1 多次元配列
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
np.ndim(A)
A.shape
A.shape[0]

# 2次元配列
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
np.ndim(B)

# 3.3.2 行列の積
A = np.array([[1, 2], [3, 4]])
A.shape # shapeは縦x横で表示

B = np.array([[5, 6], [7, 8]])
B.shape

np.dot(A, B) # 行列の積

A = np.array([[1, 2, 3], [4, 5, 6]])
A.shape

B = np.array([[1, 2], [3, 4], [5, 6]])
B.shape

# 行列の積は列と行が一致しないと計算できない
np.dot(A, B)

C = np.array([[1, 2], [3, 4]])
C.shape
A.shape
np.dot(A, C) # エラーになる

# 1次元配列でも合わせる
A = np.array([[1, 2], [3, 4], [5, 6]])
A.shape
B = np.array([7, 8])
B.shape
np.dot(A, B)

# 3.3.3 ニューラルネットワークの行列の積
X = np.array([1, 2])
X.shape

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)

# 3.4 3層ニューラルネットワークの実装
# 3.4.2 各層における信号伝達の実装
# 入力層から第1層への伝達
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1) # sigmoidはL35を使用

print(A1)
print(Z1)

# 第1層から第2層の実装
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

# 第2層から出力層への信号の伝達
# 下記は恒等関数
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # もしくはY = A3

# 3.4.3 実装のまとめ
# これまでの処理をまとめて書く
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

# 3.5 出力層の設計
# 3.5.1 恒等関数とソフトマックス関数
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 指数関数
print(exp_a)

sum_exp_a = np.sum(exp_a) # 指数関数の和
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

# ここまでがソフトマックス関数の処理
# 今後のために再定義
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 3.5.2 ソフトマックス関数の実装上の注意
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))

c = np.max(a)
a - c

np.exp(a - c) / np.sum(np.exp(a - c))

# 上記をまとめると下記
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# 3.5.3 ソフトマックス関数の特徴
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
np.sum(y)

# 3.6 手書き数字認識
# 3.6.1 MNISTデータセット
import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    # pil_img = Image.fromarray(np.uint8(img))
    pil_img = Image.fromarray(np.uint8(img))
    # pil_img.show()
    pil_img.save("/app/out_plot/mnist_sample.png")
    print("saved: /app/out_plot/mnist_sample.png")

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
