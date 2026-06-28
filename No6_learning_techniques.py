# 6.1 パラメータの更新
# 6.1.2 SGD
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 6.1.4 Momentum
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum # 慣性の強さ（過去の速度をどれだけ保持するか）
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            # v ← momentum * v - lr * grad
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            # W ← W + v
            params[key] += self.v[key]

# 6.1.5 AdaGrad


