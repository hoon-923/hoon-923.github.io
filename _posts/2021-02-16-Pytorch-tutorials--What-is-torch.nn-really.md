---
layout: post
title:  "Pytorch tutorials- What is torch.nn really?"
date:   2021-02-16 17:35:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

### What is torch.nn really?

이번 튜토리얼은 [mnist 데이터셋](https://ko.wikipedia.org/wiki/MNIST_%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4)에 대해 pytorch에서 제공하는 다양한 기능들을 이용하지 않고 은닉층이 없는 단순한 신경망을 정의해본 뒤에 `torch.nn`, `torch.optim`, `Dataset`, `DataLoader`등을 활용해서 코드를 리팩토링(refactoring) 해보는 과정입니다.

----

### Tutorial 코드



#### MNIST data setup

고전적인 손글씨 숫자(0 ~ 9) 데이터인 MNIST 데이터셋을 활용하여 실습을 진행합니다.

튜토리얼 코드가 작동을 하지 않아서 다음 링크(https://github.com/mnielsen/rmnist/blob/master/data/mnist.pkl.gz)를 통해 mnist 데이터셋을 다운 받은 후 다음 colab file 경로에 업로드 해준 후 코드를 실행 했습니다.

![colab_file.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/colab_file.PNG?raw=true)

불러오는 데이터셋은 `numpy`배열 포맷이고, 데이터를 직렬화하기 위한 파이썬 전용 포맷 `pickle`을 이용하여 저장되어 있습니다.

각 이미지는 28 x 28 형태이고, 784(28 x 28) 크기인 하나의 행으로 구성되어 있기 때문에 이를 2d 이미지로 재구성해야 합니다.

~~~python
import pickle # pickle.load()를 사용하여 파일 로드
import gzip  # 바이너리나 텍스트 모드로 gzip으로 압축된 파일을 열고, 파일 객체를 반환

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)        
~~~

실행 결과:

(50000, 784)

![MNIST_sample.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/MNIST_sample.PNG?raw=true)

pytorch는 `numpy`배열 대신에 `torch.tensor`를 사용하므로, 입력 데이터를 `map`을 이용해 변환합니다.

~~~python
import torch

# numpy를 tensor로 변환하는 과정
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape # n = 50000, c = 784
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
~~~

실행 결과:

~~~python
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]]) tensor([5, 0, 4,  ..., 8, 4, 8])
torch.Size([50000, 784])
tensor(0) tensor(9)
~~~



#### Neural net from scratch (no torch.nn)

이제 인풋할 데이터는 생성했으니 `torch.nn`을 사용하지 않고 tensor로만 간단한 신경망을 구축 해보겠습니다. 

우선 weight initialization을 Xavier initialization 기법을 통해 초기화합니다. 대표적인 weight initialization 기법은 He와 Xavier가 있는데 이에 대해서는 추후 포스트를 통해 정리하겠습니다.

또한 나중의 코드를 보면 알겠지만 iteration에서 batch마다 가중치를 초기화해주는 과정이 있습니다. 이렇게 매번 iteration마다 초기화 해준 후 `requires_grad`를 이용해서 계산해주는 이유는 각각의 step이 다음 gradient에 포함되는 것을 원치 않기 때문입니다.

~~~python
import math

weights = torch.randn(784, 10) / math.sqrt(784) # Xavier initialization
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
~~~

이제 multi-label classification을 위한 activation fuction인 `log_softmax`를 정의합니다. pytorch에서도 log_softmax를 제공하지만 여기서는 직접 정의했습니다.

log_softmax는 기존의 softmax 함수가 갖고 있는 vanishing gradient를 해결하기 위해 log를 적용한 함수입니다.

그 후 weight와 dot product를 하고 bias를 더해주는 `model`을 정의합니다.

~~~python
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias) # @ 기호는 dot product
~~~

이제 batch size를 정의하고 model 함수를 호출하여 하나의 forward pass를 실행합니다. 이 경우 시작을 무작위 가중치로 시작했기 때문에 성능이 좋지 않을껍니다. 

~~~python
bs = 64  # 배치 사이즈

xb = x_train[0:bs]  # x로부터 미니배치(mini-batch) 추출
preds = model(xb)  # 예측
preds[0], preds.shape
print(preds[0], preds.shape)
print(y_train[0])
~~~

실행 결과:

~~~python
tensor([-2.4480, -2.7134, -2.4670, -2.4524, -2.3676, -2.0937, -2.5571, -1.9191,
        -2.2606, -2.0341], grad_fn=<SelectBackward>) torch.Size([64, 10])
tensor(5)
~~~

출력 결과를 보면 모델은 첫번째 손글씨 예측으로  `preds[0]`중 가장 값이 큰 -1.9191의 인덱스인 7을 예측 했지만 실제 값은 5이다. 이 경우 예측이 틀렸음을 알 수 있다.

이제 loss function을 정의해야합니다. 예제에서는 negative log-likelihood(NLL)을 구현한 후 이를 이용해 loff 값을 구했습니다.

~~~python
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))
~~~

실행 결과:

~~~python
tensor(2.3158, grad_fn=<NegBackward>)
~~~

이번에는 모델의 정확도를 계산하기 위한 함수를 구현했습니다. 

위에서 설명한것 과 같이 각 예측에서 가장 큰 값의 인덱스와 target과 동일하면 올바른 예측을 한 것 입니다.

~~~python
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1) # Returns the indices of the maximum value of all elements in the input tensor.
    return (preds == yb).float().mean()

print(accuracy(preds, yb))
~~~

실행 결과:

~~~python
tensor(0.0469)
~~~

이제 지금까지 구현한 기능들을 이용해서 training loop를 실행할 수 있습니다. 매 반복마다 다음을 수행합니다.

* 데이터의 미니 배치 선택
* 모델을 이용해 예측
* loss 계산
* `loss.backward()`를 이용해서 `weight`와 `bias` 업데이트



~~~python
lr = 0.5  # 학습률(learning rate)
epochs = 2  # 훈련에 사용할 에포크(epoch) 수

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(0.0810, grad_fn=<NegBackward>) tensor(1.)
~~~

loss가 기존의 무작위 예측보다 작아졌음을 알 수 있습니다.



#### Using torch.nn.functional

이제 기존 코드를 pytorch의 `nn`클래스의 장점들을 이용해서 순차적으로 리팩토링(refactoring) 해보겠습니다.

이 부분에서는 `torch.nn.functional`(관례에 따라 일반적으로 ` F`로 별칭)을 통해 NLL loss fuction과 log softmax activation fucntion을 결합한 Cross Entropy loss function인 `F.cross_entropy`를 이용합니다.

~~~python
import torch.nn.functional as F

loss_func = F.cross_entropy # 기존의 loss_func은 직접 정의한 NLL Loss 

def model(xb):
    return xb @ weights + bias # log softmax를 활성화 함수로 이용한 후 loss를 구하는 과정을 한번에

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(0.0810, grad_fn=<NllLossBackward>) tensor(1.)
~~~

기존의 loss와 동일합니다. 다만 기존의 사용자 정의 함수의 경우 `grad_fn`이 `<NegBackward>`였고 여기에는 함수 이름인 `<NllLossBackward>`이라는 차이는 존재합니다.



#### Refactor using nn.Module

이번 단계에서는 좀 더 직관적이고 간결한 training loop을 위해 `nn.Module`과 `nn.Parameter`를 이용합니다.

~~~python
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic() # 모델을 인스턴스화

print(loss_func(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(2.2439, grad_fn=<NllLossBackward>)
~~~

이제 training loop마다 각 parameter들을 업데이트하고 각각의 기울기를 0으로 초기화 하는 과정을 일일이 코드로 구현했었습니다.

하지만 `model.parameters()`를 통해 이를 한번에 실행할 수 있게 됩니다.

~~~python
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): # 바뀐 부분
                    p -= p.grad * lr
                model.zero_grad()

fit()

print(loss_func(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(0.0819, grad_fn=<NllLossBackward>)
~~~

loss가 줄어들었음을 알 수 있습니다.



#### Refactor using nn.Linear

이번 단계에서는 기존 `Mnist_Logistic` 클래스에서 `self.weights`와 `self.bias`를 직접 정의 및 초기화하고 `xb @ self.weights + self.bias` 를 계산하는 대신에 이를 자동으로 해줄 pytorch 클래스인 `nn.Linear`를 이용해 리팩토링(refactoring)을 진행합니다. 

~~~python
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

fit()
print(loss_func(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(2.3367, grad_fn=<NllLossBackward>)
tensor(0.0813, grad_fn=<NllLossBackward>)
~~~



#### Refactor using optim

수동으로 각 매개변수를 업데이트 하는 대신에 `torch.optim`의 `step()` 메소드를 사용하여 코드를 더욱 간결하게 할 수 있습니다.

~~~python
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
~~~



#### Refactor using Dataset

Pytorch의 `TensorDataset`을 이용해 길이(`__len__`)와 인덱싱 방식(`__getitem__`)을 정의함으로써 텐서의 첫 번째 차원을 따라 반복, 인덱싱 및 슬라이스(slice)하는 방법도 제공합니다.

이제 `x_train`과 `y_train` 값들의 미니 배치를 인덱싱하는 코드가 간단해집니다.

~~~python
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(0.0820, grad_fn=<NllLossBackward>)
~~~



#### Refactor using DataLoader

이번에는 `DataLoader`를 이용해 미니 배치를 자동으로 생성하는 코드를 구현 했습니다.

~~~python
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
~~~

실행 결과:

~~~python
tensor(0.0817, grad_fn=<NllLossBackward>)
~~~



#### Add validation

신경망도 여타 다른 머신러닝 모델과 같이 과적합을 확인하기 위해 검증 데이터셋(validation set)이 필요합니다. 이번 단계에서는 각 epoch마다 validation loss를 계산하는 코드를 구현 했습니다.

~~~python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
~~~

훈련 데이터는 배치와 과적합 사이의 상관관계를 방지하기 위해 shuffle이 필요하지만 validation loss에 대해서는 shuffle의 여부가 아무런 영향이 없기 때문에 `train_dl`에서만 `shuffle=True`로 설정 했습니다. 

또한 검증 데이터셋의 배치 사이즈가 더 큰 이유는 검증 데이터셋에서는 backpropagation이 필요하지 않으므로 메모리를 덜 사용하기 때문입니다. 즉, 더 큰 배치 크기를 이용하여 손실을 빨리 계산하기 위해 이렇게 설정 했습니다.

~~~python
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
~~~

실행 결과:

~~~python
0 tensor(0.3011)
1 tensor(0.2982)
~~~

`eval()` 은 Batch Normalization update를 해제하고 dropout을 해제하는 기능을 갖고 있습니다. 

`eval()` 과  `with torch.no_grad()` 는 validation/test/inference 시에만 사용한다는 공통점이 있습니다.



#### Create fit() and get_data()

훈련 데이터셋과 검증 데이터셋 모두에 대한 손실을 계산하는 유사한 프로세스를 두 번 거치므로, 이를 하나의 배치에 대한 손실을 계산하는 자체 함수 `loss_batch`를 정의하여 코드를 간단하게 만들어 보려고 합니다.

`loss_batch`의 `opt`인자를 이용해서 훈련 데이터셋과 검증 데이터셋을 구분하려고 합니다.

~~~python
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
~~~



`fit` 은 모델을 훈련하고 각 에포크에 대한 훈련 및 검증 손실을 계산하는 작업을 수행합니다.

~~~python
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
~~~



`get_data` 는 학습 및 검증 데이터셋에 대한 dataloader 를 출력합니다.

~~~python
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
~~~



이제 dataloader를 이용해 데이터를 가져오고 모델을 훈련하는 전체 프로세스를 다음과 같이 3줄로 작성 가능합니다.

~~~python
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
~~~

실행 결과:

~~~python
0 0.30657379207611085
1 0.30652801516652106
~~~



#### Switch to CNN

이제 위에서 구현한 코드들을 이용해 CNN(컨볼루젼 신경망)을 학습하는 데 사용하려고 합니다.

컨볼루젼 레이어가 3개인 CNN을 정의합니다.

~~~python
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))

model = Mnist_CNN()
lr = 0.1
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
~~~

실행 결과:

~~~python
0 0.41064196791648866
1 0.2587754045128822
~~~



#### nn.Sequential

`torch.nn`에서 제공하는 `Sequential` 클래스를 이용하면 안에 포함되어 있는 모듈을 순차적으로 실행합니다. 

쉽게 설명하면 여러 `nn.Module`을 한 컨테이너에 집어넣고 한 번에 돌리는 방법입니다.  코드가 간결해지고 직관적으로 보인다는 장점이 있습니다.

이를 활용하려면 주어진 함수에서 **사용자정의 레이어(custom layer)** 를 쉽게 정의할 수 있어야 합니다. 예를 들어, pytorch에는 view 레이어가 없으므로 사용할 신경망 용으로 만들어야 합니다. `Lambda` 는 `Sequential` 로 신경망을 정의할 때 사용할 수 있는 레이어를 생성하는 역할을 합니다.

~~~python
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)
~~~

`Sequential` 로 생성된 모들은 간단하게 아래와 같습니다.

~~~python
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
~~~

실행 결과:

~~~python
0 0.32116694190502165
1 0.22053358401060105
~~~

-----

### 요약



- **torch.nn**
  - `Module`: 함수처럼 동작하지만, 또한 상태(state) (예를 들어, 신경망의 레이어 가중치)를 포함할 수 있는 호출 가능한 오브젝트를 생성합니다. 이는 포함된 `Parameter` (들)가 어떤 것인지 알고, 모든 기울기를 0으로 설정하고 가중치 업데이트 등을 위해 반복할 수 있습니다.
  - `Parameter`: `Module` 에 역전파 동안 업데이트가 필요한 가중치가 있음을 알려주는 텐서용 래퍼입니다. requires_grad 속성이 설정된 텐서만 업데이트 됩니다.
  - `functional`: 활성화 함수, 손실 함수 등을 포함하는 모듈 (관례에 따라 일반적으로 `F` 네임스페이스로 임포트 됩니다) 이고, 물론 컨볼루션 및 선형 레이어 등에 대해서 상태를 저장하지않는(non-stateful) 버전의 레이어를 포함합니다.
- `torch.optim`: 역전파 단계에서 `Parameter` 의 가중치를 업데이트하는, `SGD` 와 같은 옵티마이저를 포함합니다.
- `Dataset`: `TensorDataset` 과 같이 Pytorch와 함께 제공되는 클래스를 포함하여 `__len__` 및 `__getitem__` 이 있는 객체의 추상 인터페이스
- `DataLoader`: 모든 종류의 `Dataset` 을 기반으로 데이터의 배치들을 출력하는 반복자(iterator)를 생성합니다.

----

#### References

* pytorch tutorial: [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
* blog: [다중분류를 위한 대표적인 손실함수, torch.nn.CrossEntropyLoss](http://www.gisdeveloper.co.kr/?p=8668)