---
layout: post
title:  "Pytorch tutorials- Neural Networks"
date:   2021-02-12 19:01:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

### Neural Networks

신경망은 `torch.nn` 패키지를 사용해 생성할 수 있습니다.

`nn.Module`은 계층과 `output`을 반환하는 `forward(input)` 메서드를 포함하고 있습니다.

![Neural_Network.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/Neural_Network.PNG?raw=true)

위의 Convolution Neural Network를 보면 입력(input)을 받아 여러 계층에 차례로 전달한 후, 최종 출력(output)을 제공합니다.

신경망의 일반적인 학습 과정은 다음과 같습니다.

* 학습 가능한 매개변수(또는 weight)를 갖는 신경망을 정의
* 데이터셋(dataset) 입력을 반복
* 입력을 신경망에서 전파(process)
* 손실(loss)을 계산
* gradient를 신경망의 매개변수들에 역전파
* 신경망의 weight를 갱신($w = w - lr * gradient$, 여기서 $gradient = {dw \over dL}$)

----

### Tutorial 코드



#### Define the network

`nn.Module`을 상속하는 `Net` 클래스를 먼저 구현 했습니다.

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):# nn.Module: Base class for all neural network modules.
  
  def __init__(self):
    super(Net, self).__init__()
    # input channel 1, output channel 6, kernal(filter) size 3*3
    self.conv1 = nn.Conv2d(1, 6, 3)
    # input channel 6, output channel 16, kernal(filter) size 3*3
    self.conv2 = nn.Conv2d(6, 16 ,3)
    # affine 연산: y = Wx + b
    # affine 변환은 신경망 순전파 때 수행하는 행렬의 곱을 기하학에서 부르는 이름
    # nn.Linear: Applies a linear transformation to the incoming data: y = xA^T + by
    # A^T는 A의 전치행렬을 뜻하며 행렬의 곱을위해 대응하는 차원의 원소 수를 일치시키기 위해
    self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6은 이미지 차원
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
  
  # defines how to get the output of the neural net
  # In particular, it is called when you apply the neural net to an input Variable
  def forward(self, x):
    # (2, 2) 크기 윈도우에 대해 max pooling
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # 크기가 제곱수라면 하나의 숫자만 적어도 괜찮
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    # changing x to a vector with self.num_flat_features(x) elements
    # the size of the first dimension is inferred to be 1
    # classification을 위해 CNN 데이터 타입을 Fully Connected의 형태로 변경
    # Flatten layer에는 파라미터 x, 입력 데이터의 shape만 변경
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:] # 배치 차원을 제외한 모든 차원
    num_features = 1
    for s in size:
      num_features *= s # 배치 차원을 제외하고 모두 곱해주어서 flat 하게 펴주는 역할
    return num_features 

net = Net()
print(net)
~~~

실행 결과:

~~~python
Net(
  (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=576, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
~~~

모델의 학습 가능한 매개변수들은 `net.parameters()`에 의해 반환됩니다.

~~~python
params = list(net.parameters())
print(len(params))
for i in range(len(params)):
  print(params[i].size())
~~~

실행 결과:

~~~python
10
torch.Size([6, 1, 3, 3])
torch.Size([6])
torch.Size([16, 6, 3, 3])
torch.Size([16])
torch.Size([120, 576])
torch.Size([120])
torch.Size([84, 120])
torch.Size([84])
torch.Size([10, 84])
torch.Size([10])
~~~

여기서 `params`의 값들을 직접 확인해보면 위의 `torch.size()`에 대응하는 tensor들이 무작위 값들로 채워져 있다.

정의한 신경망은 32x32 input을 예상하고 있기 때문의 임의의 32x32 입력값을 input으로 넣어보겠습니다.

~~~python
# 임의의 32*32를 인풋으로 하고 순전파 진행

input = torch.randn(1, 1, 32, 32) # 여기서 첫번째 값은 batch 차원
out = net(input)
print(out) # 순전파 결과물
~~~

실행 결과:

~~~python
tensor([[ 0.0384, -0.0645, -0.0317,  0.0130, -0.0511, -0.0256, -0.0316,  0.1247,
         -0.0726,  0.0985]], grad_fn=<AddmmBackward>)
~~~

이제 모든 매개변수의 gradient buffer를 0으로 설정하는 과정이 필요합니다.

~~~python
# 역전파 단계를 실행하기 전 gradient를 0으로 만든다.
# network의 가중치 w는 w = w - lr * dw/dL
# 여기서 dw/dL은 gradient이고 이를 매번 역전파 시 초기화 안해주면 gradient buffer에 누적해서 계산하는 문제점 발생
net.zero_grad()
out.backward(torch.randn(1, 10))
~~~

**NOTE**

`torch.nn`은 mini-batch만 지원하기 때문에 하나의 샘플이 아닌, 샘플들의 mini-batch만을 input으로 받습니다.

만약 하나의 샘플만 있다면,  `input.unsqueeze(0)`을 이용하여 가상의 차원을 추가합니다.

> torch.unsqueeze(*input,dim*) -> Tensor

* *Returns a new tensor with a dimension of size one inserted at the specified position*



#### Loss Function

Loss Funcion은 (output, target)을 한 쌍의 입력으로 받아 output이 target으로부터 얼마나 멀리 떨어져 있는지 추정하는 값을 계산합니다.

`nn` 패키지에는 이를 계산하는 Loss Function들이 매우 많습니다. 밑의 예시 코드에서는 MSE(mean-squared error)를 사용합니다.

~~~python
output = net(input)
target = torch.randn(10)  # 예시를 위한 임의의 정답
target = target.view(1, -1)  # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
~~~

실행 결과:

~~~python
tensor(0.2333, grad_fn=<MseLossBackward>)
~~~

`.grad_fn` 속성을 이용하여 `loss`를 역방향에서 따라가면 다음과 같은 연산 그래프를 볼 수 있습니다.

~~~python
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
~~~

즉, `loss.backward()`를 사용하면 전체 그래프는 loss에 대하여 미분되며 그래프 내의 `requires_grad=True` 인 모든 Tensor는 변화도(gradient)가 누적된 `.grad` Tensor를 갖게 됩니다.



#### Backpropagation

오차를 역전파하기 위해서는 `loss.backward()`만 실행하면 됩니다. 하지만 이를 시행하기 전 gradient가 기존의 gradient에 누적되지 않도록 `.zero_grad()`를 이용해 0으로 만드는 작업이 필요합니다.

~~~python
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
~~~

실행 결과:

~~~python
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([ 0.0086,  0.0034, -0.0079, -0.0114, -0.0105,  0.0007])
~~~



#### Update the weights

~~~python
learning_rate = 0.01
for f in net.parameters():
  f.data.sub_(f.grad.data * learning_rate) # w(새로운 가중치) = w(기존 가중치) - lr * dw/dL
~~~

weight를 갱신하는 다양한 방법들은 `torch.optim`에 구현되어 있습니다.

~~~python
import torch.optim as optim

# Optimizer 생성
optimizer = optim.SGD(net.parameters(), lr=0.01)

# training_loop
optimizer.zero_grad() # gradient buffer를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()  # optimizer에 0이 아닌 값이 들어감(dw/dL)
optimizer.step() # network의 가중치가 갱신됨. w = w - lr * dw/dL

print(loss)
~~~

실행 결과:

~~~python
tensor(0.2272, grad_fn=<MseLossBackward>)
~~~

----

#### References

* pytorch tutorial: [Neural Network](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)
* pytorch documents: [TORCH.NN.FUNCTIONAL](https://pytorch.org/docs/stable/nn.functional.html), [TORCH.UNSQUEEZE](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)

