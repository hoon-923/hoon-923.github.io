---
layout: post
title:  "Pytorch tutorials- Training a Classifier"
date:   2021-02-14 02:00:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

### Training a Classifier

분류기를 학습하기 전에 우선 데이터를 불러와야 합니다. 

일반적으로 PyTorch는 `torch.utils.data.Dataset`으로 Custom Dataset을 생성하고, `torch.utils.data.DataLoader`를 이용해 데이터를 불러옵니다.

[PyTorch TORCH.UTILS.DATA 공식 문서](https://pytorch.org/docs/stable/data.html)를 보니 torch의 `dataset`은 다음과 같은 2가지 스타일이 있습니다.

* Map-style dataset
  - `__getitem__()`과 `__len__()`을 구현해야 함
  - index가 존재하여 data[index]로 데이터를 참조할 수 있음
* Iterable-style dataset
  - `__iter__()`을 구현해야 함
  - random으로 읽기에 어렵거나, data에 따라 batch size가 달라지는 데이터(dynamic batch size)에 적합



이번 튜토리얼에서 사용할 CIFAR10 예제 데이터는 Map-style dataset입니다.

특별히 영상 분야를 위한 `torchvision` 이라는 패키지가 만들어져 있습니다. 여기에는 Imagenet이나 CIFAR10, MNIST 등과 같이 일반적으로 사용하는 데이터셋을 위한 데이터 로더(data loader)인 `torchvision.datasets` 과 이미지용 데이터 변환기 (data transformer)인 `torch.utils.data.DataLoader` 가 포함되어 있습니다.

![CIFAR10_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/CIFAR10_2.PNG?raw=true)

이번 튜토리얼에서는 다음 단계로 이미지 분류기를 학습해보겠습니다.

1. `torchvision` 을 사용하여 CIFAR10의 train / test 데이터셋을 불러오고, 정규화(nomarlizing)합니다.
2. 합성곱 신경망(Convolution Neural Network)을 정의합니다.
3. 손실 함수를 정의합니다.
4. train 데이터를 사용하여 신경망을 학습합니다.
5. test 데이터를 사용하여 신경망을 검사합니다.

----

### Tutorial 코드



#### Loading and normalizing CIFAR10

위에서 언급한 것과 같이 `torchvision`을 이용해서 CIFAR10 데이터셋을 불러옵니다.

~~~python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(                               
    [transforms.ToTensor(),                                   # 이미지 데이터를 tensor로 변경
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize(mean, std) -> 이미지를 정규화

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,            
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
~~~

`transform`은 해당 image를 변환하여 module의 input으로 사용할 수 있게 변환합니다. 이때 여러 단계로 변환해야 하는 경우, `transform.Compose`를 통해서 여러 단계를 묶을 수 있습니다.

이제 불러온 train 데이터 이미지와 레이블을 출력해보겠습니다.

~~~python
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # np.transpose를 이용해 npimg의 여러 축을 변경
                                               # (1, 2, 0)의 경우 기존 tensor의 shape가 (a, b, c) 였으면 (b, c, a) 으로
    plt.show()


# train 이미지를 무작위로 가져오기
dataiter = iter(trainloader) 
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
~~~

실행 결과:

![image_label.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/image_label.PNG?raw=true)



#### Define a Convolutional Neural Network

~~~python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__() # nn.Module을 상속받아서 클래스 안에 속성들을 사용하려고 선언
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net = Net()
print(net)
~~~

실행 결과:

~~~python
Net(
  (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
~~~



#### Define a Loss function and optimizer

~~~python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
~~~

SGD와 momentum을 같이 사용해주는 이유는 단순합니다. momentum을 번역하면 '관성'이라고 할 수 있습니다. 즉, SGD를 이용해 weight를 업데이트를 할 때 이전에 내려왔던 방향도 반영을 해주자는 의미입니다. 이렇게 이전에 움직였던 방향을 고려해줌으로써 속도가 붙습니다. 결국 local minimum이나 saddle point에 걸리더라도 momentum이 없는 vanilla SGD에 비해 이를 탈출할 가능성이 커집니다.



#### Train the network

~~~python
for epoch in range(2):   # 데이터셋을 수차례 반복합니다.

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # optimizer에 0이 아닌 값이 들어감(dw/dL)
        optimizer.step() # network의 가중치가 갱신됨. w = w - lr * dw/dL

        # 통계를 출력합니다.
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
~~~

실행 결과:

~~~python
[1,  2000] loss: 2.162
[1,  4000] loss: 1.858
[1,  6000] loss: 1.668
[1,  8000] loss: 1.554
[1, 10000] loss: 1.500
[1, 12000] loss: 1.475
[2,  2000] loss: 1.392
[2,  4000] loss: 1.372
[2,  6000] loss: 1.347
[2,  8000] loss: 1.333
[2, 10000] loss: 1.304
[2, 12000] loss: 1.306
Finished Training
~~~



#### Test the network on the test data

train 데이터셋을 2회 반복하며 신경망을 학습시켰습니다(2 epoch).

신경망이 예측한 출력과 실제 정답을 비교하는 방식으로 신경망의 성능을 확인 했습니다.

~~~python
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
~~~

실행 결과:

![image_label2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/image_label2.PNG?raw=true)

~~~python
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
~~~

실행 결과:

~~~python
Predicted:   bird   car  ship  ship
~~~

이제 전체 데이터셋에 대한 정확도를 출력합니다.

~~~python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
~~~

실행 결과:

~~~python
Accuracy of the network on the 10000 test images: 53 %
~~~

성능이 썩 좋진 않지만 크기가 작은 신경망을 세부적으로 설정하지도 않았기 때문에 당연한 결과입니다.

이제 각각의 label에 대한 정확도를 출력 해보겠습니다.

~~~python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze() # Returns a tensor with all the dimensions of input of size 1 removed
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
~~~

실행 결과:

~~~python
Accuracy of plane : 55 %
Accuracy of   car : 63 %
Accuracy of  bird : 29 %
Accuracy of   cat : 36 %
Accuracy of  deer : 46 %
Accuracy of   dog : 46 %
Accuracy of  frog : 76 %
Accuracy of horse : 69 %
Accuracy of  ship : 66 %
Accuracy of truck : 48 %
~~~

----

### PyTorch model summary

튜토리얼에는 없는 내용이지만 아주 단순한 코드로 신경망의 요약을 볼 수 있어서 추가했습니다.

~~~python
net = net.to('cuda') # model을 GPU에 올리는 과정

from torchsummary import summary

summary(net, (3, 32, 32))
# parameter 수(첫 Param을 예시로 설명)
# 5 * 5(filter size) * 3(input channel) * 6(output channel) + 6(output channel bias) = 456
~~~

실행 결과:

~~~python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             456
         MaxPool2d-2            [-1, 6, 14, 14]               0
            Conv2d-3           [-1, 16, 10, 10]           2,416
         MaxPool2d-4             [-1, 16, 5, 5]               0
            Linear-5                  [-1, 120]          48,120
            Linear-6                   [-1, 84]          10,164
            Linear-7                   [-1, 10]             850
================================================================
Total params: 62,006
Trainable params: 62,006
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.06
Params size (MB): 0.24
Estimated Total Size (MB): 0.31
----------------------------------------------------------------
~~~

----

#### References

* pytorch tutorial: [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
* pytorch documents: [TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html#module-torch.utils.data)
* blog: [PyTorch가 제공하는 Learning rate scheduler 정리](https://sanghyu.tistory.com/113)

