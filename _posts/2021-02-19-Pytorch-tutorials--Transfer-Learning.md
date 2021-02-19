---
layout: post
title:  "Pytorch tutorials- Transfer Learning"
date:   2021-02-19 21:45:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

### Transfer Learning

이번 튜토리얼에서는 ResNet18을 활용한 Transfer Learning 예제를 다뤘습니다.

우선 튜토리얼 코드를 보기전에 Transfer Learning에 대해 간단하게 정리하고 넘어 가보겠습니다. 실제로 ConvNet를 학습시킬 수 있는 충분히 큰 데이터셋을 구하기 어렵기 때문에 사용 가능한 크기가 매우 큰 데이터셋(ex. ImageNet)을 이용해 학습 시킵니다. 

학습 시킨 이후 Transfer Learning은 크게 3가지 방식으로 나뉩니다.

* **ConvNet as fixed feature extractor** : 이 방식에서는 다른 데이터셋을 이용해 pret-rained 된 ConvNet을 이용하되 마지막 아웃풋을 출력하는 fully-connected layer만 변경합니다(데이터셋 마다 label 수가 다르기 때문에).
* **Fine-tuning the ConvNet** : 이 방식에서는 마지막 아웃풋을 출력하는 fully-connected layer 뿐만 아니라 pre-trained된 신경망의 weight들도 backpropagation 과정을 통해 미세 조정하는 과정을 거칩니다. 모든 weight를 미세 조정할 수도 있고, 선택적으로 weight들을 미세 조정하는 것도 가능합니다. 주로 초기 layer들의 weight는 그냥 두고 higher-level의 layer들의 weight만 미세 조정하는 경우가 많습니다.
* **Pretrained models** : 최근 ConvNet을 학습 시키는데에는 무려 2~3주의 시간이 소요되는 경우도 종종 있기 때문에 [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) 에서 다른 사람들이 학습한 신경망의 architecture와 weight들을 보고 이용할 수 있습니다.

----

### Tutorial 코드



#### Library

~~~python
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # 대화형 모드
~~~



#### Load Data

데이터를 불러오기 위해 `torchvision`과 `torch.utils.data` 패키지를 사용합니다.

튜토리얼에 필요한 데이터는 [여기](https://download.pytorch.org/tutorial/hymenoptera_data.zip)에서 다운받을 수 있습니다.

데이터를 불러오기 위해 디렉토리에 넣는 과정은 colab 환경을 가정하고 설명하겠습니다. 데이터를 받아보면 zip 형식의 압축파일 일텐데 만약 압축을 푼 후에 google drive에 업로드 하려고 하면 시간이 상당히 많이 소요됩니다. 예제 데이터는 크기가 작은편이라 약 5분 정도 소요되지만 이미지가 많은 큰 데이터의 경우 하루 종일 걸리는 불상사가 발생할 수 있습니다.

이를 방지하려면 google drive에 압축 파일 형식으로 업로드 한 후에 colab에서 압축을 풀어야 한다. 이를 실행하는 코드는 다음과 같다. `!unzip -qo (압축 해제할 파일).zip -d (압축 해제 파일 저장 경로)` 을 이용하면 됩니다.

~~~python
!unzip -qo /content/drive/MyDrive/dataset/hymenoptera_data.zip -d /content/drive/MyDrive/dataset
~~~

![unzip_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/unzip_2.PNG?raw=true)

압축이 풀린 폴더가 지정한 디렉토리 위치에 저장됬음을 알 수 있습니다.

~~~python
# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/drive/MyDrive/dataset/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
~~~

위의 코드를 보면 `RandomResizedCrop`, `RandomHorizontalFlip` 을 이용한 augmentation이 있는데 augmentation에 대해서는 추후 포스팅에 따로 정리 하겠습니다.



#### Visualize few images

~~~python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.


# 학습 데이터의 배치를 얻습니다.
inputs, classes = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지를 만듭니다.
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
~~~

실행 결과:

[few_images]

augmentation이 적용된 이미지들이 보입니다.



#### Training the model

학습을 진행하기 위해 `train_model`이라는 함수를 정의합니다. 밑의 코드를 보면 학습과정에서 learning rate를 조정할 수 있는  `scheduler`를 이용하고 있습니다. `scheduler`의 종류에 대해서도 추후 포스팅할 예정입니다.

~~~python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정, By default all the modules are initialized to train mode (self.training = True)
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 gradient를 0으로 설정
                optimizer.zero_grad()

                # forward
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'): #  informs all the layers to not to create any computational graph for test, because we do not wish to backpropagate for current computations.
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() # w = w - gradient * lr

                # 통계
                running_loss += loss.item() * inputs.size(0) # loss.item() extracts the loss’s value as a Python float
                running_corrects += torch.sum(preds == labels.data) # preds(예측)와 labels(실제값)이 같은 경우
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # state_dict는 각 계층을 매개변수 텐서로 매핑되는 Python 사전(dict) 객체

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts) # state_dict를 사용, 모델의 매개변수들을 불러옵니다
    return model
~~~



#### Visualizing the model predictions

~~~python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
~~~



#### Finetuning the ConvNet

~~~python
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# 여기서 각 출력 샘플의 크기는 2로 설정합니다.(bees와 ants)
# 기존 resnet18의 아웃풋은 1000개(ImageNet에 대해 학습한 network)
# 또는, nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.fc = nn.Linear(num_ftrs, 2)
# print(model_ft.fc)
# Linear(in_features=512, out_features=2, bias=True)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1) # step size마다 gamma 비율로 lr을 감소시킨다 (step_size 마다 gamma를 곱한다)
~~~



#### Train and evaluate

~~~python
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
~~~

실행 결과:

~~~python
Epoch 0/24
----------
train Loss: 0.6118 Acc: 0.7049
val Loss: 0.1636 Acc: 0.9608

Epoch 1/24
----------
train Loss: 0.3965 Acc: 0.8484
val Loss: 0.2612 Acc: 0.8824

Epoch 2/24
----------
train Loss: 0.8354 Acc: 0.7049
val Loss: 0.5537 Acc: 0.8627

Epoch 3/24
----------
train Loss: 0.6156 Acc: 0.7459
val Loss: 0.2724 Acc: 0.9085
        
.....

Epoch 21/24
----------
train Loss: 0.3716 Acc: 0.8238
val Loss: 0.2575 Acc: 0.9085

Epoch 22/24
----------
train Loss: 0.4059 Acc: 0.8566
val Loss: 0.2678 Acc: 0.8889

Epoch 23/24
----------
train Loss: 0.3302 Acc: 0.8689
val Loss: 0.2524 Acc: 0.8889

Epoch 24/24
----------
train Loss: 0.3153 Acc: 0.8770
val Loss: 0.2696 Acc: 0.8758

Training complete in 2m 49s
Best val Acc: 0.960784
~~~

이제 모델의 예측들 중 일부를 시각화 해봅니다.

~~~python
visualize_model(model_ft)
~~~

실행 결과:

![transfer_learning_predict1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/transfer_learning_predict1.PNG?raw=true)



#### ConvNet as fixed feature extractor

이번에는 마지막 fully-connected layer를 제외하고 기존의 pre-trained 된 신경망을 그대로 사용합니다. 이 경우 forward 과정만 필요하기 때문에 gradient를 구할 필요가 없어 학습 속도가 더 빨라집니다. 이를 위해 코드상으로는 `requires_grad = False` 를 작성 해주었습니다.

~~~python
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
~~~

~~~python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
~~~

실행 결과:

~~~python
Epoch 0/24
----------
train Loss: 0.7127 Acc: 0.5984
val Loss: 0.2325 Acc: 0.9150

Epoch 1/24
----------
train Loss: 0.5930 Acc: 0.7090
val Loss: 0.1810 Acc: 0.9281

Epoch 2/24
----------
train Loss: 0.3925 Acc: 0.8197
val Loss: 0.3210 Acc: 0.8497

Epoch 3/24
----------
train Loss: 0.4582 Acc: 0.7746
val Loss: 0.1997 Acc: 0.9542

Epoch 4/24
----------
train Loss: 0.4669 Acc: 0.8156
val Loss: 0.1977 Acc: 0.9412
        
....

Epoch 21/24
----------
train Loss: 0.3659 Acc: 0.8525
val Loss: 0.1847 Acc: 0.9412

Epoch 22/24
----------
train Loss: 0.3044 Acc: 0.8689
val Loss: 0.1876 Acc: 0.9477

Epoch 23/24
----------
train Loss: 0.3861 Acc: 0.8402
val Loss: 0.1775 Acc: 0.9542

Epoch 24/24
----------
train Loss: 0.3116 Acc: 0.8607
val Loss: 0.1683 Acc: 0.9477

Training complete in 1m 21s
Best val Acc: 0.954248
~~~

학습 속도가 빨라졌음을 알 수 있습니다.

~~~python
visualize_model(model_conv)
~~~

실행 결과: 

![transfer_learning_predict2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/transfer_learning_predict2.PNG?raw=true)

----

#### References

* pytorch tutorial: [Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* CS231n: [Transfer Learning](https://cs231n.github.io/transfer-learning/)
* 티스토리 블로그: [PyTorch가 제공하는 Learning rate scheduler 정리](https://sanghyu.tistory.com/113)
* pytorch discussion: [Why we need torch.set_grad_enabled(False) here?](https://discuss.pytorch.org/t/why-we-need-torch-set-grad-enabled-false-here/41240)