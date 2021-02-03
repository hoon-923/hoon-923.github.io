---
layout: post
title:  "Pytorch tutorials- Tensors"
date:   2021-02-03 01:15:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

#### Tensor

> Tensor는 NumPy의 ndarray와 유사하며, GPU를 사용한 연산 가속도 가능합니다.

튜토리얼 초반부에 여러 Tensor의 자료형들이 나와서 이를 표로 나타냈습니다.

![tensor_type.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/tensor_type.PNG?raw=true)

메모리의 여유에 따라 데이터 자체의 유효 숫자를 선택하면 됩니다. 특별한 설명이 없으면 default는 `torch.FloatTensor` 입니다.

밑의 코드들은 Tensor들을 생성한 예시입니다.

~~~python
x1 = torch.empty(5, 3) # 초기화되지 않은 행렬 생성
x2 = torch.rand(5, 3) # 무작위로 초기화된 행렬 생성
x3 = torch.zeros(5, 3, dtype=torch.long) # 0으로 채워진 행렬을 생성
x4 = torch.tensor([5.5, 3]) # 데이터로부터 직접 행렬 생성
~~~

이번에는 기존 Tensor를 활용해 새로운 Tensor를 생성하는 과정입니다.

~~~python
x = torch.empty(5, 3)
x = x.new_ones(5, 3, dtype=torch.double) # new_* 메소드는 크기
x = torch.randn_like(x, dtype=torch.float) # dytpe을 override
~~~

----

#### Tensor의 기본 연산

Tensor를 이용해 덧셈을 하는 방식은 다음과 같습니다.

~~~python
x = torch.rand(5, 3)
y = torch.rand(5, 3)


print(x + y) # 1

print(torch.add(x, y)) # 2

result = torch.empty(5, 3) # 3
torch.add(x, y, out=result)

y.add_(x) # 4, in-place 방식
~~~



Tensor의 크기(size) 또는 모양(shape)를 변경하고 싶으면 `torch.view` 사용

~~~python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1은 다른 차원에서 유추
print(x.size(), y.size(), z.size())
~~~



실행 결과

~~~python
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
~~~

`view` 의 특징

* 기존의 데이터와 같은 메모리 공간을 공유합니다.

`view` 의 파이토치 공식문서를 보면 다음과 같습니다.

> *Returns a new tensor with the same data as the `self` tensor but of a different `shape`.*

----

#### CUDA Tensors

`.to` 를 이용해서 Tensor를 원하는 장치(ex. GPU)로 이동시킬 수 있습니다.

밑의 코드는 CUDA가 사용 가능한 환경해서만 실행됩니다.

~~~python
if torch.cuda.is_available():
    device = torch.device("cuda")          # CUDA를 이용해서
    y = torch.ones_like(x, device=device)  # GPU 상에 직접적으로 tensor를 생성하거나
    x = x.to(device)                       # ``.to("cuda")`` 를 사용
    
print(x.device, y.device)
~~~

실행 결과

~~~python
cuda:0 cuda:0
~~~

----

#### Reference

* pytorch tutorial: [Tensors](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)
* pytorch documents: [TORCH.TENSOR](https://pytorch.org/docs/stable/tensors.html)

