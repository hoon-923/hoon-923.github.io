---
layout: post
title:  "Pytorch tutorials- Autograd"
date:   2021-02-11 23:42:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

### Autograd

`autograd` 패키지는 Tensor의 모든 연산에 대해 자동 미분을 제공하며 **define-by-run** 프레임워크 입니다.

파이토치와 같은 define-by-run은 계산 그래프가 생성됨과 동시에 결과를 얻는 방식인 반면 텐서플로우와 같은 define-and-run은 계산 그래프를 먼저 생성한 후에 결과를 계산하여 얻는 방식입니다.

------

### Tensor

Tensor의 `.requires_grad`속성을 `True`로 설정(deafault는 `False`)하면 그 tensor 에서 이뤄진 모든 연산을 추적한 후 `.backward()`를 이용해서 모든 gradient를 자동으로 계산이 가능합니다.

만약 Tensor가 기록을 추적하는 것을 중단하게 하고 싶으면 `.detach()`를 이용하여 연산 기록으로 부터 분리하여 이후 연산들에 대한 추적을 방지할 수 있습니다.

`.detach()`외에도 `with torch.no_grad():`를 이용해 코드 블럭을 감싸면 기록을 추적하는 것을 방지할 수 있습니다. 둘의 기능은 동일하지만 `with torch.no_grad()`를 이용하면 범위 내의 `requires_grad`를 한번에 `False` 로 변경하므로 상황에 맞게 사용하면 됩니다.

Autograd 구현에서 `Function` 클래스도 중요한 역할을 합니다. Tensor와 `Function`은 서로 연결되어 있으며, 모든 연산 과정을 부호화하여 acyclic graph를 생성합니다. 

Tensor는 본인을 생성한 `Function`을 참조하고 있는  `.grad_fn` 속성을 갖고 있습니다. 단, 사용자가 직접 만든 Tensor의 `.grad_fn`은 None입니다.

----

### Tutorial 코드

~~~python
import torch

x = torch.ones(2,2, requires_grad=True)
y = x+2

print(x)
print(y)
print(y.grad_fn)
~~~

실행 결과:

~~~python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
<AddBackward0 object at 0x7f7c503aa978>
~~~

`y`는 연산의 결과로 생성된 Tensor이기 때문에 `grad_fn`을 갖습니다.

~~~python
z = y*y*3
out = z.mean()

print(z)
print(out)
~~~

실행 결과:

~~~python
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
tensor(27., grad_fn=<MeanBackward0>)
~~~

이제 역전파를 위해 `out`에 `backward()`를 수행 해보겠습니다. `out`은 스칼라 값이기 때문에 `backward()`에 따로 인자를 정해주지 않아도 괜찮습니다.

여기서 `x.grad`는 $d(out) \over dx$입니다. 

~~~python
out.backward()
print(x.grad)
~~~

실행 결과:

~~~python
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
~~~

편의상 `out`을 Tensor $o$ 라고 하면, 다음과 같은 과정에 의해 $4.5$ 로 이루어진 행렬이 출력됬음을 알 수 있습니다.

$o= {1 \over 4}\sum_iz_i$와 $z_i=3(x_i+2)^2$ 을 이용하면 $z_i\bigr\rvert_{x_i=1} = 27$ 입니다. 미분 연산을 수행하면 $\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$ 이고, 최종적으로 $\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$입니다. 

이번에는 `with torch.no_grad():` 코드 블럭을 이용해서 블럭내 Tensor `.requires_grad`를 `False`로 바꾸는 코드입니다.

~~~python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
~~~

실행 결과:

~~~python
True
True
False
~~~

이번에는 `.detach()`를 이용해서 기존 Tensor를 복사하되 gradient 전파가 안되는(`.requires_grad = False`) Tensor를 생성하는 코드입니다.

단 storage를 공유하기에 `.detach()`로 생성한 Tensor가 변경되면 원본 Tensor도 똑같이 변합니다.

~~~python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
~~~

실행 결과:

~~~python
True
False
tensor(True)
~~~

`eq`를 이용해 두 Tensor가 같은지도 확인 했습니다.

----

#### References

* pytorch tutorial: [Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)
* pytorch documents: [TORCH.EQ](https://pytorch.org/docs/stable/generated/torch.eq.html)

