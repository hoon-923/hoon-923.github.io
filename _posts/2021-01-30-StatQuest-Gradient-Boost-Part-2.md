---
layout: post
title:  "StatQuest: Gradient Boost Part 2(Regression Details)"
date:   2021-01-30 01:15:45
author: Hoon
categories: 머신러닝
use_math: true
---

이번 포스트는 [StatQuest with Josh Starmer: Gradient Boost Part 2 (of 4): Regression Details](https://www.youtube.com/watch?v=2xudPOBz-vs) 영상을 보고 정리하였다.

[이전 포스트](https://hoon-923.github.io/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/2021/01/20/StatQuest-Gradient-Boost-Part-1(Regression-Main-Ideas).html)에 설명한 GBM의 전반적인 개요에 이어서 이번 포스트에서는 GBM의 작동 원리를 단계별로 설명합니다.

----

#### Input & Loss Function

![example_table.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GBM_2/example_table.PNG?raw=true)

위 테이블을 이용해 단계별로 설명 했습니다.

**Input:** Data ${(x_i,y_i)}_{i=1}^{n}$ , **Loss Function:** $L(y_i,f(x))$

위의 Input은 단순히 테이블들의 데이터를 나타내는 표현입니다. $x_i$ 는 Height, Favorite Color, Gender에 해당하고 $y_i$ 는 Weight에 해당합니다.

Loss Function은 예측하고 싶은 변수인 Weight를 얼마나 잘 측정했는지를 나타내줄 수 있는 함수입니다. Gradient Boost에서 Regression에 대한 Loss Function은 일반적으로 다음과 같습니다.

${1 \over 2}(Observed-Predicted)^2$

만약 여기서 앞의 ${1 \over 2}$ 을 빼면 이는 squared Residuals 이고 선형 회귀에서 자주 사용되는 Loss Function입니다. 

![linear_graph_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GBM_2/linear_graph_2.PNG?raw=true)

사실 위의 그래프를 보면 ${1 \over 2}$ 이 없는 경우와 있는 경우 모두 초록색 선이 분홍색 선보다 더 Loss 값이 낮은 것을 알 수 있습니다. 그래서 어떤 것을 쓰든 Loss를 측정하는데에는 상관이 없다고 판단할 수 있지만 Gradient Boost에서 ${1 \over 2}(Observed-Predicted)^2$ 을 사용하는 이유는 다음과 같습니다. 추후의 연산 과정에서 chain rule을 활용 하려면 ${1 \over 2}$ 이 있는 것이 편리하기 때문입니다.

${d \over d Predicted}{1 \over 2}(Observed-Predicted)^2$ 

$= {2 \over 2}(Observed-Predicted)*-1$

$= -(Observed-Predicted)$

단순히 음수인 Residuals만 남기 때문에 Gradient Boost에서 미분을 활용하기 더욱 편리해집니다.

----

#### Step 1: Initialize model with a constant value

Input과 Loss Function이 정해지면 가장 첫 단계로 다음과 같은 식을 통해 모델을 상수값으로 초기화 시킵니다.

$F_0(x) = \underset{\gamma}argmin\sum_{k=1}^N L(y_i,\gamma)$

시그마를 포함한 뒷부분은 단순히 Loss Function을 통한 값을 구하는 과정이고 앞의 argmin over gamma는 Loss Function을 통해 구한 값을 최소화 하는 예측값을 찾아야 한다는 의미이다.

${1 \over 2}(88-Predicted)^2 \ \Rightarrow\ -(88-Predicted)$

${1 \over 2}(76-Predicted)^2 \ \Rightarrow\ -(76-Predicted)$

${1 \over 2}(56-Predicted)^2 \ \Rightarrow\ -(56-Predicted)$

위의 예시에 Loss Function을 적용한 후 미분을 한 값들을 더해서 0이 되는 Predicted 값을 찾으면 됩니다. 

$Predicted = {88+76+56 \over 3}=73.3$ 인 평균이고, 이 값은 결국 위의 식의 좌변인 $F_0(x)$ 입니다. 이전 포스트에서 말한 것 처럼 첫 모델은 Leaf(하나의 값)이고 Regression 문제의 경우 결국 평균으로 첫 예측을 시작합니다.

----

#### Step 2: for m = 1 to M: Part A

모든 트리들을 일종의 루프를 통해 생성하는 단계입니다. 대문자 $M$은 총 트리 수를 뜻하고, 소문자 $m$ 은 개별적인 트리를 나타냅니다. 즉 $m = 1$은 첫 번째 트리를 지칭합니다.

Step 2의 첫 부분은 다음과 같습니다.

$Compute\ r_{im} = -\left[{\partial L(y_i,F(x_i)) \over \partial F(x_i)}\right]_{F(x)=F_{m-1}(x)}\ for\ i = 1,.....,n$

매우 복잡해보이지만 사실 우변의 마이너스를 제외한 부분은 이미 미분값을 계산한 $-(Observed-Predicted)$ 과 동일 합니다. 앞에 $-1$을 곱해주면서 $-$ 가 사라진 $(Observed-Predicted)$ 만 남게 됩니다. 결과적으로 저 복잡해보이는 우변은 결국 Residual 입니다. $m=1$인 경우에 대해 설명하고 있기 때문에 $F(x)=F_{m-1}(x)$ 에서 $F_{m-1}(x)$ 은 $F_0(x)$ 이 되어 $73.3$을 대입합니다.

이제 이를 이용해 우변의 $r_{im}$ 을 계산할 수 있습니다. 여기서 $r$ 은 Residual, $i$ 는 샘플 넘버, $m$ 은 몇 번째 트리 인지를 지칭합니다. $i=1$ 인 경우에 대해서만 Residual을 계산한 결과는 다음과 같습니다.

![step_2_table_3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GBM_2/step_2_table_3.PNG?raw=true)

또한 $r_{i,m}$ 은 엄밀하게 말하면 *Pseudo Residuals* 이다. 

----

#### Step 2: for m = 1 to M: Part B

이제 $i=1$ 인 경우에 대해 구한 Residuals를 예측하는 트리를 만든 후에 $R_{j,m},\ for\ j = 1...J{m}$ 에 대해 terminal region을 구합니다. terminal region은 단순히 트리에서 leaf를 뜻합니다. 즉 밑의 트리에서는 $-17.3$ 과 $14.7, 2.7$ 이 terminal region 입니다. $R_{j,m}$ 에서 $j$ 는 각 트리에서 leaf 인덱스 입니다.

![step_2_tree_4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GBM_2/step_2_tree_4.PNG?raw=true)

----

#### Step 2: for m = 1 to M: Part C

이 다음에는 각 leaf들의 output 값을 정하는 과정이 필요합니다. 특히 $R_{2,1}$ leaf는 안에는 값이 두개이기 때문에 output 값이 어떨지 불확실 합니다. 다음 식을 통해 output 값을 정할 수 있게 됩니다.

$For\ j=1...J{m}$  compute  $\gamma_{jm}=\underset{\gamma}argmin\sum_{x_i\in R{ij}} L(y_i, F_{m-1}(x_{i})+\gamma)$ 

첫 단계의 $F_0(x) = \underset{\gamma}argmin\sum_{k=1}^N L(y_i,\gamma)$ 와 매우 유사한 과정이지만 차이가 존재합니다. 첫 번째 차이는 $L(y_i, F_{m-1}(x_{i})+\gamma)$ 을 통해 전 단계의 예측을 고려하는 부분입니다. 또 다른 차이는 $\sum_{x_i\in R{ij}}$ 을 보면 모든 샘플들에 대해 계산하는 것이 아닌 그 특정 리프에 속한 값만 계산을 하는 부분입니다.

leaf $R_{1,1}$ 의 값을 구하기 위해 수식에 구체적인 값들을 입력 해보겠습니다.

$\gamma_{1,1}=\underset{\gamma}argmin {1 \over 2}(y_3- (F_{m-1}(x_{3})+\gamma))^2$ 

$=\underset{\gamma}argmin {1 \over 2}(56- (73.3+\gamma))^2$

$=\underset{\gamma}argmin {1 \over 2}(-17.3-\gamma))^2$

이제 최솟값을 구하기 위해 미분하면

${d \over d \gamma}{1 \over 2}(-17.3-\gamma)^2\ \Rightarrow\ 17.3+\gamma=0$

$\therefore \gamma_{1,1}=-17.3$

똑같은 과정을 통해 $\gamma_{2,1}$ 을 구하면 $8.7$ 이고 이는 결국 Leaf $R_{2,1}$ 에 있는 값들의 평균이다. 이는 특수한 경우가 아니고 Regression의 경우 leaf 안 값들의 평균이 항상 leaf의 output 값이 됩니다.

-----

#### Step 2: for m = 1 to M: Part D

Step 2의 마지막 과정에서는 다음 식을 통해 각 샘플들에 대해 새로운 예측을 합니다.

Update  $F_{m}(x) = F_{m-1}(x) + \nu\underset{y}{\overset{J_m}{\sum}}\gamma_{jm}I(x\in R_{jm})$

이 수식에서 주목해야 하는 $\nu$ 는 학습률이다. 

![step_2_part_d_5.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GBM_2/step_2_part_d_5.PNG?raw=true)

-----

#### Step 3: Output $F_M(x)$

위의 Step 2의 과정들을 통해 m=1인 경우의 과정들을 살펴보았습니다. 이제 이러한 과정들을 M 번 반복하는 것이 Gradient Boost의 학습 과정입니다. 이를 통해 얻어진 $F_m(x)$ 이 최종 output 입니다.

