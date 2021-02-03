---
layout: post
title:  "StatQuest: Bias and Variance"
date:   2021-02-03 14:30:45
author: Hoon
categories: 머신러닝
use_math: true
---

이번 포스트는 [StatQuest with Josh Starmer: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) 영상을 보고 정리 했습니다.

예전에 핸즈온 머신러닝을 읽으며 bias와 variance에 대한 설명을 읽은 기억이 있습니다. bias 와 variance를 모두 최소화 되게 하는 모델 학습 방식을 원하지만 일반적으로 bias와 variance는 동시에 최소화될 수 없고 이러한 현상을 *bias-variance tradeoff* 라고 한다는 글을 본적이 있습니다. 

bias, variance 각각에 대해 더 잘 이해하고 싶어 본 영상을 시청 했습니다.

----

#### Bias and Variance

![bias_variance_1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/bias_variance/bias_variance_1.PNG?raw=true)

위의 좌측 그래프는 쥐의 키와 무게에 대한 관측치들에 대한 표현이고, 우측 그래프는 관측치들을 train set(blue dots) / test set(green dots) 으로 분할한 결과입니다. 쥐의 키와 무게의 관계에 대해 설명을 잘해주는 곡선은 참고하기 위해 두었습니다.

![bias_variance_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/bias_variance/bias_variance_2.PNG?raw=true)

train set에 대해 Linear Regression을 이용해 fit을 시도하는 상황입니다. 하지만 위의 그래프에서도 볼 수 있듯이 Linear Regression을 이용하면 어떤식으로 fit을 시도해도 관측치들의 관계를 잘 설명하기 어려워 보입니다.

이처럼 머신러닝 모델이 관측치들의 진정한 관계를 파악하지 못하는 상황을 **bias** 라고 합니다.

Linear Regression과 다르게 train set의 모든 관측값에 대해 정확하게 예측 하는 구불구불한 모델을 사용해본 결과는 다음과 같습니다.

![bias_variance_4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/bias_variance/bias_variance_4.PNG?raw=true)

train set에서 두 모델(Linear Regression, 구불구불한 모델)에 대해 각각 [sums of sqaures](https://en.wikipedia.org/wiki/Sum_of_squares)를 구해보면 Linear Regression은 어느정도의 오차가 존재하는 반면 구불구불한 모델은 0입니다. 즉, train set 예측에 대해서는 구불구불한 모델이 Linear Regression 모델을 압도합니다. 하지만 이것이 과연 진짜로 좋은 모델일까요?

![bias_variance_5.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/bias_variance/bias_variance_5.PNG?raw=true)

test set에서 두 모델(Linear Regression, 구불구불한 모델)에 대해 각각 sums of squares를 구해보면 이번에는 Linear Regression의 오차가 구불구불한 모델보다 작은 것을 알 수 있습니다. 즉, 구불구불한 모델은 train set에서는 아주 훌룡한 예측을 했지만 fit을 할 데이터셋이 test set으로 바뀌면 성능이 매우 떨어짐을 알 수 있습니다.

이처럼 fit을 할 데이터셋이 바뀔 때 성능 차이가 심하게 나는 것을 머신러닝 용어로는 **variance** 라고 합니다. 

종합해보면 Linear Regression 모델은 상대적으로 high bias / low variance 이고, 구불구불한 모델은 상대적으로 low bias / high variance 입니다.

----

#### Bias and Variance Tradeoff

아쉽게도 Bias and Variance 유튜브 영상에 둘의 tradeoff 관계에 대한 언급이 전혀 없어서 이에 대해 다른 분의 티스토리 블로그 글을 참고해 정리했습니다.

$D=[(x_1,t_1),(x_2,t_2),...,(x_N,t_N)]$ 

위와 같은 데이터셋이 주어지면 이 데이터를 완벽히 표현하는 $f$ 를 찾는 것이 모델 학습이 목적이다.

$t=f(X) + \varepsilon$

위의 식에서 $t$ 는 target, $X$ 는 데이터, $\varepsilon$ 는 noise 이다. 여기서 $\varepsilon$ 는 평균이 $0$ 이고 분산이 $\sigma^2$ 인 정규분포를 따릅니다. 여기서 loss function으로 MSE(mean squared error) 를 이용하면 loss의 기댓값은 다음과 같이 bias, variance, noise로 분해가 됩니다. 여기서 $y$ 는 학습시키고자 하는 모델이다.

$E[(t-y)^2] = E[(t-f+f-y)^2]$

$= E[(t-f)^2] + E[(f-y)^2] + 2E[(f-y)(t-f)]$

$= E[\varepsilon^2] + E[(f-y)^2]$

$= E[(f-E[y]+E[y]-y)^2] + E[\varepsilon^2]$

$= E[(f-E[y])^2] + E[(E[y]-y)^2] + E[\varepsilon^2]$

최종 결과에서 $E[(f-E[y])^2]$ 는 $bias^2$, $E[(E[y]-y)^2]$ 는 $variance$, $E[\varepsilon^2]$ 는 $noise$ 이다.

$noise$ 는 $y$ 와 독립이기 때문에 최소화 하는것이 불가능하고 결국 $bias$ 와 $variance$ 를 조절해 모델의 loss를 최소화 해야 합니다.

위의 식을 보면 $bias$ 를 최소화하기 위해 $f=E[y]$ 가 되도록 모델을 학습 시키면 $variance$ 항은 $E[(f-y)^2] = E[\varepsilon^2]$ 가 됩니다. 

반대로 $variance$ 를 최소화 하기 위해 $y$ 가 입력 데이터와 상관없이 상수 $c$ 만을 반환하게 되면 $E[(E[y]-y)^2] = E[(c-c)^2] = 0$ 이 되지만, $bias$ 가 증가합니다.

이처럼 bias와 variance은 서로 tradeoff 관계임을 알 수 있습니다. 

----

#### Reference

* wikipedia: [Sum of squares](https://en.wikipedia.org/wiki/Sum_of_squares)
* statquest: [Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA)
* tistory blog: [편향-분산 트레이드오프 (Bias-Variance Tradeoff)](https://untitledtblog.tistory.com/143)

