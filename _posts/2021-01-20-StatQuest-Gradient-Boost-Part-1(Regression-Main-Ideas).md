---
layout: post
title:  "StatQuest: Gradient Boost Part 1(Regression Main Ideas)"
date:   2021-01-20 23:21:45
author: Hoon
categories: 머신러닝
use_math: true
---

이번 포스트는 [StatQuest with Josh Starmer: Gradient Boost Part 1 (of 4): Regression Main Ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6) 영상을 보고 정리하였다.

최근 데이터분석시 최종 모델링으로 부스팅 기법 중 하나인 *LightGBM* 을 채택한적이 있다. 이 모델의 원리를 자세히 알고 싶어 영상을 찾아 보게 되었다.

----

#### Gradient Boost

![GradientBoost1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost1.PNG?raw=true)

Gradient Boost는 처음에 하나의 예측을 하며(위와 같은 연속형 변수에서는 주로 평균) *leaf* 하나로 시작을 한다. 그 후에는 이전 트리에서의 잔차에 새로운 트리를 학습 시키는 과정을 반복한다. 이번 영상 예제에서는 *leaf* 가 4개가 될 때 까지만 트리가 커지지만 일반적인 학습 상황에서는 주로 8~32개가 될 때 까지 트리가 커진다.

----

#### Building trees to predict

![GradientBoost2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost2.PNG?raw=true)

Weight들의 평균값을 첫 예측값으로 두고 계산을 하여 각 값들에 대한 잔차를 구한다. 그 후에 그 잔차들과 Height, Favorite Color, Gender를 이용해 구한 트리는 다음과 같다.

![GradientBoost3-2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost3-2.PNG?raw=true)



여기서 잔차를 이용해 학습시키는 것이 살짝 의아할 수 있지만 다음 그림을 보면 쉽게 이해가 된다.

![GradientBoost9.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost9.PNG?raw=true)

Weight들의 평균값으로 구한 첫 예측(leaf)에 잔차를 이용해 구한 첫 트리를 대해 새로운 예측값들을 구한다. 이 경우 첫번째 행의 값을 구해보면 $71.2 + 16.8 = 88$ 이고 이는 실제값과 동일하다. 과연 이것이 정말 좋은 예측인가?에 대한 답은 '아니다'이다. 엄청나게 과적합된 예측이다. 이러한 이유 때문에 Gradient Boost에서는 학습률(learning rate)라는 개념을 이용한다.

![GradientBoost4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost4.PNG?raw=true)

학습률의 값은 0과 1사이이고, 위의 예시에서는 0.1이라고 두었다. 학습률을 트리에 대해서 곱해주게 되면 $71.2+16.8*0.1=72.9$ 이고 이는 학습률이 1인 경우의 예측보다는 좋지 않지만 기존 초기의 예측인 71.2에 비해선 실제값인 88에 가깝다. 즉 잔차를 이용해 학습을 한 트리에 학습률을 곱해서 예측값을 구하면 기존의 값에 비해 실제값에 조금씩 가까워지게 된다.

이제 새로 구한 예측값을 이용해 새로운 잔차를 구한 결과는 다음과 같다.

![GradientBoost8.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost8.PNG?raw=true)



새로 구한 잔차들이 기존의 잔차들에 비해서 줄어든 것을 볼 수 있고 이는 예측값이 실제값에 조금 더 가까워진것을 알 수 있다. 

이제 이러한 과정을 모델 하이퍼 파라미터에서 지정한 트리 수(`n_estimators`) 까지 진행하거나 더 이상의 성능 개선이 없을때까지 진행하면 된다.

![GradientBoost6.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/GradientBoost6.PNG?raw=true)

