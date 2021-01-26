---
layout: post
title:  "StatQuest: The Mean, Variance and Standard Deviation"
date:   2021-01-14 17:19:45
author: Hoon
categories: 통계
use_math: true
---

이번 포스트는 [StatQuest with Josh Starmer 의 Statistics Fundamentals: The Mean, Variance and Standard Deviation](https://www.youtube.com/watch?v=SzZ6GpcfoQY&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&index=5) 영상을 보고 정리하였다. 

----

#### 모평균과 모평균 추정

만약 모든값들을 다 구할 수 있다면 다음과 같이 모집단에 대한 히스토그램과 분포를 그릴 수 있고 이를 통해 모평균을 바로 구할 수 있다. 모평균은 그리스 문자 $\mu$로 표현된다.

![StatQuest5-1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Mean_Var_STD/StatQuest5-1.PNG?raw=true)

하지만 비용과 시간의 문제때문에 현실에서는 모든값들을 다 구하는것이 불가능하고 표본들을 통해 표본 평균 $\bar{X}$ 을 구한다.

![StatQuest5-2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Mean_Var_STD/StatQuest5-2.PNG?raw=true)

모평균과 표본평균 사이에 차이는 존재하지만 표본의 크기가 커질수록 이 차이는 점점 감소한다.

-----

#### 모집단의 분산과 표준편차

모집단에서 관측치들이 얼마나 퍼져있는지에 대한 값인 분산과 표준편차를 구할 수 있다.

![StatQuest5-4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Mean_Var_STD/StatQuest5-4.PNG?raw=true)

분산을 구하는 수식은 다음과 같다.

$\sigma^2 = \frac{1}{N}\sum(X-\mu)^2$

실제 관측치에서 모평균 값을 뺀 값들에 대해 제곱을 취해 음수인 값이 생기지 않도록 보장한다. 하지만 각 항이 제곱되어 있는 상태이기 때문에 분산을 바로 그래프위에 표시할 수 없고 이를 다시 루트를 씌우면 되고 이는 표준편차이다.

$\sigma = \sqrt{\frac{1}{N}\sum(X-\mu)^2}$

----

#### 분산과 표준편차의 추정

위에서 언급했듯이 현실에서 모집단 전부를 알 가능성은 매우 희박하기 때문에 모집단의 분산과 표준편차도 표본을 통해 추정해야 한다. 표본을 통해 추정할때는 모집단 전체에 대한 히스토그램이나 분포를 알 수 없으므로 데이터 값들을 통해 계산하는 방법밖에 없다.

![StatQuest5-3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Mean_Var_STD/StatQuest5-3.PNG?raw=true)

표본분산을 구하는 공식은 기존 분산을 구하는 공식과 비슷하나 모평균 $\mu$ 대신 표본평균 $\bar{X}$ 를 사용하고 분모를 N 대신 N-1을 사용한다. N-1으로 나누는 이유는 모평균 대신 표본평균을 사용하면서 발생한 차이를 보정하기 위함이다. 그렇지 않으면 모평균의 분산을 과소평가 하기 때문이다. 일반적으로 표본평균과 표본 데이터들의 차이가 모평균과 모집단 데이터들의 차이보다 작은 경향이 있다. 즉 N-1로 나누어 이 차이를 보정해주야 한다. 이에 대한 조금 더 자세한 설명은 다음 포스트에 게시하겠다.

이러한 차이를 고려한 표본분산과 표본표준편차 공식은 다음과 같다.

$s^2 = \frac{1}{N-1}\sum(X-\bar{X})^2$

$s = \sqrt{\frac{1}{N-1}\sum(X-\bar{X})^2}$

표본의 크기가 커질수록 추정된 모수(평균, 분산, 표준편차)들은 더욱 정확해지고 신뢰할 수 있게 된다.