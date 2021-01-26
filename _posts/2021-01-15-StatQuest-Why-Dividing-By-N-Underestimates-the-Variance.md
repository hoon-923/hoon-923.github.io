---
layout: post
title:  "StatQuest: Why Dividing By N Underestimates the Variance"
date:   2021-01-15 23:50:45
author: Hoon
categories: 통계
use_math: true
---

이번 포스트는 [StatQuest with Josh Starmer 의 Statistics Fundamentals: Why Dividing By N Underestimates the Variance](https://www.youtube.com/watch?v=sHRBg6BhKjI) 영상을 보고 정리하였다. 

[지난 포스트](https://hoon-923.github.io/%ED%86%B5%EA%B3%84/2021/01/14/StatQuest-The-Mean,-Variance-and-Standard-Deviation.html)에서 표본분산과 표본표준편차는 모집단과 표본집단의 차이를 보정하기 위해 N이 아닌 N-1로 나누어야 한다고 설명하였다. 이번 포스트에서는 이에 대한 구체적인 이유에 대해 정리하고자 한다.

----

#### 표본분산과 표본표준편차를 N-1으로 나누는 이유 - 그림과 예시

표본분산을 N으로 나누는 것이 왜 모집단의 분산을 과소측정하게 되는 것인지에 대해 몇 가지 예시를 들어 설명하고자 한다.

![StatQuestN-1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Dividing_By_N/StatQuestN-1.PNG?raw=true)

위의 상황에서 표본평균 $\bar{X}$ 를 임의로 0이라고 하고 N으로 나눈 표본분산을 계산하면 391이 나온다.

![StatQuestN-2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Dividing_By_N/StatQuestN-2.PNG?raw=true)

이번에는 표본평균  $\bar{X}$ 를 임의로 5라고 하고 N으로 나눈 표본분산을 계산하면 240이 나온다.

이런식으로 표본평균의 위치를 조금씩 이동하면서 N으로 나눈 표본분산을 구한 결과는 다음과 같다.

![StatQuestN-3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Dividing_By_N/StatQuestN-3.PNG?raw=true)



여기서 눈여겨 볼 점은 Variance가 가장 작은 값은 표본평균인 17.6으로부터 구했다는 것 이다. 모평균인 20으로부터 구한 Variance는 표본평균으로 구한 Variance 오른쪽 값이다.

![StatQuestN-4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Dividing_By_N/StatQuestN-4.PNG?raw=true)

실제 분포는 위와 같은 경우지만 표본분산을 N으로 나눈 경우의 값은 다음과 같다.

$\frac{1}{N}\sum(X-\bar{X})^2 = 81.4  < \frac{1}{N}\sum(X-\mu)^2 = 87.4$

즉, 표본평균으로 표본분산을 구할 때 N으로 나누게되면 모평균 근처의 분산을 과소평가하는 상황이 발생한다. 

-----

#### 표본분산과 표본표준편차를 N-1으로 나누는 이유 - 수식

위에서는 한 가지 경우에 대해서만 저런 상황이 발생하는 것을 확인했지만 표본분산과 표본표준편차를 N으로 나누게 되면 항상 위와 같은 상황이 발생함을 증명할 수 있다.

우선 위의 분산 값들의 점이 있는 분포에서 가능한 모든 값(v)들을 구한 후 선으로 이으면 다음과 같다.

![StatQuestN-5.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Dividing_By_N/StatQuestN-5.PNG?raw=true)

이제 기존의 식에서 $\bar{X}$ 대신에 $V$ 를 대입해 다음과 같이 쓸 수 있다.

$\frac{1}{N}\sum(X-V)^2$

그리고 미분값을 이용해서 위의 곡선에서 접선의 기울기들을 구할 수 있고 기울기가 0인 경우 분산을 가장 작게 만드는 $V$ 값을 찾을 수 있다. 이를 위한 미분 과정은 다음과 같다.

${d \over dV}\frac{1}{N}\sum(X-V)^2 = \frac{1}{N}\sum2(X-V)x-1 = \frac{1}{N}\sum-2(X-V) = \frac{-2}{N}\sum(X-V)$

최종결과 $\frac{-2}{N}\sum(X-V)$ 는 위 곡선에서 접선의 기울기에 해당된다. 이제 이 값이 0이 되는 분산이 최소값이 되는 점을 찾는 방법을 다음 2가지 방식으로 설명하려고 한다.

**1. 측정한 데이터들을 직접 미분값에 대입 - 예시**

관측치가 총 5개이고 이들의 값이 3, 13, 19, 24, 29이라고 하면 다음과 같이 대입한 후 $V$ 를 계산하면 된다.

$\frac{-2}{5}\left[(3-V)+(13-V)+(19-V)+(24-V)+(29-V) \right]=0$

이 경우 $V=17.6$ 이고 이는 $\bar{X}$ 의 값과 동일하다.

**2. 수식을 통한 설명 - 일반적인 경우**

이번에는 수식에 실제로 측정한 값이 아닌 $x_1,x_2, ... ,x_n$ 를 수식에 대입해서 증명하고자 한다.

$\frac{-2}{N}\left[(x_1-V)+(x_2-V)+...+(x_n-V) \right]=0$

위의 식 양변에 $-\frac{N}{2}$ 를 곱해 정리하면

$V = \frac{x_1+x_2+...+x_n}{N}$

즉, 이 경우에도 $V$ 는 값들의 평균이고 이는 $\bar{X}$ 의 값과 동일하다. 이를 통해 어떤 값들을 위의 공식에 대입해도 결과적으로  $\bar{X}$ 의 값과 동일해짐을 알 수 있다.

----

#### 공식에서 제곱을 사용하는 이유

만약 분산을 구하는 공식에서 제곱 대신 절댓값을 사용하면 분산의 분포 그래프는 다음과 같을 것이다.

![StatQuestN-6.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Dividing_By_N/StatQuestN-6.PNG?raw=true)

최소값이 첨점이 되어버려 미분이 불가능해져 이러한 이유 때문에 제곱을 해준다.