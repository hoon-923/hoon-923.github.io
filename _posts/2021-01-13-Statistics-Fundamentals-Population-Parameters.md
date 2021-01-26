---
layout: post
title:  "StatQuest: Population Parameters"
date:   2021-01-13 12:15:45
author: Hoon
categories: 통계
---

이번 포스트는 [StatQuest with Josh Starmer 의 Statistics Fundamentals: Population Parameters](https://www.youtube.com/watch?v=vikkiwjQqfU&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&index=4) 영상을 보고 정리하였다. 

Population Parameters(모수)에 대해 히스토그램과 분포를 이용해서 설명하였다. 모수 추정은 통계에서 가장 기본적이면서도 중요한 개념이라고 생각된다. 모수 추정 방식과 이를 통해서 얻을 수 있는 이점에 대해 정리하고자 한다.

----

#### 샘플과 모집단

![Population_Parameters1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Population_Parameters/Population_Parameters1.PNG?raw=true)

위의 그림은 Gene X의 5개 관측치이다. 만약 값들의 일부가 아닌 모든 값을 관측한다면 다음과 같은 그림이 그려질 것이다(모든 점을 표현하지 못했지만 편의상 2400억개의 점이라 가정).

![Population_Parameters2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Population_Parameters/Population_Parameters2.PNG?raw=true)

위와 같이 모든 값들을 구하게되면 이를 통해 우리는 모집단에 대한 히스토그램과 분포를 알 수 있다. 또한 이를 통해서 Gene X의 값 하나를 선택했을 때 30 이상인 값일 확률을 구할 수 있다.

![Population_Parameters3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Population_Parameters/Population_Parameters3.PNG?raw=true)

모집단은 모든 값들을 관측한 결과이기 때문에 모집단을 나타내는 곡선의 평균과 표준편차는 **모수**라고 불린다.

-----

#### 모수 추정

위와 같이 모든 값들을 관측하면 제일 좋겠지만 현실적으로는 시간과 비용의 문제때문에 모든 값들을 관측하는 것이 불가능하다. 그래서 대부분의 경우 표본집단을 사용하여 모수를 추정한다.

![Population_Parameters4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/statistics/statquest/Population_Parameters/Population_Parameters4.PNG?raw=true)

이와 같이 5개의 관측치를  통해 모수를 추정해보는 과정을 생각해볼 수 있다. 모수를 알고 싶은 이유는 실험에서 도출된 결과가 모집단을 모사할 수 있는지 알기 위해서 이다. 5개의 또 다른 관측치가 있다고 할 때 추정된 모수를 이용해서 실험(위에서 예시를 든 30 이상 값인 확률이 일종의 실험이 될 수 있다)에 대한 결과들에 대한 근거로 사용하기 위함이다. 

여기서 한 가지 혼란이 올 수 있는 부분은 5개의 관측치를 매번 측정할 때마다 표본들의 추정값(표본평균, 표본표준편차)들이 매번 달라진다는 점이다. 우선 첫번째로는 관측치의 수를 2개부터 하나씩 차차 늘려보면 관측치의 수가 커질수록 모수의 값과 유사해짐을 알 수 있다. 즉, 데이터가 많을수록 추정 값에 대한 신뢰가 높아진다고 생각할 수 있다. 통계학의 주요 목표 중 하나는 모수 추정에 대하 얼마나 확신을 할 수 있는지에 대한 정량적인 측정을 하는 것이다. 이를 수치화 하기 위해 p-value 또는 신뢰구간을 이용해 계산한다.

