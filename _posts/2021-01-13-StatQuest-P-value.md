---
layout: post
title:  "StatQuest: P-value"
date:   2021-01-13 16:50:45
author: Hoon
categories: 통계
---

이번 포스트는 [StatQuest with Josh Starmer 의 Statistics Fundamentals: How to calculate p-values](https://www.youtube.com/watch?v=JQc3yx0-Q9E&feature=youtu.be) 영상을 보고 정리하였다. 

평소에 p-value에 대해 기본적인 이해는 하고 있는 상태였지만 더욱 정확하게 알고 싶어 영상을 보고 정리하게 되었다.

----

#### 동전 던지기 예시(discrete variable)와 p-value의 구성요소

동전을 던져서 두번 연속 앞면이 나오면 이 동전이 특별한가? 라는 물음에 대해 p-value를 계산하면 통계적으로 그렇다 또는 아니다라고 답할 수 있다.

![pvalue2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/pvalue2.PNG?raw=true)

위의 그림을 보면 동전을 두번 던졌을 시 모든 경우의 수 와 확률을 알 수 있다. p-value를 계산하기 전 p-value가 어떠한 값들로 구성되어 있는지 알아야 한다.

**p-value 구성**

1. The probability random chance would result in the observation
2. The probability of observing something else that is equally rare
3. The probability of observing something rarer or more extreme

동전을 두번 던지는 경우 앞면이 두번 연속 나오는 경우의 p-value를 계산해보면 0.25(앞면 두번) + 0.25(뒷면 두번) + 0(앞면 두번보다 확률이 낮은 경우 x)로 0.5이다. 2번 구성요소를 더해주는 이유는 동일하게 희귀한 확률을 가진 다른 것들이 있다면 두번 연속 앞면이 나온 것이 특별하지 않기 때문이다. 3번 구성요소도 비슷한 이유에 의해서 더해준다.

p-value를 이용해 통계적으로 검증하기 위해서 '이 동전이 일반 동전과 다른 것이 없다' 라는 귀무가설을 설정하고 이를 기각하게 되면 이 동전이 특별하다고 말할 수 있다. 일반적으로 귀무가설을 기각하려면 p-value가 0.05보다 작아야 하지만 이 경우 0.05보다 큰 값인 0.5이기 때문에 귀무가설은 기각되지 못하고 이 동전은 일반 동전과 다르지 않다는 결론이 난다.

----

#### 사람 키 예시(continous variable)와 p-value 계산

사람의 키와 같은 연속형 변수에 대한 확률과 p-value를 계산할때는 분포를 사용한다.

![pvalue3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/pvalue3.PNG?raw=true)

곡선 아래의 영역(넓이)는 어떤 사람의 키가 해당 범위 안에 들어갈 확률을 의미한다. 위의 1996년에 측정한 브라질 여성의 키 예시에서는 어떤 사람의 키가 142cm와 169cm 사이에 위치할 확률이 95%이라는 것을 의미한다. 또한 2.5%의 확률로 키가 169cm 보다 클 것이고 마찬가지로, 2.5%의 확률로 142cm 보다 작을 것이다.

분포를 통해 p-value를 계산하려면, 곡선 아래 영역의 확률(%)을 더하면 된다. 위의 예시에서 키가 142cm로 측정된 사람이 평균 값이 155.7인 분포로부터 나온 것인지에 대한 의문을 p-value를 통해 구할 수 있다. 이 경우 귀무가설은 위의 질문에 대해 그렇다일 것이고 이를 p-value를 통해 기각 또는 채택할 수 있다.

![pvalue4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/pvalue4.PNG?raw=true)

p-value 구성 요소들을 생각해보면 142cm 이하인 값만큼 169cm 이상인 값들도 동일하거나 더 극단적인 값으로 간주되기 때문에 p-value값에 더해주어야 한다. 즉, p-value는 0.025 + 0.025로 결과적으로 0.05이고 '키가 142cm인 사람이 위의 분포로부터 나왔다'라는 가설에 대한 p-value는 0.05이다. 일반적으로 유의성을 검정하는 p-value 기준값이 0.05이므로 귀무가설을 기각하기도 채택하기도 애매한 상황이다. 만약 142cm이 아닌 141cm 이였다면 최종적인 p-value는 0.032였을 것이고 이 경우에는 귀무 가설을 기각할 수 있다.

-----

#### One-Sided p-values의 위험성

![pvalue5.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/pvalue5.PNG?raw=true)

위의 예시는 사람들이 특정 질병으로부터 회복하는데 걸리는 시간에 대한 분포이다. 이 상황에서 신약을 개발해 회복하는데 시간을 단축시키는지를 확인하고 싶은 상황이다. 

기존의 양측 p-value 검정에서 만약 신약으로 인해 평균 회복일이 4.5일이라면 p-value는 0.32일 것이고 '신약이 효과가 없을 것'이라는 귀무가설을 기각해 이 경우 신약이 사람들의 회복 시간을 단축시켰다고 말할 수 있다.

단측 p-value 검정에서는 어느 방향으로 변화가 일어남을 볼건지 결정해야 한다. 이 경우에서는 신약이 회복시간을 줄여주는 방향의 효과를 보고 싶을 것이다. 방향을 그쪽으로만 정해두었기 때문에 더 극단적인 값들은 4.5일 이하의 값들 뿐이다. 이 경우 p-value는 0.016이고 0.05보다 작기 때문에 양측 검정과 동일하고 귀무가설을 기각할 수 있게 된다.

만약 신약이 오히려 회복을 악화시키는 경우여서 오히려 평균 회복일이 15.5일인 경우를 가정해보자. 이 경우에도 양측 검정은 0.016 + 0.016으로 0.032이다. 다시 말해서 신약이 효과가 있든 병세를 악화를 시키든 양측 p-value 검정에서는 무언가 특별한 일이 발생했다는것을 알 수 있다.

단측 p-value에서는 이 경우 신약이 회복시간을 줄여주는 방향의 효과를 보게되면 p-value 값이 0.98이 나올 것이고 신약이 어떤 특별한 영향이 있었는지에 대해 알 수 없게 된다. 단측 p-value 검정은 관련 분야에 대한 도메인 지식이 있는 전문가에 의해서만 사용되는 것이 바람직하다.