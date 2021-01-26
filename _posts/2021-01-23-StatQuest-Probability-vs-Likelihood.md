---
layout: post
title:  "StatQuest: Probability vs Likelihood"
date:   2021-01-23 19:44:45
author: Hoon
categories: 통계
use_math: true
---

이번 포스트는 [StatQuest with Josh Starmer 의 Statistics Fundamentals: Probability vs Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4) 영상을 보고 정리하였다.

다소 헷갈릴 수 있는 개념인 Probability(확률)와 Likelihood(가능도)를 확실히 구분해서 정리하기 위해 본 포스트를 작성하였다.

----

#### Probability

확률을 한마디로 정의하면 주어진 확률분포가 있을 때, 관측값 혹은 관측 구간이 분포내에서 얼마의 확률로 존재하는지를 나타낸 값이다. 밑의 쥐의 무게를 측정한 예시를 통해 확률의 개념을 설명해 보았다.

![3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/statistics/statquest/probability_vs_likelihood/3.PNG?raw=true)



위의 분포에서 빨간 면적은 쥐의 무게가 32grams 과 34grams 사이에 위치할 **확률**을 의미한다.  이를 수학적으로 표현하면 다음과 같다.

**pr(weight between 32 and 34 gramsㅣmean = 32 and standard deviation = 2.5)**

여기서 핵심은 '고정된' 분포(위의 수식에서는 mean = 32 and standard deviation = 2.5)에서 앞 부분에 상승하는 부분의 area under the curve를 구하는 것이다.

----

#### Likelihood

가능도는 확률과 반대로 고정되는 요소가 분포가 아닌 관측값들이다. 쉽게 풀어 쓰면 가능도는 어떤 값이 관측되었을 때, 이 값이 어떤 확률 분포에서 왔을지에 대한 확률이다. 이를 밑의 분포를 통해 쉽게 설명해보았다.

![4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/statistics/statquest/probability_vs_likelihood/4.PNG?raw=true)

34 grams의 쥐가 위의 확률분포에서 왔을 확률을 의미하는 것이 **가능도**이다. 이를 수학적으로 표현하면 다음과 같다.

**L(mean = 32 and standard deviation = 2.5 ㅣ mouse weighs 34 grams)**

여기서 핵심은 '고정된' 관측치(위의 수식에서는 mouse weighs 34 grams)가 주어진 분포에서 왔을 확률이 얼마일지를 구하는 것이다.