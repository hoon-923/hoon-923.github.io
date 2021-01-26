---
layout: post
title:  "Quantile Regression"
date:   2021-01-08 00:40:45
author: Hoon
categories: 머신러닝
---

 머신러닝에 관심이 있다면 대부분 일반적인 [Regression Analysis](https://en.wikipedia.org/wiki/Regression_analysis) 는 다들 한번씩은 들어 보셨을꺼라 생각됩니다.  이는 Ordinary Least Square(OLS)의 예측으로 다음과 같은 conditional mean: Y = E[YㅣX]+e 을 추청합니다. 이 경우 특정한 하나의 값을 예측하게 됩니다. 예를 들어 야구 선수의 연봉을 예측하는 Regression Analysis를 진행한 결과 그 선수의 연봉을 $124,189.45 이라고 예측한다면, 과연 그 값이 토시 하나 안틀리고 맞을것이라고 예상하기는 다소 힘듭니다. 다음의 두 가지 경우 중에 어떠한 정보가 더욱 유용하다고 생각이 드나요?

* 선수의 연봉을 $124,189.45 이라고 예측
* 90%의 확신을 갖고 선수의 연봉이 $80,000~$150,000 사이이면서 평균은 $120,000 정도일 것이며 50%의 확신을 갖고 선수의 연봉이 $100,000~$135,000 사이라고 예측

 상식적으로 생각해보면 두 번째 경우가 더욱 유용합니다. 첫 번째는 Ordinary Least Square(OLS) 예측의 Regression Analysis이고 두 번째는 [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression)입니다. Conditional mean을 예측하는 Regression Analysis의 아이디어에서 출발하여 'conditional median이나 아니면 다른 percentile을 예측하는 것은 어떨까?'라는 아이디어에서 출발한 것이 Quantile Regression입니다.

-----

Qunatile Regression은 다음과 같은 장점을 갖고 있습니다.

1. target의 분포에 대한 가정을 하지 않고 직접 modeling 하기 때문에 robust한 모델입니다.
2. 이상치들에 대해 민감하게 반응하지 않습니다.
3. 단조변환(monotonic transformations)에 대해 불변입니다.

----

Quantile Regression을 사용하는 대표적인 두 가지 경우는 다음과 같습니다. 

1. Prediction Interval에 관심이 있는 경우
2. target의 분포가 heteroskedasticity(이분산성)인 경우

여기서 Prediction Interval은 미래의 값들의 어떠한 범위에 올지에 대해 관심을 갖는 Interval 입니다. 반면 흔히 알고 있는 Confidence Interval은 모수가 어떤 범위에 있는지 확률적으로 보여주는 Interval 입니다. Heteroskedasticity에 대해서는 다음 시각화 자료를 활용해 설명 하겠습니다.

![이분산성_등분산성.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/Quantile_Regression/%EC%9D%B4%EB%B6%84%EC%82%B0%EC%84%B1_%EB%93%B1%EB%B6%84%EC%82%B0%EC%84%B1.PNG?raw=true)

Heteroskedasticity(이분산성)와 반대되는 개념으로 Homoscedasticity(등분산성)이 있습니다. 모집단에서 각 독립변수에 대해 종속변수의 값들은 정규분포를 이루는데 이러한 각 정규분포의 표준편차가 전부같은 경우 이를 Homoscedasticity(등분산성)이라고 합니다. 반면 각 독립변수의 값이 증가할수록 표준편차가 증가하는 경우 이를Heteroskedasticity(이분산성)이라고 합니다.

-----

마지막으로는 제가 현재 제가 참가하고 있는 [DACON 공모전 [태양광 발전량 예측 AI 경진대회]](https://dacon.io/competitions/official/235680/overview/)에서 실제로 Quantile Regression을 진행한 결과의 일부입니다.

![QR.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/Quantile_Regression/QR.PNG?raw=true)

