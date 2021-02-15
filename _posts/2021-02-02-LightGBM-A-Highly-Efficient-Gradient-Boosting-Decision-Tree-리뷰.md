---
layout: post
title:  "LightGBM: A Highly Efficient Gradient Boosting Decision Tree 리뷰"
date:   2021-02-02 21:15:45
author: Hoon
categories: 머신러닝
use_math: true
---

이번 포스트는 Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu의 *[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)*를 읽고 정리 했습니다.

----

#### Introduction

GBDT는 효율성, 정확성, 해석 가능성 등의 이유로 다양한 머신러닝 작업에서 널리 사용되고 있는 알고리즘 이다. 하지만 갈수록 데이터가 방대해지면서(고차원 변수 + 데이터의 크기), GDBT는 효율성과 확장성에 대해 의문부호가 붙는 상황이다. 각 변수마다 가능한 모든 분할점에 대해 정보 획득을 평가하기 위해 데이터 전체를 모두 봐야한다는 문제점이 존재한다. 이를 위해서는 컴퓨터 연산량이 변수(컬럼)와 행에 의해 모두 영향을 받기 때문에 엄청난 시간이 소요되기 때문에 크기가 큰 데이터를 다루기에는 적합하지 않다. 이러한 문제 때문에 변수의 수와 데이터 행의 수를 줄여야 한다는 생각이 직관적으로 들게 된다. 기존에는 데이터들의 가중치를 이용해 가중치가 높은 중요한 데이터들로만 학습하는 방법이 있으나 GBDT 에는 가중치라는 개념이 존재하지 않아 적용하기 어렵다. 본 논문은 이를 다른 방법으로 해결하기 위한 다음의 두 해결책을 제시한다.

첫 번째는 *Gradient-based One-Side Sampling* (GOSS) 입니다. GDBT에 weight가 존재하지 않아 데이터 크기를 못줄이는 문제를 gradient를 이용해 해결한다. 데이터들은 다른 gradient 크기를 갖고 있고 이에 따라information gain 과정에서 중요도가 달라진다라는 점에 착안했다. gradient 크기가 클수록 중요한 역할을 수행하기 때문에 크기가 큰 gradient 데이터는 그대로 두고 상대적으로 작은 크기의 gradient들의 데이터들을 랜덤하게 드랍한다. 

두 번째는 *Exclusive Feature Bundling* (EFB) 입니다. 실제 대부분의 데이터들이 변수의 수는 매우 많지만 대부분의 변수들이 희소(sparse)하다는 점에 착안한 아이디어이다. 동시에 nonzero 값을 갖는 경우가 거의 없는 상호 배타적인 변수들을 묶음으로써 변수들의 수를 줄일 수 있다.  최적 묶음 문제를 그래프 색칠 문제(변수를 각 꼭짓점에 두고 두 변수가 상호 배타적이지 않으면 두 변수를 잇는 변을 추가함으로써)로 바꾸는 효율적인 알고리즘을 설계한 후에 일정 근사 비율을 갖는 탐욕 알고리즘으로 문제를 해결 했다.

*Gradient-based One-Side Sampling* (GOSS)와 *Exclusive Feature Bundling* (EFB) 를 이용해 GBDT를 새롭게 구현한 머신러닝 기법을 *LightGBM* 이라고 한다. *LightGBM* 은 기존의 방식보다 학습속도를 약 20배 향상시킴과 동시에 정확도를 거의 비슷한 수준으로 유지할 수 있다.

----

#### GBDT and Its Complexity Analysis

GBDT에서 가장 시간이 많이 소요되는 작업은 최적 분할점을 찾는 과정이다. 최적 분할점을 찾는 가장 유명한 알고리즘 중 하나는 *사전 정렬 (pre-sorted)* 알고리즘이다. 가능한 모든 분할점들을 사전적으로 정렬하여 열거하는 방식이다. 이 방식을 통해 최적의 분할점을 찾을 수 있지만 시간이 많이 소요되고 메모리도 많이 사용해서 비효율적이다. 또 다른 유명한 알고리즘은 *히스토그램 기반 (histrogram-based)* 알고리즘이다. 연속적인 변수 값을 개별 구간으로 나누고 이 구간을 사용하여 훈련 시 변수 히스토그램을 만든다. 히스토그램 기반 알고리즘에서 히스토그램을 만드는 일에는 $O($ #$data ×$ #$feature)$ , 분할점을 찾는 일에는 $O($ #$bins ×$ #$feature)$가 소요된다. #$bins$ 은 작기 때문에 결국 #$data$ 와 #$feature$ 가 GDBT 속도의 관건이다.

-----

#### Gradient-based One-Side Sampling

[AdaBoost](https://ko.wikipedia.org/wiki/%EC%97%90%EC%9D%B4%EB%8B%A4%EB%B6%80%EC%8A%A4%ED%8A%B8) 와 다르게 GBDT에는 데이터의 중요도를 확인할 수 있는 weight가 존재하지 않아서 Adaboost의 데이터 샘플링 방식을 사용할 수 없다. 대신에 GBDT에 있는 gradient가 데이터 샘플링시에 유용한 정보를 제공해준다. 쉽게 생각하면 만약 gradient 값이 작으면 학습 에러가 낮다는 뜻이고 이는 이미 학습이 잘 되어 있다는 뜻이다. 여기서 바로 gradient 값이 작은 데이터들을 삭제해버릴 수 있지만 이러면 데이터의 분포가 변화해서 예측의 정확도에 악영향을 미칠 수 있다. 이러한 이유 때문에 *Gradient-based One-Side Sampling* (GOSS) 방식이 탄생하였다.

GOSS 는 gradient가 큰 데이터들은 그대로 두고 gradient가 작은 데이터들을 랜덤하게 샘플링한다. 하지만 이대로 진행하면 분포가 변화하기 때문에 이를 막기 위해서 샘플링 한 gradient가 작은 데이터들에 $1-a \over b$을 곱해준다($a$ 는 상위 gradient 데이터 비중, $b$ 는 샘플링 데이터 비중). 이러한 방식을 채택함으로써 기존 데이터의 분포를 크게 왜곡하지 않고 gradient가 큰 데이터 위주로 학습할 수 있게 된다. 본 논문에서는 $a$ 와 $b$  의 비중을 정하는 것은 아직 과제로 남아있다고 한다. 



