---
layout: post
title:  "[MySQL] Window 함수"
date:   2021-02-25 16:00:45
author: Hoon
categories: SQL
tag: sql
---

### Window 함수

Window 함수는 **행과 행 사이의 관계**를 쉽게 정의하기 위해 만든 함수입니다. Window 함수를 이용해서 **순위, 합계, 평균** 등을 구할 수 있습니다.

Window 함수의 대표적인 종류는 다음과 같습니다.

* 순위 관련: RANK, DENSE_RANK, ROW_NUMBER
* 집계 관련: SUM, MAX, MIN, AVG, COUNT

이 외에도 다양한 종류가 존재하지만 아직 직접 사용해본적이 없어서 일단 정리하지 않았습니다.

----

### Window 함수 구조

<script src="https://gist.github.com/hoon-923/d0e4947ea71d83cdf7e91d1ca55c064f.js"></script>

Window 함수의 기본 구조는 위와 같습니다. 위의 코드를 하나하나 자세히 살펴보면 다음과 같습니다. 편의상 말로 풀어서 설명해보겠습니다. (이 부분은 다른분의 블로그를 많이 참고했습니다. Reference에도 작성) `WINDOW_FUNCTION`(ex. SUM(), AVG(), RANK() etc.) 을 이용해서 column1을 집계합니다. 하지만 `GROUP BY` 로 집계하여 레코드를 줄이고 싶진 않고, column2에서 고려된 레코드의 이력까지 추적해가면서 보고싶습니다. `PARTITION BY` 에 `GROUP BY` 에 사용하려고 했던 column2를 넣어주어 column2를 기준으로 묶습니다. 정렬을 하고 싶으면 `ORDER BY`를 이용해서 합니다. 만약, 모든 레코드에 대해 하기 싫으면 `WINDOWING OPTIONS`로 레코드의 범위를 지정해줍니다.

위의 설명을 보면 Window 함수의 가장 큰 특징인 기존 레코드 수가 그대로 유지된다는 점을 알 수 있습니다. 

**Window 함수 구조 내부 각각의 역할**

* WINDOWS_FUNCTION(): Window 함수에 따라서 인수를 설정
* PARTITION BY: 특정 컬럼에 의해 소그룹으로 분할
* ORDER BY: 특정 컬럼을 이용해 정렬
* WINDOWING: 행 기준의 범위 지정



**WINDOWING**

* ROWS: 물리적 단위로 범위를 지정
* RANGE: 논리적인 값으로 범위를 지정
* BETWEEN ~ AND ~: 시작과 끝을 지정
* UNBOUNDED PRECEDING: 윈도우 시작 위치가 첫 번째 레코드임을 의미
* UNBOUNDED FOLLOWING: 윈도우 마지막 위치가 마지막 레코드임을 의미
* CURRENT ROW: 윈도우 시작 위치가 현재 레코드임을 의미

----

#### Reference

* 네이버 블로그: [SQL 기본 및 활용 - 윈도 함수 (Window Function)](https://m.blog.naver.com/pikachups/221968972177)

