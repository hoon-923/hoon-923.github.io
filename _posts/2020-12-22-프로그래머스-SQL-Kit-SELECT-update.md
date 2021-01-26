---
layout: post
title:  "[프로그래머스] SQL Kit - SELECT"
date:   2020-12-22 17:35:45
author: Hoon
categories: SQL
tag: sql DB
---

#### [프로그래머스] SELECT

--------

프로그래머스 코딩테스트 연습 SQL Kit에 있는 SELECT 문제 풀이 입니다.

1. 모든 레코드 조회하기
2. 역순 정렬하기
3. 아픈 동물 찾기
4. 어린 동물 찾기
5. 동물의 아이디와 이름
6. 여러 기준으로 정렬하기
7. 상위 n개 레코드

![프로그래머스table2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4table2.PNG?raw=true)

모든 문제는 다음의 `ANIMAL_INS` 테이블을 바탕으로 주어집니다.

--------

#### 1. 모든 레코드 조회하기

**문제**

동물 보호소에 들어온 **모든 동물의 정보**를 **ANIMAL_ID순**으로 조회하는 SQL문을 작성해주세요.

~~~sql
SELECT *
FROM ANIMAL_INS
ORDER BY ANIMAL_ID;
~~~

모든 동물의 정보 즉, 모든 컬럼들을 출력하기 위해 `*` 를 사용하였고 ANIMAL_ID순으로 조회 하기 위해 `OREDER BY` 를 사용 했습니다. `ORDER BY` 의 default는 오름차순인 `ASC` 이기 때문에 이 문제의 경우 따로 설정해주지 않아도 괜찮지만 내림차순으로 정렬하고 싶은 경우 `ORDER BY column_name DESC` 를 사용하면 됩니다.

______

#### 2. 역순 정렬하기

**문제**

동물 보호소에 들어온 **모든 동물의 이름과 보호 시작일**을 조회하는 SQL문을 작성해주세요. 이때 결과는 **ANIMAL_ID 역순**으로 보여주세요.

~~~sql
SELECT NAME, DATETIME
FROM ANIMAL_INS
ORDER BY ANIMAL_ID DESC;
~~~

모든 정보가 아닌 동물의 이름과 보호 시작일만 조회하라고 했으므로 `SELECT` 뒤에 두 컬럼명만 입력 한 후 내림차순으로 정렬하기 위해 `DESC` 를 사용 했습니다.

------

#### 3. 아픈 동물 찾기

**문제**

동물 보호소에 들어온 동물 중 **아픈 동물의 아이디와 이름**을 조회하는 SQL 문을 작성해주세요. 이때 결과는 **아이디 순**으로 조회해주세요.

~~~sql
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE INTAKE_CONDITION = 'Sick'
ORDER BY ANIMAL_ID;
~~~

아픈 동물이라는 조건을 고려하기 위해 `WHERE column_name 부등호 조건` 를 이용 했습니다. `WHERE` 문에서 `and` 또는 `or` 를 이용해 여러 조건을 고려할수도 있습니다.

------

#### 4. 어린 동물 찾기

**문제**

동물 보호소에 들어온 동물 중 **젊은 동물의 아이디와 이름**을 조회하는 SQL 문을 작성해주세요. 이때 결과는 **아이디 순**으로 조회해주세요.

~~~sql
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE INTAKE_CONDITION != 'Aged'
ORDER BY ANIMAL_ID;
~~~

이전 문제와 동일하게 조건을 고려하기 위해 `WHERE` 문을 사용하였고, 이 문제에서는 같지 않다라는 조건을 고려하기 위해 `!=` 를 사용 했습니다.

------

#### 5. 동물의 아이디와 이름

**문제**

동물 보호소에 들어온 **모든 동물의 아이디와 이름**을 **ANIMAL_ID순**으로 조회하는 SQL문을 작성해주세요. 

~~~sql
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
ORDER BY ANIMAL_ID;
~~~

1, 2번 문제와 매우 동일한 문제입니다.

------

#### 6. 여러 기준으로 정렬하기

**문제**

동물 보호소에 들어온 **모든 동물의 아이디와 이름, 보호 시작일**을 **이름 순**으로 조회하는 SQL문을 작성해주세요. 단, **이름이 같은 동물 중**에서는 **보호를 나중에 시작한 동물을 먼저** 보여줘야 합니다.

~~~sql
SELECT ANIMAL_ID, NAME, DATETIME
FROM ANIMAL_INS
ORDER BY NAME, DATETIME DESC;
~~~

첫 번째 조건인 이름 순으로 조회(오름차순)하기 위해서 `ORDER BY` 문 뒤에 NAME 컬럼을 먼저 입력한 후 이름이 같은 동물 중에서는 보호를 나중에 시작한 동물을 먼저(내림차순) 보여주기 위해 `DESC` 를 이용 했습니다.

------

#### 7. 상위 n개 레코드

**문제**

동물 보호소에 가장 먼저 들어온 동물의 이름을 조회하는 SQL 문을 작성해주세요.

~~~sql
SELECT NAME
FROM ANIMAL_INS
ORDER BY DATETIME
LIMIT 1;
~~~

가장 먼저 들어온 동물의 이름만 조회하도록 출력 행의 수를 제한하는 `LIMIT` 를 사용 했습니다. 물론 이 문제는 `MIN` 을 사용해서 풀어도 무방합니다.

