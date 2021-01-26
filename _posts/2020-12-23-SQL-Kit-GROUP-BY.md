---
layout: post
title:  "[프로그래머스] SQL Kit - GROUP BY"
date:   2020-12-23 14:45:45
author: Hoon
categories: SQL
tag: sql
---

#### [프로그래머스] GROUP BY

------

프로그래머스 코딩테스트 연습 SQL Kit에 있는 ORDER BY 문제 풀이 입니다.

1. 고양이와 개는 몇 마리 있을까
2. 동명 동물 수 찾기
3. 입양 시각 구하기(1)
4. 입양 시각 구하기(2)

![프로그래머스table3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4table3.PNG?raw=true)

1, 2번 문제는 좌측의  `ANIMAL_INS` 테이블을 바탕으로 주어지고 3, 4번 문제는 우측의  `ANIMAL_OUTS` 테이블을 바탕으로 주어집니다.

------

#### 1. 고양이와 개는 몇 마리 있을까

**문제**

동물 보호소에 들어온 동물 중 **고양이와 개가 각각 몇 마리**인지 조회하는 SQL문을 작성해주세요. 이때 **고양이를 개보다 먼저** 조회해주세요.

~~~sql
SELECT ANIMAL_TYPE, COUNT(ANIMAL_TYPE) AS 'count'
FROM ANIMAL_INS
GROUP BY ANIMAL_TYPE
ORDER BY FIELD (ANIMAL_TYPE, 'Cat', 'Dog');
~~~

조회한 결과 테이블에 count 라는 새로운 컬럼명이 필요하므로 `AS` 를 이용하여 컬럼명을 count로 변경했습니다. 또한 `ORDER BY FIELD` 를 활용하여 특정 값을 우선 정렬하도록 설정했습니다.

------

#### 2. 동명 동물 수 찾기

**문제**

동물 보호소에 들어온 **동물 이름 중 두 번 이상 쓰인 이름**과 **해당 이름이 쓰인 횟수**를 조회하는 SQL문을 작성해주세요. 이때 결과는 **이름이 없는 동물은 집계에서 제외**하며, **결과는 이름 순**으로 조회해주세요.

~~~sql
SELECT NAME, COUNT(NAME) AS 'COUNT'
FROM ANIMAL_INS
WHERE 'NAME' IS NOT NULL
GROUP BY NAME HAVING COUNT(NAME) > 1
ORDER BY NAME;
~~~

이름이 없는 동물은 집계에서 제외하기 위해 `WHERE` 조건에서 `NOT NULL` 인 동물들만 집계하도록 설정했습니다. `WHERE` 절 내부에서는 `COUNT` 를 이용한 집계함수를 사용할 수 없기 때문에 이름 중 두번 이상 쓰인 이름만 출력하는 조건은 `HAVING COUNT` 절을 `GROUP BY`  절과 함께 사용하여 문제의 조건을 고려했습니다.

-----

#### 3. 입양 시각 구하기(1)

**문제**

보호소에서는 몇 시에 입양이 가장 활발하게 일어나는지 알아보려 합니다. **09:00부터 19:59까지, 각 시간대별로 입양이 몇 건**이나 발생했는지 조회하는 SQL문을 작성해주세요. 이때 **결과는 시간대 순**으로 정렬해야 합니다.

~~~sql
SELECT HOUR(DATETIME) AS 'HOUR', COUNT(HOUR(DATETIME)) AS 'COUNT'
FROM ANIMAL_OUTS 
WHERE HOUR(DATETIME) >= 9 and HOUR(DATETIME) <=19
GROUP BY HOUR(DATETIME)
ORDER BY HOUR(DATETIME);
~~~

기존 `DATETIME` 변수는 날짜, 시간, 분을 모두 나타내고 있기때문에 문제의 조건에 따라 시간대별로 조회하기 위해 `HOUR(DATETIME)` 으로 시간만 추출했습니다. 이후에 09:00부터 19:59사이의 값들만 조회하는 조건을 고려 하기 위해 `WHERE` 문을 활용 했습니다.

------

#### 4. 입양 시각 구하기(2)

**문제**

보호소에서는 몇 시에 입양이 가장 활발하게 일어나는지 알아보려 합니다. **0시부터 23시까지, 각 시간대별로 입양이 몇 건**이나 발생했는지 조회하는 SQL문을 작성해주세요. 이때 **결과는 시간대 순**으로 정렬해야 합니다.

~~~sql
SET @hour := -1;
SELECT (@hour := @hour + 1) AS HOUR, 
(SELECT COUNT(*) FROM ANIMAL_OUTS WHERE HOUR(DATETIME) = @hour) AS COUNT
FROM ANIMAL_OUTS
WHERE @hour < 23;
~~~

위 문제와 다른 조건은 다 동일하지만 모든 시간대(0시~23시)에 대해 조회해야 한다는 차이점이 있습니다. 이 문제에서는 로컬 변수를 활용해야 합니다. `SET` 을 이용해 변수명과 초기값을 설정할 수 있습니다. `@` 가 붙은 변수는 프로시저가 종료되어도 유지가 되고 이를 통해 값을 누적하여0부터 23까지 표현할 수 있습니다. `SELECT (@hour := @hour + 1)` 를 이용하여 `@hour` 값 1씩 증가하면서 `SELECT` 문이 실행되고 `WHERE @hour < 23` 일 때까지만 증가합니다.

-------



