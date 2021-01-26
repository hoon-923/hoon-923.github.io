---
layout: post
title:  "[프로그래머스] SQL Kit - JOIN"
date:   2020-12-25 01:34:45
author: Hoon
categories: SQL
tag: sql
---

#### [프로그래머스] JOIN

---------

프로그래머스 코딩테스트 연습 SQL Kit에 있는 JOIN 문제 풀이 입니다.

1. 없어진 기록찾기
2. 있었는데요 없었습니다
3. 오랜 기간 보호한 동물(1)
4. 보호소에서 중성화한 동물

![프로그래머스table3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4table3.PNG?raw=true)

모든 문제는 좌측의  `ANIMAL_INS` 테이블과 우측의  `ANIMAL_OUTS` 테이블을 바탕으로 주어집니다.

------

#### 1. 없어진 기록찾기

**문제**

천재지변으로 인해 **일부 데이터가 유실**되었습니다. **입양을 간 기록은 있는데, 보호소에 들어온 기록이 없는** 동물의 **ID와 이름**을 **ID 순**으로 조회하는 SQL문을 작성해주세요.

~~~sql
SELECT ANIMAL_OUTS.ANIMAL_ID, ANIMAL_OUTS.NAME
FROM ANIMAL_OUTS
LEFT JOIN ANIMAL_INS
ON ANIMAL_OUTS.ANIMAL_ID = ANIMAL_INS.ANIMAL_ID
WHERE ANIMAL_INS.ANIMAL_ID IS NULL
ORDER BY ANIMAL_ID;
~~~

`table_a LEFT JOIN table_b` 를 이용하여 합치면 앞의 테이블의 기준으로 합쳐집니다. 즉 ` ANIMAL_OUTS`의 값을 기준으로 합쳐지기 때문에` ANIMAL_OUTS`에는 값이 있지만` ANIMAL_INS`에 값이 없는 경우` NULL`으로 표시가 됩니다. 이를 이용하여 두 테이블을 합친 후`WHERE ANIMAL_INS.ANIMAL_ID IS NULL` 를 이용해 문제의 조건처럼 입양을 간 기록은 있는데, 보호소에 들어온 기록이 없는’ 동물들만 조회할 수 있게 됩니다.

-----

#### 2. 있었는데요 없었습니다

**문제**

관리자의 실수로 **일부 동물의 입양일이 잘못 입력**되었습니다. **보호 시작일보다 입양일이 더 빠른** 동물의 **아이디와 이름**을 조회하는 SQL문을 작성해주세요. 이때 결과는 **보호 시작일이 빠른 순**으로 조회해야합니다.

~~~sql
SELECT ANIMAL_OUTS.ANIMAL_ID, ANIMAL_OUTS.NAME
FROM ANIMAL_OUTS
LEFT JOIN ANIMAL_INS
ON ANIMAL_OUTS.ANIMAL_ID = ANIMAL_INS.ANIMAL_ID
WHERE ANIMAL_OUTS.DATETIME < ANIMAL_INS.DATETIME
ORDER BY ANIMAL_INS.DATETIME;
~~~

 ` LEFT JOIN` 을 이용하여 합친 후 문제의 조건처럼 '보호 시작일보다 입양일이 더 빠른' 동물을 조회하기 위해 ` WHERE ANIMAL_OUTS.DATETIME < ANIMAL_INS.DATETIME` 를 이용했습니다.

-----

#### 3. 오랜 기간 보호한 동물(1)

**문제**

**아직 입양을 못 간 동물 중, 가장 오래 보호소에 있었던 동물 3마리**의 **이름과 보호 시작일**을 조회하는 SQL문을 작성해주세요. 이때 결과는 **보호 시작일 순**으로 조회해야 합니다.

~~~sql
SELECT ANIMAL_INS.NAME, ANIMAL_INS.DATETIME
FROM ANIMAL_INS
LEFT JOIN ANIMAL_OUTS
ON ANIMAL_INS.ANIMAL_ID = ANIMAL_OUTS.ANIMAL_ID
WHERE ANIMAL_OUTS.ANIMAL_ID IS NULL
ORDER BY ANIMAL_INS.DATETIME
LIMIT 3;
~~~

 ` table_a LEFT JOIN table_b` 를 이용하여 합치면 앞의 테이블의 기준으로 합쳐집니다. 즉, ` ANIMAL_INS` 의 값을 기준으로 합쳐지기 때문에  ` ANIMAL_INS` 에는 값이 있지만  ` ANIMAL_OUTS` 에 값이 없는 경우  ` NULL` 으로 표시가 됩니다. 이를 이용하여 두 테이블을 합친 후  ` WHERE ANIMAL_OUTS.ANIMAL_ID IS NULL` 를 이용해 보호소에는 들어왔지만 아직 입양을 못 간 동물들을 조회할 수 있습니다.

------

#### 4. 보호소에서 중성화한 동물

**문제**

보호소에서 중성화 수술을 거친 동물 정보를 알아보려 합니다. **보호소에 들어올 당시에는 중성화되지 않았지만, 보호소를 나갈 당시에는 중성화된** 동물의 **아이디와 생물 종, 이름**을 조회하는 **아이디 순**으로 조회하는 SQL 문을 작성해주세요.

~~~sql
SELECT ANIMAL_OUTS.ANIMAL_ID, ANIMAL_OUTS.ANIMAL_TYPE, ANIMAL_OUTS.NAME 
FROM ANIMAL_OUTS 
LEFT JOIN ANIMAL_INS 
ON ANIMAL_OUTS.ANIMAL_ID=ANIMAL_INS.ANIMAL_ID 
WHERE ANIMAL_INS.SEX_UPON_INTAKE LIKE 'Intact%' AND (ANIMAL_OUTS.SEX_UPON_OUTCOME LIKE 'Spayed%' OR ANIMAL_OUTS.SEX_UPON_OUTCOME LIKE 'Neutered%')
~~~

`FROM ANIMAL_OUTS LEFT JOIN ANIMAL_INS` 을 이용하는 이유는 `ANIMAL_INS` 에만 존재하고  `ANIMAL_OUTS` 에는 없는 관측치들은 필요 없기 때문이다.  보호소에 들어오기 전 중성화 수술을 하지 않은 동물들은 `ANIMAL_INS` 테이블 `SEX_UPON_INTAKE` 컬럼의 값에 `Intact` 를 포함하고 있을 것이기 때문에 부분적으로 일치하는 값을 찾는 `LIKE` 를 이용했습니다. 반면 보호소를 나갈 시 중성화된 동물들은 `ANIMAL_OUTS` 테이블 `SEX_UPON_INTAKE` 컬럼의 값에 `Spayed`  또는 `Neutered` 를 포함하고 있을 것이기 때문에 마찬가지로 `LIKE`  를 이용 했습니다.

------

