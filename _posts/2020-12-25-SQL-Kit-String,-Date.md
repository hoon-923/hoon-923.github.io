---
layout: post
title:  "[프로그래머스] SQL Kit - String, Date"
date:   2020-12-25 01:34:45
author: Hoon
categories: SQL
tag: sql
---

#### [프로그래머스] String, Date

---------

프로그래머스 코딩테스트 연습 SQL Kit에 있는 String, Date문제 풀이 입니다.

1. 루시와 엘라 찾기
2. 이름에 el이 들어가는 동물 찾기
3. 중성화 여부 파악하기
4. 오랜 기간 보호한 동물(2)
5. DATETIME에서 DATE로 형 변환

![프로그래머스table3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4table3.PNG?raw=true)

문제 1~3, 5는 좌측의  `ANIMAL_INS` 테이블을 통해 주어지고 문제 4는 좌측의  `ANIMAL_INS` 테이블과 우측의  `ANIMAL_OUTS` 테이블을 통해 주어집니다.

-----

#### 1. 루시와 엘라 찾기

**문제**

동물 보호소에 들어온 동물 중 **이름이 Lucy, Ella, Pickle, Rogan, Sabrina, Mitty**인 동물의 **아이디와 이름, 성별 및 중성화 여부**를 조회하는 SQL 문을 작성해주세요.

~~~sql
SELECT ANIMAL_ID, NAME, SEX_UPON_INTAKE
FROM ANIMAL_INS
WHERE NAME IN ('Lucy', 'Ella', 'Pickle', 'Rogan', 'Sabrina', 'Mitty');
~~~

`WHERE column_명 IN ('A','B')`  를 사용하면 여러 값을 or 관계로 묶어 나열하는 조건을 사용할 수 있습니다. 이를 이용해서 문제의 조건에 주어진 이름들의 동물들을 조회했습니다.

-----

#### 2. 이름에 el이 들어가는 동물 찾기

**문제**

보호소에 돌아가신 할머니가 기르던 개를 찾는 사람이 찾아왔습니다. 이 사람이 말하길 할머니가 기르던 개는 이름에 'el'이 들어간다고 합니다. 동물 보호소에 들어온 동물 이름 중, **이름에 EL이 들어가는 개의 아이디와 이름**을 조회하는 SQL문을 작성해주세요. 이때 결과는 **이름 순**으로 조회해주세요. 단, **이름의 대소문자는 구분하지 않습니다**.

~~~sql
SELECT ANIMAL_ID, NAME
FROM ANIMAL_INS
WHERE NAME LIKE '%el%' AND ANIMAL_TYPE = 'DOG'
ORDER BY NAME;
~~~

`LIKE %el%` 를 이용하여 이름에 el이 포함된 동물들을 조회한 후 `ANIMAL_TYPE = 'DOG'` 를 이용해 이 중 강아지인 값들만 조회했습니다. 두 가지 조건을 동시에 만족 시켜야 했기 때문에 `AND` 를 사용 했습니다.

-----

#### 3. 중성화 여부 파악하기

**문제**

보호소의 동물이 중성화되었는지 아닌지 파악하려 합니다. 중성화된 동물은 `SEX_UPON_INTAKE` 컬럼에 'Neutered' 또는 'Spayed'라는 단어가 들어있습니다. 동물의 아이디와 이름, 중성화 여부를 아이디 순으로 조회하는 SQL문을 작성해주세요. 이때 중성화가 되어있다면 'O', 아니라면 'X'라고 표시해주세요.

~~~sql
SELECT ANIMAL_ID, NAME, 
CASE WHEN SEX_UPON_INTAKE LIKE "%Neutered%" OR SEX_UPON_INTAKE LIKE "%Spayed%" 
THEN "O" ELSE 'X' 
END AS "중성화" FROM ANIMAL_INS
~~~

문제의 조건에 따라 값을 지정해주기 위에 `CASE column_명 WHEN 조건 THEN 값` 을 활용했습니다. 

-----

#### 4. 오랜 기간 보호한 동물(2)

**문제**

**입양을 간 동물 중, 보호 기간이 가장 길었던 동물 두 마리의 아이디와 이름**을 조회하는 SQL문을 작성해주세요. 이때 결과는 **보호 기간이 긴 순**으로 조회해야 합니다.

~~~sql
SELECT ANIMAL_OUTS.ANIMAL_ID, ANIMAL_OUTS.NAME
FROM ANIMAL_OUTS
LEFT JOIN ANIMAL_INS
ON ANIMAL_OUTS.ANIMAL_ID = ANIMAL_INS.ANIMAL_ID
ORDER BY ANIMAL_OUTS.DATETIME - ANIMAL_INS.DATETIME DESC
LIMIT 2;
~~~

문제의 '보호 기간이 가장 길었던' 이라는 조건을 충족하기 위해 `ANIMAL_OUTS.DATETIME - ANIMAL_INS.DATETIME` 이 값이 큰 순으로 정렬한 후 `LIMIT` 를 통해 두 행만 조회하도록 설정 했습니다.

-----

#### 5. DATETIME에서 DATE로 형 변환

**문제**

`ANIMAL_INS` 테이블에 등록된 모든 레코드에 대해, **각 동물의 아이디와 이름, 들어온 날짜**를 조회하는 SQL문을 작성해주세요. 이때 결과는 **아이디 순**으로 조회해야 합니다.

~~~sql
SELECT ANIMAL_ID, NAME, DATE_FORMAT(DATETIME, '%Y-%m-%d') AS '날짜'
FROM ANIMAL_INS
ORDER BY ANIMAL_ID;
~~~

기존의 시간, 분, 초 까지 존재했던 `DATETIME` 컬럼을 `DATE_FORMAT(DATETIME, '%Y-%m-%d')` 을 이용해 들어온 날짜만 남도록 변환 했습니다.

-----

