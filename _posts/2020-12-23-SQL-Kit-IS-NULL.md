---
layout: post
title:  "[프로그래머스] SQL Kit - IS NULL"
date:   2020-12-23 16:13:45
author: Hoon
categories: SQL
tag: sql
---

#### [프로그래머스] IS NULL

------

1. 이름이 없는 동물의 아이디
2. 이름이 있는 동물의 아이디
3. NULL 처리하기

![프로그래머스table2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4table2.PNG?raw=true)

모든 문제는 위 `ANIMAL_INS` 테이블을 바탕으로 주어집니다.

------

#### 1. 이름이 없는 동물의 아이디

**문제**

동물 보호소에 들어온 동물 중, **이름이 없는 채로 들어온 동물의 ID**를 조회하는 SQL 문을 작성해주세요. 단, **ID는 오름차순 정렬**되어야 합니다.

~~~sql
SELECT ANIMAL_ID 
FROM ANIMAL_INS 
WHERE NAME IS NULL 
ORDER BY ANIMAL_ID;
~~~

이름이 없는 채로 들어온 동물의 ID만 조회하기 위해 `WHERE NAME IS NULL` 을 사용했습니다.

-----

#### 2. 이름이 있는 동물의 아이디

**문제**

동물 보호소에 들어온 동물 중, **이름이 있는 동물의 ID**를 조회하는 SQL 문을 작성해주세요. 단, **ID는 오름차순 정렬**되어야 합니다.

~~~sql
SELECT ANIMAL_ID 
FROM ANIMAL_INS 
WHERE NAME IS NOT NULL
ORDER BY ANIMAL_ID;
~~~

이번에는 이름이 있는 동물의 ID를 조회하기 위해 `WHERE NAME IS NOT NULL` 을 사용했습니다.

-----

#### 3. NULL 처리하기

**문제**

입양 게시판에 동물 정보를 게시하려 합니다. 동물의 **생물 종, 이름, 성별 및 중성화 여부**를 **아이디 순**으로 조회하는 SQL문을 작성해주세요. 이때 프로그래밍을 모르는 사람들은 NULL이라는 기호를 모르기 때문에, **이름이 없는 동물의 이름은 No name**으로 표시해 주세요.

~~~sql
SELECT ANIMAL_TYPE, IFNULL(NAME, 'No name') AS NAME, SEX_UPON_INTAKE
FROM ANIMAL_INS
ORDER BY ANIMAL_ID
~~~

`IFNULL` 은 해당 컬럼의 값이 NULL인 경우 다른 값으로 출력할 수 있도록 하는 함수입니다. 

`IFNULL(컬럼명, '병경할 값')` 

------

