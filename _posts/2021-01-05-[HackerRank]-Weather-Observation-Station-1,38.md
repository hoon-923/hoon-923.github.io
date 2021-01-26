---
layout: post
title:  "[HackerRank] Weather Observation Station 1,3~8"
date:   2021-01-05 23:16:45
author: Hoon
categories: SQL
tag: sql
---

#### [HackerRank] Weather Observation Station 1, 3~8

------

[HackerRank 사이트](https://www.hackerrank.com/dashboard)에 있는 Sql Basic Select - Weather Observation Station 1, 3~8 문제 풀이입니다.

![HackerRank_table1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_table1.PNG?raw=true)

모든 문제는 위 `STATION` 테이블을 바탕으로 주어집니다.

-----

#### [Weather Observation Station 1](https://www.hackerrank.com/challenges/weather-observation-station-1/problem)

**문제**

Query a list of **CITY** and **STATE** from the **STATION** table.

~~~sql
SELECT CITY,STATE
FROM STATION
~~~

`SELECT` 와 `FROM` 의 기본적인 기능만 숙지하고 있으면 풀 수 있는 문제입니다.

----

#### [Weather Observation Station 3](https://www.hackerrank.com/challenges/weather-observation-station-3/problem)

**문제**

Query a list of **CITY** names from **STATION** for cities that have an even **ID** number. Print the results in any order, but exclude duplicates from the answer.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE MOD(ID,2) = 0
~~~

중복을 제거하기 위해  `DISTINCT` 를 사용하였고 나머지를 반환하는 연산인 `MOD` 를 이용해 짝수만 조회하도록 했습니다. `MOD(N,M)` 은 N을 M으로 나눈 나머지를 반환합니다.

----

#### [Weather Observation Station 4](https://www.hackerrank.com/challenges/weather-observation-station-4/problem)

**문제**

Find the difference between the total number of **CITY** entries in the table and the number of distinct **CITY** entries in the table.

~~~sql
SELECT COUNT(CITY) - COUNT(DISTINCT CITY)
FROM STATION
~~~

중복을 제거하지 않은 `CITY`  row수와 중복을 제거한 `CITY` row수의 차이를 구하기 위해 `COUNT` 로 각각을 구한 후 `-` 를 이용해 빼 주었습니다.

-----

#### [Weather Observation Station 5](https://www.hackerrank.com/challenges/weather-observation-station-5/problem)

**문제**

Query the two cities in **STATION** with the shortest and longest *CITY* names, as well as their respective lengths (i.e.: number of characters in the name). If there is more than one smallest or largest city, choose the one that comes first when ordered alphabetically.

~~~sql
SELECT CITY, CHAR_LENGTH(CITY)
FROM STATION
ORDER BY CHAR_LENGTH(CITY), CITY
LIMIT 1;

SELECT CITY, CHAR_LENGTH(CITY)
FROM STATION
ORDER BY CHAR_LENGTH(CITY) DESC, CITY
LIMIT 1;
~~~

`CHAR_LENGTH` 를 이용해 `CITY` 변수의 길이를 구한 변수를 조회하도록 한 후 이를 이용해 한번은 오름차순, 한번은 내림차순으로 정렬하도록 했습니다. 그 후 길이가 같은 경우 도시 이름으로 정렬하기 위해 `ORDER BY` 에 `CITY` 도 추가 했습니다. 마지막으로 각각 가장 첫 row만 조회하도록 `LIMIT 1` 을 사용 했습니다.

----

#### [Weather Observation Station 6](https://www.hackerrank.com/challenges/weather-observation-station-6/problem)

**문제**

Query the list of *CITY* names starting with vowels (i.e., `a`, `e`, `i`, `o`, or `u`) from **STATION**. Your result *cannot* contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY LIKE 'a%' OR CITY LIKE 'e%' OR CITY LIKE 'i%' OR CITY LIKE 'u%' OR CITY LIKE 'o%'
~~~

`LIKE` 를 이용해 영어의 모음(a,e,i,o,u) 중 하나로 시작하는 도시의 값들만 조회하도록 했습니다.

----

#### [Weather Observation Station 7](https://www.hackerrank.com/challenges/weather-observation-station-7/problem)

**문제**

Query the list of *CITY* names ending with vowels (a, e, i, o, u) from **STATION**. Your result *cannot* contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY LIKE '%a' OR CITY LIKE '%e' OR CITY LIKE '%i' OR CITY LIKE '%u' OR CITY LIKE '%o'
~~~

`LIKE` 를 이용해 영어의 모음(a,e,i,o,u) 중 하나로 끝나는 도시의 값들만 조회하도록 했습니다.

----

#### [Weather Observation Station 8](https://www.hackerrank.com/challenges/weather-observation-station-8/problem)

**문제**

Query the list of *CITY* names from **STATION** which have vowels (i.e., *a*, *e*, *i*, *o*, and *u*) as both their first *and* last characters. Your result cannot contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY REGEXP '^[aeiou].*[aeiou]$';
~~~

정규 표현식인 `REGEXP` 를 이용해 문제를 해결 했습니다. 정규 표현식에서 `^` 는 시작을 의미하는 것으로 `^[aeiou]` 는 모음으로 시작하는 단어를 조회합니다.  `.` 는 무슨 글자든지 상관없이 하나,  `*` 는 앞의 글자가 0개 이상 있는 경우를 뜻합니다. 즉 둘을 이은 `.*` 은 어떠한 글자든지 길이에 상관없이 올 수 있다는 의미입니다. 마지막으로 `$` 은 끝을 의미하는 것으로 `[aeiou]$` 는 모음으로 끝나는 단어를 조회합니다.