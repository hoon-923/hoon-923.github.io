---
layout: post
title:  "[HackerRank] Weather Observation Station 9~12"
date:   2021-01-06 16:50:45
author: Hoon
categories: SQL
tag: sql
---

#### [HackerRank] Weather Observation Station 9~12

------

[HackerRank 사이트](https://www.hackerrank.com/dashboard)에 있는 Sql Basic Select - Weather Observation Station 9~12 문제 풀이입니다.

![HackerRank_table1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_table1.PNG?raw=true)

모든 문제는 위 `STATION` 테이블을 바탕으로 주어집니다.

-----

#### [Weather Observation Station 9](https://www.hackerrank.com/challenges/weather-observation-station-9/problem)

**문제**

Query the list of *CITY* names from **STATION** that *do not start* with vowels. Your result cannot contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY REGEXP '^[^aeiou]';
~~~

정규표현식에서 `[]` 안의 `^` 는 `^ ` 뒤 문자들을 포함하지 않는 문자열을 찾는 기능을 합니다.

-----

#### [Weather Observation Station 10](https://www.hackerrank.com/challenges/weather-observation-station-10/problem)

**문제**

Query the list of *CITY* names from **STATION** that *do not end* with vowels. Your result cannot contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY REGEXP '[^aeiou]$';
~~~

위의 9번 문제는 모음으로 시작하지 않는 도시들을 찾는 문제이고 이번 10번 문제는 모음으로 끝나지 않는 도시들을 찾는 문제이기 때문에 풀이가 거의 동일합니다.

----

#### [Weather Observation Station 11](https://www.hackerrank.com/challenges/weather-observation-station-11/problem)

**문제**

Query the list of *CITY* names from **STATION** that either do not start with vowels or do not end with vowels. Your result cannot contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY REGEXP '^[^aeiou]|[^aeiou]$';
~~~

문제의 조건을 충족시키기 위해 MySQL 정규표현식에서 or 기능을 하는 `|` 을 사용 했습니다.

-----

#### [Weather Observation Station 12](https://www.hackerrank.com/challenges/weather-observation-station-12/problem)

**문제**

Query the list of *CITY* names from **STATION** that *do not start* with vowels and *do not end* with vowels. Your result cannot contain duplicates.

~~~sql
SELECT DISTINCT CITY
FROM STATION
WHERE CITY REGEXP '^[^aeiou].*[^aeiou]$';
~~~

조회하고자 하는 도시가 동시에 모음으로 시작하지 않고 끝나지 않도록 하기 위해 시작과 끝 조건을 정했습니다.

