---
layout: post
title:  "[MySQL] 서브쿼리(Subquery)"
date:   2021-01-08 15:30:45
author: Hoon
categories: SQL
tag: sql
---

 이번 포스트에서는 MySQL에서 종종 유용하게 쓰이는 서브쿼리에 대해 정리 했습니다.

-----

#### 서브쿼리(Subquery)란?

서브쿼리는 하나의 쿼리 안에 포함되어 있는 또 다른 쿼리를 뜻합니다. 서브쿼리에서는 메인쿼리의 컬럼을 사용 가능하지만, 메인쿼리에서는 서브쿼리의 컬럼을 사용할 수 없습니다. 쿼리를 여러번 수행해 결과를 얻어야 하는 경우 서브쿼리를 사용하면 하나의 중첩된 쿼리문으로 작성할 수 있게 됩니다. 

**서브쿼리의 종류**

* WHERE절에 사용하는 Nested Subquery
* FROM절에 사용하는 Inline View
* SELECT절에 사용하는 Scalar Subquery 

**서브쿼리 사용시 주의할 점**

* 서브쿼리는 괄호()안에 작성하여야 한다.
* SELECT문으로만 작성할 수 있다.
* 괄호()가 끝난 뒤에 세미콜론(;)을 사용하지 않는다.
* 서브쿼리에서는 ORDER BY를 사용할 수 없다.

**서브쿼리 사용이 가능한 위치**

* SELECT
* FROM
* WHERE
* HAVING
* ORDER BY
* INSERT문의 VALUES
* UPDATE문의 SET

------

#### WHERE절에 사용하는 Nested Subquery

Nested Subquery의 종류

* 단일행 서브쿼리
* 다중행 서브쿼리
* 다중컬럼 서브쿼리

단일행 서브쿼리의 경우 하나의 행을 리턴하게 되며 코드 예시는 다음과 같습니다.

~~~sql
# 단일행 서브쿼리

SELECT *
FROM table_name
WHERE col_1 = (
		SELECT col_1
		FROM table_name
		WHERE col_2='a')
ORDER BY col_3;
~~~

다중행 서브쿼리의 경우 동시에 여러 행을 리턴하게 되며 코드 예시는 다음과 같습니다.

~~~sql
# 다중행 서브쿼리

SELECT *
FROM table_name
WHERE col_1 IN (
		SELECT col_1
		FROM table_name
		WHERE col_4='a')
ORDER BY col_3;
~~~

다중컬럼 서브쿼리의 결과가 여러 컬럼인 경우이며 코드 예시는 다음과 같습니다.

~~~sql
# 다중컬럼 서브쿼리

SELECT *
FROM table_name
WHERE (col_1, col_2) IN (
			SELECT col_1, MIN(col_2)
			FROM table_name)
ORDER BY col_1;
~~~

다중컬럼 서브쿼리에서는 `WHERE` 절의 컬럼수와 서브쿼리에서 반환하는 컬럼수가 반드시 일치해야 합니다.

-----

#### FROM절에 사용하는 Inline View

일반적으로 FROM 절 뒤에는 테이블 명이 오도록 되어 있습니다. 마찬가지로 서브쿼리가 FROM 절에 사용되면 일종의 뷰(View)처럼 결과가 동적으로 생성된 테이블로 사용할 수 있습니다. 이와 같이 동적으로 생성된 테이블의 컬럼은 자유롭게 참조가 가능합니다. 하지만 이는 임시적인 테이블이기 때문에 데이터베이스에 저장되어지지 않습니다. Inline View 쿼리를 작성할 때 주의할 점은 임시로 생성된 테이블엔 반드시 별칭을 정해주어야 합니다. 이를 지정해주지 않을 시 에러가 발생합니다. 코드 예시는 다음과 같습니다.

~~~sql
# 인라인 뷰

SELECT table_별칭.col_1, table_별칭.col_2
FROM (SELECT *
     	FROM table_명
     	WHERE table_명.col_3='a') table_별칭;
~~~

-----

#### SELECT절에 사용하는 Scalar Subquery

`SELECT` 절에 사용된 서브 쿼리를 사용하면 그 서브 쿼리는 항상 레코드와 컬럼이 각각 하나인 결과를 반환합니다. 코드 예시는 다음과 같습니다.

~~~sql
SELECT col_1, col_2, (
		SELECT COUNT(*)
		FROM table_name) as col_3
FROM table_name
~~~



