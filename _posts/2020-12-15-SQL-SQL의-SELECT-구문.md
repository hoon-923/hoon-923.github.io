---
layout: post
title:  "SQL - SQL의 SELECT 구문(생활코딩)"
date:   2020-12-15 19:48:45
author: Hoon
categories: SQL
---

[이번 영상](https://opentutorials.org/course/3161/19540)에서는 SELECT 구문을 이용해서  데이터를 읽는 방법에 대해 학습하였다. 데이터를 읽는 방법에 대한 문법은 추가 삭제 보다 훨씬 복잡할 수 있어 많은 학습이 요구될 수 있다. 테이블 전체를 보고 싶은 경우 `SELECT * FROM table명`을 입력하면 된다.  만약 특정 컬럼만 선택하고 싶은 경우 `SELECT col1,col2,col3 FROM table명` 을 입력하면 table중 col1,col2,col3의 부분만 읽어온다.

그 외에도 특정 값들만 보고 싶은 경우 WHERE를 이용해 조건을 걸고, 정렬을 하고 싶은 경우 ORDER BY를 이용하면 된다.

MySQL의 경우 한번에 10억건의 데이터 이상도 table에 저장이 가능하다. `SELECT * FROM table명`을 사용해서 한꺼번에 모든 데이터를 읽어오려고 할 경우 컴퓨터가 과부하에 걸릴 가능성이 있다. 이럴시에는 제약을 걸어서 읽어올 필요성 존재하고 `LIMIT`로 row의 수를 제한 가능하다. 

SELECT는 수련이 좀 필요한 문법이기 때문에 이 영상 외의 내용도 자체적으로 학습해볼 필요성이 있다. 다양한

다양한 조건을 적용하는 방법은 직접 [구글링](https://dev.mysql.com/doc/refman/8.0/en/select.html) 해보면서 학습이 가능하다.

