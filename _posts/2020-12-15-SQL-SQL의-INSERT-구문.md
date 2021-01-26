---
layout: post
title:  "SQL - SQL의 INSERT 구문(생활코딩)"
date:   2020-12-15 19:48:45
author: Hoon
categories: SQL
---

[이번 영상](https://opentutorials.org/course/3161/19539)에서는 MySQL에서 row를 생성하는 방법에 대해 학습하였다.

row를 생성하기 전에 row를 생성하는 테이블을 확인하기 위해서는 `SHOW TABLES;`를 입력하면 존재하는 table들을 볼 수 있고, `DESC table명;` 를 입력하면 특정 table의 구조를  볼 수 있다.

table에 row를 추가하기 위해서는 `INSERT INTO table명(col1, col2, ········ , coln) VALUES` (col1에 할당할 값, col2에 할당할 값, ········, coln에 할당할 값); 을 실행하면 된다. 시간 입력에서 특정한 값 대신 `NOW()` 라는 함수를 입력하면 현재 시간이 입력된다. 

row가 table에 잘 추가 되었는지 확인하려면 `SELECT * FROM table 명;` 를 입력하면 table의 모든 데이터를 볼 수 있다.