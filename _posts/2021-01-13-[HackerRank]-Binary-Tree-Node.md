---
layout: post
title:  "[HackerRank] Binary Tree Node"
date:   2021-01-13 00:10:45
author: Hoon
categories: SQL
tag: sql
---

문제링크: [https://www.hackerrank.com/challenges/binary-search-tree-1/problem](https://www.hackerrank.com/challenges/binary-search-tree-1/problem)

----

You are given a table, *BST*, containing two columns: *N* and *P,* where *N* represents the value of a node in *Binary Tree*, and *P* is the parent of *N*.

![HackerRank_BTN1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/HackerRank_BTN1.PNG?raw=true)

Write a query to find the node type of *Binary Tree* ordered by the value of the node. Output one of the following for each node:

- *Root*: If node is root node.
- *Leaf*: If node is leaf node.
- *Inner*: If node is neither root nor leaf node.

**Sample Input**

![HackerRank_BTN2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/HackerRank_BTN2.PNG?raw=true)

**Sample Output**

```
1 Leaf
2 Inner
3 Leaf
5 Root
6 Leaf
8 Inner
9 Leaf
```

**Explanation**

The *Binary Tree* below illustrates the sample:

![HackerRank_BTN3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/master/_images/HackerRank_BTN3.PNG?raw=true)

---

**코드**

~~~mysql
SELECT N, (
    CASE WHEN P IS NULL THEN 'Root'
    WHEN N NOT IN (SELECT P FROM BST WHERE P IS NOT NULL) THEN 'Leaf'
    ELSE 'Inner' END) AS nodetype
FROM BST
ORDER BY N
~~~

**Key Point**

1. 서브쿼리
2. NULL 처리

----

**해설**

nodetype을 결정하는 값을 반환하는 서브쿼리를 작성하여 문제를 풀이 했습니다. nodetype을 결정하는 논리구조는 다음과 같습니다.

* Parent node가 Null이면 Root
* Parent 컬럼에 값이 없으면 Leaf
* 그 외에는 모두 Inner

기존의 `Leaf` 를 결정하는 코드를 단순하게 `WHEN N NOT IN (SELECT P FROM BST) THEN 'Leaf'` 이라고 작성했었지만 syntax error가 발생했었다. 구글링 해본 결과 `IN` 에서는 `NULL` 이 없어야 정상적으로 작동함을 알 수 있었다. 이를 고려해서 코드를 수정하였다.