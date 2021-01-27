---
layout: post
title:  "[MySQL] JOIN(INNER,LEFT,RIGHT,OUTER)"
date:   2021-01-27 14:02:45
author: Hoon
categories: SQL
tag: sql
---

이번 포스트에서는 MySQL에서 가장 핵심적인 기능인 `JOIN`에 대해 설명하려고 한다. `JOIN` 기능 덕분에 MySQL을 관계형 데이터베이스(RDB)라고 할 수 있다고 생각한다. 

-----

#### 예제  테이블 생성

`JOIN` 을 실습해보기 전에 실습에 사용할 예제 테이블들을 생성하였다. 평소에 축구를 좋아해서 축구 선수의 정보가 담긴 player_table과 축구팀의 정보가 담긴 team_table을 생성하여 실습에 사용하였다.

~~~sql
# player_table

CREATE TABLE player_table
(
  Name VARCHAR(50) NOT NULL,
  Team VARCHAR(100) NOT NULL,
  Appearances INT NULL,
  Rating INT NULL,
  Goals INT NULL,
  Assists INT NULL
);

INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Son Heung-Min', 'Tottenham', 28, 7.24, 12, 6);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Kevin De Bruyne',  'Manchester City', 32, 7.97, 13, 20);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Marcus Rashford',  'Manchester United', 31, 7.35, 17, 7);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Mohamed Salah', 'Liverpool', 33, 7.40, 19, 10);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Lionel Messi', 'Barcelona', 32, 8.71, 25, 21);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Karim Benzema', 'Real Madrid', 36, 7.44, 21, 8);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Robert Lewandowski', 'Bayern Munich', 31, 8.13, 34, 4);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Thomas Muller', 'Bayern Munich', 26, 7.42, 8, 21);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Cristiano Ronaldo', 'Juventus', 33, 7.82, 31, 5);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Ciro Immobile', 'Lazio', 36, 7.57, 36, 9);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Neymar', 'PSG', 15, 8.58, 13, 6);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Kylian Mbappe', 'PSG', 17, 8.14, 18, 5);
INSERT INTO player_table (Name, Team, Appearances, Rating, Goals, Assists) VALUES ('Andres Iniesta', 'Vissel Kobe', 22, NULL, 4, 0);
~~~

![player_table.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/SQL_%EB%AC%B8%EB%B2%95/JOIN_1/player_table.PNG?raw=true)

~~~sql
# team_table

CREATE TABLE team_table
(
  Team VARCHAR(50) NOT NULL,
  League VARCHAR(100) NOT NULL,
  Standings INT NOT NULL
);

INSERT INTO team_table (Team, League, Standings) VALUES ('Liverpool', 'Premier League', 1);
INSERT INTO team_table (Team, League, Standings) VALUES ('Manchester City', 'Premier League', 2);
INSERT INTO team_table (Team, League, Standings) VALUES ('Manchester United', 'Premier League', 3);
INSERT INTO team_table (Team, League, Standings) VALUES ('Chelsea', 'Premier League', 4);
INSERT INTO team_table (Team, League, Standings) VALUES ('Tottenham', 'Premier League', 6);
INSERT INTO team_table (Team, League, Standings) VALUES ('Real Madrid', 'LaLiga', 1);
INSERT INTO team_table (Team, League, Standings) VALUES ('Barcelona', 'LaLiga', 2);
INSERT INTO team_table (Team, League, Standings) VALUES ('Atletico Madrid', 'LaLiga', 3);
INSERT INTO team_table (Team, League, Standings) VALUES ('Bayern Munich', 'Bundesliga', 1);
INSERT INTO team_table (Team, League, Standings) VALUES ('RB Leipzig', 'Bundesliga', 2);
INSERT INTO team_table (Team, League, Standings) VALUES ('Juventus', 'Serie A', 1);
INSERT INTO team_table (Team, League, Standings) VALUES ('Inter', 'Serie A', 2);
INSERT INTO team_table (Team, League, Standings) VALUES ('PSG', 'League 1', 1);
INSERT INTO team_table (Team, League, Standings) VALUES ('Marseille', 'League 1', 2);
~~~

![team_table.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/SQL_%EB%AC%B8%EB%B2%95/JOIN_1/team_table.PNG?raw=true)

----

#### INNER JOIN

`INNER JOIN` 은 가장 일반적으로 생각하는 JOIN 방식이다. 

~~~sql
SELECT P.Name, P.Team, T.league
FROM player_table AS P
INNER JOIN team_table AS T
ON P.Team = T.Team;
~~~

![INNER_JOIN.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/SQL_%EB%AC%B8%EB%B2%95/JOIN_1/INNER_JOIN.PNG?raw=true)

위의 그림처럼 두 테이블 모두 포함하는 레코드들을 합쳐서 나타낸다. MySQL에서 `INNER JOIN` 은 그냥 `JOIN` 으로 써도 무방하며 `JOIN` 뒤에 `ON` 을 이용하여 어떤 조건으로 결합할지 결정해주면 된다. 

만약 조건을 걸어주지 않으면 모든 경우의 수에 대해 결합을 하게 되고 row의 수가 엄청 많아지는것을 볼 수 있을 것이다. 이 경우는 대부분의 경우의 원치 않는 결과이기 때문에 꼭 `ON` 을 이용해 결합할 조건을 명시해주어야 한다.

나중에 효율적인 쿼리를 작성하는 방법에 대해 포스팅할 때도 언급하겠지만 3개 이상의 테이블을 `INNER JOIN` 할 때는, 크기가 가장 큰 테이블을 `FROM` 절에 배치하고, `INNER JOIN` 절에는 남은 테이블을 작은 순서대로 배치하는 것이 좋습니다.

-----

#### LEFT JOIN & RIGHT JOIN

`LEFT JOIN` 은 `FROM` 절에 오는 table을 기준으로 합치는 결합 방식이고, `RIGHT JOIN` 은 `JOIN` 절에 오는 table을 기준으로 합치는 결합 방식이다.

예제 테이블을 이용해 쉽게 설명하면 다음과 같다. `FROM` 절 뒤에 player_table, `JOIN` 절에 team_table이 온다고 했을 때, `LEFT JOIN` 을 하면 특정 선수의 팀 정보가 없어도 선수 정보가 결합된 테이블에 표시가 되고 `RIGHT JOIN` 을 하면 특정 팀에 소속된 선수의 팀 정보가 없어도 팀 정보가 결합된 테이블에 표시가 된다.

~~~sql
SELECT P.Name, P.Team, T.league
FROM player_table AS P
LEFT JOIN team_table AS T
ON P.Team = T.Team;
~~~

![LEFT_JOIN.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/SQL_%EB%AC%B8%EB%B2%95/JOIN_1/LEFT_JOIN.PNG?raw=true)

~~~sql
SELECT P.Name, P.Team, T.league
FROM player_table AS P
RIGHT JOIN team_table AS T
ON P.Team = T.Team;
~~~

![RIGHT_JOIN.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/SQL_%EB%AC%B8%EB%B2%95/JOIN_1/RIGHT_JOIN.PNG?raw=true)

----

#### OUTER JOIN

`OUTER JOIN` 은 `LEFT OUTER JOIN` , `RIGHT OUTER JOIN` , `FULL OUTER JOIN` 이 존재한다. 위의 JOIN에서는 INNER Table에 일치하는 레코드가 있으면 가져오고, 일치하는게 없으면 버리는 걸 알 수 있다. 이와 다르게 OUTER JOIN에선 INNER Table에 일치하는 레코드가 없으면 모두 NULL로 채워서 가져온다.

MySQL에서는 `FULL OUTER JOIN` 을 직접 지원하진 않지만 다음과 같이 `UNION` 을 이용해 `OUTER JOIN` 을 구현할 수 있다.

~~~sql
SELECT p.Name, p.Team, t.league
FROM player_table AS p
LEFT JOIN team_table AS t
ON p.Team = t.Team
UNION
SELECT p.Name, p.Team, t.league
FROM player_table AS p
RIGHT JOIN team_table AS t
ON p.Team = t.Team;
~~~

![FULL_OUTER_JOIN.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/SQL_%EB%AC%B8%EB%B2%95/JOIN_1/FULL_OUTER_JOIN.PNG?raw=true)

