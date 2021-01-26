---
layout: post
title:  "[HackerRank] New Companies"
date:   2021-01-10 00:07:45
author: Hoon
categories: SQL
tag: sql
---

문제링크: [https://www.hackerrank.com/challenges/the-company/problem](https://www.hackerrank.com/challenges/the-company/problem)

Amber's conglomerate corporation just acquired some new companies. Each of the companies follows this hierarchy:

![HackerRank_New_Companies.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_New_Companies.PNG?raw=true)

Given the table schemas below, write a query to print the *company_code*, *founder* name, total number of *lead* managers, total number of *senior* managers, total number of *managers*, and total number of *employees*. Order your output by ascending *company_code*.

**Note:**

* The tables may contain duplicate records.
* The *company_code* is string, so the sorting should not be **numeric**. For example, if the *company_codes* are *C_1*, *C_2*, and *C_10*, then the ascending *company_codes* will be *C_1*, *C_10*, and *C_2*.

---

**Input Format**

The following tables contain company data:

* *Company:* The *company_code* is the code of the company and *founder* is the founder of the company.

![HackerRank_Company.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_Company.PNG?raw=true)

* *Lead_Manager:* The *lead_manager_code* is the code of the lead manager, and the *company_code* is the code of the working company.

![HackerRank_Lead_Manager.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_Lead_Manager.PNG?raw=true)

* *Senior_Manager:* The *senior_manager_code* is the code of the senior manager, the *lead_manager_code* is the code of its lead manager, and the *company_code* is the code of the working company.

![HackerRank_Manager.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_Manager.PNG?raw=true)

* *Manager:* The *manager_code* is the code of the manager, the *senior_manager_code* is the code of its senior manager, the *lead_manager_code* is the code of its lead manager, and the *company_code* is the code of the working company.

![HackerRank_Manager.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_Manager.PNG?raw=true)

* *Employee:* The *employee_code* is the code of the employee, the *manager_code* is the code of its manager, the *senior_manager_code* is the code of its senior manager, the *lead_manager_code* is the code of its lead manager, and the *company_code* is the code of the working company.

![HackerRank_Employee.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/HackerRank_Employee.PNG?raw=true)

----

**Sample Input**

Company Table: 

![Sample_1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/Sample_1.PNG?raw=true)

Lead_Manager Table: 

![Sample_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/Sample_2.PNG?raw=true)

 Senior_Manager Table: 

![Sample_3.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/Sample_3.PNG?raw=true)

 Manager Table: 

![Sample_4.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/Sample_4.PNG?raw=true)

 Employee Table: 

![Sample_5.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/SQL/HackerRank/Sample_5.PNG?raw=true)

**Sample Output**

~~~
C1 Monika 1 2 1 2
C2 Samantha 1 1 2 2
~~~

**Explanation**

In company *C1*, the only lead manager is *LM1*. There are two senior managers, *SM1* and *SM2*, under *LM1*. There is one manager, *M1*, under senior manager *SM1*. There are two employees, *E1* and *E2*, under manager *M1*.

In company *C2*, the only lead manager is *LM2*. There is one senior manager, *SM3*, under *LM2*. There are two managers, *M2* and *M3*, under senior manager *SM3*. There is one employee, *E3*, under manager *M2*, and another employee, *E4*, under manager, *M3*.

-----

**코드**

~~~sql
SELECT c.company_code, c.founder,
    COUNT(DISTINCT l.lead_manager_code), COUNT(DISTINCT s.senior_manager_code),
    COUNT(DISTINCT m.manager_code), COUNT(DISTINCT e.employee_code)
FROM Company c, Lead_Manager l, Senior_Manager s, Manager m, Employee e
WHERE c.company_code = l.company_code AND
    l.lead_manager_code = s.lead_manager_code AND
    s.senior_manager_code = m.senior_manager_code AND
    m.manager_code = e.manager_code
GROUP BY c.company_code, c.founder ORDER BY c.company_code;
~~~

