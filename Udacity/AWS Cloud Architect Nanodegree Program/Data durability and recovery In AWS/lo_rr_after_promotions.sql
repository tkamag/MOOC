[ec2-user@ip-10-0-11-154 ~]$ 
[ec2-user@ip-10-0-11-154 ~]$ mysql -u admin -h myudacity.cybbr3mrysva.us-west-1.rds.amazonaws.com -p
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MySQL connection id is 10
Server version: 8.0.28 Source distribution

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MySQL [(none)]> show databases;
+--------------------+
| Database           |
+--------------------+
| information_schema |
| mysql              |
| performance_schema |
| sys                |
| udacity            |
+--------------------+
5 rows in set (0.00 sec)

MySQL [(none)]> use udacity;
Reading table information for completion of table and column names
You can turn off this feature to get a quicker startup with -A

Database changed
MySQL [udacity]> SELECT * FROM authors;
+------+-------+---------------+
| id   | name  | email         |
+------+-------+---------------+
|    1 | Vivek | xuz@abc.com   |
|    2 | Priya | p@gmail.com   |
|    3 | Tom   | tom@yahoo.com |
+------+-------+---------------+
3 rows in set (0.00 sec)

MySQL [udacity]> INSERT INTO authors (id,name,email) VALUES(4,"Thierry","udacity@udacity.com");
Query OK, 1 row affected (0.01 sec)

MySQL [udacity]> SELECT * FROM authors;
+------+---------+---------------------+
| id   | name    | email               |
+------+---------+---------------------+
|    1 | Vivek   | xuz@abc.com         |
|    2 | Priya   | p@gmail.com         |
|    3 | Tom     | tom@yahoo.com       |
|    4 | Thierry | udacity@udacity.com |
+------+---------+---------------------+
4 rows in set (0.00 sec)

MySQL [udacity]> 