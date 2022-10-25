[ec2-user@ip-10-1-10-83 ~]$ 
[ec2-user@ip-10-1-10-83 ~]$ mysql -u admin -h udacity.cafyz46jtz3u.us-east-1.rds.amazonaws.com -p
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MySQL connection id is 43
Server version: 8.0.28 Source distribution

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MySQL [(none)]> use udacity
Database changed
MySQL [udacity]> show tables;
Empty set (0.01 sec)

MySQL [udacity]> CREATE TABLE authors (id INT, name VARCHAR(20), email VARCHAR(20));
Query OK, 0 rows affected (0.05 sec)

MySQL [udacity]> INSERT INTO authors (id,name,email) VALUES(1,"Vivek","xuz@abc.com");
Query OK, 1 row affected (0.01 sec)

MySQL [udacity]> INSERT INTO authors (id,name,email) VALUES(2,"Priya","p@gmail.com");
Query OK, 1 row affected (0.01 sec)

MySQL [udacity]> INSERT INTO authors (id,name,email) VALUES(3,"Tom","tom@yahoo.com");
Query OK, 1 row affected (0.02 sec)

MySQL [udacity]> show tables;
+-------------------+
| Tables_in_udacity |
+-------------------+
| authors           |
+-------------------+
1 row in set (0.00 sec)

MySQL [udacity]> SELECT * FROM authors;
+------+-------+---------------+
| id   | name  | email         |
+------+-------+---------------+
|    1 | Vivek | xuz@abc.com   |
|    2 | Priya | p@gmail.com   |
|    3 | Tom   | tom@yahoo.com |
+------+-------+---------------+
3 rows in set (0.00 sec)

MySQL [udacity]> exit;
Bye
[ec2-user@ip-10-1-10-83 ~]$ 