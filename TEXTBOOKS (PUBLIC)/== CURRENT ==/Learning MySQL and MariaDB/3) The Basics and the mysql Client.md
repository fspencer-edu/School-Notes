## The mysql Client


## Connecting to the Server

```python
# connect to server
mysql -u fiona -p
```

- Commands end with a semicolon or a slash-g (`\g)

```python
# helpful commands
help
help contents
hep Data Manipulation
help SHOW DATABASES
```
- `\c`
	- Cancel a SQL statement mid line
- `>`
	- A prompt
- `->`
	- Continued prompt

```python
# change SQL prompt
prompt SQL Command \d>\_
```

## Starting to Explore Databases

- Data is stored in a table
- MySQL is not case sensitive
- Names of databases, tables, and columns may be case sensitive

**Default DBs**
- `information_schema`
	- Contains information about the server
- `mysql`
	- Stores usernames, passwords, and user privileges

```python
# create a table
# database.table

CREATE DATABASE test;

CREATE TABLE test.books (
	book_id INT,
	title TEXT,
	status INT
);

SHOW TABLES from test;

USE test

DESCRIBE books;
```

### Inserting and Manipulating Data

```python
INSERT INTO books VALUES(100, 'Heart of Darkness', 0);
INSERT INTO books VALUES(101, 'The Catcher of the Rye', 1);
INSERT INTO books VALUES(102, 'My Antonia', 0);

SELECT * FROM books;

SELECT * FROM books WHERE status = 1;

SELECT * FROM books WHERE status = 0 \G

UPDATE books SET status = 1 WHERE book_id = 102;

UPDATE books
SET title = 'The Catcher in the Rye', Status = 1
WHERE book_id = 101
```

- Instead of a semicolon, a `\G` shows the results as a batch of lines instead of table format


- Create another table and insert rows of data
```python
CREATE TABLE status_names (
	status_id INT,
	status_name CHAR9(8)
);

INSERT INTO status_names VALUES(0, 'Inactive'), (1,'Active');
```
- Combing the tables with the books table

```python
SELECT book_id, title, status_name
FROM books 
JOIN status_names
WHERE status = status_id;

+---------+------------------------+-------------+
| book_id | title                  | status_name |
+---------+------------------------+-------------+
|     100 | Heart of Darkness      | Inactive    |
|     101 | The Catcher in the Rye | Active      |
|     102 | My Antonia             | Active      |
+---------+------------------------+-------------+
```

## Summary