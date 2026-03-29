
## Updating Data

- UPDATE
	- Changes the data in particular columns of existing records


```python
UPDATE table
SET column = value, ... ;

UPDATE birdwatchers.humans
SET country_id = 'us';
```

### Updating Specific Rows

- Use the SELECT statement to test the conditions

```python
SELECT human_id, name_first, name_last
FROM humans
WHERE name_first = 'rusty'
AND name_last = 'Osborne'

UPDATE humans
SET name_last = 'Johnson'
WHERE human_id = 3

UPDATE humans
SET formal_title = 'Ms.'
WHERE human_id IN(24, 32);
```
- IN
	- Matches specific rows in the table

```python
SHOW FULL COLUMNS
FROM humans
LIKE 'formal_title' \G

*************************** 1. row ***************************
     Field: formal_title
      Type: enum('Mr.','Miss','Mrs.','Ms.')
 Collation: latin1_bin
      Null: YES
       Key:
   Default: NULL
     Extra:
Privileges: select,insert,update,references
   Comment: 

# changes all title values to Ms IN
UPDATE humans
SET formal_title = 'Ms.'
WHERE formal_title IN ('Miss', 'Mrs.');

ALTER TABLE humans
CHANGE COLUMN formal_title formal_title ENUM('Mr', 'Ms.');
```
- Alter table before updating valuues
	- The new values are the same as the first two characters of the old values

```python
ALTER TABLE humans
CHANGE COLUMN formal_title formal_title ENUM('Mr.','Ms.','Mr','Ms');

UPDATE humans
SET formal_title = SUBSTRING(formal_title, 1, 2);

ALTER TABLE humans
CHANGE COLUMN formal_title formal_title ENUM('Mr','Ms');
```

- SUBSTRING(column, start, length)
	- Extracts the text
	- Takes only the first 2 characters of the current value


### Limiting Updates

- LIMIT

```python
CREATE TABLE prize_winners (
	winner_id INT AUTO_INCREMENT PRIMARY KEY,
	human_id INT,
	winner_date DATE,
	prize_chosen VARCHAR(255),
	prize_sent DATE
);

```

- Select the winners and insert then in the new table

### Ordering to Make a Difference

```python
UPDATE prize_winners
SET winner_date = CURDARTE()
WHERE winner_date IS NULL
ORDER BY RAND()
LIMIT 2;
```

- RAND()
	- Can return the same results
	- Slow in an ORDER BY clause
- LIMIT
	- Limit with order by can cause problem with MySQL replication

1774816730883@@127.0.0.1@3308@mydb/test.sql
### Updating Multiple Tables

```python
UPDATE prize_winners, humans
SET winner_date = NULL,
    prize_chosen = NULL,
    prize_sent = NULL
WHERE country_id = 'uk'
AND prize_winners.human_id = humans.human_id;

UPDATE prize_winners
SET winner_date = CURDATE()
WHERE winner_date IS NULL
AND human_id IN
  (SELECT human_id
   FROM humans
   WHERE country_id = 'uk'
   ORDER BY RAND())
LIMIT 2;
```


### Handling Duplicates


## Deleting Data

### Deleting in Multiple Tables