
## Prudence When Altering Tables

- Make backup of the tables before changing
- Map a copy of the table within the same database to use as a backup
- Better
	- Copy table and then alter copy
- `mysqldump` utility

```python
# backup of table
mysqldump --user='fiona' -p \
rookery brids > /tmp/birds.sql

# backup of database
mysqldump --user='fiona' -p \
rookery > rookerys.sql

# restore database
mysql --user='fiona' -p \
rookery < rookery-new . sql
```

- Delete the `rookery` database with its tables and data before restoring the back up copy with its tables and data

## Essential Changes

- Add or change an index to improve query speed

```python
ALTER TABLE table_name changes;

# add a column to table
ALTER TABLE bird_families
ADD COLUMN order_id INT;

# copy birds table schema
CREATE TABLE test.birds_new LIKE birds;

USE test

DESCRIBE birds_new;

INSERT INTO birds_new
SELECT * FROM rookery.birds;

# another method to copy table
CREATE TABLE birds_alt
SELECT * FROM rookery.birds;

# Delete table
DROP TABLE birds_new_alternative;

ALTER TABLE birds_new
ADD COLUMN wing_id CHAR(2);
```

- A add column is added to the end of the table
- Specify the location of a new column

```python
ALTER TABLE birds_new
DROP COLUMN wing_id;

ALTER TABLE birds_new
ADD COLUMN wing_id CHAR(2) AFTER family_id;

ALTER TABLE birds_new
ADD COLUMN body_id CHAR(2) AFTER wing_id,
ADD COLUMN bill_id CHAR(2) AFTER body_id,
ADD COLUMN endangered BIT DEFAULT b'1' AFTER bill_id,
CHANGE COLUMN common_name common_name VARCHAR(255);
```
- For bit values
	- Stores the bit, does not display the value
	- If unset (0), there is a blank space
- CHANGE COLUMN
	- Old name, new name and type
- MySQL executes the clauses of an `ALTER TABLE` in sequence
	- Creates a temporary copy of the table, and alters that copy based on the statement instructions

```python
UPDATE birds_new SET endangered = 0
WHERE birds_id IN(1,2,3,4);
```

- Set the values given the column number
- NOT
	- Reverse filter

```python
ALTER TABLE birds_new
MODIFY COLUMN endangered
ENUM('Extinct',
     'Extinct in Wild',
     'Threatened - Critically Endangered',
     'Threatened - Endangered',
     'Threatened - Vulnerable',
     'Lower Risk - Conservation Dependent',
     'Lower Risk - Near Threatened',
     'Lower Risk - Least Concern')
AFTER family_id;

SHOW COLUMNS FROM birds_new LIKE 'endangered' \G
```
- MODIFY COLUMN
	- Does not allow name change
- CHANGE COLUMN
	- Changes the column names and data type

```python
UPDATE birds_new
SET endangered = 7;
```

### Dynamic Columns

- Only available in MariaDB
	- Similar to `ENUM`, but with key/value pairs instead of plain list of options
- Add preferences of bird-watchers

```python
USE birdwatchers;

CREATE TABLE surverys (
	survey_id INT AUTO_INCREMENT KEY,
	survey_name VARCHAR(255)
);

CREATE TABLE survey_questions (
	question_id INT AUTO_INCREMENT KEY,
	survey_id INT,
	question VARCHAR(255),
	choices BLOB
);

CREATE TABLE survey_answers (
	answer_id INT AUTO_INCREMENT KEY,
	human_id INT,
	question_id INT,
	date_answered DATETIME,
	answer VARCHAR(255)
);

INSERT INTO surveys (survey_name)
VALUES("Favorite Birding Location");

INSERT INTO survey_questions
(survey_id, question, choices)
VALUES(LAST_INSERT_ID(),
"What's your favorite setting for bird-watching?",
COLUMN_CREATE('1', 'forest', '2', 'shore', '3', 'backyard') );

INSERT INTO surveys (survey_name)
VALUES("Preferred Birds");

INSERT INTO survey_questions
(survey_id, question, choices)
VALUES(LAST_INSERT_ID(),
"Which type of birds do you like best?",
COLUMN_CREATE('1', 'perching', '2', 'shore', '3', 'fowl', '4', 'rapture') );

```
- The data type used has to be able to hold the data given to it
- `choices` will be a dynamic column
- `COLUMN_CREATE`
	- Create the enumerated list of choices
	- Each choice has a key and a value

```python
SELECT COLUMN_GET(choices, 3 AS CHAR)
AS 'Location'
FROM survey_questions
WHERE survey_id = 1

+----------+
| Location |
+----------+
| backyard |
+----------+
```
- This returns the third choice

```python
INSERT INTO survey_answers
(human_id, question_id, date_answered, answer)
VALUES
(29, 1, NOW(), 2),
(29, 2, NOW(), 2),
(35, 1, NOW(), 1),
(35, 2, NOW(), 1),
(26, 1, NOW(), 2),
(26, 2, NOW(), 1),
(27, 1, NOW(), 2),
(27, 2, NOW(), 4),
(16, 1, NOW(), 3),
(3, 1, NOW(), 1),
(3, 2, NOW(), 1);

# count the votes for the first survey
SELECT IFNULL(COLUMN_GET(choices, answer AS CHAR), 'total')
AS 'Birding Site', COUNT(*) AS 'Votes'
FROM survey_answers
JOIN survey_questions USING(question_id)
WHERE survey_id = 1
AND question_id = 1
GROUP BY answer WITH ROLLUP;

+--------------+-------+
| Birding Site | Votes |
+--------------+-------+
| forest       |     2 |
| shore        |     3 |
| backyard     |     1 |
| total        |     6 |
+--------------+-------+
```
- Choose `survey_id` from questions and `question_id` answers
- Group, and count rows for each answer to find how many bird-watchers vote for each one
- IFNULL
	- `IFNULL(value, replacement)`
	- If the choice is missing, then use 'total'
- ROLLUP
	- Adds a summary row (total) to the results
- total in the `birding sites` represents the total null values specified by IFNULL

## Optional Changes

**Setting a Column's Default Value**

- ALTER TABLE
	- Set options of existing tables and its columns
	- Set the value of tables variables, and default values of columns

```python
CREATE TABLE rookery.conservation_status (
	status_id INT AUTO_INCREMENT PRIMARY KEY,
	conservation_category CHAR(10),
	conservation_state CHAR(25)
);

INSERT INTO rookery.conservation_status
(conservation_category, conservation_state)
VALUES('Extinct','Extinct'),
('Extinct','Extinct in Wild'),
('Threatened','Critically Endangered'),
('Threatened','Endangered'),
('Threatened','Vulnerable'),
('Lower Risk','Conservation Dependent'),
('Lower Risk','Near Threatened'),
('Lower Risk','Least Concern');

SELECT * FROM rookery.conservation_status;

+-----------+-----------------------+------------------------+
| status_id | conservation_category | conservation_state     |
+-----------+-----------------------+------------------------+
|         1 | Extinct               | Extinct                |
|         2 | Extinct               | Extinct in Wild        |
|         3 | Threatened            | Critically Endangered  |
|         4 | Threatened            | Endangered             |
|         5 | Threatened            | Vulnerable             |
|         6 | Lower Risk            | Conservation Dependent |
|         7 | Lower Risk            | Near Threatened        |
|         8 | Lower Risk            | Least Concern          |
+-----------+-----------------------+------------------------+

ALTER TABLE birds_new
CHANGE COLUMN endangered conservation_status_id INT DEFAULT 8;

ALTER TABLE birds_new
ALTER conservation_status_id SET DEFAULT 7;

SHOW COLUMNS FROM birds_new LIKE 'conservation_status_id' \G

ALTER TABLE birds_new
ALTER conservation_status_id DROP DEFAULT;
```

- CHANGE COLUMN
	- Modified an existing column
	- Adds a default value
	- 

**Setting the Value of AUTO_INCREMENT**

```python
SELECT auto_increment
FROM information_schema.tables
WHERE table_name = 'birds';

+----------------+
| auto_increment |
+----------------+
|              7 |
+----------------+
```

**Another Method to Alter and Create a Table**

- If a table has a column that uses `AUTO_INCREMENT` the counter will be set to 0 for the new table

```python
CREATE TABLE birds_new LIKE birds;

SHOW CREATE TABLE brids \G
```

- Create a new table and copy specific column's settings and data
```python
CREATE TABLE birds_details
SELECT bird_id, description
FROM birds;

ALTER TABLE birds
DROP COLUMN description
```
- `birds_id` does not use `AUTO_INCREMENT`

**Renaming a Table**

- Change an existing table
- Delete the existing table, and rename the replacement table

```python
RENAME TABLE table_alt
TO table1

RENAME TABLE rookery.birds TO rookery.birds_old,
test.birds_new TO rookery.birds;

SHOW TABLES IN rookery LIKE 'birds%';

DROP TABLE birds_olds;
```
- Rename the birds table to birds old, then rename and relocate the new bird table from the `test` database to `birds` in the `rookery` database

**Reordering a Table**
- SELECT statement has an `ORDER BY` clause
	- Alphabetical
	- Numerical

```python
SELECT * FROM country_codes
LIMIT 5;

ALTER TABLE country_codes
ORDER BY counry_code;

SELECT * FROM country_codes
ORDER BY country_name
LIMIT 5;
```

## Indexes

- Renaming a column that is indexed by using only an `ALTER TABLE` statement

```python
ALTER TABLE conservation_status
CHANGE status_id conservation_status_id INT AUTO_INCREMENT PRIMARY KEY;

Error
```

- An index is separate from the column upon which the index is based
- Indexes are used to locate data quickly
- Without an index, rows are searched sequentially
- Index can jump directly to the row that matches the search pattern

```python
SHOW INDEX FROM birdwatchers.humans \G

*************************** 1. row ***************************
       Table: humans
  Non_unique: 0
    Key_name: PRIMARY
Seq_in_index: 1
 Column_name: human_id
   Collation: A
 Cardinality: 0
    Sub_part: NULL
      Packed: NULL
        Null:
  Index_type: BTREE
     Comment:
```

- EXPLAIN
	- Return information on how the `SELECT` statement searches the table and on what basis
	- Execution type

```python
EXPLAIN SELECT * FROM birdwatchers.humans
WHERE name_last = 'Hoolar' \G
*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: humans
         type: ALL
possible_keys: NULL
          key: NULL
      key_len: NULL
          ref: NULL
         rows: 4
        Extra: Using where
```
- `possible_keys` field 
	- Shows the keys that the `SELECT` statement could have used
- `key` field
	- A key is the column on which a table is indexed


- To improve performance, create an index that combined the two columns
- The key, or index, is called `human_names` and based on the values of the two columns

```python
ALTER TABLE birdwatchers.humans
ADD INDEX human_names (name_last, name_first);

SHOW CREATE TABLE birdwatches.humans \G

*************************** 1. row ***************************
       Table: humans
Create Table: CREATE TABLE `humans` (
  `human_id` int(11) NOT NULL AUTO_INCREMENT,
  `formal_title` varchar(25) COLLATE latin1_bin DEFAULT NULL,
  `name_first` varchar(25) COLLATE latin1_bin DEFAULT NULL,
  `name_last` varchar(25) COLLATE latin1_bin DEFAULT NULL,
  `email_address` varchar(255) COLLATE latin1_bin DEFAULT NULL,
  PRIMARY KEY (`human_id`),
  KEY `human_names` (`name_last`,`name_first`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_bin

SHOW INDEX FROM birdwatchers.humans
WHERE Key_name = 'human_names' \G

*************************** 1. row ***************************
       Table: humans
  Non_unique: 1
    Key_name: human_names
Seq_in_index: 1
 Column_name: name_last
   Collation: A
 Cardinality: NULL
    Sub_part: NULL
      Packed: NULL
        Null: YES
  Index_type: BTREE
     Comment:
*************************** 2. row ***************************
       Table: humans
  Non_unique: 1
    Key_name: human_names
Seq_in_index: 2
 Column_name: name_first
   Collation: A
 Cardinality: NULL
    Sub_part: NULL
      Packed: NULL
        Null: YES
  Index_type: BTREE
     Comment: 
     
EXPLAIN SELECT * FROM birdwatchers.humans
WHERE name_last = 'Hollar' \G

*************************** 1. row ***************************
           id: 1
  select_type: SIMPLE
        table: humans
         type: ref
possible_keys: human_names
          key: human_names
      key_len: 28
          ref: const
         rows: 1
        Extra: Using where
```

- Because the index is associated with the column, we need to remove that association in the index
	- Otherwise, the index will be associated with a column that does not exist
	- Tied to old name
- Delete the index and rename the column, the add a new index on the new column name

```python
ALTER TABLE conservation_status
DROP PRIMARY KEY,
CHANGE status_id conservation_status_id INT PRIMARY KEY AUTO_INCREMENT;
```

- There is and can by only one primary key per table

## Summary