
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

## Indexes
## Summary