
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
	- 

## Optional Changes

## Indexes
## Summary