
- Separate databases are created for each separate business or organization
	- Further separated for departments or projects
- Schema
	- Specific fields or columns in a data object

---

- Creating a structure for data
	- Number of tables in db
	- Table names
	- For each table
		- Columns name, contents, and type

## Creating a Database

- `DATABASE = SCHEMA`

```python
CREATE DATABASE rookery;
CREATE SCHEMA database;

DROP DATABASE rookery;

CREATE DATABASE rookery
CHARACTER SET latin1
COLLATE latin1_bin;
```
## Creating Tables

```python
CREATE TABLE birds (
	bird_id INT AUTO_INCREMENT PRIMARY KEY,
	scientific_name VARCHAR(255) UNIQUE,
	common_name VARCHAR(50),
	family_id INT,
	description TEXT
);

DESCRIBE birds;

ALTER TABLE;
DROP TABLE;
```

- DESCRIBE
	- Field
	- Type
	- Null
	- Key
	- Default
	- Extra
 - `AUTO_INCREMENT`
	 - Automatically increment the value of the field
- Null
	- Indicates whether each field may contain NULL values

## Inserting Data

```python
INSERT INTO birds (scientific_name, common_name)
VALUES
('Charadrius vociferus', 'Killdeer'),
('Gavia immer', 'Great Northern Loon'),
('Aix sponsa', 'Wood Duck'),
('Chordeiles minor', 'Common Nighthawk'),
('Sitta carolinensis', ' White-breasted Nuthatch'),
('Apteryx mantelli', 'North Island Brown Kiwi');
```

- Create another table for bird watchers
```python
CREATE DATABASE birdwatchers;

CREATE TABLE birdwatchers.humans(
	human_id INT AUTO_INCREMENT PRIMARY KEY,
	formal_title VARCHAR(25),
	name_first VARCHAR(25),
	name_last VARCHAR(25),
	email_address VARCHAR(255)
);

INSERT INTO birdwatchers.humans
(name_first, name_last, email_address)
VALUES
('Mr.', 'Russell', 'Dyer', 'russell@mysqlresources.com'),
('Mr.', 'Richard', 'Stringer', 'richard@mysqlresources.com'),
('Ms.', 'Rusty', 'Osborne', 'rusty@mysqlresources.com'),
('Ms.', 'Lexi', 'Hollar', 'alexandra@mysqlresources.com');
```
## More Perspectives on tables

```python
SHOW CREATE TABLE birds \G
*************************** 1. row ***************************
       Table: birds
Create Table: CREATE TABLE `birds` (
  `bird_id` int(11) NOT NULL AUTO_INCREMENT,
  `scientific_name` varchar(255) COLLATE latin1_bin DEFAULT NULL,
  `common_name` varchar(50) COLLATE latin1_bin DEFAULT NULL,
  `family_id` int(11) DEFAULT NULL,
  `description` text COLLATE latin1_bin,
  PRIMARY KEY (`bird_id`),
  UNIQUE KEY `scientific_name` (`scientific_name`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1 COLLATE=latin1_bin
```

- SHOW CREATE TABLE
	- Shows the table schema and intended specifications
- MyISAM
	- Default storage engine used for many servers
- Data is stored and handled in different ways by different storage engines
- CHARACTER SET
	- Defines the set of allowed characters in the db and encoding scheme
- COLLATION
	- Collation controls how text is compared and stored

```python
CREATE TABLE bird_families (
	family_id INT AUTO_INCREMENT PRIMARY KEY,
	scientific_name VARCHAR(255) UNIQUE,
	brief_description VARCHAR(255)
)
```
- The `bird` and `bird_family` tables will be joined based on the `fanily_id` columns in both

```python
CREATE TABLE bird_orders (
  order_id INT AUTO_INCREMENT PRIMARY KEY,
  scientific_name VARCHAR(255) UNIQUE,
  brief_description VARCHAR(255),
  order_image BLOB
) DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
```

- BLOB
	- Binary large object
	- Stored image files
	- Can make tables large
	- Better to store in on the server and them store a file path or URL address in the database
- CHARSET
	- UTF-8
	- Names are not part of the default latin1 character set
- COLLATE


## Summary

## Exercises

