
## The Syntax

- INSERT
	- Adds rows of data into a table

```python
INSERT INTO table [(column, …)]
  VALUES (value, …), (…), …;
  
INSERT INTO books
VALUES('The Big Sleep', 'Raymond Chandler', '1934');

# with default values
INSERT INTO books
VALUES('The Thirty-Nine Steps', 'John Buchan', DEFAULT);

# specify column names
INSERT INTO books
(author, title)
VALUES('Evelyn Waugh','Brideshead Revisited');

# multiple insert
INSERT INTO books
(title, author, year)
VALUES('Visitation of Spirits','Randall Kenan','1989'),
      ('Heart of Darkness','Joseph Conrad','1902'),
      ('The Idiot','Fyodor Dostoevsky','1871');
```

## Practical Examples

- Database development
	- Change schema of existing tables
	- Shift large blocks of data from one table to another

```python
DESCRIBE bird_orders;

+-------------------+--------------+------+-----+---------+----------------+
| Field             | Type         | Null | Key | Default | Extra          |
+-------------------+--------------+------+-----+---------+----------------+
| order_id          | int(11)      | NO   | PRI | NULL    | auto_increment |
| scientific_name   | varchar(255) | YES  | UNI | NULL    |                |
| brief_description | varchar(255) | YES  |     | NULL    |                |
| order_image       | blob         | YES  |     | NULL    |                |
+-------------------+--------------+------+-----+---------+----------------+

ALTER TABLE bird_orders
AUTO_INCREMENT = 100;
```

- Start inc. at 100, for a 3 digit id length
- Use the `order_id` to connect with bird families table

```python
DESCRIBE bird_families;

SELECT order_id FROM bird_orders
WHERE Scientific_name = 'Gaviiformes';

INSERT INTO bird_families
VALUES(100, 'Gaviidae',
"Loons or divers are aquatic birds found mainly in the Northern Hemisphere.",
103);
```

- 100 is used instead of 1
	- Represents the first column
- Non-ordered data may be accepted

```python
SHOW WARNINGS \G

*************************** 1. row ***************************
  Level: Warning
   Code: 1366
Message: Incorrect integer value: 'Anatidae' for column 'family_id' at row 1
1 row in set (0.15 sec)

SELECT * FROM bird_families \G

*************************** 1. row ***************************
        family_id: 100
  scientific_name: Gaviidae
brief_description: Loons or divers are aquatic birds
                   found mainly in the Northern Hemisphere.
         order_id: 103
*************************** 2. row ***************************
        family_id: 101
  scientific_name: This family includes ducks, geese and swans.
brief_description: NULL
         order_id: 103
         
DELETE FROM birds_families
WHERE family_id = 101;

INSERT INTO bird_families
(scientific_name, order_id, brief_description)
VALUES('Anatidae', 103, "This family includes ducks, geese and swans.");

SELECT order_id, scientific_name FROM bird_orders;
```
- There is no undo statement
- Values can be correct if ordered based on the insert value order (different from schema)

```python
INSERT INTO bird_families
(scientific_name, order_id)
VALUES('Charadriidae', 109),
      ('Laridae', 102),
      ('Sternidae', 102),
      ('Caprimulgidae', 122),
      ('Sittidae', 128),
      ('Picidae', 125),
      ('Accipitridae', 112),
      ('Tyrannidae', 128),
      ('Formicariidae', 128),
      ('Laniidae', 128);
      
SELECT family_id, scientific_name
FROM bird_families
ORDER BY scientific_name;

+-----------+-----------------+
| family_id | scientific_name |
+-----------+-----------------+
|       109 | Accipitridae    |
|       102 | Anatidae        |
|       106 | Caprimulgidae   |
|       103 | Charadriidae    |
|       111 | Formicariidae   |
|       100 | Gaviidae        |
|       112 | Laniidae        |
|       104 | Laridae         |
|       108 | Picidae         |
|       107 | Sittidae        |
|       105 | Sternidae       |
|       110 | Tyrannidae      |
+-----------+-----------------+

SHOW COLUMNS FROM birds;

+------------------------+--------------+------+-----+-------+----------------+
| Field                  | Type         | Null | Key |Default| Extra          |
+------------------------+--------------+------+-----+-------+----------------+
| bird_id                | int(11)      | NO   | PRI | NULL  | auto_increment |
| scientific_name        | varchar(100) | YES  | UNI | NULL  |                |
| common_name            | varchar(255) | YES  |     | NULL  |                |
| family_id              | int(11)      | YES  |     | NULL  |                |
| conservation_status_id | int(11)      | YES  |     | NULL  |                |
| wing_id                | char(2)      | YES  |     | NULL  |                |
| body_id                | char(2)      | YES  |     | NULL  |                |
| bill_id                | char(2)      | YES  |     | NULL  |                |
| description            | text         | YES  |     | NULL  |                |
+------------------------+--------------+------+-----+-------+----------------+

SHOW COLUMNS FROM birds LIKE '%id';

+------------------------+---------+------+-----+---------+----------------+
| Field                  | Type    | Null | Key | Default | Extra          |
+------------------------+---------+------+-----+---------+----------------+
| bird_id                | int(11) | NO   | PRI | NULL    | auto_increment |
| family_id              | int(11) | YES  |     | NULL    |                |
| conservation_status_id | int(11) | YES  |     | NULL    |                |
| wing_id                | char(2) | YES  |     | NULL    |                |
| body_id                | char(2) | YES  |     | NULL    |                |
| bill_id                | char(2) | YES  |     | NULL    |                |
+------------------------+---------+------+-----+---------+----------------+

# more detailed desc.
SHOW FULL COLUMNS FROM birds;
```

- `SHOW COLUMNS = DESCRIBE`
	- Show, can retrieve a list of columns based on a pattern

### Table for Birds

```python
INSERT INTO birds
(common_name, scientific_name, family_id)
VALUES('Mountain Plover', 'Charadrius montanus', 103);
```

```python
SELECT common_name AS 'Bird',
	birds.scientific_names AS 'Scientific Name',
	bird_families.scientific_name AS 'Family',
	bird_orders.scientific_name AS 'Order'
FROM birds,
	bird_families,
	bird_orders
WHERE birds.family_id = bird_familiies.family_id
AND bird_families.order_id = bird_orders.order_id;

+-----------------------+----------------------+--------------+---------------+
| Bird                  | Scientific Name      | Family       | Orders        |
+-----------------------+----------------------+--------------+---------------+
| Mountain Plover       | Charadrius montanus  | Charadriidae | Ciconiiformes |
| Snowy Plover          | Charadrius alex...   | Charadriidae | Ciconiiformes |
| Black-bellied Plover  | Pluvialis squatarola | Charadriidae | Ciconiiformes |
| Pacific Golden Plover | Pluvialis fulva      | Charadriidae | Ciconiiformes |
+-----------------------+----------------------+--------------+---------------+
```

- Connecting 3 tables
- FROM
	- `birds, bird_families, bird_orders`
- WHERE
	- Joins the bird and family tables based on `family_id`
	- Joins the family and order tables based on `order_id`
- AS
	- Changes the column headings

## Other Possibilities

### Inserting Emphatically

- Maps individual columns to given data

```python
INSERT INTO bird_families
SET scientific_name = 'Rallidae',
order_id = 113;
```

### Inserting Data From Another Table

```python
DESCRIBE cornell_birds_families_orders;

+-------------+--------------+------+-----+---------+----------------+
| Field       | Type         | Null | Key | Default | Extra          |
+-------------+--------------+------+-----+---------+----------------+
| fid         | int(11)      | NO   | PRI | NULL    | auto_increment |
| bird_family | varchar(255) | YES  |     | NULL    |                |
| examples    | varchar(255) | YES  |     | NULL    |                |
| bird_order  | varchar(255) | YES  |     | NULL    |                |
+-------------+--------------+------+-----+---------+----------------+

SELECT * FROM cornell_birds_families_orders
LIMIT 1;

+-----+---------------+----------+------------------+
| fid | bird_family   | examples | bird_order       |
+-----+---------------+----------+------------------+
|   1 | Struthionidae | Ostrich  | Struthioniformes |
+-----+---------------+----------+------------------+
```
- Take the family names, use the examples and insert into the `bird_familities` table

- Add another column to `bird_families` table to take in the `bird_order`
- Copy the data from the Cornell table to the table containing data on bird families

```python
ALTER TABLE bird_families
ADD COLUMN cornell_bird_order VARCHAR(255);

INSERT INGORE INTO bird_families
(scientific_name, brief_description, cornell_bird_order)
SELECT bird_family, examples, bird_order
FROM cornell_birds_families_orders;

SELECT * FROM bird_families
ORDER BY family_id DESC LIMIT 1;

+-----------+-----------------+-----------------+----------+-------------------+
| family_id | scientific_name |brief_description| order_id | cornell_bird_order|
+-----------+-----------------+-----------------+----------+-------------------+
|       330 | Viduidae        | Indigobirds     |     NULL | Passeriformes     |
+-----------+-----------------+-----------------+----------+-------------------+
```

- IGNORE
	- Flag instructs the server to ignore any errors

### A Digression: Setting the Right ID

- The Cornell data did not have id numbers assigned
- Set the value of the `order_id` column to the right `order_id` from the `bird_orders` tables
- Join the `bird_orders` and Cornell data
- UPDATE
	- Changes any data

```python
# test name equality
SELECT DISTINCT bird_orders.order_id,
cornell_bird_order AS "Cornell's Order",
bird_orders.scientific_name AS 'My Order',
FROM bird_families, bird_orders
WHERE bird_families.order_id IS NULL
AND cornell_bird_order = bird_orders.scientific_name
LIMIT 5;

+----------+------------------+------------------+
| order_id | Cornell's Order  | My Order         |
+----------+------------------+------------------+
|      120 | Struthioniformes | Struthioniformes |
|      121 | Tinamiformes     | Tinamiformes     |
|      100 | Anseriformes     | Anseriformes     |
|      101 | Galliformes      | Galliformes      |
|      104 | Podicipediformes | Podicipediformes |
+----------+------------------+------------------+
```

- Get the two tables
- Where the family order id is null
- And Cornell and bird order scientific names are equal
- Select order id, Cornell order, and org. order
- UPDATE
	- Change or update rows

```python
UPDATE bird_families, bird_orders
SET bird_families.order_id = bird_orders.order_id
WHERE bird_families.order_id IS NULL
AND cornell_bird_order = bird_orders.scientific_name;

SELECT * FROM bird_families
ORDER BY family_id DESC LIMIT 4;

+-----------+-----------------+---------------------+----------+
| family_id | scientific_name | brief_description   | order_id |
+-----------+-----------------+---------------------+----------+
|       330 | Viduidae        | Indigobirds         |      128 |
|       329 | Estrildidae     | Waxbills and Allies |      128 |
|       328 | Ploceidae       | Weavers and Allies  |      128 |
|       327 | Passeridae      | Old World Sparrows  |      128 |
+-----------+-----------------+---------------------+----------+

SELECT * FROM bird_orders
WHERE order_id = 128;

+----------+-----------------+-------------------+-------------+
| order_id | scientific_name | brief_description | order_image |
+----------+-----------------+-------------------+-------------+
|      128 | Passeriformes   | Passerines        | NULL        |
+----------+-----------------+-------------------+-------------+

SELECT family_id, scientific_name, brief_description
FROM bird_families
WHERE order_id IS NULL;

+-----------+-------------------+----------------------+
| family_id | scientific_name   | brief_description    |
+-----------+-------------------+----------------------+
|       136 | Fregatidae        | Frigatebirds         |
|       137 | Sulidae           | Boobies and Gannets  |
|       138 | Phalacrocoracidae | Cormorants and Shags |
|       139 | Anhingidae        | Anhingas             |
|       145 | Cathartidae       | New World Vultures   |
```

- Set the order id in the family table to the value of the order id
- Only update where the Cornell order equals the scientific_name


```python
ALTER TABLE bird_families
DROP COLUMN cornell_bird_order;

DROP TABLE cornell_brids_families_orders;
```

### Replacing Data

- For duplicate data with a `UNIQUE` constraint, the duplicates must be removed to complete transaction
- `REPLACE`
	- New rows of data will be inserted
	- Replace the matching row already in the table
	- Duplicates are overwritten

```python
REPLACE INTO bird_families
(scientific_name, brief_description, order_id)
VALUES('Viduidae', 'Indigobirds & Whydahs', 128),
('Estrildidae', 'Waxbills, Weaver Finches, & Allies', 128),
('Ploceidae', 'Weavers, Malimbe, & Bishops', 128);

Query OK, 6 rows affected (0.39 sec)
Records: 3  Duplicates: 3  Warnings: 0
```
- Replace changes the entire row, unlike update
- Affected rows = added + duplicated

```python
SELECT * FROM bird_families
WHERE scientific_name = 'Viduidae' \G

*************************** 1. row ***************************
        family_id: 331
  scientific_name: Viduidae
brief_description: Indigobirds & Whydahs
         order_id: 128
```

### Priorities When Inserting Data

- On a busy server there may be many transactions at the same time
	- INSERT, UPDATE, DELETE
- Take priority over read statements
	- SELECT
- InnoDB
	- Locks the rows, rather than the entire table
- Locking a table could cause delays
- Set the priorities for an `INSERT`
	- `LOW_PRIORITY`
	- `DELAYED`
	- `HIGH_PRIORITY`

```python
INSERT LOW_PRIORITY INTO bird_signtings

INSERT DELAYED INTO bird_sightings
```

**Lowering the priority of an insert**

- `LOW_PRIORITY`
	- Large data dumps
	- Puts the `INSERT` statement in a queue, waiting for all of the current and pending requests to be completed
	- If new requests are made, they are put ahead of it in queue
	- Locks the table once a low priority statement has being to prevent inconsistent data writes



**Delaying an INSERT**

- `DELAYED`
	- May be deprecated
	- Similar to low priority
	- Client is never informed whether the delayed insertion is made
	- Stored in the server's memory
	- Insertions are lost if server dies


**Raising the priority of an INSERT**

- `HIGH_PRIORITY`
	- `INSERT` statements by default are higher priority than read-only SQL statements
	- Makes writes less priority, and reads higher priority

## Summary