
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
DESCRIBE cornell_birds_families_orde
```


## Summary