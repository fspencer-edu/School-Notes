
## Basic Selection

```python
SELECT column FROM table;

USE rookery;

SELECT * FROM birds;

SELECT bird_id, scientific_name, common_name
FROM birds;
```

## Selecting by a Criteria

```python
SELECT common_name, scientific_name
FROM birds WHERE family_id = 103
LIMIT 3;

+----------------------+-------------------------+
| common_name          | scientific_name         |
+----------------------+-------------------------+
| Mountain Plover      | Charadrius montanus     |
| Snowy Plover         | Charadrius alexandrinus |
| Black-bellied Plover | Pluvialis squatarola    |
+----------------------+-------------------------+
```

## Ordering Results

- LIMIT
- ORDER BY
	- Ascending by default

```python
SELECT common_name, scientific_name
FROM birds WHERE family_id = 103
ORDER BY common_name
LIMIT 3;

+-----------------------+----------------------+
| common_name           | scientific_name      |
+-----------------------+----------------------+
| Black-bellied Plover  | Pluvialis squatarola |
| Mountain Plover       | Charadrius montanus  |
| Pacific Golden Plover | Pluvialis fulva      |
+-----------------------+----------------------+
```
- ORDER BY ASC/DESC


- IN
	- Determine if a specified column's value matches any values within a list

```python
SELECT * FROM bird_families
WHERE scientific_name
IN('Charadriidae','Haematopodidae','Recurvirostridae','Scolopacidae');

+-----------+------------------+------------------------------+----------+
| family_id | scientific_name  | brief_description            | order_id |
+-----------+------------------+------------------------------+----------+
|       103 | Charadriidae     | Plovers, Dotterels, Lapwings |      102 |
|       160 | Haematopodidae   | Oystercatchers               |      102 |
|       162 | Recurvirostridae | Stilts and Avocets           |      102 |
|       164 | Scolopacidae     | Sandpipers and Allies        |      102 |
+-----------+------------------+------------------------------+----------+

SELECT common_name, scientific_name, family_id
FROM birds
WHERE family_id IN(103, 160, 162, 164)
ORDER BY common_name
LIMIT 3;

SELECT common_names, scientific_name, family_id
FROM birds
WHERE family_id IN(103, 160, 162, 164)
AND common_name != ''
ORDER BY common_name
LIMIT 3;
```

- !=
	- NOT operator
	- Non-Null columns
	- <>

## Limiting Results

```python
SELECT common_name, scientific_name, family_id
FROM birds
WHERE family_id IN(103, 160, 162, 164)
AND common_name != ''
ORDER BY common_name
LIMIT 3, 2;
```

- LIMIT x, y
	- start, number of rows

## Combining Tables

```python
SELECT common_name AS 'Bird',
bird_families.scientific_name AS 'Family'
FROM birds, bird_families
WHERE birds.family_id = bird_families.family_id
AND order_id = 102
AND common_name != ''
ORDER BY common_name LIMIT 10;

+------------------------+------------------+
| Bird                   | Family           |
+------------------------+------------------+
| African Jacana         | Jacanidae        |
| African Oystercatcher  | Haematopodidae   |
```

- Put tables named before the the column name (dot)
- AS
	- Substitutes name for the heading column
	- Also used to specify a table name

```python
SELECT common_name AS 'Bird',
families.scientific_name AS 'Family'
FROM birds, bird_families AS families
WHERE birds.family_id = families.family_id
AND order_id = 102
AND common_name != ''
ORDER BY common_name LIMIT 10;
```

- After setting the alias, for the table name, the query is referred to the new table name

```python
SELECT common_name AS 'Bird',
families.scientific_name AS 'Family',
orders.scientific_name AS 'Order'
FROM birds, bird_families AS families, bird_orders AS orders
WHERE birds.family_id = families.family_id
AND families.order_id = orders.order_id
AND families.order_id = 102
AND common_name != ''
ORDER BY common_name LIMIT 10, 5;

+------------------+------------------+-----------------+
| Bird             | Family           | Order           |
+------------------+------------------+-----------------+
| Ancient Murrelet | Alcidae          | Charadriiformes |
| Andean Avocet    | Recurvirostridae | Charadriiformes |
| Andean Gull      | Laridae          | Charadriiformes |
| Andean Lapwing   | Charadriidae     | Charadriiformes |
| Andean Snipe     | Scolopacidae     | Charadriiformes |
+------------------+------------------+-----------------+
```

## Expressions and the Like

- LIKE
	- Select multiple names that are similar

```python
SELECT common_name AS 'Bird',
families.scientific_name AS 'Family',
orders.scientific_name AS 'Order'
FROM birds, bird_families AS families, bird_orders AS orders
WHERE birds.family_id = families.family_id
AND families.order_id = orders.order_id
AND common_name LIKE 'Least%'
ORDER BY orders.scientific_name, families.scientific_name, common_name
LIMIT 10;
```

- `%`
	- Wildcard
- Cannot use alias names for columns in the `ORDER BY` clause, only in table names in the FROM clause
- REGEXP
	- Regular expression
- ^
	- Start values
- |
	- Or operator

```python
SELECT common_name AS 'Birds Great and Small'
FROM birds
WHERE common_name REGEXP 'Great|Least'
ORDER BY family_id LIMIT 10;

# NOT REGEXP
SELECT common_name AS 'Birds Great and Small'
FROM birds
WHERE common_name REGEXP 'Great|Least'
AND common_name NOT REGEXP 'Greater'
ORDER BY family_id LIMIT 10;
```

- REGEXP BINARY
	- Case sensitive

```python
SELECT common_name AS 'Hawks'
FROM birds
WHERE common_name REGEXP BINARY 'Hawk'
AND common_name NOT REGEXP 'Hawk-Owl'
ORDER BY family_id LIMIT 10;
```

- Use a character class and character name
	- `[[:alpha:]]` for alphabetic
	- `[[.hyphen.]]` for a hyphen


```python
SELECT common_name AS 'Hawks'
FROM birds
WHERE common_name REGEXP '[[:space:]]Hawk|[[.hyphen.]]Hawk'
AND common_name NOT REGEXP 'Hawk-Owl|Hawk Owl'
ORDER BY family_id;
```

## Counting and Grouping Results

- COUNT()
	- Adding function
	- Does not count null values
	- Counts empty or blank values

```python
SELECT COUNT(*) FROM birds;

SELECT families.scientific_name AS 'Family',
COUNT(*) AS 'Number of Birds'
FROM birds, bird_families AS families
WHERE birds.family_id = families.family_id
AND families.scientific_name = 'Pelecanidae'

+-------------+-----------------+
| Family      | Number of Birds |
+-------------+-----------------+
| Pelecanidae |              10 |
+-------------+-----------------+

SELECT orders.scientific_name AS 'Order',
families.scientific_name AS 'Family',
COUNT(*) AS 'Number of Birds'
FROM birds, bird_families AS families, bird_orders AS orders
WHERE birds.family_id = families.family_id
AND families.order_id = orders.order_id
AND orders.scientific_name = 'Pelecaniformes'
GROUP BY Family;

+----------------+-------------------+-----------------+
| Order          | Family            | Number of Birds |
+----------------+-------------------+-----------------+
| Pelecaniformes | Ardeidae          |             157 |
| Pelecaniformes | Balaenicipitidae  |               1 |
| Pelecaniformes | Pelecanidae       |              10 |
| Pelecaniformes | Scopidae          |               3 |
| Pelecaniformes | Threskiornithidae |              53 |
+----------------+-------------------+-----------------+
```

## Summary