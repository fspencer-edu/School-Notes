
## Terminology

- Used to express and define the special ideas and conecpts of the relational database model
- Used to express and define database-design process itself
- Used anywhere a relational database or RDBMS

## Value-Related Terms

### Data

- The values stored in databases
- Data is static
### Information

- Information is data that you process in a manner that makes it meaningful and useful
- Dynamic
	- Changes relative to the data stored in the database
- Information is what is retrieved

### Null

- Null is a condition that represents a missing or unknown value
	- Zero
	- Empty/blank text string
	- A zero length string
### Value of Null

- Missing values
- Unknown values
- If none of its values applies to a particular record
- Not applicable

### Problem with Null

- Adverse effect on mathematical operations
- Causes undetected errors

## Structured-Related Terms
### Table

- Data is a relational database is stored in relations, perceived as tables
- Each relation is composed of tuples and attributes
- Tables are structures in the database and always represent a single, specific subject
- The order of tuples have no important
- Primary key
	- Uniquely identifies each of its records
- A table can represent an object or event
	- Object
		- Table represents a tangible person, place, or thing
	- Event
		- table represents something that occurs at a given point in time
- Data table
	- Most common type of table in a relational database
	- Dynamic
- Validation table (lookup table)
	- Stores data that is specifically used to data integrity
	- Static
	- Used indirectly to validate values from other data tables

### Field

- A field is the smallest structure in the database
	- Also called attribute
- Represents a characteristic of the subject of the table to which it belongs
- Every field is a properly designed database that contains one and only one value
	- Name and type identifies the values it hold

3 Types
- Multipart/composite field
- Multivalued field
- Calculated field

### Record

- A record/tuple represents a unique instance of the subject of a table
- Composed of the entire set of fields in the table

### View

- A view is a "virtual" table composed of fields from one or more tables in the database
- The tables that compose the view are known as base tables
- Draws from base tables rather than storing data on its own
- The only information stored in the view is its structure
- Also called saved queries
- See information in database from different aspects

3 Reasons for Views
- Enable to work with data from multiple tables simultaneously
	- Tables must have connections/relationships
- Enable to prevent users from viewing or manipulating specific fields within a table or group of tables
- Used to implement data integrity
	- Validation view

### Keys

- Keys are special fields that determines its purpose within the table
	- Primary key
	- Foreign key

**Primary Key**
- A field of group of fields that uniquely identifies each record within a table

- Primary key value
	- Identifies a specific record throughout the entire database
- Primary key field
	- Identified a given table throughout the entire database
- Primary key enforces table-level integrity and helps establish relationships with other tables in the database

**Foreign Key**
- The second table already has a primary key of its own, and the primary key introducing from the first table is "foreign" to the second
- Ensures relationship-level integrity
- Avoid orphaned records
	- An order record without an associated customer

### Index

- An index is a structure an RDBMS provides to improve data processing
- An index has nothing to do with the logical database structure
- Keys
	- Logical structures to identify records within a table
- Indexes
	- Physical structures used to optimize data processing

## Relationship-Related Terms

### Relationships

- A relationship exists between two tables when you can associate the records of the first with the second table
- Established with a set of primary and foreign keys
- Linking/associative table (third)

**Importance of **


## Types of Relationships
### One-to-One
### One-to-Many
### Many-to-Many
## Types of Participation

## Degree of Participation

## Integrity-Related Terms

### Field Specification

### Data Integrity

