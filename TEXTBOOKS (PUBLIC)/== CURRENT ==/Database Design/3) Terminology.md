
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

**Importance of Relationships**
- Create multi table views
- Crucial to data integrity because it helps reduce redundant data and eliminate duplicate data

## Types of Relationships
### One-to-One

- A single record in the first table is related to zero or one and only one record in the second table
- Same for the second table
- One table serves as the parent table, and other child
- Take a copy of the parents table's primary key and add to structure of child table, as a foreign key
- Both tables may share the same primary key

<img src="/images/Pasted image 20260409140136.png" alt="image" width="500">

### One-to-Many

- Exists between a pair of tables when a single record in the first table can be related to zero, one, or many records in the second table
- A single record in the second table can be related to only one record in the first
- The "one" table is the parent
- And the "many" represent the children
- Take a copy of the parent's table's primary key and add to the children's tables, as foreign keys


<img src="/images/Pasted image 20260409140150.png" alt="image" width="500">


### Many-to-Many

- Relationship when a single record in the first table can be related to zero, one, or many records in the second table, and vice versa
- Establish relationship with a linking table
	- Associate records from one table with another
	- Take a copy of the primary key of each table, and use them to form the structure of the new table

<img src="/images/Pasted image 20260409140332.png" alt="image" width="500">

## Types of Participation

- A table's participation within a relationship can be either mandatory or optional

- Mandatory
	- If you must enter at least one record into A before you can enter into B
- Options
	- No required to enter any records into A before B

## Degree of Participation

- Degree of participation determines the minimum number of records that a given table must have associated with a single record in the related table
- Max number of records that a given table is allowed to have associated with. single record in the related table

## Integrity-Related Terms

### Field Specification

- A field specification (domain) represents all the elements of a field
- Each field specification incorporates 3 types of elements
	- General
		- Fundamental information
		- Field Name, Description, Parent Table
	- Physical
		- Data Type, Length, Character Support
	- Logical
		- Required Value, Range of Values, Null Support


### Data Integrity

- Data integrity refers to the validity, consistency, and accuracy of the data in a database
- The level of accuracy of information retrieved is in direct proportion of the level of data integrity imposed upon the database

**4 Types of Data Integrity**
- Table level integrity
	- Also called entity integrity
	- No duplicate record exists within the table
	- The field that identifies each record within the table is unique and never Null
- Field level integrity
	- Domain integrity
	- Ensures that the structure of every field is sound
	- Values in each field are valid, consistent, and accurate
	- Fields of the same type are consistently defined
- Relationship level integrity
	- Referential integrity
	- Ensures that the relationship between a pair of tables is sound and synchronized when data is entered into, updated in, or deleted from either table
- Business rules
	- Impose restrictions or limitations on certain aspects of a database based on the ways an organization perceived and uses its data
	- Range and types of values stored
	- Type of participation and degree of participation
	- Type of syncrhonization