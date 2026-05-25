
## What is a Database

- Database
	- An organized collection of data used for modeling some type of organization or organizational process
- 2 types
	- Operational
	- Analytical

### Operational Databases

- Online transaction processing (OLTP)
- Collect, modify, and maintain data on a daily basis
- Dynamic
- Retail, manufacturing companies, hospitals and clinics

### Analytics Databases

- Online analytical processing (OLAP)
- Store and track historical and time-dependent data
- Track trends, view statistical data, make tactical or business projections
- Static
- Chemical labs, geological companies, marketing-analysis firms
- Used operational databases as their main data source

## Relational Database

- 1969
- New database based on two branches of math
	- Set theory
	- First-order predicate logic
- A relational database stores data in relations
- Each relation is composed of tuples
- The physical order is immaterial, and each record in the table is identified by a field that contains a unique value
- Data exists independently of the way it is stored in the computer
- Relationship types
	- One-to-one
	- One-to-many
	- Many-to-many


### Retrieving Data

- Retrieve data in a relational database by using Structured Query Language (SQL)
	- Create, modify, maintain, and query relational databases

```sql
SELECT ... FROM
WHERE ... ORDER BY
```

### Advantages of a Relational Database

- Built-in multilevel integrity
- Logical and physical data independence from database applications
- Guaranteed data consistency and accuracy
- Easy data retrieval

### Relational Database Management Systems (RDBMS)

- A software application program used to create, maintain, modify, and manipulate a relational database
- IBM DB2
- IBM informix
- Microsoft Access
- Microsoft SQL Server
- MySQL
- Oracle RDBMS
- PostgreSQL
- SAP SQL Anywhere
- SQP Sybase ASE
- SQLite


- Alternative storage
	- Photos
	- Read-only data
	- Graph data
	- Geospatial data
	- Analytics data

**NoSQL Databases**
- MongoDB
- CouchBase
- HBase
- Cassandra
- Redis