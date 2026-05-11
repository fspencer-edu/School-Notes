
- MySQL and MariaDB database servers and client software OS
	- Linux
	- Mac OS
	- FreeBSD
	- Sun Solaris
	- Windows

## The Installation Packages

- `mysqld` daemon
	- The software the stores and maintains control over all of the data in the databases
	- Listens on 3306 (default)
- Standard client
	- `mysql`
- MySQL wrappers
	- `mysqld_safe`
	- Restart on crash
- `mysqlaccess`
	- Creates user accounts and sets their privileges
- `mysqladmin`
	- Manage the database server
- `mysqlshow`
	- Example a server's status
	- Database and table information 
- `mysqldump`
	- Exporting data and table structures to a plain-text file
	- Backing up data
	- Copying databases between servers


## Licensing

- MySQL
	- Oracle allows free use without redistribution
	- Redistribution with a license

## Finding the Software

## Choosing a Distribution

- Binary distribution
	- Easier to install
	- Recommended
- Source distribution

## The \_AMP Alternatives

- Apache, MySQL/MariaDB, PHP/Perl/Python (AMP)

- Apache
	- Most populat web servier
- PHP
	- Popular programming language used with MySQL
- AMP package or stack is based on an OS
	- Linux stack => LAMP
	- Mac => MAMP
	- Windows => WAMP


**Linux Binary Distributions**
- RPM or DEB package
	- Use binary distribution

### Mac

```bash
brew install mariadb

cd /usr/local
tar xvfz mysql-version.tar.gz
```

## Post-Installation

### Special Configuration

- `/etc/my.cnf`
	- Contains error logging and settings

```python
# Set root password
mysqladmin -u root -p flush-privliedges password "new_pwd"

# List of usernames and host on server
mysql -u -root -p -e "SELECT User, Host FROM mysql.user;"
```

- Privileges are set based on a combination of the user's name and the user's host
- Host
	- `%` - wildcard
		- Can be access from any location
	- `localhost`
		- Local host
- Flush privileges to save the new passwords


### Creating a User

- Do not user root for general database management
- Creates a user that has access to all databases and tables, and sets a password
```python
# creating a user
mysql -u root p -e "GRANT USAGE ON *.*
TO 'fiona'@'localhost'
IDENTIFIED By 'pass1234';"
```

- This use has no privileges
- To view data add `SELECT`

```python
# Get privileges for users
mysql -u root -p -e "SHOW GRANTS FOR 'fiona'@'localhost' \G"
```

- Grant Options

```python
GRANT privilege_list ON database.table To 'user'@'host;

# Access
SELECT -- read data
INSERT -- add rows
UPDATE -- modify rows
DELETE -- remove rows

# Structure
CREATE
DROP
ALTER
INDEX -- create/drop indexes

# Admin
GRANT OPTION -- grant priv. to others
SUPER -- superlevel conrol
PROCESS -- view running queries
SHUTDOWN
RELOAD -- reload priv. cache

# Other
EXECUTE -- run stored procedures
LOCK TABLES -- lock tables
REFERENCES -- foreign keys
```


- Daemon
	- Background process that runs continuously
- Tar
	- Combined multiple files and folders into one file
	- Does not compress by default
