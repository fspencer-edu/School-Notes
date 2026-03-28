
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