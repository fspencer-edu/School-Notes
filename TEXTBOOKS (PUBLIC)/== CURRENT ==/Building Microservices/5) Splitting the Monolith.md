# It's All About Seams

- Services should be highly cohesive and loosely coupled
- Working Effectively with Legacy Code
	- Defines the concept of a seam
	- Portion of the code that can be treated in isolation and worked on without impacting the rest of the codebase
	- Service boundaries
- Namespaces groups similar code together
	- `package` in Java

# Breaking Apart MusicCorp

- Identify high-level bounded contexts
- Backend of MusicCorp
	- Catalog
	- Finance
	- Warehouse
	- Recommendations
- Create packages representing these contexts
	- Move the existing code into them
	- Refactoring
- Remaining code will identify misses bounded contexts
- Analyze dependencies between packages
- Packages should represent similar contexts in real-life organizational groups

# The Reasons to Split the Monolith

- Incremental approach

## Pace of Change

- Split large or potential large code bases

## Team Structure

- Split given geographical regions

## Security

- Split to tighten up protection of sensitive information

## Technology

- Split highly used services

# Tangled Dependencies

- Acyclical graph of dependencies
- 

# The Database

# Getting the Grips with the Problm

# Example: Breaking Foreign Key Relationships

# Example: Shared Static Data

# Example: Shared Data

# Example: Shared Tables

# Refactoring Databases
# Transactional Boundaries

# Reporting

# The Reporting Database

# Data Retrieval via Service Calls

# Data Pump

# Event Data Pump

# Backup Data Pump

# Toward Real Time

# Cost of Change


# Understanding Root Causes

# Summary