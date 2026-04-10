
- Integrity policies focus on integrity rather than confidentiality

# Goals

Commercial requirements
1. Users will not write their own programs
2. Programmers will develop and test programs on a non-production system
3. A special process must be followed to install a program from development into production
4. The process must be controlled and audited
5. Mangers and auditors must have access to both the system state and system logs

- Separation of duty
	- IF two or more steps are required to perform a critical function, at least two different people should perform the steps
- Separation of functipn
	- Developers do not develop new programs on production systems because potential threat of production data
- Auditing
	- Process of analyzing systems to determine what actions took place and who performed them
- Military environment is centralized
	- Tight control
- Commercial is decentralized
	- Limited amount of information for public
	- Aggregation can deduce sensitive information

# Biba Integrity Model

- System consists of a set of S subjects, O objects, and a set I of integrity levels
- 1977
- Higher the level, the more confidence one has that a program will execute correctly
- Data at a higher level is more accurate and reliable
- Ensure information cannot flow from a less trustworthy source to a more trustworthy one
- No write up, no read down

Biba's Model
![[Pasted image 20251104121711.png]]

- OS LOCUS
- Limit execution domains for each program to prevent untrusted software from altering data or other software

# Clark-Wilson Integrity Model

- David Clark and David Wilson developed integrity model (1987) which uses transactions as the basic operation
- Models systems more realistically
- Data is in a consistent state

D = money deposited
W = money withdrawn
YB = money in all accounts (yesterday)
TB = money in all accounts (today)

$D + TB - W = TB$

- Before and after each action, consistency conditions must hold
- Well-formed transaction is a series of operations that transition the system from one consistent state to another
- Requiring more than one person to handle process (principle of separation of duty)
- Computer-based transactions
	- Someone must certify that the transactions are correct

## The Model

- Define data subject to its integrity controls as constrained data items (CDIs)
	- Balances of accounts
- Data not subject are unconstrained data items (UDI)
	- Gifts
- A set of integrity constraints constrain the values of the CDI

2 Procedures
1) Integrity verification procedures (IVP)
	1) Test that the CDI conform to the integrity constraints
	2) Valid state
2) Transformation procedure (TP)
	1) Change the state of the data

IVP
- Checking the accounts are balanced
TP
- Depositing money
- Withdrawing money
- Transferring money

2 Certification Rules
1) When any IVP is run, it must ensure that all CDIs are in a valid state
2) For some associated set of CDIs, a TP must transform those CDIs in a valid state into a (different) valid state
3) 3) The allowed relations must meet the requirements imposed by the principle of separation of duty
4) All TPs must append enough information to reconstruct the operation to an append-only CDI
5) Any TP that takes as input a UDI may perform only valid transformations, or no transformations

Enforcement Rule
1) The system must maintain the certified relations, and must ensure that only TPs certified to run on a CDI manipulate that CDI
2) The system must associate a user with each TP and set of CDI
3) The system must authenticate each user attempting to execute a TP
4) Only the certifier of a TP may change the list of entities associates with that TP

- Firms do not classify data using a multilevel scheme
- Enforce separation of duty
- Certification is distinct from the notion of enforcement

Applications
- Banking
- Accounting
## Comparison with the Requirements

5 Requirements in Clark-Wilson Model
1) If users are not allowed to perform certifications of TPs, instead "trusted personnel" are
2) Control is to omit interpreter and compilers from production system
3) Installing a program from development system onto a production system requires a TP to do the installation and "trusted personnel" to certify
4) Auditing of program installation
5) Management and auditors can have access to system logs

Clark-Wilson model meets Lipner's requirements

## Comparison with Other Models

- Biba model attaches integrity levels to objects and subjects
- In CW-model each objects has two levels
	- Constrained (CDI)
	- Unconstrained (UDI)
- Subject have two levels
	- Certified (TP)
	- Uncertified (other procedures)
- Two models lie in the certification rules

Biba Model
- Biba model has none, asserts that trusted subjects exist to ensure that the actions are obeys
- Requires a security officer, to pass on every input sent to a process running at integrity level higher than that of an input
- Confidentiality

Clark-Wilson Model
- CW-model provide explicit requirement that actions must meet
- Requires trusted entity certify the method of upgrading data to a higher integrity level
- Integrity
- More parical

Bell-LaPadual Model
- Entities must change integrity levels to be trusted

# Summary
