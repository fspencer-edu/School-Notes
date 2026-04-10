
- Chinese Wall model is derived from the British laws concerning conflict of interest
	- Mixture of confidentiality and integrity models
- Clinical Information Systems security model is derived from medical ethics and laws about dissemination of patient data
- Originator controlled access
- Role-based access control

# Chinese Wall Model

- Refers to the equally to confidentiality and integrity
- Policies that involve a conflict of interest in business
	- Stock exchange
	- Investment
- Prevent a conflict of interest which a trader represents two clients, and the best interest of the clients conflict
- Information flow and access control model designed to prevent conflicts of interest in organizations

Definitions
- The objects of the database are items of information related to a company
- A company dataset (CD) contains objects related to a single company
- A conflict of interest (COI) class contains the datasets of companies in competition

- COI(O) represent the COI class that contains object O

![[Pasted image 20251104130134.png]]

- Model distinguishes between sanitized data and unsanitized data

## Bell-LaPadula and Chinese Wall Models

Chinese Wall Model
- Subjects have no associated security labels
- Past access is central to model's controls
- Each object assigned clearance
- Tracks the history of accesses

Bell-LaPadula Model
- Subjects have associated labels
- No notion of past accesses
- Cannot capture change over time


![[Pasted image 20251104130527.png]]

## Clark-Wilson and Chinese Wall Models

- Deals with many aspects of integrity, such as validation and verification, and access control

# Clinical Information System Security Policy

- Medical records require policies that combine confidentiality and integrity
- Conflict of interest is not a critical problem
- Patient confidentiality, authentication of records and personnel entries are critical

3 Types of entities
1) A patient is the subject of medical records, or an agent for that person who can give consent for the person to be treated
2) Personal health information is information about a patient's health or treatment
3) A clinician is a health-care professional whose has access to personal health information during job

Principles
1. Each medical record has an access control list naming the individuals or groups
2. One of the clinicians on the access control list must have the right to add other clinicians to the access control list
3. The responsible clinician must notify patient of the names on the access control list
4. Name of clinician, data, time of access of medial information is recorded

Creation Principle
- A clinician may open a record, with the patient on the access control list
- Referral

Deletion Principle
- Clinical information cannot be deleted, until sufficient time has passed

Confinement Principle
- Information form one medial record may be appended to a different medical record

Aggregation Principle
- Measures for preventing the aggregation of patient data must be effective

Enforcement Principle
- System that handles medical records must have a subsystem that enforces the preceding principles

## Bell-LaPadula and Clark Wilson Models

IVP Certify Items
- A person identified as a clinician
- A clinician validates information in the medical record
- Someone is to be notified of an event when occurred
- Operation cannot proceed without consent

# Originator Controlled Access Control

- Mandatory and discretionary access control (MAC and DACs) do not handle environments in which the originators of the document retain control over then after disseminated
- ORGCON ORCON (ORiginator CONtrolled)
	- Subject can given another subject rights to any object only with the approval of the creator of that object
	- Secretary of Defense of US
- Single author does not control dissemination
	- Organizations do
- Organizations that use categories grant access to individuals on a "need to know" basis
- ORCON is a decentralized system of access control
	- Originator determines who needs access to data
1. Owner of an object cannot change the access controls of the object
2. When an object is copies, the access control restrictions of that source are copied and bound to the target of the copy
3. The creator can alter the access control restrictions on a per-subject and per-object bases

- Owner may not override the originator

# Role-Based Access Control (RBAC)

- The ability, or need, to access information may depend on one's job functions

Access to the job of the user
1. A role is a collection of job functions
2. The active role of a subject is the role that is currently performing
3. The authorized roles of a subject is the set of roles that is authorized to assume
4. The predicate `canexect(s,t)` s true if and only if the subject can execute the transaction t at the current time


- Axiom implied that if a subject can execute any transaction, then that subject has an active role
- RBAC is a form of MAC
