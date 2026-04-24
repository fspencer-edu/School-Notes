
- A security policy defined "secure" for a system or a set of systems

# Security Policies

- A security policy is a statement that partitions the states of the system into
	- Authorized
	- Secure
- A secure system is a system that starts in an authorized state and cannot enter an unauthorized state

<img src="/images/Pasted image 20251104103416.png" alt="image" width="500">

- Finite-state machine consists of 4 states and 5 transitions
	- A = {s1, s2}
	- UA = {s3, s4}
- System is not secure, because can enter unauthorized state
- A breach of security occurs when a system enter an unauthorized state
- Confidentiality implied that information must not be disclosed to some set of entities
- Information flow is the leakage of rights, and ilicit transformation without leakage of rights
- Confidentiality policy
- Separation of duties
- A security mechanisms is an entity or procedure that enforces some part of the security policy
- A security model is a model that represents a particular policy or set of policies

# Types of Security Policies

- A military security policy (governmental security policy) is a security policy developed to provide confidentiality
- A commercial security policy is a security policy developed primarily to provide integrity
- Transaction-oriented integrity security policies are a type of information security policy that focus on ensuring that data integrity is maintain across entire transactions
	- Database
- A confidentialty policy is a security policy dealing only with confidentialiy
- An integrity is a security policy dealing with only integrity

# The Role of Trust

Security patch assumptions
1. Patch came from verified vendor
2. Vendor has tested patch
3. Vendor's test environment correspond with the local environment
4. Patch is installed correctly


Installation Assumptions
1. Formal verification of S (security related program) is correct
2. Assumption made in the formal verification of S are correct
3. The program will be transformed into an executable
4. Hardware will execute the program

# Types of Access Control

- 2 types of access controls
	- Alone 
	- Combination

- If an individual user can set an access control mechanism, it is a discretionary access control (DAC), also called an identity-based access control (IBAC)
- Identity is the key

- When a system mechanism controls access to an object and an individual user cannot alter that access, the control is a mandatory access control (MAC), also called a rule-based access control
- OS enforces MAC
- An originator controlled access control (ORCON or ORGCON) bases access on the creator of an object
- Allow originator to control the dissemination of the information

# Example: Academic Computer Security Policy

- Security policies have many or few details
- Explicitness depends on environment

## General University Policy

- Acceptable Use Policy (AUP)
- Generic constraints for individual units
- Written informally and aimed at the user community that is to abide by it

## Electronic Mail Policy

- Constraints imposed on access to, and use of, electronic mail
- 3 parts
	- AUP
	- Full policy
	- General electronic mail policy

### The Electronic Mail Policy Summary

- Warns users that electronic mail is not private

### The Full Policy

- Description of the context of the policy, its purpose and score

### Implementation at UC Davis

- Interpretation of the policy
- Adds campus-specific requirements and procedures to university's policies

# Summary

- 