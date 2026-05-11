
- Sniffers were placed on many computers run by network service providers and recorded login names and passwords
- Cryptography provides a mechanism for performing specific functions
	- Preventing unauthorized people from reading and altering messages on a network
- Computer security is based on mathematical constructions, analyses, and proofs

- Security mechanisms detect and prevent attacks and recover from those that succeed

# The Basic Components

- Computer security rests on
	- Confidentiality
	- Integrity
	- Availability

## Confidentiality

- Confidentiality is the concealment of information or resources
	- Military
	- Government
	- Proprietary
- Access control mechanisms support confidentiality
- A cryptographic key controls access to the unscrambled data
- Resource hiding is another important aspect of confidentiality

## Integrity

- Integrity refers to the trustworthiness of data or resources
- Integrity mechanisms fails into two classes
	- Prevention
	- Detection
- Prevention mechanisms seek to maintain the integrity of the data by blocking any unauthorized ways
- Detection mechanisms do not try to prevent violations of integrity, they report the data integrity is no longer trustworthy
	- Analyze system events
- Correctness and trustworthiness of data

## Availability

- Availability refers to the ability to use the information or resource desired
- Attempts to block availability, called denial of service attack
- System designs usually assume a statistical model to analyze expected patterns of use
	- Atypical events need to be taken into account

# Threats

- A threat is a potential violation of security
- Actions are called attacks

4 Classes
1. Disclosure
	1. Unauthorized access to information
2. Deception
	1. Acceptance of false data
3. Disruption
	1. Interruption or prevention of correct operation
4. Usurpation
	1. Unauthorized control of some part of a system

- Encompass common threats

- Snooping
	- The unauthorized interception of information, is a form of disclosure
	- Wiretapping, or passive wiretapping
	- Confidentiality counters the threat
- Modification or altering
	- Unauthorized change of information
	- Disruption and usurpation
	- Active wiretapping
	- Man-in-the-middle attack
	- Integrity services counter this threat
- Masquerading or spoofing
	- An impersonation of on entity by another
	- Deception and usurpation
	- Integrity service (authentication) counter this threat

- Delegation occurs (masquerading) when one entity authorizes a second entity to perform functions on its behalf

 - Repudiation of origin
	 - A false denial that an entity sent (or created) something
	 - Form of deception
	 - Integrity mechanisms counter this threat
- Denial of receipt
	- False denial that an entity received some information or message
	- Form of deception
	- Integrity and availability counter this threat
- Delay
	- A temporary inhibition of service
	- Form of usurpation, and deception
	- Attack can force delay, and gain authorization
	- Availability counters the threat
- Denial of service
	- Long-term inhibition of service
	- Form of usurpation, used to deceive
	- Availability counters this threat

# Policy and Mechanism

- A security policy is a statement of what is, and what is not, allowed
- A security mechanism is a method, tool, or procedure for enforcing a security policy


## Goals of Security

- Security mechanisms can prevent the attack, detect the attack, or recover data from the attack

- Prevention
	- Attack will fail
	- Implementation of mechanisms users cannot override
	- Passwords
- Detection
	- Useful when an attack cannot be prevented
	- Attack is monitored, and data is collected
	- Warning
	- System log reports
- Recovery
	- Stop an attack and assess and repair damage

# Assumptions and Trust

- Security rests on assumptions specific to the type of security required and the environment
- A policy consists of a set of axioms that the policy makers believe can be enforced
- Secure and nonsecure state
- Mechanisms are
	- Secure
	- Precise
	- Broad

$P$ = all possible state
$Q$ = set of secure state
$R$ = security mechanisms

$R ⊆ P$

- A security mechanism is security if $R ⊆ Q$
- Precise is $R = Q$
- Board if there are state $r$ such that $r ∈ R$ and $r ∉ Q$

- The union of all security mechanisms active on a system would provide a single precise mechanisms (R = Q)
- In reality, security mechanisms are broad

Assumptions
1. Each mechanisms is designed to implement one or more parts of the security policy
2. The union implements all aspects of the security policy
3. The mechanisms are implemented correctly
4. Mechanisms are installed and administered correctly


# Assurance

- Assurance is the aspect of trust in a system
- Technology for medicine
	- Certification
	- Manufacturing standards
	- Preventative seal
- Assurance specifies steps to ensure that computer will function properly
- Detailed specification of desired or undesirable behaviour
- A system is said to satisfy a specification if the specification correctly state how the system will function

## Specification

- A specification (formal or informal) is a statement of the desired functioning of the system
- A derivation of specifications is determination of the set of requirements relevant to the system's planned use

## Design

- The design of a system translates the specifications into components that will implement them
- The design is said to satisfy the specifications if, under all relevant circumstances, the design will not permit the system to violate those specification

## Implementation

- The implementation creates a system that satisfies that design
- Satisfy the specification
- A program is correct if it implementation performs as specified
- Proofs of correctness require each line of source code to be checked
- Precondition and postconditions correctness
- Complexity of programs makes verification difficult
- Program has pre-condition derived from environment of system
- Program verification assumes that the programs are complied correctly
- Hardware failure, buggy code, and failure in other tools may invalidate the preconditions


- Posteriori verification techniques known as testing are used
- Execute the program on data to determine output
	- Introduce errors to determine affect of output

# Operational Issues

## Cost-Benefit Analysis

- Benefit of computer security are weighted against total cost
- Overlapping benefits are also a consideration
- Integrity protection can be augmented to provide confidentiality

## Risk Analysis

- Risk analysis asses potential threats and the likelihood that they will materialize
- Level of protection is a function of the probability of an attack occurring
- Risk is a function of environment
- Risk changes with time
- Many risks are remote
- Analysis paralysis, refers to making risk analyses with no effort to act on those analyses

## Laws and Customs

- Laws restrict the availability and use of technology and affect procedural controls
- Most site require users to give permission for system administrator to read their files
- Society distinguished between legal and acceptable practices

# Human Issues


## Organizational Problems

- Losses occur when security protections are in place
- Security provide no direct financial reward to the user

## People Problems

- Many successful break-ins are from social engineering

# Trying It All Together

- Human issues pervade each stage of the cycle

<img src="/images/Pasted image 20251104094500.png" alt="image" width="500">

