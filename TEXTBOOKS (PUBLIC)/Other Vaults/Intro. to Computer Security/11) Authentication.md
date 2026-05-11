- Authentication is the binding of an identity to a principal
- Network-based authentication mechanisms require a principal to authentication to a single system (local or remote)

# Authentication Basics

- Authentication in the binding of an identity to a subject

External information
1. Entity knowns (password)
2. Entity has (badge)
3. What entity is (fingerprints)
4. Where entity is (directory)

- Computer must store information about the entity

5 Components of Authentication system
1) Authentication information
2) Complementary information
3) Complementary functions
4) Authentication function
5) Selection functions

- A user authenticates by entering a password

## Passwords

- Password is information associated with an entity that confirms the entity's identity
- Sequence of characters
- UNIX password does not store the passwords
	- Instead hash password

2 Approaches to protect password
1. Hide enough information
	1. Readable only by root
	2. Shadow password files
2. Prevent access to the authentication functions
	1. Cannot login from a network

### Attacking a Password System

- A dictionary attack is the guessing of a password by repeated trial and error

### Countering Password Guessing

$P >= TG/N$

$P$ -> probability of guessed password in time
$G$ -> number of guesses
$T$ -> number of time units
$N$ -> number of possible password


_Example_

$R$ -> number of bytes per minute
$E$ -> number of character exchanged when loggin
$S$ -> length of password


#### Random Selection of Password


#### Pronounceable and Other Computer-Generated Passwords

#### User Selection of Passwords

#### Reusable Passwords and Dictionary Attacks

#### Guessing Through Authentication Functions

### Password Aging



# Challenge-Response

## Pass Algorithm

## One-Time Passwords

### Hardware-Supported Challenge-Response Procedures


### Challenge-Response and Dictionary Attacks


# Biometrics

## Fingerprints

## Voices
## Eyes
## Faces
## Keystrokes
## Combinations

## Caution


# Location

# Multiple Methods

# Summary

pg 220