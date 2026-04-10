
- Key management refers to the distribution of cryptographic keys
- Authentication and key distribution
- Key generation and the binding of an identity to a key using certificates
- Key storage and revocation

$X$ -> $Y : {Z} k$

- Entity X sends entity Y a message Z enciphered with key k
$k$ -> secret key
$e$ -> public key
$d$ -> private key


# Session and Interchange Keys

- An interchange key is a cryptographic key associated with a principal to a communication
- A session key is a cryptographic key associated with the communication itself
- Session key prevent forward search
- A forward search attack occurs when the set of plaintext messages is small
- Ciphertext is compared with the precomputed texts to get public key

# Key Exchange

- Key cannot be transmitted in the clear
- A third party can be trusted
- Cryptosystems and protocols are publicly known
	- Only secrete data is to be the cryptographic keys

## Classical Cryptographic Key Exchange and Authentication

- Classical protocols rely on a trusted third party with a shared secret key

![[Pasted image 20251105115711.png]]

**Needham-Schroeder Protocol**

![[Pasted image 20251105115804.png]]
![[Pasted image 20251105115815.png]]

- The two random numbers are generated, and cannot repeat between different protocol exchanges
	- Nonces
- Assumes that all cryptographic keys are secure


**Otway-Rees Protocol**
![[Pasted image 20251105115953.png]]

- Uses timestamps
## Kerberos

- Kerberos uses the Needham-Schroeder protocol. as modified by Denning and Saccor

![[Pasted image 20251105120050.png]]

- The ticket is the issuer's voucher for the identity of the requester of the service
- The authenticator contains the identity of the sender of the ticket and is used

## Public Key Cryptographic Key Exchange and Authentication

Sally -> Bob : {$k_{session}$}$e_{bob}$




# Cryptographic Key Infrastructure

- Classical cryptosystems use shared keys
- Not possible to bind an identity to a key
- Originator to sign the public key with her private key
- Kohnfelder suggests creating a message containing a representation of identity, the corresponding public key, a timestamp, and having a trusted authority sign it
- A certificate is a token that binds an identity to a cryptographic key

## Certificate Signature Chains

- The usual form of certification is for the issuer to encipher a hash of the identity of the subject, the public key, and information of time of issue or expiration data
- Two approaches to this problem are to construct a tree-like hierarchy, with the public key of the root known out of band, or to allow an arbitrary arrangement of certifiers and rely on each individual's knowledge of the certifiers

### X.509: Certification Signatures Chains


- The Directory Authentication Framework is the basis for many other protocols
- Defines certificates formats and certification validation in a generic context
- New version X.509v3

X.509 Structure
1. Version
2. Serial number
3. Signature algorithm identifier
4. Issuer's Distinguished Name
5. Validity interval
6. Subject's Distinguished Name
7. Subject's public key information
8. Issuer's unique identifier
9. Subject's unique identifier
10. Extensions
11. Signature


- To validate the certificate, the user obtains the issuer's public key for a signature algorithm and deciphers the signature
- Uses the information in the signature field to recompute the hash value
- If it matches the deciphered signature, the signature is valid
- Certification authority (CA) is an entity that issues certification
- Two CAs are cross-certified if each has issued a certificate for the other
- Certificates can be revoked, or cancelled


### PGP Certificate Signature Chains

- PGP is en encipherment program use to provide privacy for electronic mail
	- Sign files digitally
- OpenPGP certificate is composed of packets
- A packet is a record with a tag describing its purpose
- A certificate contains a public key packet followed by zero ro more signature packets

PGP Structure
1. Version
2. Time of creation
3. Validity period
4. Public key algorithm and parameters
5. Public key

PGP Structure (version 3)
1. Version
2. Signature type
3. Creation time
4. Key identifier of the signer
5. Public key algorithm
6. Hash algorithm
7. Part of a signed hash value
8. Signature


## Summary

- Most protocols use certificates
- Infrastructure that manages public key and certification authorities is called a public key infrastructure

# Storing and Revoking Keys

## Key Storage

- OS access control mechanisms can often be evaded of defeated
- A feasible solution is to put the key onto one or more physical devices

## Key Revocation

- Certificate formats contain a key expiration data
- An expired certificate has reached a pre-designated period after which it is no longer valid
- X.509 and Internet public key infrastructure (PKIs) use lists of certificates
- A certificate revocation list is a list of certificates that are no longer valid
- Contains serial numbers of the revoked certificates and date
- Contains name of the issuer, data issued
- PGP allows signers of certificates to revoke their signatures

# Digital Signatures

- A digital signature is a construct that authenticates both the origin and contents of a message in a manner that is provable to a disinterested third party
- A digital signature provides the service of non-repudiation

## Classical Signatures

- All classical digital signatures schemes rely on a trusted third party
- Merkle's scheme

## Public Key Signatures

- Digital signature scheme based on the RSA system

