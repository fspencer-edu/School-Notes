- Cryptography is the science and art of protecting information by transforming it into a form

# What Is Cryptography?

- Cryptanalysis is the breaking of code
- A cryptosystem $(E, D, M, K, C)$

$E: M \times K$ -> enciphering functions
$D: C \times K -> M$ -> deciphering functions
$M$ -> plaintext
$K$ -> keys
$C$ -> ciphertext

- Caesar cipher
	- Letters are shifted
	- Hello -> KHOOR

- Goal of cryptography is to keep enciphered information secret
- Ciphertext only attack
	- Adversary has only ciphertext
	- Finds plaintext, and key
- Known plaintext
	- Find key
- Chosen plaintext
	- Find key

- Attacks use moth mathematics and statistics
- Assumptions are collectively called a model of the language
- 1-gram model
	- Language
- 2-gram model
	- Frequency
- Markov model
- Word models

# Classical Cryptosystems

- Classical cryptosystems (single-key or symmetric cryptosystems) use the same key for encipherment and decipherment

$E_k ∈ C$
$k ∈ K$
$D_k ∈ D$
$D_k = E^{-1}$


- Enciphering functions belong to ciphertext

<img src="/images/Pasted image 20251104133232.png" alt="image" width="500">

- Caesar has a key of 3
- To decipher KHOOR use decipherment

2 Basic Types of Classical Ciphers
1. Transposition
2. Subsitution


## Transposition Ciphers

- A transposition cipher rearranges the characters in the plaintext to form the cipher processed down, then across, and reading the ciphertext across, then down

```c
M -> HELLO, WORLD”

HLOOL
ELWRD

C -> HLOOLELWRD
```

- A key to a transposition cipher is a permutation function
- Does not alter the frequency
- Attacking transposition cipher requires rearrangement of letters
	- Anagramming, uses tables of n-gram frequencies to identity common n-grams

```c
C -> HLOOLELWRD

HE -> 0.0305
LL
OW
OR
LD

```
- The frequency suggests that E follow H

## Substitution Ciphers

- A substation cipher changes the characters in the plaintext

<img src="/images/Pasted image 20251104134000.png" alt="image" width="500">

```c
KHOOR ZRUOG -> Hello World
```

- Apply character-based model
- The correlation should be a maximum when the key k translates the ciphertext into English
- Using Konheim's model of single-character frequencies

<img src="/images/Pasted image 20251104134304.png" alt="image" width="500">


Konheim Model
- Access control and safety
- Rights propagation
- System commands
- Undecidability (safety problem)


Denning's Model (Information Flow Model)
- Information flow and classification
- Lattices and security labels
- Confidentiality and integrity
- Preventing covert or implicit leaks

Transposition Cipher
- Rearranges the order
- Rail Fence
- Columnar

Substitution Cipher
- Replaces each symbol with another
- Caesar
- Mono-alphabetic
- Vigenere

### Vigenere Cipher

- Chooses a sequence of keys, represented by a string
- Length of the key is called the period
- Polyalphabetic
- Multiple Caesar shifts

<img src="/images/Pasted image 20251104134631.png" alt="image" width="500">


- Encrypts text by shifting each letter of plaintext by an amount by a key or phrase
- Key is repeated to match messages length
- Letter converted to number
- H => 7 + K => 10
- (7 + 10) = 17 => R

```c
HELLO WORLD

KEY

KEYKEYKEYKE

RIJVSUYVJN
```

$C = (P + K) mod 26$

$P = (C + K) mod 26$


<img src="/images/Pasted image 20251104135124.png" alt="image" width="500">

- The index of coincidence measures the difference in the frequencies of the letters in the ciphertext
- Probability that two randomly chosen letters from ciphertext will be the same

<img src="/images/Pasted image 20251104135210.png" alt="image" width="500">

- Lower the index of coincidence, the less variation
- Considered unbreakable
- Prussian cavalry noticed repetitions to identify key

<img src="/images/Pasted image 20251104135313.png" alt="image" width="500">

<img src="/images/Pasted image 20251104135325.png" alt="image" width="500">

- OPK appears twice
- Ciphertext repetitions are 9 characters apart

<img src="/images/Pasted image 20251104135413.png" alt="image" width="500">

- GCD of 30 and 72 is 6
- Compute the IC for each alphabet

<img src="/images/Pasted image 20251104135522.png" alt="image" width="500">


<img src="/images/Pasted image 20251104135505.png" alt="image" width="500">

_Example: Decryption of Vigenere_

```c
C -> LXFOPVEFRNHR

// Repated sequence FR appears every 6 letter
Key = length(6)

// Divide cipher into k groups
LXFOPV
EFRNHR

P -> ATTACK AT DAWN
```

IC = 0.066

- Perform frequency analysis to get key
- If key is LEMON

$P = (C - K) mod 26$

L => A: $P = (11 - 11) mod 26 = 0$
X => T: $P = (25 - 4) mod 26 = 0$


### One-Time Pad

- Variant of the Vigenere cipher
- Key string is chosen at random
- At least as long as the message
	- Does not repeat
- Random generation of the key and key distribution

## Data Encryption Standard

- Data Encryption Standard (DES) was designed to encipher sensitive but non-classified data
- Bit-oriented
	- Uses transposition and substitution
- Referred to as produce cipher
- Input, output, and key are each 64-bits long (blocks)
- Cipher consists of 16 rounds
- Each round uses a separate key of 48 bits
	- From block by dropping the parity bits
	- 64 - 16 = 48
- Right half produce 32 bits, then xor'ed into the left

<img src="/images/Pasted image 20251104140535.png" alt="image" width="500">

<img src="/images/Pasted image 20251104140551.png" alt="image" width="500">


- Each set of 8 bits are put through substitution table (s-box)
	- Catenated into 32-bit quantity
- Complementary property

<img src="/images/Pasted image 20251104140708.png" alt="image" width="500">


<img src="/images/Pasted image 20251104140726.png" alt="image" width="500">


- Biham and Shamir applied a technique called differential cryptanalysis to the DES

<img src="/images/Pasted image 20251104140834.png" alt="image" width="500">


DES Process
1. Initial permutation
	- Rearrange input bits
2. 16 Feistel rounds
	- Substitution and permutation using round keys
3. Swap halves
	1. Final round output
4. Final permutation
	1. Inverse of initial step

AES (Advanced Encryption Standard)
- 128-bit size

EDE Mode (Encrypt-Decrypt-Encrypt)
- 3DES

## Other Classical Ciphers

- NewDES has a block size of 64 bits and a key length of 120 bits
- FEAL-8
	- Differential cryptanalysis

# Public Key Cryptography

- Distinguishes between encipherment and decipherment
- 1976
- One of the keys is publicly known
- Other is private
- Classical cryptography requires the send and recipient to share a common key
- Complementary key must remain secret

1. Computationally easy to encipher or decipher a message given the appraise key
2. Must be computationally unfeasible to derive the private key from the public key
3. Must be computationally infeasible to determine the private key from a choice plaintext attack

- RSA cipher provides both secrecy and authentication

## RSA

- RSA is an exponentiation cipher
- Two large prime numbers, $p$ and $q$
$n = pq$
Totient $\phi(n)$ of $n$ if the number of numbers less than $n$ with no factors in common with $n$

_Example_

$n = 10$
$n = 1, 3, 7, 9$
$\phi(10) = 4$

$n = 21$
$n = 1, 2, 4, 5, 8, 10, 11, 13, 16, 17, 19, 20$
$\phi(10) = 12$


- Choose an integer $e < n$ that is relatively prime to $\phi(n)$
- Find second integer $d$, $ed$ mod $\phi(n) = 1$
- Public key = $(e, n)$
- Private key = $d$
- Message = $m$

$c = m^e$ mod $n$

$c = c^d$ mod $n$

_Example_

$p = 7$
$q = 11$
$n = 77$
$\phi(10) = 60$


- If Sally chooses $e = 17$, then her private key is $d = 53$
- Plaintext is represented by a number 00 (A) and 25 (Z)
- Bob wants to send the message "HELLO WORLD" with Sally's public key
- `07 04 11 11 14 26 22 14 17 11 03`

$07^{17}$ mod $77 = 22$
$04^{17}$ mod $77 = 16$
$11^{17}$ mod $77 = 44$

-> `28 16 44 44 42 38 22 42 19 44 75`

- In addition to confidentiality, RSA provides data and origin authentication
- If Sally enciphers her message using private key, anyone can read it, but if anyone alters it, the ciphertext cannot be deciphered correctly

- For Sally to send message to Bob
- Enciphers the message with her private key and sends
- `07 04 11 11 14 26 22 14 17 11 03`
- To send Sally enciphers the message with private key

-> `35 09 44 44 93 12 24 94 04 05`

- Bob can be sure that no letter were altered

- The use of public key system provides a non-repudiation of origin
- Public is inverse of private key
- Only the private key could have enciphered the message
- RSA can be broken using techniques to break classical substitution ciphers
- Attacker can use forward search or pre-computation to break cipher

# Cryptographic Checksums

- Sally wants to send Bob a message of n bits
- Wants to verify that the message he receives is the same one sent
- Applied checksum function, to generate a smaller set of k bits from the original n bits
- The smaller set is called the checksum or message digest
- When Bob receives the message recomputes the checksum and compares it to the one send
- If matches, then message is not altered

_Example_
- Parity bit in ASCII representation is used as a single-bit checksum
- If odd parity, the sum of the 1-bit

- A cryptographic checksum function (strong hash function/strong one-way function) is a function that has the following properties (h: A -> B)

1. For any $x ∈ A, h(x)$ is easy to compute
2. For any $y ∈ B$, infeasible to find $x ∈ A$ such that $h(x) = y$
3. Computationally infeasible to find collision pair


- Pigeonhole principle state that if there are n containers for n + 1 objects, at least one container will hold two objects
- A keyed cryptographic checksum function requires a cryptographic key, keyless does not

CBC (Cipher Block Chaining) Mode
- Use to message authentication code if 64-bits or fewer
- Uses initialization vector (IV)
- Chaining
	- Block depends on the previous one


MD2 (Message Digest)
- Obsolete
- Outputs a fixed-length digest
- Collisions

SHA Checksum (Secure Hash Algorithm)
- Inputs message processed in 512-bit blocks
- Output message digest
- One-way

## HMAC (Hash-Based Message Authentication Code)

- HMAC is a generic term for an algorithm that uses keyless hash functions and cryptographic key to produce keyed hash functions
- Many countries restrict the import and export of software that implements cryptographic algorithms with keyed hash functions

<img src="/images/Pasted image 20251104143955.png" alt="image" width="500">



- Uses hash operations and two special padding constants
	- ipad
	- opad

Applications
- AWS
- TLS