- Cryptographic systems are sensitive to environment
- Stream and block ciphers
- Network layers

# Problems

## Precomputing the Possible Messages

- Entropy (uncertainty) in a message
- If plaintext corresponding to intercepted ciphertext is drawn from a small set of possible plaintext, the cryptanalyst can encipher a set of plaintext
- Attack is similar to attacks to derive the cryptographic key of symmetric ciphers based on chosen plaintext
- Does not reveal the private key

## Misordered Blocks

- Parts of a ciphertext message can be deleted, replayed, or recorded
- One solution is to generates a cryptographic checksum of the entire message

## Statistical Regularities

- Independence of parts of ciphertext can give information relating to the structure of the enciphered messages
- Code book mode
	- Each part is effectively looked up in a list

## Summary

# Stream and Block Ciphers

- Some ciphers divide a message into sequence of parts, or blocks, and encipher each block with the same key
- A block cipher is a cipher for 

**Block Cipher**

$E_k(m) = E_k(b_1)E_k(b_2)...$

E -> encipherment algorithm
$E_k(b)$ -> encipherment of message b with key k

- DES is a block cipher
- Breaks the message into 64-bit blocks and uses the same 56-bit key to encipher each block


**Stream Cipher**

- If a key stream of a stream cipher repeats, it is a periodic cipher
- Vigenere cipher is a periodic stream cipher
- One-tine pad is a stream cipher, but not periodic

## Stream Ciphers

- One-time pad
- Bit-oriented cipher implement the one-time pad by XOR'ing each bit of the key with one bit of the message

m -> 00101
k -> 10010
c -> 10111

### Synchronous Stream Ciphers

- To simulate a random, infinitely long key, synchronous stream ciphers generate bit from a source other than the message itself
- An n-stage linear feedback shift register (LFSR) consists of an n-bit register and b-bit tap sequence
- 

### Self-Synchronous Stream Ciphers

- Self-synchronous ciphers obtains the key from the message itself
- Autokey

## Block Ciphers

### Multiple Encryption

# Networks and Cryptography

- ISO/OSI model provides an abstract representation of networks suitable
	- Composed of a series of layers
	- Each host has a principal at each layer that communicates with a peer on other hosts
	- Layer 1, 2, 3 principals interact only with similar principals at neighbouring hosts
	- 4, 5, 6, 7, interact with hosts at other end of the communication
	- Host forwards message to nearest host


![[Pasted image 20251105123920.png]]

- Difference between end-to-end protocol and a link protocol is that the intermediate hosts play no part in an end-to-end protocol other than forwarding messages
- Link protocol describes how each pair of intermediate hosts processes each message
- Telnet protocol is an application layer protocol that allows users to obtains a virtual terminal on a remote host
- IP is a network layer protocol that guides messages from a host to one of its immediate neighbours
- When encryption is used, called end-to-end encryption and link encryption
- PPP Encryption Control Protocol enciphers messages in telnet
- Traffic analysis deduces information from the sender and receiver of a message


# Example Protocols

- PEM is a privacy-enhanced electronic mail protocol at the application layers
- PGP is used as a security-enhanced electronic mail protocol
- SSL provides transport layer security
- HTTP can use SSL to ensure secure connections
- IPsec provides security mechanisms at the network

## Secure Electronic Mail: PEM

- UA (user agent) hands message to the MTA (message transport, transfer agent)
- MTA transfers the message to destination or another MTA

![[Pasted image 20251105124547.png]]

Privacy-Enhanced Electronic Mail (PEM)
1. Confidentiality
2. Origin authentication
3. Data integrity
4. Non-repudiation

### Design Principles

Design goals of PEM
1. Not to redesign existing mail system or protocol
2. Compatible with a range of MTAs, UAs, and other computers
3. Make privacy enragements
4. Enable two parties to use the protocol

### Basic Design

- PEM defined two types of keys
- Message to be sent is enciphered with a data encipherment key (DEK), corresponding to a session key
	- Enciphered with an interchange key
- Hash function value is a message integrity check (MIC)


### Other Considerations

- PEM suggests the use of a certificate-base key management scheme
- Internet electronic mail contains only ASCII characters
- ASCII is unreadable on EBCDIC-based system

1. Local representation of characters
2. Message integrity check
3. Message is treated as a stream of bits

- ASCII message has PEM header prepended

### Conclusion


## Security at the Network Layer: IPsec

- IPsec is a collection of protocols and mechanisms that provide confidentiality, authentication, message integrity, and replay detection at the IP layer
- IPsec mechanisms protect all messages send along a path
- Reside on an intermediate host (firewall or gateway)
	- Security gateway
- IPsec has two mode
	- Transport mode
		- Encapsulates the IP packet data areas in an IPsec envelope and uses IP to send the IPsec-wrapped ticket
		- IP header is not protected
		- Both endpoints support IPsec
	- Tunnel mode
		- Encapsulates an entire IP packet in an IPsec envelope and then forwards it using IP
		- IP is protected
		- Either or both endpoints do not support IPsec

![[Pasted image 20251105125433.png]]

- Two protocols provides message security
- Authentication header (AH) protocols provides message integrity and origin authentication
- Encapsulating security payload (ESP) protocol provides confidentiality and same services as AH
- Both protocols use key from Internet Key Exchange (IKE)

### IPsec Architecture

- Use security policy database (SPD) to determine how to handle messages
- Action depends on information in the IP and transport layer headers
- When packets arrives, IPsec mechanisms consults SPD for relevant network interface
	- Source port
	- Destination port
	- Address

SPD has two entries for destinations
- 10.1.2.3
- 10.1.2.103

- First applies to packet with destination port 25
- Second transporting the protocol HTTP
- If a packet arrives with destination address 10.1.2.50, and its destination port is 25, then first entry applied
- If 80, the second entry

- Entries are check in order
- If no entry matches in the incoming packets, it is discarded

- A security association (SA) is an association between peers for security services
	- Unidirectional
	- Destination address
	- Security protocol (AH or ESP)
	- 32-bit security parameter index (SPI)

- A security association bundle (SA bundle) is a sequence of security associations that the IPsec mechanisms apply to packets
- Iterated tunnelling occurs when multiple hosts build tunnels through which they are send traffic
- Transport adjacency is used with both AH and ESP protocols are used
	- ESP should precede AH
- If AH precedes, then it protects the IP packets, but ESP would not protect IP headers

![[Pasted image 20251105130330.png]]

- The SA directs the packet to the encapsulated and forwarded to firewall
- The innermost IP packet is forwarded to the equity and processed

### Authentication Header Protocol

- Goal of AH is to provide origin authentication, message integrity, and protection against replay
- Checks that replay is not occurring
- Checks the authentication data

### Encapsulating Security Payload Protocol

- The goal of encapsulating security payload (ESP) is to provide confidentiality, origin authentication, message integrity, protection against replay, and a limited form of traffic flow confidentiality
- 

# Conclusion