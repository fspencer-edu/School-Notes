- A protection system describes the conditions under which a system is secure
- The access control matrix model arose both in OS research and in database research
	- Describes allowed accesses

# Protection State

- The state of a system is the collection of the current values of all memory locations, all secondary storage, and all registers of other components
- Subset of this collection is the protection state
- An access control matrix is one tool that can describe the current protection state

$P$ = protection state
$Q of P$ = authorized subset
$Q$ = secure
$P - Q$ = not secure

- The access control matrix model is the most precise model used to describe a protection state
- Characterizes the rights of each subset with respect to every other eneity

$A$ = specification

- Protection state change with system changes
	- State transition
- Any operation on a real system causes multiple state transition
- Variable altered that affect privileges of a process, then the program does alter the protection state needs to be accounted for

# Access Control Matrix Model

- Describes rights of users over files in a matrix
- Butler Lampson (1971)
- Graham and Denning refined
- The set of all protected entities is called the set of objects O
- The set of subjects S is the set of active objects
	- Processes
	- users
- Control matrix model, A
	- Right, R
- The set of protection states of the system represented by (S, O, A)

<img src="/images/Pasted image 20251104095335.png" alt="image" width="500">

Process 1
- Can read/write file 1
- Read file 2
- Communicate with process 2 with write

Process 2
- Append file 1
- Read file 2

- Each process are treated as both subjects and objects
- Abstract model of the protection state
- The own right is a distinguished right

UNIX
- read, write, execute rights

- Message sent between processes on a local area network (LAN)
- Network protocols
	- own (ability to add servers)
	- FTP (File Transfer Protocol)
	- NFS (Network File System)
	- SMTP (Simple Mail Transfer Protocol)

- The subject nob is configured to provided NFS service to a set of clients that does not include the host toadflax

<img src="/images/Pasted image 20251104095846.png" alt="image" width="500">

- Access control matrices can model programming language accesses
- Locking function to enforce the Bernstein conditions
	- A process is writing to a file, other processes cannot access
	- When done, the process is accessible again
	- Ensure data consistency
- 
<img src="/images/Pasted image 20251104100034.png" alt="image" width="500">


# Protection State Transitions

- Initial state $X_0 = (S_0, O_0, A_0)$
- Set of transitions $τ_1, τ_2 ...$
- Sequences of state transitions are represented as single commands, or transformation procedures, that update the access control matrix
- For every command, there is a sequence of state transition operations
- Defined a set of primitive commands that alter the access control matrix

Primitive commands
1. Create subject $s$
2. Create object $o$
3. Enter $r$ into $a[s, o]$
4. Delete $r$ from $a[s, o]$
5. destroy subject $s$
6. destroy subject $o$

- Primitive operations are combined into commands
- UNIX
	- p = process
	- f = file
	- r, w, e - read, write, execute


```c
// Adds r and r permission to file f
command create•file(p, f)
	create objectf;
	enter own into a[p,f];
	enter r into a[p,f];
	enter w into a[p,f];
end

// create new process q in p, r and w rights enable the partent and child to signal each other
command spawn•process(p, q)
	create subject q;
	enter own into a[p, q];
	enter r into a[p, q];
	enter w into a[p, q];
	enter r into a[q, p];
	enter w into a[q, p];
end

// mono-operation
caommand make•owner(p,f)
	enter own into a[p,f]
end
```

## Conditional Commands

```c
// p give q right and read for file f
command grant•read•file•1(p, f, q)
	if own in a[p, f]
	then
		enter r into a[q, f];
end

// Give any other subject r rights over that object
command grant•read•file•2(p, f, q)
	if r in a[p, f] and c in a[p, f]
	then
		enter r into a[q, f];
end
```

- Commands with one condition are called mono-conditional
- Two, bi-conditional
- All conditions are joined by and, and never by or

```c
// 
if own in a[p, f] or a in a[p, f]
then
	enter r into a[q, f];
	
//
command grant•write•file•1(p, f, q)
	if own in a[p, f]
	then
		enter r into a[q, f];
	end
command grant•write•file•2(p, f, q)
	if a in a[p, f]
	then
		enter r into a[q, f];
end
```

# Summary

- Access control matrix is the purest form to express security policy
- Transitions are express in terms of commands
- A command consists of possible condition followed by one or more primitive operations
- Access control matrix sometimes referred to as an authorization matrix (legacy)