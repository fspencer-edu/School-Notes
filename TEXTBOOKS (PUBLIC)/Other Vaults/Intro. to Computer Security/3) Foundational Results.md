
- Harrison, Ruzzo, and Ullman (HRU) (1976) proivdes that in the most general abstract case, security was undecidable 

# The General Question

- R = set of generic (primitive) rights of the system
- When a generic right, r, is added to an element of the acces control matrix, the right is said to be leaked
- No generic rights can be added to the matrix
- There is no authorized transfer of rights
- If a system can never lead the right, r, the system is called safe with respect to the right r
- If can leak, it is unsafe with respect to right r
- Safety refers to the abstract model and security refers to the actual implementation
- A secure system correspond to a model safe with respect to all rights
- Model safe with respect to all rights does not ensure a secure system

# Basic Results

- Mono-operation; single primitive commands
- There exists an algorithm that will determine whether a given mono-operational protection system with initial state $s_0$ is safe with respect to a generic right $r$
- Omit the delete and destroy commands
- All creates are unnecessary except possible the first, and entering rights into the new subject are rewritten to enter the new right into the lone create subject
- By enumerating all possible state, determine whether the system is safe
- A Turing machine T consists of a head and an infinite tape divided into cell numbered 1, 2, ..., from left to right
- It is undecidable whether a given state of a given protection system is safe for a given generic right

<img src="/images/Pasted image 20251104102155.png" alt="image" width="500">

**Computability Theory**
- It is undecidable whether a given state of a protection system is safe for a given generic right
- Reduction from the halting problem


```c
command cp, A(si , si–1)
	if own in a[si–1, si ] and p in a[si , si ] and A in a[si , si ]
	then 
		delete p from a[si , si ]; 
		delete A from a[si , si ]; 
		enter B into a[si , si ]; 
		enter q into a[si–1, si–1]; 
end

command cp, A(si , si+1) 
	if own in a[si , si+1] and p in a[si , si ] and A in a[si , si ]
	then 
		delete p from a[si , si ]; 
		delete A from a[si , si ]; 
		enter B into a[si , si ]; 
		enter q into a[si+1, si+1];
end

command crightmostp, A(sk, sk+1)
	if end in a[si , si ] and p in a[si , si ] and A in a[si , si ] then
		delete end from a[sk, sk];
		create new subject sk+1;
		enter own into a[sk, sk+1];
		enter end into a[sk+1, sk+1]; delete p from a[si , si ];
		delete A from a[si , si ];
		enter B into a[si , si ];
		enter q into a[si+1, si+1]; 
end
```

- The set of unsafe systems is recursively enumerable
- Deleting the create primitive makes the safety question decidable
- Monotonic, increase in size and complexity, cannot decrease
- The safety question for bi-conditional monotonic protection system is undecidable
- The safety question for mono-conditional monotonic protection system is decidable
- The safety question for mono-conditional monotonic protection system with create, enter, and delete (no destroy) is decidable
- Safety question is undecidable for generic protection models
- Decidable if the protection system is restricted

# Summary

- Safety question is concerned with ensuring that a computer system remains in a secure and safe state
- Impossible to create an algorithm that can always return an answer for every system

# Further Reading
