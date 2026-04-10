
<font style="color:#96DED1">Memory hierarchy</font> - test
<font style="color:#96DED1">P</font> - test


- [x] [[#^493d64|1. Computer System Overview]]
- [x] [[#^8de730|2. Operating System Overview]]
- [x] [[#^e4bee5|3. Process Description and Control]]
- [x] [[#^090406|4. Threads]]
- [ ] [[#^e78bc7|5. Concurrency: Mutual Exclusion and Synchronization]]
- [ ] [[#^17da49|6. Concurrency and Starvation]]
- [ ] [[#^17a4c6|7. Memory Management]]
- [ ] [[#^6ee6a5|8. Virtual Memory]]
- [ ] [[#^f3a001|9. Uniprocessor Multithreading]]
- [ ] [[#^a0c7e7|10. Multiprocessors]]
- [ ] [[#^a2d488|11. I/O Management]]
- [ ] [[#^cf3bfc|12. File Management]]
- [ ] [[#^b127fb|13. Embedded Operating Systems]]
- [ ] [[#^9f0cf8|14. Virtual Machines]]
- [ ] [[#^34d101|15. Operating System Security]]
- [ ] [[#^dc25b5|16. Cloud and IoT Operating System]]


## 1. <font style="color:#FCD12A">Computer System Overview</font>  

^493d64

- An operating system (OS) uses the hardware resources of one or more processors to provide services to system users
- OS manages secondary memory and I/O (input/output) decides


1. <font style="color:#A8A8A8; text-decoration: underline;">Basic Elements</font>  

- At the top level, a computer consists of:
	- Processor
	- Memory
	- I/O components
	- one or more modules of each type
- 4 Main structural elements:
	- Processor
	- Main memory
	- I/O modules
	- System bus

<font style="color:#96DED1">Processor</font> - controls the operation of the computer and performs its data processing function, the processor is referred to as the _central processing unit_ (CPU)
<font style="color:#96DED1">Main memory</font> - stores data and programs, typically volatile, but also includes disk memory that is stored
<font style="color:#96DED1">I/O modules</font> - move data between the computer and its external environment
<font style="color:#96DED1">System bus</font> - provides for communication among processors, main memory, and I/O modules

- Processor exchanges data with memory
- Uses two internal register: 
	- Memory address register (MAR)
		- Specifies the address in memory for the next read or write
	- Memory buffer register (MBR)
		- Contains the data to be written into memory, or receives the data read from memory
	- I/O address register (I/OAR)
		- Specifies a particular I/O device
	- I/O buffer register (I/OBR)
		- Used to exchange of data between an I/O module and the processor
- A memory module consists of a set of locations, defined sequentially numbered addresses, containing a bit pattern that is instruction or data
- I/O module transfers data from external devices to processor and memory, vice versa






1. <font style="color:#A8A8A8; text-decoration: underline;">Evolution of the Microprocessor</font>  
- The microprocessor is the CPU of a computer system integrated into a single semi-conductor chip
	- Uses sub-nanosecond timeframes
	- Small size has brought portable computers
- Microprocessors have become the fastest general-purpose processor
	- Multiprocessors have chips (socket) containing multiple processors (called cores)
- Graphical Processing Units (GPUs) provide efficient computation on arrays of data using Single-Instruction Multiple Data (SIMD) techniques from supercomputers
- Digital Signal Processors (DSPs) are used to deal with streaming signals such as audio or video
- DSPs are embedded in I/O devices, like modems
- Handheld devices, use microprocessor in a System on a Chip (SoC), where many of other components of the system, such as GPUs, I/O devices, and main memory are location on the same chip


2. <font style="color:#A8A8A8; text-decoration: underline;">Instruction Execution</font>  
- Instruction processing consists of two steps (Instruction cycle):
	- Fetch stage
	- Execute stage
- Program execution halts only if the processor is turned off, or program instruction error that halt the processor
- Processor begins by fetching from program counter (PC) which holds the address of the next instruction
- PC increments after each instruction fetch
- The fetched instruction is loaded into the instruction register (IR)
- Instruction contains bits that specify the action
- 4 Categories of Instructions:
	- Processor-memory
		- Data may be transferred from processor to memory, or from memory to processor
	- Processor-I/O
		- Data may be transferred to or from a peripheral device by transferring between the processor and an I/O module
	- Data processing
		- The processor may perform some arithmetic or logic operation on data
	- Control
		- An instruction may specify that the sequence of execution be altered
- An instruction's execution may involve a combination of these action
- The accumulator (AC) is a register that temporarily stores intermediate results during arithmetic and logical operations
- Both the instructions and data are 16 bits long, and memory is organized as a sequence of 16-bit words
- Instruction format is in 4 bits for the opcode, allowing $2^4$ = 16 opcodes (represented by a single hexadecimal)
- With the remaining 12 bit of instruction, $2^12$ = 4,096 (4K) words of memory

![[Pasted image 20250123083930.png|500]]

- PC contains 300, the address of the first instruction
- The instruction (value of 1940 in hex) is loaded into the IR and PC is incremented
- First 4 bits in IR indicate that the AC is to be loaded from memory, the remaining 12 specify the address, which is 940
- The next instruction (5941) is fetched from location 301 and the PC is incremented
- The old contents of the AC and the contents of location 941 are added, and the results stored in the AC
- The next instruction (2941) is fetched from location 203, and PC is incremented
- The contents of the AC are stored in location 941

- Most modern processors include instructions that contain more than one address
- And execution stage of an instruction may involve more than one reference to memory

1. <font style="color:#A8A8A8; text-decoration: underline;">Interrupts</font>  
- All computes provide mechanisms by which other modules (I/O, memory) may interrupt the normal sequencing of the processor
Classes of Interrupts
- Program
	- Generated by some condition that occurs as a result of an instruction executions
		- Arithmetic overflow, division by zero, execute illegal machine instruction, or reference outside a user's allowed memory space
	- Timer
		- Generated by a timer within the processor
		- Allow the operating system to perform functions on a regular basis
	- I/O
		- Generated by an I/O controller, to signal normal completion of an operation
	- Hardware failure
		- Generated by a failure
			- Power failure or memory parity error
- Interrupts are a way to improve processor utilization
- A typical hard disk has rotational speed of 7200 revolutions per minute for a half-track rotation time of 4 ms, which is 4 M times slower than the processor

I/O consists of 3 Sections:
1) A sequence of instructions to prepare for the actual I/O operation
		- Copying the data to be output into a special buffer and preparing the parameters for a device command
2) I/O command
		- Program must wait for the I/O device to perform the requested function, with no interrupts
3) A sequence of instructions to complete the operation
		- Flag indicating the success of failure of the operation

![[Pasted image 20250123085251.png|700]]

- Solid lines = program execution
- Dashed lines = path of execution followed by the processor

Interrupts and the Instruction Cycle

- With interrupts, the processor can be engages in executing other instructions while the I/O operation is in process
- The interrupts allow I/O operation that is concurrent with the execution of instructions in the user program
- When the external device becomes ready to be services the I/O module for that external device sends an _interrupt request_ signal to the processor
- The processor responds by suspending operation of the current program; branching off to a routine to service that I/O device (interrupt handler)
- Resumes the original execution after the device has been services
- The processor and the OS are responsible for suspending the user program, and resuming at the same point
- An _interrupt stage_ is added to the instruction cycle

![[Pasted image 20250123090144.png|700]]

- The _interrupt-handler_ routine is part of the OS, a determines the interrupt and actions needed
- Extra instructions must be executed in the interrupt handler, therefore some overhead

9 Steps In Interrupt Processing
1) The device issues an interrupt signal to the processor
2) The processor finished execution of the current instruction before responding to the interrupt
3) The processor test for a pending interrupt request, sends a signal to the device that issues the interrupt
	1) Removes the interrupt signal is it already sent one
4) Processor transfer control to the interrupt routine
	1) Save information needed to resume the current program at point of interrupt
	2) The minimum information is the program status word (PSW) and location of the next instruction in the PC; this is pushed onto a control stack
5) Processor loads the program counter with the entry location of the interrupt0handling routine
	1) Processor must determine which interrupt to invoke if there are multiple
6) Program counter and PSW relating to the interrupted program are saved on the control stack
	1) The contents of the register plus the address of the next instruction (N + 1), of total M words
	2) The stack pointer is updated to point to the new top of stack, and program counter is updated to point to the beginning of interrupt service routing
7) Interrupt handler may now proceed to process the interrupt
	1) Examination of status information relating to the I/O operation
	2) Sending addition commands or acknowledgments to the I/O device
8) Then interrupt processing is complete, the saved register values are retrieved from the stack and restored to the registers
9) The final act is to restore the PSW and program counter values form the stack

![[Pasted image 20250123091512.png|600]]

![[Pasted image 20250123091546.png|600]]


Multiple Interrupts
- Two approaches:
	- Disable interrupt (sequential or nested)
		- Process ignores any new interrupt request signal
		- Remains pending and will be checks by the processor after the processor has reenabled interrupts
		- Interrupts are handled in strict sequential order
		- Does not take into account relative priority or time-critical needs
	- Time sequence
		- Define priorities for interrupts
		- Interrupts with higher priority requests are honours, with lower priority being held
![[Pasted image 20250123092313.png|600]]
![[Pasted image 20250123092255.png|600]]

11. <font style="color:#A8A8A8; text-decoration: underline;">The Memory Hierarchy</font>  
- Computer memory
	- Cost
	- Speed
	- Amount

![[Pasted image 20250123093508.png|600]]

- Memory must be able to keep up with the processors' speed of execution
- Cost of memory must be reasonable to the relationship to other components
	- Faster access time, greater cost per bit
	- Greater capacity, smaller cost per bit
	- Greater capacity, slower access speed
- Memory heiarchy
	- Decreasing cost per bit
	- Increasing capacity
	- Increasing access time
	- Decreasing frequency of access to the memory by the processor
- Key to success of organization is the decreasing frequency of access at lower levels
- Hit ratio (H), where H is defined as the fraction of all memory accesses that are found in the faster memory (cache)
- $T_1$ is the access time to level 1, $T_2$ is the access time to level 2
- 95% if the memory address are found in the cache (H = 0.95)
- Average time to access a byte:
(0.95)()

$(0.95)(0.1 µs) + (0.05)(0.1 µs + 1 µs) = 0.095 + 0.055 = 0.15 µs$

- The result is close to the access time of the faster memory
- The basis of validity is the principle, _locality of reference_
- Memory references by the processor, for instructions and data, tend to cluster
- Current clusters can be temporarily places in level 1, for

nces will be to instructions and data contained in level 1
- The fastest, smallest, and most expensive type of memory consists of the registers internal to the processor
	- Inboard memory (volatile)
		- A processor will contain a few dozen registers (cache)
		- Main memory is the principal internal memory system (RAM/NVDIMM)
	- Outboard storage/secondary memory (Non-volatile)
		- External memory (HDD, SSD, flash drives, optical discs, memory cards, network, cloud)
		- Hard disk cal provide an extension to main memory known as virtual memory

1. <font style="color:#A8A8A8; text-decoration: underline;">Cache Memory</font>  
- Cache memory is invisible to the OS
- Interacts with other memory management hardware
- Virtual memory schemes are also applied to cache memory
- On all instruction cycles, the processor access memory at least once, to fetch the instruction or operands and store results
- Rate of the processor is limited by the memory cycle (Von Neumann Bottleneck)
- Face a trade-off among speed, cost and size
- Main memory should be built with the same technology at the processor registers, but too expensive
- The solution is to exploit the principle of locality by providing a small, fast memory between the processor and main memory, in cache 

Cache Principle
- Cache memory is intended to provide memory access time approaching that of the fastest memories available

![[Pasted image 20250123094844.png|600]]

- Main memory consists of up to $2^n$ addressable words, with each word having a unique n-bit address
- This memory is considered to consist o a number of fixed-length blocks of _K_ words each
- M = $2^n/K$
- Cache consists of C slots (also referred to as lines) of K words each, and the number of slots is considerably less than the number of main memory blocks (C << M)
- Some subset of the blocks of main memory resides in the slots of the cache
- If a word in a block of memory that is not in cache is read, that block is transferred to one of the slots of the cache
- There are more blocks than slots, therefore each slot includes a tag that identifies which particular block is currently being stored
- The processor generates the address, RA, of a word to read
	- If in caches, it is delivered to the processor
	- Otherwise, the block containing that word is loaded into the cache, and then delivered to processor

![[Screenshot 2025-01-23 at 10.08.37 AM.png|500]]


- Key Elements of Virtual Memory and Disk Cache Design
	- Cache size
	- Block size
	- Mapping function
	- Replacement algorithm
	- Write policy
	- Number of cache levels
- Block size: the unit of data exchanged between cache and main memory
- As the block size increases, more useful data are brought into the cache with each block transfer
- Hit ratio increases because of the principle of locality
- The hit ratio will begin to decrease as the block becomes larger
- The mapping function determines which cache location the block will occupy
- 2 constraints of the mapping function
	- Minimize replacing blocks needed in the future
	- More complex the circuity requires to search the cache to determine if block is in the cache
- The replacement algorithm chooses (within constraints of the mapping function) which block to replace when a new block is to be loaded into the cache
- Replace the block that is less likely to be needed in the future
- Reasonably effective strategy is to replace the block that has been in the cache longest with no reference; least-recently used (LRU)
- The write policy dictates when the memory write operation takes place
- Multiple levels of cache is commonplace to maximize efficiency


1. <font style="color:#A8A8A8; text-decoration: underline;">Direct Memory Access</font>  
- Techniques for I/O operations:
	- Programmed I/O
		- I/O module performs the requested actions and sets bits in I/O status register, but takes no further action
		- It does not interrupt the processor
		- Processor has to wait for the I/O module to be ready or reception/transmission of more data
		- The repeated check to the I/O module degrades the performance level of the CPU
	- Interrupt-driven I/O
		- Processor issues an I/O command to a module then does useful work
		- I/O module will interrupt to request service when it is ready to exchange data
		- Processor then executes the data transfer, and resumes former processing
		- More efficient than simple I/O programming
		- Still requires the active intervention of the processor
	- Direct memory access (DMA)
		- Performed by a separate module on the system but, or incorporated into an I/O module
		- When the processor reads/writes to block of data, it issues a command to the DMA module
			- Whether a read or write is requested
			- The address of the I/O device involved
			- The starting location in memory to read data from or write data to
			- The number of words to be read or written
		- Delegates this I/O operation to the DMA module
		- DMA module sends an interrupt signal to processor
		- Only interacts with processor at the beginning or end of the transfer
		- DMA uses bus, and may block processor
		- DMA is more efficient for large volumes of data when moved

- Drawbacks of programmed and interrupt-driven I/O:
	- I/O transfer rate is limited by the speed of the processor's test and service to device
	- The processor is tied up in managing an I/O transfer

1. <font style="color:#A8A8A8; text-decoration: underline;">Multiprocessor and Multicore Organization</font>  

- Traditionally, computers have been viewed as sequential machines
- Each instruction is executed in a sequence of operations
- At the micro-operation level, multiple control signals are generated at the same time
- Instruction pipelining, executes in parallel
- Parallelism by replicating processors
	- Symmetric multiprocessors (SMPs)
	- Multicore computers
	- Clusters

**Symmetric Multiprocessors**
- An SMP can be defined as a stand-alone computer system with the following characteristics:
	- Two or more similar processors of comparable capability
	- Processors share the same main memory and I/O facilities and are interconnected by a bus or other internal connection scheme, such that memory access time is approximately the same for each processor
	- All processors share access to I/O devices, either through the same channels or through different channels that provide paths to the same device
	- All processors can perform the same funtions
	- The system is controlled by an integrated operating system that provides interaction between processors and their programs at the job, task, file, and data element levels

![[Screenshot 2025-01-23 at 4.27.04 PM.png|600]]

- Advantages of SMP organization:
	- Performance
		- Parallel work; yields greater performance
	- Availability
		- Processors can perform same functions, failure of a single processor does not halt machine
	- Incremental growth
		- A user can enhance the performance of a system by adding an additional processor
	- Scaling
		- Vendors can offer a range of products with different system configurations
- The operating system takes care of scheduling of tasks on individual processors, and synchronization among processors
- Each processor and its dedicated caches are housed on a separate chip
- Each processor has access to a shared main memory and have a interconnected I/O devices
- Processors can communicate through memory (messages and statuses)
- Multiple simultaneous accesses to separate blocks of memory
- Local cache contains an image of a portion of main memory, do all processors must be alerted when an update occurs (cache coherence)
- Experiences a steady/exponential performance increase
	- Increased clock frequency
	- Increased cache locality/miniaturization

**Multicore Computers**
- A multicore computer, aka a chip multiprocessor, combines two or more processors (called cores) on a single piece of silicon (called a die)
- Each core consists of all the components on an independent processor, such as registers, ALU, pipeline hardware, and control unit, plus L1 instruction and data caches
- Some newer multicore chips include L2 cache and L3 cache
- Intel Core i7-5960, includes six x86 processors, each with a dedicated L2 cache. and a shared L3 cache
	- Intel uses _prefetching_, fills the cache speculatively with data that is likely to be requested
	- Supports two forms of external communication to other chips
	- DDR4 memory controller brings the memory controller for the DDR (double data rate) main memory onto the chip
	- Supports 4 channels of up to 64 GB/s
	- PCI express is a peripheral bus that enables high-speed communication between processor chips
	- PCI express link operates at 8 GT/s (transfer per second)
	- At 40 bits per transfer; 40 GB/s

![[Pasted image 20250208100905.png|500]]

<font style="color:#AA0F02; text-decoration: underline;">Key Terms</font>  

| address register <br>auxiliary memory<br>block<br>cache memory<br>cache slot<br>central processing unit<br>chip multiprocessor<br>data register<br>direct memory access (DMA)<br>hit<br>hit ratio<br>input/output<br>instruction | instruction cycle<br>instruction register<br>interrupt<br>interrupt-driven I/O<br>I/O module<br>locality of reference<br>main memory<br>memory hierarchy<br>miss<br>multicore<br>multiprocessor<br>processor<br>program counter | programmed I/O<br>register<br>replacement algorithm<br>secondary memory<br>slot<br>spatial locality<br>stack<br>stack frame<br>stack pointer<br>system but<br>temporal locality |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

Spatial locality refers to the tendency of execution to involve a number of memory locations that are clustered

Temporal locality refers to the tendency for a processor to access memory location that have been used recently


**Operation of Two-Level Memory (Calculations)**





## 2. <font style="color:#FCD12A">Operating System Overview</font>  

^8de730

2.1 <font style="color:#A8A8A8; text-decoration: underline;">Operating System Objectives and Functions</font>  
- An OS is a program that controls the execution of application programs, and acts as an interface between applications and the computer hardware

3 Objectives of OS:
1) Convenience
	1) An OS makes a computer more convenient to use
2) Efficiency: An OS allow the computer system resources to be used in an efficient manner
3) Ability to evolve:
	1) An OS should be constructed in such a way as to permit the effective development, testing, and introduction of new system functions without interfering with service

The Operating System as a User/Computer Interface
- The user interacts with applications
- The system program is made up of utilities, and library programs
- System programs help in program creation, the management of files, and the control of I/O devices
- OS masks the details of the hardware from the programmer, and provides a convenient interface for using the system

OS Services:
- Program development
	- Editor, debugger, to assist creating programs
- Program execution
	- Instructions and data must be loaded into main memory, I/O devices and files initialized, and other resources prepared
- Access to I/O devices
	- Each I/O devices has its set of instructions or control signals for operation
- Controlled access to files
	- Account for multiple file access and protection mechanisms to control access
- System access:
	- For shared or public systems, the OS controls access to the system to specify system resources
	- Provide protection of resources and data form unauthorized users, and resolve resource contention
- Error detection and response
	- Internal and external hardware errors (memory error, or a device failure)
	- Software errors (stack overflow, division by zero)
	- Provide a clear response for each error condition
- Accounting
	- Collect usage statistics for resources and monitor performance parameters (e.g. response time)
- Instruction set architecture (ISA)
	- Defines the machine language instructions the computer can follow
	- Interface is the boundary between hardware and software
	- Both application programs and utilizes may access the ISA directly
	- OS has access to additional machines language instruction that deals with managing system resources
- Application binary interface (ABI)
	- Defines a standard for binary portability across programs
	- Defines the system call interface to the operation system, and the hardware resources and services available in a system through the user ISA
- Application programming interface (API)
	- Gives a program access to the hardware resources and services available in a system through the user ISA with high-level language (HLL) library calls

![[Pasted image 20250127082909.png|600]]


The Operating System as Resource Manager
- Two parts of OS
	- Program is executed by the processor
	- OS depends on the processor to give/regain control
- While executing the OS decides how processor time is to be allocated and which computer resources are available for use
- A portion of the OS is in main memory:
	- Kernel/nucleus
		- Contains most frequently used functions in the OS
	- User and utility programs
	- Data

![[Pasted image 20250127083754.png|500]]

Evolution of an Operating System
- Reason for OS to Evolve:
	- Hardware upgrades plus new types of hardware
		- Paging in UNIX and Macintosh OS
	- New services
		- New measurements and control tools are added for better performance
	- Fixes


- OS system should be modular in construction, with clearly defined interfaces between the modules
- OS needs to be thoroughly documents


2.2. <font style="color:#A8A8A8; text-decoration: underline;">The Evolution of Operating Systems</font>  

Serial Processing
- Programmers interacted directly with the computer hardware (1940s to mid 1950s)
- Computers ran from a console consisting of display lights, toggle switches, and input devices
- Programs in machine code were loaded via the input device
- Two main problems
	- Scheduling
		- User programs had to wait in a queue to process
	- Setup time
		- A single problem (job) involved loading the compiler plus the high-level language program (source program) into memory, saving the compiled program (object program), then loading and linking together the object program and common function

Simple Batch Systems
- Minimize wasted time by increasing processor utilization
- The first batch OS was developed in the mid-1950s by General Motors for use on the IBM 701 [WEIZ81]
- The user submits the job on cards or tape to a computer operator, who batches the jobs together sequentially for the monitor
- Each program was constructed to brach back to the monitor when it completes processing, and start the next program
- Monitor point of view
	- Controls the sequence of events
	- Much of the monitor is always in main memory and available for execution (resident monitor)
	- The rest of the monitor consists of utilities and common functions that are loaded as subroutines to the user program at the beginning of any job that requires them
	- Current job is places in the user program area, and control is passed to this job
- Processor point of view
	- Once a job has been read in, the processor will encounter a branch instruction in the monitor that instructs the processor to continue execution at the start of the user program

![[Pasted image 20250127085612.png|500]]

- Job control language (JCL) is a programming language used to provide instructions to the monitor

```FORTAN
$JOB
-- FORTRAN instructions --
$FTN
.
.
.
$LOAD
-- Data --
$RUN
.
.
.
$END
```

- `$FTN` loads the language compiler from its mass storage (usually tape)
- The compiler translates the user's program into object code, and is stored in memory
- If stored on tape the `$LOAD` instruction loads on tape
- After being loaded the monitor transfers control to it, and runs
- A large segment of main memory can be shares, but only one subsystem can be executing
- Any input instruction causes one line of data to be read
- Other hardware features
	- Memory protection
		- Memory area containing the monitor cannot be altered during program execution
	- Timer
		- A timer is used to prevent a single job from monopolizing the system
		- Control returns to the monitor when timer expires
	- Privileged instructions
		- Certain machine level instructions are executed only by the monitor
		- When encountered the processor returns to the monitor to execute
	- Interrupts
		- Gives OS more flexibility in relinquishing control to, and regaining control fro, user programs
- A user program execute in a user mode, where certain areas of memory are protected from the user's use
- The monitor executes in a system mode, or kernel mode
- With a batch OS, processor time alternates between execution of user programs and execution of the monitor 
- Overhead: main memory is given to monitors and processor time is consumed by the monitor

Multiprogrammed Batch Systems
- I/O devices are slow compared to the processor

![[Pasted image 20250127091524.png|500]]
![[Pasted image 20250127091544.png|500]]
![[Pasted image 20250127091601.png|500]]

- There us enough memory to hold the OS (resident monitor) and two user programs
- When one job is waiting for I/O the processor can switch to the other job
- This approach is known as _multiprogramming/multitasking_

![[Pasted image 20250127091927.png|600]]
![[Pasted image 20250127091941.png|600]]

- Running concurrently, all three jobs can run in a minimum time on the same processor
- For useful multiprogramming, hardware must support I/O interrupts and DMA (direct memory access)
- To have several jobs ready to run, they must be kept in main memory, requiring some form of memory management
- An algorithm for scheduling is used to decide how the jobs are sequenced
- Today most computers requires an interactive computing facility
- Multiprogramming can also be used to handle multiple interactive jobs (time sharing)
- Time-sharing shares the resources of a computer system among multiple users or tasks simultaneously
	- OS interleaving the execution of each user program in a short burst of quantum of computation
- There are _n_ users actively requesting service, each user till only see $1/n$ of the effective computer capacity, not counting OS overhead
- CTSS
	- One of the first time-sharing operation systems was the Compatible Time-Sharing System (CTSS) [CORB62], developed at MIT by Project MAC
	- Developed for the IBM 709 in 1961 and later used in the IBM 7094
	- Monitor consumed 5,000 words of main memory
	- Program was loaded at the start of the 5000th word
	- System clock generated interrupts at a rate of every 0.2 seconds
	- Each clock interrupt, the OS regained control (time slicing)
	- To preserve the old user program status, the old user program and data were written out to disk before the new user programs and data were read in

![[Pasted image 20250127093450.png|500]]
![[Pasted image 20250127093502.png|600]]

- To minimize disk traffic, user memory was only written out when the incoming program would overwrite it
- Minimized the size of the monitor, since job is always loaded into the same location in memory

2.3. <font style="color:#A8A8A8; text-decoration: underline;">Major Achievements</font>  

4 Theoretical Advances In OS Development:
1) Processes
2) Memory management
3) Information protection and security
4) Scheduling and resource management

The Process
- Central to the design of operating systems is the concept of the process
	- A program in execution
	- An instance of a program running on a computer
	- The entity that can be assigned to and executed on a processor
	- A unit of activity characterized by a single sequential thread of execution, a current state, and an associated set of system resources

3 Major Lines of Computer System Development
1) Multiprogramming batch operation
	1) Keep the processor and I/O devices simultaneously running to achieve maximum efficiency
2) Time-sharing
	1) Multiple users or tasks share the computing resource of a single system/cpu
3) Real-time transaction systems
	1) Process transactions immediately as they occur, ensuring near-instant responses and updates

4 Main Causes of System Errors:
1) Improper synchronization
	1) Improper design of the signalling mechanism can result in signals being lost or duplicated signals being received
2) Failed mutual exclusion
	1) Two users will attempt to edit the same file
3) Nondeterminate program operation
	1) Overwriting common memory areas in unpredictable ways
4) Deadlocks
	1) Two of more programs to be hung up waiting for each other
	2) Dependent on the chance timing of resource allocation and release

3 Components of the Process
1) An executable program
2) The associated data needed by the program (variables, work space, buffers, etc.)
3) The execution context of the program

- The execution context/process state is the internal data that the OS is able to supervise and control
- The internal information is separated from the process, because the OS has information the OS needs to manage the process, and the processor needs to execute the process properly
- The context includes the processor registers (program counter and data registers), and information of use to the OS (priority access)

![[Pasted image 20250127100452.png|500]]

- The process index register contains the index into the process list of the process currently in control
- The program counter points to the next instruction in the process to be executed
- The base and limit registers define the region in memory occupied by the process
- The base register is the starting address of the region of memory
- Limit is the size of the region
- The process is realized as a data structure
- A process can either be executing or awaiting execution
- The entire state of the process at any instant is contained in its context
- This structure allows the development of techniques for ensuring coordination and cooperation among processes
- The process structure is used to solve problems by multiprogramming and resource sharing
- A single process, which is assigned certain resources, can be broken up into multiple, concurrent threads that execute cooperatively to perform the work of the process

Memory Management

5 Principle Storage Management Responsibilities:
1) Process isolation
	1) Prevent independent process from interfering with each other's memory, both data and instructions
2) Automatic allocation and management
	1) Dynamically allocated across the memory hierarchy
	2) Allocation is transparent to the programmer
3) Support of modular programming
	1) Programmer should be able to define program modules, and to dynamically create, destroy, and alter the size of modules
4) Protection and access control
	1) Sharing of memory
	2) The OS must allow portions of memory to be accessible in various ways by various users
5) Long-term storage
	1) Application programs require means for storing information for extended periods of time, after being shut off

- Operating systems meet these requirements with virtual memory and file system facilities
- Information is stored in named objects called files
- The files is a convenient concept for the programmer, and is a useful unit of access control and protection for the OS
- _Virtual memory_ is a facility that allows programs to address memory from a logical point of view, without regard to the amount of main memory physically available
- Paging systems allow processes to be comprised of a number of fixed-sized blocks, called pages
- A program references a word by means of virtual address consisting of a page number and an offset within the page
- Each page of a process can be located anywhere in main memory
- The paging system provides a dynamic mapping between the virtual address used in the program and a real address, or physical address, in main memory
- All the pages of a process are maintained on disk
- Some of its pages are in main memory when executed
- Reference is made to a page that is not in main memory, the memory management hardware detects this and, in coordination with OS, arranges for the missing page to be loaded (virtual memory)

![[Pasted image 20250127110859.png|500]]
![[Pasted image 20250127110912.png | 500]]

- Process isolation can be achieved by giving each process a unique, non-overlapping virtual memory
- Memory sharing is achieved by overlapping portions of two virtual memory spaces
- Files are maintained in a long-term store
- Files and portions of files may be copies into the virtual memory for manipulation
- The OS designer needs to develop:
	- an address translation mechanism that generates little overhead
	- a storage policy that minimizes the traffic between memory levels


Information Protection and Security
- Controlling access to computer systems and the information stored in them
	- Availability
		- Concerned with protecting the system against interruption
	- Confidentiality
		- Assures that uses cannot read data for which access is unauthorized
	- Data integrity
		- Protection of data from unauthorized modification
	- Authenticity
		- Concerned with the proper verification of the identity of users and the validity of messages or data

Scheduling and Resource Management
1) Fairness
	1) All processes that are competing for the use of a particular resource to be given equal and fair access to that resource
2) Differential responsiveness
	1) OS may need to discriminate among different classes of jobs with different service requirements
3) Efficiency
	1) Maximize throughput, minimize response time, and, in the case of time sharing, accommodate as many users as possible

![[Pasted image 20250127111702.png|600]]

- Scheduling and resource management are essentially operations-research problems and the mathematical results of that discipline
- Measurement of system activity is important to be able to monitor performance and make adjustments
- The short-term/dispatcher chooses with processor to process next
- A common strategy is to give each process in the queue some time in turn; _round-robin_ technique
- Round-robin technique uses a circular queue
- Another strategy is to assign priority levels to the various processes, when the scheduler selecting processes in priority order
- The long-term queue is a list of new jobs waiting to use the processor
- The OS adds jobs to the system by transferring a process from the long-term queue to the short-term queue
- A portion of main memory must be allocated from incoming processes
- Research and development in operating systems has been directed at picking algorithms and data structures for this function that provides fairness, differential responsiveness, and efficiency
   
2.4. <font style="color:#A8A8A8; text-decoration: underline;">Developments Leading to Modern Operating Systems</font>  
- Modern operating systems respond to new developments in hardware, new applications, and new security threats
- Multiprocessor systems increase in speed, new high-speed network attachments, and increasing size and variety of memory storage devices
- Multimedia applications, Internet and Web access, and client/server computing have influenced OS design
- Internet access has increased potential threats
	- Viruses, worms, and hacking techniques

Features of Operating Systems:
- Microkernel architecture
- Multithreading
- Symmetric multiprocessing
- Distributed operating systems
- Object-oriented design

- Most operating systems featured a large monolithic kernel
- Kernels provide scheduling, file system, networking, device drivers, memory management, etc.
- Monolithic is in a single process, with all elements sharing the same address space
- A microkernel architecture assigns only a few essential functions to the kernel
	- Address space management
	- Interprocess communication (IPC)
	- Basic scheduling
- Other OS services are provided by processes/servers that run in user mode and treated like application by the microkernel
- This approach decouples kernel and server development
- Microkernel approach simplifies implementation, provides flexibility, and is well suited to distributed environment
- Microkernel interacts with local and remove server processes in the same way, facilitating construction of distributed systems

- _Multithreading_ is a technique in which a process, executing an application, is divided into threads that can run concurrently
	- _Thread:_ A dispatchable unit of work
		- Processor context (program counter and stack point)
		- Data area for a stack (to enable subroutine branching)
		- Executes sequentially and is interruptible, for processor to turn to another thread
	- _Process:_ A collection of one or more threads and associated system resources
		- Memory
			- Code, data, open files, and devices
- By breaking a single application into multiple threads it is more modular so the programmer has more control over the timing of application-related events

- Multithreading is useful for applications that perform independent tasks and do not need to be serialized
	- Database server that listens for and processes numerous client requests
- Multiple threads switching back and forth involves less processor overhead than a major process
- Useful for structuring processes that are part of the OS kernel

- _Symmetric multiprocessing (SMP)_
	- A term that refers to a computer hardware architecture and also to the OS behaviour that exploits that architecture
	- Potential advantages over uniprocessor architecture
		- Performance
			- More than one process can be running simultaneously, on different processors
		- Availability
			- All processors can perform the same functions
			- Failure of a single processor does not halt the system
		- Incremental growth
			- User can enhance the performance of a system by adding additional processor
		- Scaling
			- Vendors can offer a range of products with different price and performance characteristics
- Multithreading and SMP are independent facilities
- Multithreading is useful on uniprocessor systems
	- Structuring applications and kernel processes
- A SMP system is useful for non-threaded processes
	- Parallel running processes
- SMP has transparent multi processors to the user

![[Pasted image 20250128123015.png|600]]


- A _distributed operating system_ provides the illusion of a single main memory space and a single secondary memory space, plus file systems
- Object-oriented design adds modular extensions to a small kernel
	- Allows programmers to customize an OS without disrupting system integrity

   
2.5. <font style="color:#A8A8A8; text-decoration: underline;">Fault Tolerance</font>  
- Fault tolerance refers to the ability of a system or component to continue normal operation despite the presence of hardware or software faults
- Involved some degree of redundancy
- Intended to increase the reliability of a system
- Increased fault tolerance, increases cost (financial, performance, or both)

**Fundamental Concepts**
- 3 basic measures of quality
	- Reliability $R(t)$
		- The probability of its correct operation up to time $t$ given that the system was operating correctly at time $t=0$
	- Mean time to failure (MTTF)
		- $MTTF=$ $\int_0^\infty R(t)$
	- Availability
		- The fraction of time the system is available to service users' requests
		- Probability that an entity os operating correctly under given conditions at a given instant of time

- Mean time to repair (MTTR)
	- The average time it takes to repair or replace a faulty element
- Downtime
	- Time in which the system is not available
- Uptime
	- Time during which is system is available

$A=\Large \frac{MTTF}{MTTF + MTTF}$

$MTTF=\Large \frac{B1 + B2 + B3}{3}$

$A=\Large \frac{A1 = A2 + A3}{3}$

- Often the mean uptime is a better indicator than availability

![[Pasted image 20250128124347.png|500]]
![[Pasted image 20250128124405.png|500]]

**Faults**
- A fault is an erroneous hardware or software state resulting from:
	- component failure, operator error, physical interference from the environment, design error, program error, or data structure error
- 2 ways a fault occurs
	- A defect in a hardware device or component
		- Short circuit or broken wire
	- An incorrect step, process, or data definition in a computer program

Fault Categories
- Permanent
	- A fault that is always present
	- Fault persists until the faulty component is replaced of repaired
		- Disk head crashes, software bugs, and a burn-out communications component
- Temporary
	- A Fault that is not present all the time for all operating conditions
		- Transient
			- A fault that occurs only once
				- Bit transmission errors sue to an impulse noise, power supply disturbances, and radiation that alters a memory bit
		- Intermittent
			- A fault that occurs at multiple, unpredictable times
				- Loose connection

Methods of Redundancy:
- Spatial (physical redundancy)
	- Physical redundancy involves the use of multiple components that either perform the same function simultaneously (parallel circuitry), or are a backup component is available (backup server)
- Temporal redundancy
	- Repeating a function or operation when an error is detected
	- Effective with temporary faults, but not permanent
		- Retransmission of a block of data when an error is detected
			- Data link control protocols
- Information redundancy
	- Provides fault tolerance by replicating or coding data so that bit errors can be detected and corrected
		- Error-control coding circuitry using memory systems
		- Error-correction techniques using RAID disks


**Operating System Mechanisms**
- Process isolation
	- Isolated structure provides a certain level of protection for other processes from a process that produces a fault
- Concurrency controls
	- Faults can occur when processes communicate or cooperate (deadlock)
- Virtual machines
	- Provide greater degree of application isolation and hence fault isolation
	- Provides redundancy, with one virtual machine serving as a backup for another
- Checkpoints and rollbacks
	- A checkpoint is a copy of an application's state saved in storage that is immune to failures
	- A rollback restarts the execution from a previously saves checkpoint
	- Used to recover from transient as well as permanent hardware failures, and certain types of software failures

2.6. <font style="color:#A8A8A8; text-decoration: underline;">OS Design Considerations for Multiprocessor and Multicore</font>  

**Symmetric Multiprocessor OS Considerations**
- In an SMP system, the kernel can execute on any processor, which each processor self-scheduling from the pool of available processes of threads
- Kernel constructed as multiple processes or multiple threads
	- Kernel executes in parallel
- A multiprocessor OS must provide all the functionality of a multiprogramming system, plus features to accommodate multiple processors

Key Design Issues
- Simultaneous concurrent processes of threads
	- Kernel tables and management structures must be managed properly to avoid data corruption or invalid operations
-  
	- Kernel-level multithreading must schedule multiple threads from the same process simultaneously on multiple processors
- Synchronization
	- Effective shared address spaces or shared I/O resource
	- Enforces mutual exclusion and event ordering
		- Locks
- Memory management
	- OS exploits the available hardware parallelism to achieve increased performance
	- Paging mechanisms on different processors are coordinates
		- Page sharing/replacement
	- Physical page can no longer be accessed with its old contents before the page is put to a new use
- Reliability and fault tolerance
	- Provides graceful degradation in the face of processor failure
	- Restructure management tables

**Multicore OS Considerations**
- 3 Potential for parallelism in contemporary multicore systems
	- Hardware/Instruction level parallelism
		- May or may not be exploited by application programmers and compilers
	- Multiprogramming and multithreaded execution
	- Single application to execute in concurrent processes or threads across multiple cores

_PARALLELISM WITH APPLICATION_
- Developer must decide how to split an application into independently executable tasks
- Compiler and programming language features support parallel programming design
- OS can support this design process, at minimum, by efficiently allocation resources among parallel tasks
- Grand Central Dispatch (GCD), implemented in UNIX-based OS X and iOS operating system, use to split off expensive tasks
- GCD is a thread pool mechanism, in which the OS maps tasks onto threads representing an available degree of concurrency
- Thread pool mechanisms have been used in Windows for server applications
- Anonymous functions (blocks) used to  specify tasks

_VIRTUAL MACHINE APPROACH_
- Dedicate particular processes to one or more core to avoid overhead of task switching and scheduling decisions
- Multiprogramming is based on the concept of a process, which is an abstraction of an execution environment
- The OS requires protected space, free from user and a program interference
- Kernel mode and user mode was developed
- Kernel mode an user model abstracted the processor into two processors
- The overhead of switching processors starts to grow to the point where responsiveness suffer (especially when the number of multicores increases)


2.7. <font style="color:#A8A8A8; text-decoration: underline;">Microsoft Windows Overview</font>  

**Background**
- Microsoft originally used Windows (1985) for an operating environment extension to MS-DOS operating system
- Windows/MS-DOS was replaced by Windows NT, where the OS has evolved to todays Windows 10

**Architecture**
- Windows separates application-oriented software from the core OS software
	- Executive, the Kernel, device drivers, and the hardware abstraction layer, runs in kernel mode
	- Kernel-mode software has access to system data and to the hardware
	- Remaining software, in user mode, has limited access to system data


_OPERATING SYSTEM OPERATION_
- Windows has a module architecture
- Each system is managed by one component of the OS
- Rest of the OS and all application share information using standard interfaces
- Any module can be removed, upgrades, or replaced without rewriting the entire system or its standard application program interfaces (APIs)

Window Components
1) Executive
	1) Contains the core OS services
		1) Memory management
		2) Process and thread management
		3) Security
		4) I/O
		5) Interprocess communication
2) Kernel
	1) Controls execution of the processors
		1) Manages thread scheduling
		2) Process switching
		3) Exception and interrupt handing
		4) Multiprocessor synchronization
	*Kernel's own code does not run in threads
3) Hardware abstraction layer (HAL)
	1) Maps between generic hardware commands and responses (unique to specific platform)
	2) Isolates the OS from platform-specific hardware differences
	3) Makes components look the same to Executive and kernel components
		1) System bus
		2) Direct memory access (DMA) controller
		3) Interrupt controller
		4) System timers
		5) Memory controller
4) Device drivers
	1) Dynamic libraries that extend the functionality of the Executive
		1) Hardware device drivers
			1) Translates user I/O calls into hardware device I/O requests
		2) Software components for file systems
		3) Network protocols
		4) Other systems extensions that run in kernel mode
5) Window and graphic system
	1) Implements the GUI functions
		1) Windows
		2) user interface controls
		3) Drawing

![[Pasted image 20250128135440.png|600]]

Executive Modules
1) I/O manager
	1) Provides a framework through which I/O devices are accessible to applications, and is responsible for dispatching to the appropriate device drivers for further processing
	2) Implements all the Windows I/O APIs and enforces security and naming for devices, network protocols, and file systems
2) Cache manager
	1) Improves the performance of file-based I/O by causing recently referenced file data to reside in main memory for quick access
	2) Deferring disk writes by holding the updates in memory for a short time before sending them to the disk in more efficient batches
3) Object manager
	1) Creates, manages, and deletes Windows Executive objects that are used to represent resources such as processes, threads, and synchronization objects
	2) Enforces uniform rules for retaining, naming, and setting the security of objects
	3) Creates the entries in each process's handle table, which consist of access control information and a pointer to the object
4) Plug-and-play manager
	1) Determines which drivers are required to support a particular device and loads those drivers
5) Power manager
	1) Coordinates power management among various devices
	2) Reduce power consumption by shutting down idle devices, putting the processor to sleep, and even writing all of memory to disk and shutting off power to the entire system
6) Security reference monitor
	1) Enforces access-validation and audit-generation rules
	2) Consistent view of security, as lower as the fundamental entities that make up the Executive
	3) Windows uses the same routines for access validation and for audit checks for all protected objects, including files, processes, address spaces, and I/O devices
7) Virtual memory manager
	1) Manages virtual addresses, physical memory, and the paging files on disk
	2) Controls the memory management hardware and data structures which map virtual addresses in the process's address space to physical pages in the computer's memory
8) Process/thread manager
	1) Creates, manages, and deletes process and thread objects
9) Configuration manager
	1) Implementing and managing the system registry, which is the repository for both system-wide and per-user settings of various parameters
10) Advanced local procedure call (ALPC) facility
	1) Implements an efficient cross-process procedure call mechanism for communication between local processes implementing services and subsystems
	2) Similar to the remote procedure call (RPC) facility used for distributed processing

_USER-MODE PROCESSES_
1) Special system processes
	1) User-mode services that manage the system
		1) Session manager
		2) Authentication subsystem
		3) Service manager
		4) Logon process
2) Service processes
	1) Printer spooler
	2) Even logger
	3) User-mode components that cooperate with device drivers
	4) Network services
	*Used by Microsoft and external software to extend system functionality
3) Environmental subsystems
	1) Provide different OS personalities (environments)
	2) Subsystems: Win32 and POSIX
	3) Each environment subsystem includes a subsystem process shared with applications using the subsystem and dynamic link libraries (DLL) that convert the user application calls to ALPC calls on the subsystem process, and/or native Windows calls
4) User applications
	1) Executables (.exe) and DLLs that provide the functionality users run to make use of the system
	2) Targeted at a specific environment subsystem; some programs use the native system interfaces (NT API)
	3) Support for running 32-bit programs on 64-bit systems

- Windows supports applications written for multiple OS personalities
- Common set of kernel-mode components that underlie the OS environment subsystem
- An environmental subsystem provides a graphical or command-line user interface that defines the look and feel of the OS for a user
- Source code can be written on any Windows machines, only need to be recompiled

**Client/Server Model**
- Windows OS services, the environment subsystem, and the applications are structured using the client/server computing model
- Native NT API is a set of kernel-based service which provide the core abstractions
	- Processes
	- Threads
	- Virtual memory
	- I/O
	- Communication
- Windows provides a set of services using the client/server model to implement functionality in user-mode processes
	- Communicate via RPC (Remove Procedure Call)
- Server wait for a request from a client for one of its services
	- Memory services
	- Process creation services
	- Networking services
- A client, an application or another server program, requests a service by sending a message
- The message is routed through the Executive to the appropriate server
- The server performs the requested operation and returns the results of status information in the form of a message, routed through the Executive back to the client

Advantages of Client/Server Architecture:
- Simplifies the Executive
	- New APIs are added easily without conflicts of duplications
- Improves reliability
	- Server has separate memory from kernel and is protected from other servers
	- A single server can fail without crashing or corrupting the rest of the OS
- Provides a uniform means for applications to communicate with services via RPCs without restricting flexibility
	- Message-passing process is hidden from the client application by function stubs
		- Code that wraps the RPC call
	- When an API call is made, the stub in the client application packages the parameters for the call and sends them as a message to the server process that implements the call
- Provides a suitable base for distributed computer
	- A distributed computer makes use of client/server model, with remove procedure calls
	- With Windows, a local server can pass a message on to a remove server for processing on behalf of local client applications

**Threads and SMP**
- 2 important characteristics of Windows
	- Threads
	- Symmetric multiprocessing (SMP)
- Features of Windows that support threads and SMP
	- OS routines can run on any available processor, and different routines can execute simultaneously on different processors
	- Windows supports the use of multiple thread of execution within a single process
	- Server processes may use multiple threads to process requests from more then one client simultaneously
	- Windows provides mechanisms for sharing data and resources between processes and flexible interprocess communication capabilities

**Window Objects**
- Windows is written in C, a uses object-oriented design
- Facilitates resource sharing and data among processes
- Protection of resources from unauthorized access

Window's Object-Oriented Concepts
- Encapsulation
	- An object consists of one or more items of data (attributes) and one or more procedures that may be performed (services)
	- Data in objects are easily protected from unauthorized and incorrect use
- Object class and instance
	- Template that lists the attributes and services of an object, and defines certain object characteristics
	- OS can create specific instances of an object class as needed
		- Single process object class, and one process object for every currently active process
	- Simplifies object creation and management
- Inheritance
	- Extend object classes by adding new features
	- Every Executive class is based on a base class which specifies virtual methods that support creating, naming, securing, and deleting objects
		- Dispatcher objects are Executive objects the inherit the properties of an even types
			- Device class, inherits from the base class, and add additional data and methods
- Polymorphism
	- Windows uses a common set of API functions to manipulate objects of any type
	- Not entirely polymorphic, many APIs are specific to a single object type

- Objects are used in cases where data are intended for user-mode access, or when data access is shared or restricted
	- Objects: files, processes, threads, semaphores, timers, and graphical windows
- Object manager is responsible for creating and destroying objects on behalf of applications and granting access to services and data
- Each object in the Executive (kernel object) exists as a memory block allocated by the kernel and is only accessible by kernel-mode components
- Applications manipulate object indirectly through the set of object manipulation functions to access otherwise unreachable address space
- A handle returned when an objected is created
- The handle is an index into a per-process Executive table containing a pointer to the referenced object
- The handle can be used by any thread within the same process to invoke Win32 function that work with objects, or can be duplicated into other processes
- Objects use Security Descriptors (SD)  to restrict access to object based on contents of a token object which describes a particular user
	- SD for semaphores object can list users that are allowed/denied access to the semaphore object and access permitted (read, write, change, etc.)
- Objects can be named or unnamed
- The handle is the only reference, when an unnamed objects is created
- Handles can be inherited by child processes or duplicated between
- Names objects are given a name that other processes can use to obtain a handle to the object
	- When A synchronizes with B, B receives a named even object
	- When A synchronizes two threads within itself, it creates an unnamed even object
- 2 Categories of objects use by Windows for synchronizing
	- Dispatcher object
		- The subset of Executive objects which threads can wait on to control the dispatching and synchronization of thread-based system operations
	- Control objects
		- Used by the Kernel component to manage the operation of the processor in areas not managed by normal thread scheduling


| Window Kernel Control Objects | Description                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Asynchronous procedure call   | Used to break into the execution of a specified thread and to cause a procedure to be called in a specified processor mode                                                                                                                                                                                                                                      |
| **Procedure call**<br>        | Used to postpone interrupt processing to avoid delaying hardware interrupts. Also used to implement timers and interprocessor communication                                                                                                                                                                                                                     |
| Interrupt                     | Used to connect an interrupt source to an interrupt service routine by means of an entry in an Interrupt Dispatch Table (IDT). Each processor has an IDT that is used to dispatch interrupts that occur on the processor                                                                                                                                        |
| Process                       | Represents the virtual address space and control information necessary for the execution of a set of thread objects. A process contains a pointer to an address map, a list of ready threads containing thread objects, a list of thread belonging to the process, the total accumulated time for all threads executing within the process, and a base priority |
| Thread                        | Represents thread objects, including scheduling priority and quantum, and which processors the thread may run on                                                                                                                                                                                                                                                |
| Profile                       | Used to measure the distribution of run time within a block of code. Both user and system codes can be profiled                                                                                                                                                                                                                                                 |


2.8. <font style="color:#A8A8A8; text-decoration: underline;">Traditional UNIX Systems</font>  


**History**
- UNIX was initially developed at Bell Labs and become operational on a PDP-7 in 1970
- UNIX was rewritten in C language, from assembly language
- Reasons it was written in assembly
	- Memory (RAM and secondary storage) was small and expensive
	- Computer industry was skeptical of automatically generated code
	- Processor and bus speed were relatively slow, wanted efficient use of clock cycles
- C implementation demonstrated the advantages of a high-level language
- Today, mostly all of UNIX is in C

**Description**
- 3 levels of UNIX architecture:
	- Hardware
	- Kernel
	- User
- UNIX kernel with user service and interfaces are called the shell
- The shell supports system calls from applications, other interface software, and the components of the C compiler (compiler, assembler, loader)
- The level above the kernel is the user application and interface
- User programs can invoke OS services directly, or through library programs
- The system call interface is the boundary with the user and allows high-level software to gain access to specific kernel function
- OS contains primitive routines that interact with the hardware
- Two main parts of the kernel:
	- Process control
		- Memory management
		- Scheduling and dispatching of processes
		- Synchronization and interprocess communication of processes
	- File management and I/O
		- Exchanges data between memory and external devices
		- For block-oriented transfers, a disk cache approach is used
- Traditional UNIX systems are designed to run on a single processor, and lacks the ability to protect its data structure from concurrent access my multiple processors
- Kernel only support a single type of file system, process scheduling policy, and executable file format

   
![[Pasted image 20250128174036.png|600]]
   
2.9. <font style="color:#A8A8A8; text-decoration: underline;">Modern UNIX Systems</font>  
- Today the UNIX has become more modular


**System V Release 4 (SVR4)**
 - Developed jointly by AT&T and Sun Microsystems
 - New features in the release include real-time processing support, process scheduling classes, dynamically allocated data structures, virtual memory management, virtual file systems, and preemptive kernel
 - Uses both commercial and academic design scheme, and provides a uniform platform for commercial UNIX development
 - Runs on 32-bit microprocessors up to supercomputers



![[Pasted image 20250128174104.png|600]]

**Berkeley Software Distribution (BSD)**
- Series of UNIX played a key role in development of OS design theory
- Responsible for the popularity of UNIX and most enhancements
- 4.4BSD was the final version and included a new virtual memory system, changes in the kernel structure, etc.
- Open-source versions (FreeBSD, OpenBDSD) is popular for internet-based servers and firewalls on embedded systems
- Machintosh OS is based on FreeBSD 5.0 and the Mach 3.0 microkernel

![[Pasted image 20250128193546.png|600]]

**Solaris 11**
- Oracle's SVR4-based UNIX
- Includes fully pre-emptable, multithreaded kernel, full support for SMP, and an object-oriented interface to file systems
   
2.10. <font style="color:#A8A8A8; text-decoration: underline;">Linux</font>  

**History**
- Linux started out as a UNIX variant for the IBM PC (Intel 80386) architecture
- By Linus Torvalds in 1991
- Free and open-source, it became an early alternative to other UNIX workstations
- Free Software Foundation (FSF) is a non-profit organization to promote and protect free software
	- GNU Project provides tools for software developers
	- GNU Public License (GPL) is the FSF seal of approval
- Linux is highly modular and easily configured
- Vendors can change applications and utilities to meet specific requirements
- Patches are sent by developers from individuals to commercial companies (Intel, Red Hat, Google, Samsung, etc)
- Patches are sent and reviewed to meet qualifications and convention standards
- Subsystem maintainer sends a pull request of his tree to the main kernel tree, when a change has occurred
- A new kernel version is released every 7-10 weeks, and has 5-8 release candidates versions (RC)


**Modular Structure**
- Most UNIX kernels are monolithic
	- Includes virtually all of the OS functionality in one large block of code that runs as a single process with a single address space
	- All components have access to all internal data structures and routines
- Modules and routines must be relinked and reinstalled, and system rebooted, if there is a change
- Modification is difficult
- Linux is structured as a collection of independent modules (_loadable modules_), a number of which can be automatically loaded and unloaded on demand
- A module is an object file that is liked to and unlinked from the kernel at runtime
- A module does not execute on its own process or thread, instead it is executed in kernel mode by the current process
- Two Characteristics
	- Dynamic linking
		- A kernel module can be loaded and linked into the kernel while the kernel is already in memory and executing
		- Module can be unlinked and removed from memory at any time
	- Stackable modules
		- Modules are arranged in a hierarchy
		- Individual modules serve as libraries when they are referenced by client modulesX higher up in the hierarchy, and as clients when they reference modules further down

![[Pasted image 20250128195808.png|600]]

- Dynamic linking facilitates configuration and save kernel memory
- Programs can explicitly load and unload kernel modules using `insmod`, `modprobe`, and `rmmod` commands
- Benefits of stackable modules
	- Code common to a set of similar modules can be moves into a single module, reducing replication
	- The kernel can make sure that needed modules are present, reducing unloading and loading unused modules

Module Table Elements
- Name: Module name
- refcnt: Module counter
	- Incremented when an operation involving a module's function is started and decremented when the operation terminates
- num_syms: Number of exported symbols
- \*syms: Pointer to this module's symbol table


**Kernel Components**
- Main components of a typical Linux kernel implementation has several processes running on top of the kernel
- Each box represents a separate process
- Squiggly line is a thread of execution

| Principle Kernel Components | Description                                                                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Signals                     | The kernel uses signals to call into a process<br>Ex. signals are used to notify a process of certain faults, such as division by zero                                                      |
| System calls                | The system call is the mean by which a process requests a specific kernel service<br>6 types: file system, process, schedulling, interprocess communication, socket (networking), and misc. |
| Processes and scheduler     | Creates, manages, and schedules processes                                                                                                                                                   |
| Virtual memory              | Allocates and manages virtual memory for processes                                                                                                                                          |
| File system                 | Provide a global, hierarchical namespace for files, directories, and other file-related object and provide file system functions                                                            |
| Network protocols           | Support the Socket interface to users of the TCP/IP protocol suite                                                                                                                          |
| Character device drivers    | Manage device that require the kernel to send or receive data one byte at a time, such as terminals, modems, and printers                                                                   |
| Block device drivers        | Manage devices that read and write data in blocks, such as various forms of secondary memory (magnetic disks, CD-ROMs)                                                                      |
| Network device drivers      | Manage network interface cards and communications ports that connect to network devices, such as bridges and routers                                                                        |
| Traps and faults            | Handle traps and faults generated by the processor, such as memory fault                                                                                                                    |
| Physical memory             | Manages the pool of page frames in real memory and allocates pages for virtual memory                                                                                                       |
| Interrupts                  | Handle interrupts from peripheral devices                                                                                                                                                   |



![[Pasted image 20250128200314.png|600]]
![[Pasted image 20250128200331.png|600]]
![[Pasted image 20250128200351.png|600]]
![[Pasted image 20250128200408.png|600]]

  
2.11. <font style="color:#A8A8A8; text-decoration: underline;">Android</font>  
- Android operating system in a Linux-based system designed for mobile phones
- Increasingly, the OS of any device with a computer chip other than servers and PC
- Android Open Source Project (AOSP) helps the active community develop and distribute modified version of the operating system

**Android Software Architecture**
- Defined as a software stack that includes a modified version of the Linux kernel, middleware, and key application
- Android is a complete software stack, not just an OS

_APPLICATIONS_
- All the applications that the user interacts with is part of the application layer
	- General-purpose applications
		- E-mail client
		- SMS program
		- Calendar
		- Maps
		- Browser
		- Contacts
		- Other app. standard on a mobile device

_APPLICATION FRAMEWORK_
- Provides high-level building blocks, accessible through standardized APIs, that programmers use to create new apps
- Simplify and reuse components

| Application Framework Components | Descriptions                                                                                                                                                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Activity Manager                 | Manages lifecycle of application<br>Responsible for starting, pausing, and resuming the various applications                                                                      |
| Window Manager                   | Java abstraction of the underlying Surface Manager<br>Handles the frame buffer interaction and low-level drawing, whereas the Window Manager provides a layer on top (status bar) |
| Package Manager                  | Installs and removes application                                                                                                                                                  |
| Telephony Manager                | Allows interaction with phone, SMS, and MMS services                                                                                                                              |
| Content Providers                | These functions encapsulate application data that need to be shared between application, such as contacts                                                                         |
| Resource Manager                 | Manages application resources, such as localized strings and bitmaps                                                                                                              |
| View System                      | Provides the user interface (UI) primitives, such as buttons, list-boxes, date pickers, and other controls (UI Events such as touch and gestures)                                 |
| Location Mananger                | Allows developers to tap into location-based services, whether by GPS, cell tower IDs, or local Wi-Fi databases                                                                   |
| Notification Manager             | Manages events, such as arriving messages and appointments                                                                                                                        |
| XMPP                             | Provides standardized messaging functions between applications                                                                                                                    |
|                                  |                                                                                                                                                                                   |

_SYSTEM LIBRARIES_
- Consists of two parts: System Libraries, Android Runtime
	- System Libraries
		- A collection of useful system function, in C or C++, and used by various components of the Android system
		- Called through Java interface

| Key Libraries                 | Description                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Surface Manager               | Window manager similar to Vista or Compiz<br>Drawing commands go into off-screen bitmaps that are then combined to form the screen content (window transparency, and transitions) |
| OpenGL (Open Graphic Library) | Cross-language, multi-platform API for rendering 2D and 3D computer graphics                                                                                                      |
| Media Framework               | Supports video recording and playing in many formats, AAC, AVC (H.264), H.263, MP3, and MPEG-4                                                                                    |
| SQL Database                  | Android includes a lightweight SQLite database engine for storing persistent data                                                                                                 |
| Browser Engine                | For fast display of HTML content, Android uses WebKit library (same library used in Safari and iPhone)                                                                            |
| Bionic LibC                   | Stripped-down version of the standard C system library, tuned for embedded Linux-based devices<br>The interface is the standard Java Native Interface (JNI)                       |

LINUX KERNEL
- OS kernel for Android is similar, bot not identical, to standard Linux kernel distribution
- Android kernel lacks drivers not applicable in mobile environments, making the kernel smaller
- Android kernel has featured tailored to the mobile environment
- Relies on Linux kernel for core systems services
	- Security
	- Memory management
	- Process management
	- Netwrok stack
	- Driver model
- Kernel is an abstraction layer between the hardware and the rest of the software stack

**Android Runtime**
- Most operating systems on mobile devices (ex. iOS and Windows) use software that is compiled directly to the hardware
- Android software is mapped into a byte code format, which is then transformed into native instruction on the device itself
- Early scheme are known as Dalvik
	- Limitations in terms of scaling up to larger memory and multicore architecture
- More recent schemes use Android runtime (ART)
	- Compatible with Dalvik, dex (Dalvik Executable)

![[Pasted image 20250130085119.png|600]]


_THE DALVIK VIRTUAL MACHINE (DVM)_
- Executes files in the .dex format, optimized for efficient storage and memory-mappable execution
- VM can run classed compiled by Java language compiler that have been transformed into its native format using the included "dx" tool
- VM runs on top of Linux kernel
- Dalvik core class library provides a familiar development base for those used to programming with Java Standard Edition


_THE DEX FILE FORMAT_
- The DVM runs applications and code written in Java
- A standard Java compiler turns source code (written as text files) into bytecode
- Bytecde is compiled into a .dex file that the DVM can read and use
- Class files are converted into .dex files and then read and executed by the DVM
	- Similar to .jar files in a standard Java VM
- Duplicate data used in class files are included only once in the .dex file, saves space and uses less overhead
- Executables are modified again during application installation


_ANDROID RUNTIME CONCEPTS_
- For both Dalvik and ART, all Android application written in Java are compiled o dex bytecode
- Dalvik has to compile dex code to machine code to run
- Dalvik runtime complied during JIT (just-in-time) compilation
- JIT compiles a segment of code, and has a smaller memory footprint
- The pages of physical memory that store the cached code are not swappable/pageable
- JIT-compilation is done every time an app is ran
- ART compiles the bytecode to native machine code during install time of the app
	- Ahead-of-time (AOT) compilation
- ART uses the "dex2oat" tool to have compilation at install time

- Android Package Kit (APK) is an application package that comes from the developer to the user
- Source code in compiled into .dex format and combined with any support code to form an APK
- The user unpacks the APK, and installed into the application directory
- Dalvik
	- A function `dexpopt` is applied to produce an optimized version of dex referred to as quickened dex
- ART
	- Uses `dex2oat` that provides similar optimization
	- Compiles the dex code to produce native code on the target device
	- The output is an Executable and Linkable Format (ELF) file, that can fun without an interpreter

![[Pasted image 20250128202947.png|600]]

_ADVANTAGES AND DISADVANTAGES_
- Advantages
	- Reduces startup time of applications as native code is directly executed
	- Improves battery life because processor usage for JIT is avoided
	- Lesser RAM footprint to run application (no JIT cache)
	- Garbage Collection optimizations and debug enhancements
- Disadvantages
	- Application installation takes more time
		- Increased time for testing that requires app load
	- All applications installed on a device are compiled to native code take longer to reach Home Screen in ART, compared to Dalvik
	- Native code is stored on internal storage that requires a significant amount of addition internal storage space


**Android System Architecture**

| Android Layers                   | Description                                                                                                                                                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Applications and Framework       | Application developers are primarily concerned with this layer and the APIs that allow access to lower-layer services                                                                                                                       |
| Binder IPC                       | The Binder Inter-Process Communication mechanism allows the application framework to cross process boundaries and call into the Android system services code<br>Allows high-level framework APIs to interact with Android's system services |
| Android System Services          | Most functionality is exposed through the application framework APIs that invokes system services, that access hardware and kernel function<br>2 groups: Media services and System services                                                 |
| Hardware Abstraction Layer (HAL) | Standard interface to kernel-layer device drivers, to upper-layer code need not be concerned with the details of the implementation of specific drivers and hardware                                                                        |
| Linux Kernel                     | Linux kernel is tailored to meet the demands of a mobile environment                                                                                                                                                                        |
![[Pasted image 20250128203017.png|600]]

**Activities**
- A single visual user interface component
	- Menu selections
	- Icons
	- Checkboxes
- Every screen in an application is an extension of the Activity clss
- Activities use Views to form graphical user interfaces that display information and respond to user actions

**Power Management**
- Android adds two features to the Linux kernel to enhance power management: alarms, and wakelocks
- Alarms visible to the app developer through the AlarmManager in the RunTime core libraries
- Alarm can request a timed service
	- Sleep mode, saving power, wake up
- The wakelock prevents an Android system from entering into sleep mode
- Locks are requested through the API whenever an application requires one of the managed peripheral to remain powered on

- ==Full_Wake_Lock:== Processor on, full scree brightness, keyboard bright
- ==Partial_Wake_Lock:== Processor on, screen off, keyboard off
- ==Screen_Dim_Wake_Lock:== Processor on, screen dim, keyboard off
- ==Screen_Bright_Wake_Lock:== Processor on, scree bright, keyboard off

  
2.12. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>
<font style="color:#AA0F02; text-decoration: underline;">Key Terms</font>  

| batching processing<br>batch system<br>execution context<br>distributed operating system<br>downtime<br>fault<br>interrupt<br>job | job control language (JCL)<br>kernel<br>kernel mode<br>loadable modules<br>mean time to failure (MTTF)<br>memory management<br>microkernel<br>monitor | monolithic kernel<br>multiprogrammed batch system<br>multiprogramming<br>multitasking<br>multithreading<br>nucleus<br>object-oriented design<br>operating system | physical address<br>privileged instruction<br>process<br>process state<br>real address<br>reliability<br>resident monitor<br>round-robin<br>scheduling<br>serial processing<br>state | symmetric multiprocessing (SMP)<br>task<br>thread<br>time sharing<br>time-sharing system<br>time slicing<br>uni-programming<br>uptime<br>user mode<br>virtual address<br>virtual machine<br>virtual memory |
| --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |


## 3. <font style="color:#FCD12A">Process Description and Control</font>  

^e4bee5

- All multiprogramming operating systems are built around the concept of the process
	- OS must interleave the execution of multiple processes, to maximize processor utilization while providing reasonable response time
	- OS must allocate resource to processes in conformance with a specific policy, and avoid deadlock
		- Certain functions/applications of high priority
	- Support interprocess communication and user creation of processes, that aid in the structuring of application
- Data structures represent the state of each process and data structures that record other characteristics of processes that the OS needs


3.1. <font style="color:#A8A8A8; text-decoration: underline;">What is a Process?</font> 
Background
1) A computer platform consists of a collection of hardware resource
	1) Processor, main memory, I/O modules, timers, disk drivers
2) Computer applications are developed to perform some task
	1) Accept input, perform processing, and generate output
3) Inefficient for applications to be written directly for a given hardware platform
	1) Numerous applications can be developed from the same platform
	2) Processor provides limited support for multiprogramming and needs to manage the sharing of the processor and other resources by multiple applications
	3) Protect data, I/O use, and other resource use of each application from the others
4) OS provides a convenient, feature-rich, secure, and consistent interface for applications to use
	1) OS is a layer of software between the applications and the hardware the supports applications and utilities
5) OS provides a uniform, abstract representation of resources that can be requests and accessed by applications

- OS management
	- Resources are made available to multiple applications
	- Physical processor is switched among multiple applications
	- Processor and I/O devices can be used efficiently

Processes and Process Control Blocks

 
3.2. <font style="color:#A8A8A8; text-decoration: underline;">Process States</font> 
-    What is a _process_
	- A program in execution
	- An instance of a program running on a computer
	- The entity that can be assigned to and executed on a processor
	- A unit of activity by execution of a sequence of instructions, a current state, and an associated set of system resource
- Process consists of two essential elements: program code and set of data
- The information in the list below is stored in a data structure (process control block)
- Process control block contains sufficient information to be interrupted and later resumed
- When interrupted, the current values of the program counter and processor registers (context data) are saves in the appropriate fields
- State is changed to _blocked or ready_
- OS is free to put another process in the running state
- Process consists of program code and associated data plus a control block
- For a single-processor, at most one process is in running state

| Process Elements       | Description                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Identifier             | A unique identifier associated with the process, to distinguish it from all other processes                                    |
| State                  | If the process is currently executing, it is in the running state                                                              |
| Priority               | Priority level relative to other processes                                                                                     |
| Program counter        | The address of the next instruction in the program to be executed                                                              |
| Memory pointers        | Include pointers to the program code and data associated with this process, plus any memory blocks shared with other processes |
| Context data           | These are data that are present in registers in the processor while the process is executing                                   |
| I/O status information | Includes outstanding I/O requests, I/O devices assigned to this process, a list of files in use by the process, and so on      |
| Accounting information | May include the amount of processor time and clock time used, time limits, account numbers, etc.                               |
![[Screenshot 2025-01-30 at 9.52.56 AM.png|300]]

3.2 <font style="color:#A8A8A8; text-decoration: underline;">Process States</font>
- A process or task is create for a program to be executed
- The processor executes instructions by changing values in the program counter register
- Program counter may refer to code in different programs that are part of different processes
- Within an individual program, its execution involves a sequence of instructions within that program
- The _trace_ of a process, is the listing of processes from an individual process
- All three processes are represented by programs that are fully loaded in main memory
- There are small dispatcher programs that switched the processor from one process to another
![[Pasted image 20250128203226.png|500]]

- Shaded area represents code executed by the dispatcher
- OS only allows a process to continue execution for a maximum of six instruction cycles
	- Prevents any single process from monopolizing processor time

![[Pasted image 20250128203243.png|500]]


**A Two-State Process Model**
- Operating system determines the interleaving pattern for execution and allocating resources to processes
- Each entry in the queue is a pointer to the process control block of a particular process
- Queue consists of a linked list of data blocks

![[Pasted image 20250128203301.png|600]]

**The Creating and Termination of Processes**
- The life of a process is bounded by its creation and termination


_PROCESS CREATION_
- OS builds the data structures used to manage the process, and allocates address memory space in main memory
- In a batch environment, a process is created in response to the submission of a job
- In an interactive environment, a process is created when a new user attempts to log on
- OS is responsible for the creation of the new process
- _Process spawning_, is the creation at the explicit request of another process
- When a process spawns another (_parent process_), and the spawned (_child process_)
![[Pasted image 20250128203325.png]]

_PROCESS TERMINATION_
- Any computer system must provide a means for a process to indicate its completion
- A batch job should indicate a Halt instruction or an explicit OS service call for termination
- In some operating systems, a process may be terminated by the process that created it, or when the parent process is itself terminated

![[Pasted image 20250128203347.png]]

**A Five-State Model**
- The queue is a first-in-first-out list and the processor operates in _round-robin_ fashion on the available processes
- Some process in the Not Running state are ready, when others are blocked, waiting for an I/O operation
- A single queue cannot account for different process states
- Dispatcher would have to scan the list for processes that are not blocked and that have been in the queue the longest
- Better way is to split the Not Running state into two: Ready and Blocked

1) Running
	1) The process that is currently being executed
2) Ready
	1) A process that is prepared to execute when given the opportunity
3) Blocked/Waiting
	1) A process that cannot execute until some event occurs
		1) Completion of an I/O operation
4) New
	1) A process that just been created but has not yet been admitted to the pool of executable processes by the OS
		1) Process control block is created, but not loaded into main memory
5) Exit
	1) A process that has been released from the pool of executable processes by the OS
		1) Halted or aborted

![[Pasted image 20250128203358.png]]
- New and Exit states are constructs for process management
- When a new state is defined
	- The OS allocates and builds tables, and information concerning the process is maintained in control tables in main memory
	- However, process itself is not in main memory, and remains in secondary storage (disk storage)
- When a process exits
	- A process is terminated when it reaches a natural completion point, or aborts
	- Termination moves the process to the Exit state
	- Tables and associated information are temporarily preserved by the OS, where program extracts any needed information

| Each State Transition for a Process | Description                                                                                                                                                                                                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Null → New                          | A new process is created to execute a program                                                                                                                                                                          |
| New → Ready                         | The OS will move a process when it is prepared to take on an additional process                                                                                                                                        |
| Ready → Running                     | OS chooses one of the processes in the Ready state<br>Job of the scheduler or dispatcher                                                                                                                               |
| Running → Exit                      | The currently running process is terminated by the OS if the process indicates that it has completed or if it aborts                                                                                                   |
| Running → Ready                     | Running process has reached the maximum allowable time for uninterrupted execution<br>Preempted process A, if B has higher priority<br>Process may voluntarily release control of the processor (background processes) |
| Running → Blocked                   | If it requests something for which it must wait; request to the OS is in the form of a system service call                                                                                                             |
| Blocked → Ready                     | A process is moved when the event for which it has been waiting occurs                                                                                                                                                 |
| Ready → Exit                        | A parent may terminate a child process at any time<br>If a parent terminated, all child process associated with that parent may be terminated                                                                          |
| Blocked → Exit                      | The comments under the preceding item apply                                                                                                                                                                            |

- When an even occurs, the OS must scan the entire blocked queue, searching for those processes waiting on the event
- More efficient to have multiple queues; then the entire list of processes in a queue can be moved to the Ready state
- If the dispatching of processes uses a priority scheme, the Ready queues should be numbered by priority level
- OS can then determine which is the highest-priority ready process than has been waiting the longest


![[Pasted image 20250128203428.png|600]]
![[Pasted image 20250128205857.png|600]]


**Suspended Processes**
_THE NEED FOR SWAPPING_
- In the three principle states, all the queues must be resident in main memory
- I/O activities are much slower than computation, and therefore the processor in a uni-programming system is idle most of the time
- Larger memory results in larger processes, not more processes
- Swapping, moves parts of all of a process from main memory to disk
- When all of the processes in main memory are in Blocked state, the OS can suspend one process by putting it in the Suspend state and transferring it to disk
	- This frees main memory to bring in another process
- The OS can bring a newly created process, or a previously suspended one
- Two independent concepts, 2 x 2 combinations

| 4 Process State Transition | Description                                                                                                 |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Ready                      | The process is in main memory and available for execution                                                   |
| Blocks                     | The process is in main memory and awaiting an event                                                         |
| Blocked/Suspend            | The process is in secondary memory and awaiting an event                                                    |
| Ready/Suspend              | The process is in secondary memory, but it available for execution as soon as it is loaded into main memory |

- With a virtual memory scheme, it is possible to execute a process that is only partially in main memory
- Virtual memory eliminates the need for explicit swapping

![[Pasted image 20250128205920.png|600]]

| Possible Process Transitions        | Description                                                                                                                                                                                                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Blocked → Blocked/Suspend           | If there are no ready processes, then at least one blocked process is swapped out to make room for another process that is not blocked<br>OS determines that the currently running process, or a ready process that it would like to dispatch, requires more main memory to maintain performance, a blocked process will be suspended |
| Blocked/Suspend → Ready/Suspend     | A process when the even for which it has been waiting occurs<br>Requires that the state information concerning suspended processes must be accessible to the OS                                                                                                                                                                       |
| Ready/Suspend → Ready               | When there are no ready processes in main memory, the OS will need to bring one in to continue execution<br>Ready/Suspend state has higher priority than any of the processes in the Ready state                                                                                                                                      |
| Ready → Ready/Suspend               | OS prefers to suspend a blocked process rather than a ready one<br>Suspend a ready process to free up large block of main memory<br>Suspend lower-priority if the blocked process will be ready soon                                                                                                                                  |
| New → Ready/Suspend and New → Ready | A new process is added to either queue<br>OS must create a process control block and allocate an address space<br>Preferable for OS to perform housekeeping at early time, so it can maintain more processes that are not blocked                                                                                                     |
| Blocked/Suspend → Blocked           | When there is a process in the Blocked/Suspend queue with a higher priority than any of the processes in the Ready/Suspend queue                                                                                                                                                                                                      |
| Running → Ready/Suspend             | OS is preempting the process because a higher-priority process on the Blocked/Suspend queue has become unblocked, OS could move the running process directly to the Ready/Suspend queue to free main memory                                                                                                                           |
| Any State → Exit                    | A process may be terminated by the process that created it or when the parent process is terminated                                                                                                                                                                                                                                   |

_OTHER USES OF SUSPENSION_
-  A process that is not in main memory is not immediately available for execution, whether or not it is awaiting an event
- Characteristics of suspended processes
	- The process is not immediately available for execution
	- The process may or may not be waiting on an event
		- Block condition is independent of the suspend condition
	- The process was placed in a suspended state by an agent; itself, parent process, or the OS, for the purpose of preventing its execution
	- The process may not be removed from this state until the agent explicitly orders the removal

![[Pasted image 20250130122551.png]]

3.3. <font style="color:#A8A8A8; text-decoration: underline;">Process Description</font>  
- OS schedules and dispatches processes for execution by the processor, allocates resources to processes, and responds to requests by user processes for basic services
- OS is the entity that manages the use of system resources by processes
- In a multiprogramming environment, there are a number or processes $(P_1,...,P_n)$ that have been created and exist in virtual memory
- $P_1$ is partially in main memory, and has control of two I/O devices
- $P_2$ is also in main memory, but is blocked waiting for an I/O device allocated to $P_1$ 
- $P_n$ has been swapped out and is suspended

![[Pasted image 20250128205951.png|500]]

**Operating System Control Structures**
- OS needs information about the current status of each process and resource
- The OS constructs and maintains tables of information about each entity that it is managing
- 4 types of OS Tables
	- Memory
	- I/O
	- File
	- Process
- Memory tables are used to keep track of main (real) and secondary (virtual) memory
- Some main memory is reserved for use by the OS, the remainder is available for use by processes
- Memory tables must include:
	- Allocation of main memory to processes
	- Allocation of secondary memory to processes
	- Any protection attributes of blocks of main or virtual memory
		- shared memory regions
	- Any information needed to manage virtual memory
	- I/O tables are used by the OS to manage I/O devices and channels of the computer system
	- OS also maintains file tables
	- File tables provide information about the existence of files, their location on secondary memory, their current status, and other attributes
	- OS maintains process tables to manage processes
	- All 4 tables must be linked or cross-referenced
	- The files referred to in the file tables are accessible via an I/O device and will be in main or virtual memory
	- Tables are subject to memory management
	- When OS is initialized, it must have access to some configuration data that defines the basic environment
		- Data must be created outside the OS, which human assistance or by some auto configuration software

![[Pasted image 20250128210008.png|500]]

**Process Control Structures**
- OS needs the process locations, and attributes of the process for its management
	- Process ID and process state

_PROCESS LOCATION_
- A process must include a program or set of programs to be executed
- Associated with these programs is a set of data locations for local and global variables and any defined constant
- Therefore, a process has sufficient memory to hold programs and data of that process
- Execution uses a stack that keeps track of procedure calls and parameter passing between procedures
- Each process has attributes that are used (process control block)
- The collection of program, data, stack, and attributes is the _process image_
- The location of the process image depends on the memory management scheme used
- Process image is maintained as a contiguous block of memory
	- This block is maintained in secondary memory, usually disk
- To execute, the entire process image must be loaded into main memory, or at least virtual memory
- OS need to know location of each process on disk, and location of process in main memory
- Modern operating system presume paging hardware that allows noncontiguous physical memory to support partially resident processes
- Process tables maintained by the OS must show the location of each page of each process image

![[Pasted image 20250130150610.png]]
![[Pasted image 20250130152025.png]]

_PROCESS ATTRIBUTES_
- A complex multiprogramming system requires more information about each process
- Different system will organize this differently
- Process control block information
	- Process identification
	- Processor state information
	- Process control information

- Each process is assigned a unique numeric identifier, which may be indexed into the primary process table; otherwise there must be a mapping that allows the OS to locate the appropriate tables based on the process identifier
- OS may use process identifiers to cross-reference tables
	- Memory tables provide a map of main memory with an indication of which process is assigned to each region
	- Similar references will appear in I/O and file tables
	- Identifiers inform OS of destination during process communication
	- Identifiers indicate the parent and descendants of each process

- Processor state information consists of the content of process registers
- When a process is interrupts, register information must be saved to be used when restored
- The register set will include user-visible registers, control and status register, and stack pointers

- A program status word (PSW) is a register or set that contained status information
- Contained condition codes and other status info.
- `EFLAGS` register is used on Intel x86 processors

- Process control information is the additional information needed to control and coordinate active processes

- Process control block contains structuring information, including pointers that allow the linking of process control blocks
- Queue are linked lists of process control blocks

![[Pasted image 20250130152051.png|335]]![[Pasted image 20250130152315.png|300]]

![[Pasted image 20250130152108.png]]


_THE ROLE OF THE PROCESS CONTROL BLOCK_
- The process control block is the most important data structure in an OS
- Each process control block contains all of the information about a process that is needed by the OS
- The blocks are read and/or modified by virtually every module in the OS
	- Scheduling, resource allocation, interrupt processing, and performance monitoring and analysis
- The set of process control blocks defines that state of the OS
- Each process has a unique ID, and can be used as an index into a table of pointers to the process control block
- 2 problems
	- A bug in a single routine can damage process control blocks
	- A design change in the structure or semantics of the process control block could affect a number of modules in the OS
- Problems can be resolved by requiring all routines in the OS to go through a handler routine, to protect process control blocks
- Trade-off is performance and the degree to which the remainder of the system software can be trusted to be correct


![[Pasted image 20250130152441.png]]

   
3.4. <font style="color:#A8A8A8; text-decoration: underline;">Process Control</font>  
**Modes of Execution**
- Most processors support at least two modes of execution
- Certain instructions can only be accessed on a more-privileged mode
	- Reading or altering a control register
		- PSW, primitive I/O instructions, and instructions that relate to memory management
- Certain regions of memory can only be accessed in the more-privileged mode

![[Pasted image 20250130154403.png]]

- The less-privileged mode is referred to as the _user mode_
- More-privileged mode is the _system mode, control mode, or kernel mode_
- Using two modes protects the OS and key operating system tables
	- Keeps control blocks from interference by user programs
- In the kernel mode, the software has complete control of the processor and all its instructions, registers, and memory
- PSW indicates the mode of execution
- When user makes a call to an operating system service or when an interrupt triggers execution of an operating system routine, the modes is set to the kernel mode
- When return to the user process, the mode is set to user
	- Intel Itanium processor (64-bit IA-64 architecture)
	- Processor has a PSR that includes a 2-bit CPR (current privilege level) field
	- Level 0 to 3 (most to least privilege)
- Most operating systems, Linux, use level 0 for the kernel and 1 for the user
- The final instruction during the interrupt-handling routine is the IRT (interrupt return) that restores the PSR, and privilege level

**Process Creation**
1) Assign an unique process identifier to the new process
	1) A new entry is added to the primary process table, which contains one entry per process
2) Allocate space for the process
	1) OS must know space needed for the private user address space (programs and data) and the user stack
		1) Assigned by default or set based ono user request at job creation time
	2) If any existing address space, it is shared by this new process, and appropriate linkage is set
	3) Space for the process control block must be allocated
3) Initialize the process control block
	1) Initialized with most entries zero, except for the program counter and system stack pointers
	2) Portion is initialized based on standard default values plus attributes that have been requested for this process
		1) Process state sets to Ready or Ready/Suspend
		2) Priority set to lowest unless requested otherwise
	3) Process may have no resources unless explicit requests or inherited from the parent
4) Set the appropriate linkages
	1) If the OS maintains each scheduling queue as a linked list, then the new process must be put in the Ready or Ready/Suspend list
5) Create or expand other data structures
	1) OS may maintain an accounting file on each process to be used subsequently for billing and/or performance assessment purposes

**Process Switching**
_WHEN TO SWTIVH PROCESSES_
- A process switch occurs any time that the OS has gained control from the current running process
- 2 system interrupts
	- Interrupt
		- Event that is external to and independent of the currently running process
			- Completion of I/O operation
	- Trap
		- Error or exception condition generated within the currently running process
			- Illegal file access attempt
- Types of interrupts
	- Clock interrupt
		- OS determines whether the currently running process has been executing for the maximum allowable unit of time (_time slice_)
		- Time slice is the maximum amount of time that a process can execute before being interrupted
	- I/O interrupt
		- OS determines what I/O action has occurred
		- OS moves all corresponding blocked processes to the Ready state (and Blocked/Suspend processes to the Ready/Suspend state)
		- OS must then decide whether to resume execution of the process currently in the Running state, or to preempt that process for a higher-priority Ready process
	- Memory fault
		- The processor encounters a virtual memory address reference for a work that is not in main memory
		- OS brings in block (page or segment) of memory containing the reference from secondary memory to main memory
		- After the I/O request, the process with memory fault is placed in a blocked state; the OS then switches to resume execution of another process
		- After the block is brought into memory, that process is places in the Ready state
- With a trap, the OS determines if the error or exception condition is fatal
- If fatal, the current running process is moves to the Exit state and a process switch occurs
- If not, the action of the OS will depend on the nature of the error and the design of the OS
- Attempt some recovery procedure or simply notify the user
- May perform a process switch or resume the currently running process
- OS may be activated by _supervisor call_ from the program being executed
	- A system call may place the user process in the Blocked state

![[Pasted image 20250130155936.png]]


_MODE SWITCHING_
- In the interrupt stage, the processor checks to see if any interrupts are pending (interrupt signal)
- If not interrupts are pending, the processor fetches the next instruction of the current program in the current process
- If an interrupt is pending, the processor does the following:
	1) It sets the program counter to the starting address of an interrupt-handler program
	2) It switch from user mode to kernel mode to the interrupt processing code may include privileged instructions
- The context of the process that has been interrupted is saved into that process control block of the interrupted program
- Portion of the process control block that was referred to the as the processor state information must be saves
	- Program counter, other processor registers, and stack information
- The interrupt program
	- Resets the flag or indicator that signals the presence of an interrupt
	- Send an acknowledgement to the entity that issued the interrupt
		- I/O module
	- Basic housekeeping relating to the effects of the event that caused the interrupt
- If the interrupt is by the clock, then the handler will hand control over to the dispatch, which will want to pass control to another process because the time slice allotted to the currently running process has expired
- After ending an interrupt and resuming execution, all that is necessary is to save the processor state information when the interrupt occurs and restore that information when control is returned to the program that was running

_CHANGE OF PROCESS STATE_
- Mode switch is different from that of the process switch
- A mode switch may occur without changing the state of the process that is currently in the Running State
- The context saving and restoral involve little overhead
- If currently running process is move to another state (Ready, Blocked, etc.) then the OS must make substantial changes to its environment

- Steps in a full process switch
1) Save the context of the processor
	1) Program counter and other register
2) Update the process control block of the process that is currently in the Running state
	1) Other relevant fields must also be updated, including the reason for leaving the Running state and accounting information
3) Move the process control block of this process to the appropriate queue
	1) Ready, Blocked on Event $i$, or Ready/Suspend
4) Select another process for execution
5) Update the process control block of the process selected
	1) Changing the state of this process to Running
6) Update memory management data structures
7) Restore the context of the processor to that which existed at the time the selected process was last switched out of the Running state
	1) Load the previous values of the program counter and other registers


3.5. <font style="color:#A8A8A8; text-decoration: underline;">Execution of the Operating System</font>  
- The OS functions in the same way as ordinary computer software, in the sense that the OS is a set of programs executed by the processor
- The OS frequently relinquishes control and depends on the processor to restore control to the OS

**Non-process Kernel**
- Common on older operating systems, is to execute the kernel of the OS outside of any process
- When the currently running process is interrupted or issues a supervisor call, the mode context of this process is saved and control is passed to the kernel
- OS has its own region of memory to use and its own system stack for controlling procedure calls and returns
- The process is only applied to user programs
- The operating system code is executed as a separate entity that operates in privileged mode

**Execution within User Processes**
- Common on smaller computers is to execute virtually all OS software in the context of a user process
- OS is a collection of routines the user called to perform various functions
- OS manages $n$ process image, which includes program, data, and stack areas for kernel programs
- A separate kernel stack is used to manage calls/returns while the process is in kernel mode
- OS code and data are in the shared address space and are shared by all user processes
- A process switch does not occur, just a mode switch within the same process
- Key advantage: During an interrupt, there is no need for two process switches
- Switching routing puts one process in a running state, and the other in a non-running state
- Therefore, execution is taking place outside of all processes
- Disadvantage: Executed code is shared with operating system code, and user cannot interfere with the operating system routines
- Within a process, both a user program and operating system programs may execute, and the operating system programs that execute in the various user processes are identical

![[Pasted image 20250130195342.png | 300]]

**Process-Based Operating System**
- OS is a collection of system processes
- In the other options, the software that is part of the kernel executes in a kernel mode
- In this case, major kernel functions are organized as separate processes
- Advantages
	- Imposes a program design discipline that encourages the use of a modular OS with minimal, clear interfaces between the modules
	- Some non-critical OS functions are separated
		- A monitor program can run in the background at low-priority and interleave with other processes
- OS processes can have dedicated processors to improve performance
   
   
3.6. <font style="color:#A8A8A8; text-decoration: underline;">UNIX SVR 4 Process Management</font>  
- UNIX System V uses an OS that executes within the environment of a user process
- UNIX uses two categories of processes
	- System processes
		- Run kernel mode
		- Executes operating system code to perform administrative and housekeeping functions
			- Memory allocation
			- Process swapping
	- User processes
		- Operate in user mode
		- Executes user programs and utilities
		- Executes in kernel mode to execute instructions that belong to the kernel
		- A user process enters kernel mode by a system call

**Process States**
- 9 process states are recognized by the UNIX SVR4 OS
- With the two UNIX sleeping states corresponding to the two blocked states
- Differences
	- UNIX employs two Running states to indicate whether the process is executing in a use more or kernel mode
	- A distinction is made between the two states: (Ready to Run, In Memory) and (Preempted)
- Preemption can only occur when a process is about to move from kernel mode to use mode
- UNIX is unsuitable for real-time processing, because it cannot preempt in kernel mode
- Two unique processes in UNIX
	- Process 0 is created when the system boots
		- Predefined as a data structure loaded at boot
		- time
	- Process 0 spawns process 1, referred to as the init process
	- All other processes in the system have process 1 as an ancestor

| UNIX Process State      | Description                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| User Running            | Executing in user mode                                                                                                           |
| Kernel Running          | Executing in kernel mode                                                                                                         |
| Ready to Run, in Memory | Ready to run as soon as the kernel schedules it                                                                                  |
| Asleep in Memory        | Unable to execute until an event occurs; process is in main memory (a blocked state)                                             |
| Ready to Run, Swapped   | Process is ready to run, but the swapper must swap the process into main memory before the kernel can schedule it to execult     |
| Sleeping, Swapped       | The process is awaiting an event and has been swapped to secondary storage (a blocked state)                                     |
| Preempted               | Process is returning from kernel to user mode, but the kernel preempts it and does a process switch to schedules another process |
| Created                 | Process is newly created and not yet ready to run                                                                                |
| Zombie                  | Process no longer exists, but it leaves a record for its parent process to collect                                               |

![[Pasted image 20250130201131.png]]

**Process Description**
- 3 Parts of a UNIX Process image
	- User-level content
		- Basic elements of user's program and can be generated directly from a compiled object file
		- Separated into text and data areas
		- Processor uses the user stack area for procedure calls, returns, and parameter passing
		- The shared memory area is a data area shared with other processes
	- Register content
		- When a process is not running, the processor status information is stored in the register context area
	- System-level content
		- Contains the remaining information that the OS needs to manage
		- Static and dynamic part
		- One element of the static part is the process table entry
		- The process table contains one entry per process, and control information that is accessible to the kernel (virtual memory)
		- The third static portion of the system-level is the per process region table, which is used by the memory management system
		- Kernel stack is dynamic, used for saved and restored data during interrupts

![[Pasted image 20250203134628.png|600]]


| UNIX U Area                | Description                                                                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Process table pointer      | Indicates entry that corresponds to the U area                                                                                                                |
| User identifiers           | Real and effective user IDs used to determine user privileges                                                                                                 |
| Timers                     | Record time that the process (and its descendants) spend executing in user mode and in kernel mode                                                            |
| Signal-handler array       | For each type of signal defined in the system, indicates how the process will react to receipt of that signal (exit, ignore, execute specified user function) |
| Control terminal           | Indicated login terminal for this process, if one exists                                                                                                      |
| Error field                | Records errors encountered during a system call                                                                                                               |
| Return value               | Contains the result of system call                                                                                                                            |
| I/O parameter              | Describe the amount of data to transfer, the address of the source (or target) data array in user space, and file offset for I/O                              |
| File parameter             | Current directory and current root described the file system environment of the process                                                                       |
| User file descriptor table | Records the files the process has opened                                                                                                                      |
| Limit fields               | Restrict the size of the process and the size of a file it can write                                                                                          |
| Permission modes fields    | Mask mode setting on files the process creates                                                                                                                |

**Process Control**
- Process creation is made by the kernel system call, `fork()`
- When a fork request is issues, the following occur:
1) It allocates a slot in the process table for the new process
2) Is assigns a unique process ID to the child process
3) It makes a copy of the process image of the parent, with the exception of any shared memory
4) It increments counters for any files owned by the parent, to reflect that an additional process now also owns those files
5) It assigns the child process to the Ready to Run state
6) It returns the ID number of the child to the parent process, and a 0 value to the child process

- When the parent process kernel finishes process creation it can do the following as a dispatcher routine:
	- Stay in the parent process
	- Transfer control to the child process
	- Transfer control to another process


- When the return from the fork occurs, the return parameter is tested
- If the value is 0, this is the child process, and a branch is executed
- If the value is non-zero, this is the parent process, and the main line of execution can continue

3.7. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
- The principle function of the OS is to create, manage, and terminate processes
- While processes are active, OS must allocate time for execution by the processor, coordinate activities, manage conflicting demands, and allocate system resources to processes
- To perform management, OS maintains a description of each process
	- Address space of process execution
	- Process control block
		- Information required to manage process
			- Current state
			- Resources allocated to it
			- Priority
			- Other relevant data
- The most important state are Ready, Running, and Blocked
- A running process is interrupted by an interrupt, which is an event that occurs outside the process, or by executing a supervisor call to the OS
- The processor performs a mode switch, transferring control to an operating system routine

3.8. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| blocked state<br>child process<br>dispatcher<br>exit state<br>interrupt<br>kernel mode<br>mode switch<br>new state | parent process<br>preempt<br>privileged mode<br>process<br>process control block<br>process control information<br>process image | process spawning<br>process switch<br>program status word<br>read state<br>round-robin<br>running state<br>suspend state | swapping<br>system mode<br>task<br>time slice<br>trace<br>trap<br>user mode |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |



VAX/VMS Process States

| Process State                 | Process Condition                                                                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Currently Executing           | Running process                                                                                                                                                                    |
| Computable (resident)         | Ready and resident in main memory                                                                                                                                                  |
| Computable (outswapped)       | Ready, but swapped out of main memory                                                                                                                                              |
| Page Fault Wait               | Process has referenced a page not in main memory and must wait for the page to be read in                                                                                          |
| Collieded Page Wait           | Process has referenced a shared page that is the cause of an existing page fault wait in another process, or a private page that is in the process of being read in or written out |
| Common Page Wait              | Waiting for shared event flag (event flags are single-bit interprocess signalling mechanisms)                                                                                      |
| Free Page Wait                | Waiting for a free page in main memory to be added to the colleciton of pages in main memory devoted to this process (the working set of the process)                              |
| Hibernate Wait (resident)     | Process put itself in a wait state                                                                                                                                                 |
| Hibernate Wait (outswapped)   | Hibernating process is swapped out of main memory                                                                                                                                  |
| Local Even Wait (resident)    | Process in main memory and waiting for local event flag (usually I/O completion)                                                                                                   |
| Local Event Wait (outswapped) | Process in local event wait is swapped out of main memory                                                                                                                          |
| Supsended Wait (resident)     | Process is put into a wait state by another process                                                                                                                                |
| Suspended Wait (outswapped)   | Suspended process is swapped out of main memory                                                                                                                                    |
| Resource Wait                 | Process waiting for miscellaneous system resource                                                                                                                                  |


## 4. <font style="color:#FCD12A">Threads</font>  
^090406
- Contemporary operating systems has two separate and potentially independent concepts
	- One relating to resource ownership
	- Another relating to execution

4.1. <font style="color:#A8A8A8; text-decoration: underline;">Processes and Threads</font>  
- Concept of a process has two characteristics:
	- Resource ownership
		- Process has a virtual address space that holds the process image
		- The process image is the collection of program, data, stack, and attributed defined in the control block
		- From time to time, a process may be allocated control or ownership of resource
			- Such as main memory, I/O channels, I/O devices, and files
	- Scheduling/execution
		- Process follows an execution path (trace) through one or more programs
		- May be interleaved with that of other processes
		- Process has an execution state and a dispatching priority
- The unit of dispatching is referred to as the _thread_ or _lightweight process_, while the unit of resource ownership is the _process_ or _task_
   
**Multithreading**
- Multithreading refers to the ability of an OS to support multiple, concurrent paths of execution within a single process
- MS-DOS is an OS that supports a single-user process and a single thread
- UNIX supports multiple user processes, but only support on thread per process
- A Java runtime environment uses multiple threads in a single process
- Windows, Solaris, and modern versions of UNIX use multiple processes, that each support multiple threads
- In a multithreading environment, a process is defined as the unit of resource allocation and a unit of productions
- Processes:
	- A virtual space that holds the process image
	- Protects access to processors, and processes, files, and I/O resources
- In a process, a or more than one thread has:
- A thread execution state (Running, Ready, etc.)
- A saved thread context when not running
	- Independent program counter operating within a process
- An execution stack
- Some per-thread static storage for local variables
- Access to the memory and resources of its process, shared with all other threads in the process

![[Pasted image 20250130202309.png|400]]
- In a single-threaded process model there is no distinct concept of thread
- In a multithreaded environment, there is still a single process control block and user address space associated with the process, but are separate stacks for each thread, and separate control blocks for each thread containing registers values, priority, and other thread-related state information
![[Pasted image 20250130202328.png|500]]

- All of the threads of a process share the state and resources of that process
- When one thread alters data in memory, other threads see the result when they access that item
- Benefits of threads
	- Takes less time to create a new thread in an existing process, than to create a brand-new process
		- Thread creation is 10x faster than process creation
	- Takes less time to terminate
	- Takes less time to switch between two threads within the same process
	- Threads enhance efficiency in communication between different executing programs
		- Does not need kernel protection for tradition communication
		- Communicate between threads without invoking the kernel
- A files server can be implemented as a collection of threads
	- Server will handle many requests for file creation and termination in a short period of time
	- Faster to use threads and shared memory than processes and message passing
- Uses of threads in a single-user multiprocessing system
	- Foreground and background work
	- Asynchronous processing
		- Write RAM buffer to disk every minute (autosave in word)
	- Speed of execution
	- Modular program structure
		- Efficient for programs that share a variety of sources and destinations of input and output

- In an OS that support threads, scheduling and dispatching is done on a thread basis
- Most of the state information is maintained in thread-level data structures
- Suspension involved swapping the address space of one process out of main memory to make room for the address space of another process
	- All threads in a process shares the same address space, and therefor suspended at the same time
- Termination of a process terminates all threads within that process


**Thread Functionality**
- Threads have execution state that synchronize with one another

_THREAD STATES_
- Key states for a thread are Running, Ready, and Blocks
- If a process is swapped out, all of its threads are also swapped out
- 4 thread operations associated with a change in thread state
	- Spawn
		- New process is spawns, a thread is also spawned
		- A thread may also spawn another thread with an instruction pointer and arguments
	- Block
		- When a thread needs to wait for an event, it will block
	- Unblock
		- When the event of a blocked thread occurs, moved to Ready queue
	- Finish
		- When a thread completes, its register context and stacks are deallocated

- The image below performs to remove procedure calls $(RPCs)^2$ to two different hosts

![[Pasted image 20250130202414.png|600]]
![[Pasted image 20250130202428.png|600]]

- On a uniprocessor, multiprogramming enables the interleaving of multiple threads within multiple processes

_THREAD SYNCHRONIZATION_
- Any alteration of a resource by one thread affects the environments of the other threads in the same process
- Therefore, it is necessary to synchronize the activities of all the threads

4.2. <font style="color:#A8A8A8; text-decoration: underline;">Types of Threads</font>  
   
User-Level and Kernel-Level Threads
- 2 broad categories of thread implementation
	- User-level threads (ULTs)
	- Kernel-level threads (KLTs)
		- _Kernel-supported threads_ or _lightweight processes_

_USER-LEVEL THREADS_
- All of the work of thread management is done by the application and the kernel is not aware
- Any application can be programmed to be multithreaded by using a threads library, which is a package of routines for ULT management
- An application begins with a single thread and begins running in that thread
- This application and its thread are allocated to a single process managed by the kernel
- The application may spawn a new thread to run
- Control is passes to that utility by a procedure call
- The threads library create a data structure for the new thread and passes control to one of the threads within this process that is in the Ready state, using a scheduling algorithm
- The context consists of the content of user registers, the program counter, and stack pointers

![[Pasted image 20250130202454.png|600]]

- The following clarifies the relationship between thread scheduling and process scheduling
- (a to b) 
	- The application executing in thread 2 makes a system call that blocks B
	- This causes control to transfer to the kernel
	- The kernel invokes the I/O action, places process B in the Block state, and switches to another process
	- Thread 2 of process B is still in the Running state by the threads library
- (a to c)
	- A clock interrupt passes control to the kernel, and the kernel determines that the currently running process B has exhausted its time slice
	- The kernel places B in the Ready state and switches processes
	- Thread  of B is still in the Running state
- (a to d)
	- Thread 2 has reached a point it needs an action by thread 1 of process B
	- Thread 2 enters a block state and thread 1 transitions from Read to Running
	- The process B itself remains in the Running state
- A process may be in the midst of a thread switch from one thread to another when interrupted
- When that process is resumes. execution continues within the threads library, which completes the thread switch and transfers control to another thread within that process

![[Screenshot 2025-01-30 at 8.25.08 PM.png|600]]

- Advantages of the use of ULTs instead of KTLs
	- Thread switching does not requires kernel-mode privileges because management data structures are within the user address space of a single process
		- Saves overhead of two mode switches
	- Scheduling can be application specific
		- Applications can have their own scheduling algorithm
			- Round-robin
			- Priority-based
	- ULTs can run on only OS
- Disadvantages
	- In a typical OS, many system calls are blocking
		- All threads are blocked within the process when a system call is performed
	- In a pure ULT, a multithreaded application cannot take advantage of multiprocessing
		- A kernel can only assign one process to one processor at a time

- To avoid these problems, _jacketing_ is used
- Jacketing converts a blocking system call into a non-blocking system call
	- A thread calls an application-level I/O jacket routine
	- The jacket routine checks to determine if the I/O device is busy
	- If it is, the thread enters the Blocked state and passes control to another thread
	- Jacket routine checks until thread is Ready


_KERNEL-LEVEL THREADS_
- In a pure KLT, all of the work of thread management is done by the kernel
- There is no thread management code in the application level, simply an application programming interface (API) to the kernel thread
- Windows uses this approach
- Scheduling is done by the kernel on a thread basis
- This approach overcomes the 2 disadvantages of the ULT
- Kernel can simultaneously schedule multiple threads from the same process on multiple processors
- If one thread is blocked, the kernel can schedule another thread of the same process
- Another advantage is that the kernel routines can be multithreaded
- Disadvantage
	- The transfer of control from one thread to another requires a mode switch to the kernel

![[Pasted image 20250130202627.png|600]]
- Null fork: time to create, schedule, execute, and complete a process/thread that invokes the null procedure
	- Overhead of forking a process/thread
- Signal-Wait: time for a process/thread to signal a waiting process/thread and then wait on a condition
	- Overhead of synchronizing two processes/thread together

- Order or magnitude or more difference between ULTs and KLTs, and similarly between KLTs and processes
- There is a significant speedup by using KLT multithreading compared to single-threaded processes, similarly for ULTs
- If most of the thread switches in an application require kernel-mode access, then a ULT-based scheme may not perform much better than a KLT-based scheme

_COMBINED APPROACHES_
- Some OS use a combined ULT/KLT facility
- Thread creation is done completely in user spaces, as it is the bulk of the scheduling and synchronization of threads within an application
- Multiple ULTs from a single application are mapped onto some (smaller or equal) number of KLTs
- Multiple threads within the same application can run in parallel on multiple processors, and a blocking system call need not block the entire process
- This approach combines the advantages of the pure ULT and KLT, while minimizing the disadvantages
- Solaris uses this OS

**Other Arrangements**
- Resource allocation and dispatching traditionally have a 1 : 1 relationship between threads and processes
- Recently, many-to-one relationship have been used

![[Pasted image 20250130202651.png|600]]

_MANY-TO-MANY RELATIONSHIP_
- Used in experimental OS TRIX
- Uses concepts of domain and thread
- The domain is static, consisting of an address space and "ports" through which messages are sent and receivef
- A thread is a single execution path, with an execution stack, processor state, and scheduling information
- Multiple threads may execute in a single domain
- A single-user activity, or application may be performed in multiple domains
- Threads exists and can move from one domain to another

- An entire program implements as a single process
	- Entire program must be loaded into main memory
	- Some of the program is subject to swapping
	- This memory management also occurs in a two threaded system that has the same address space
- The main program and I/O subprogram can be implemented as two separate processes
	- Creates overhead of creating the subordinate process
	- If the I/O activity is frequent is consumes management resource, or creates and destroys the subprogram inefficiently
- The main program and the I/O subprogram is a single activity that is implemented as a single thread
	- An address for main program and I/O subprogram
	- The thread can be moved between two address spaces
	- OS can manage two address space independently; no overhead
	- I/O address space can be shared by other simple I/O programs
	- This is the most effective solution for some applications

_ONE-TO-MANY RELATIONSHIP_
- Distributed OS
	- Designed to control distributed computer systems, is a concept of a thread as primarily an entity that can move among address spaces
	- Clouds OS, and its kernel, RA
	- Emerald system
	- A thread in Clouds is a unit of activity from the user's perspective
	- A process is a virtual address space with an associated process control block
	- During creation, a thread starts executing a process by invoking a program in that process
	- Threads move from one address space to another, and span computer boundaries
	- Threads carry information, such as the controlling terminal, global parameters, and scheduling guidance
	- The Clouds approach provides an effective way of insulating both users and programmers from the details of the distributed environment

4.3. <font style="color:#A8A8A8; text-decoration: underline;">Multicore and Multithreading</font>  
- The use of a multicore system to support a single application with multiple threads is used in workstations, video game consoles, or personal computers

 **Performance of Software on Multicore**
- The potential performance benefits of a multicore organization depends on the ability to effectively exploit the parallel resources available
- Amdahl's law states:
![[Pasted image 20250203232435.png|700]]

- The law assumes a program in which a fraction $(1-f)$ of the execution time involves code that is inherently serial, and a fraction $f$ that is infinitely parallelizable with no scheduling overhead
- Small amounts of serial code have noticeable negative impact on performance
- If only 10% of cade in serial $f= 0.9$, running a program on a multicore system with 8 processors yields a performance gain of a factor of only 4.7
- Software incurs overhead as a result of communication and distribution of work to multiple processors and cache coherence overhead
- This results in a curve where performance peaks and then degrades because of the increase of overhead

![[Pasted image 20250130202718.png|500]]
- Although there are ways to exploit a multicore system
- A set of database application, where reducing serial fraction within hardware architectures, OS, middleware, and the software
- Many servers can use parallel multicore organization, because servers typically handle numerous relatively independent transactions in parallel

![[Pasted image 20250130202733.png|500]]

- In addition to general-purpose server software, a number of classes of applications benefit directly from the ability to scale throughput with the number of cores
	- Multithreaded native application
		- Small number of highly threaded processes
		- Lotus Domino
		- Siebel CRM (Customer Relationship Manager)
	- Multiprocess applications
		- Many single-threaded processes
		- Oracle database, SAP, and PeopleSoft
	- Java applications
		- Use threading in a fundamental way
		- Java language facilitates multithreaded applications
		- Java Virtual Machine is a multithreaded process that provides scheduling and memory management for Java Applications
		- Java applications benefit from multicore resources from servers such as Oracle's Java Application Server, BEA;s Weblogic, IBM's Websphere, and open-source Tomcat
	- Multi-instance application
		- Possible to gain scaling through multicore architecture by running multiple instance of an application in parallel
		- Virtualization technology can be used to provide application instance with separate and secure environments

**Application Example: Valve Game Software**
- Valve is an entertainment and technology company that has developed games, and Source Engine (a popular game engine)
- Valve has reprogrammed the Source engine software to use multithreading to exploit the power of multicore processor chips from Intel and AMD
- Threading Options:
	- Coarse threading
		- Individual modules, called systems, are assigned to individual processors
		- Rendering, AU, physics, have there own processor
		- Each module is single-threaded and is synchronized will all the threads
	- Fine-grained threading
		- Similar or identical tasks are spread across multiple processors
		- Loop that iterates over an array of data can be split up into a number of smaller parallel loops in individual threads that can be scheduled in parallel
	- Hybrid threading
		- Selective use of fine-grained threading for some systems, and single-threaded for other systems

- Valve uses coarse threading to achieve up to twice the performance across two processors compared to a single processor
- On real-world cases the improvement was a factor of 1.2
- Effective fine-grained threading was difficult
	- The time-per-work unit can be variable, and managing the timeline of outcomes was complex
- Hybrid threading was the most promising and scaled the best, with multicore system with 8 or 16 processors
- Systems that operate very effectively being permanently assigned to a single processor
	- Sound mixing

![[Pasted image 20250130202753.png|500]]
- Higher-level threads spawn lower-level threads as needed
- Key elements of the threading strategy for the rendering module:
	- Construct scene rendering list for multiple scenes in parallel
	- Overlap graphic simulations
	- Compute character bone transformations for all characters in all scenes in parallel
	- Allow multiple threads to draw in parallel
- Over 95% of the time, a thread is trying to read from a data set, and 5% is spent writing to a data set
- Concurrency mechanisms known as the _single-writer-multiple-reader_ models works effectively
   
4.4. <font style="color:#A8A8A8; text-decoration: underline;">Windows Process and Thread Management</font>  
- An application consists of one or more processes
- Each process provides the resources need to execute a program
- A process has a virtual address space, executable code, open handles to system objects, a security context, a unique process identifier, environment variables, a priority class, min and max working set sizes, and at least one thread of execution
- A process states with a single thread/primary thread, a can create addition threads
- A thread is the entity within a process that can be scheduled for execution
- All threads share a virtual address space and system resource
- A thread maintain exception handlers, a scheduling priority, thread local storage, a unique thread identifier, and a set of structures the system will use to save the thread context until it is scheduled
- On on multiprocessor, the system can execute as many threads as there are processors on the computer
- A job object allows groups of processes to be managed as a unit
- Job objects are namable, securable, sharable object that control attributes of the processes associated with them
- Operation affect all processes associated with the job object
	- Working set size
	- Process priority
	- Termination
- A thread pool is a collection of worker threads that efficiently execute asynchronous callbacks on behalf of the application
	- Use to reduce the number of application threads and provide management of worker threads
- A fiber is a unit of execution that must be manually scheduled by the application
- Fibers run in the context of the threads the schedule them
- Each thread can schedule multiple fibers
- Fiber do not provide overall advantages, but make it easiter to port applications
- A fiber assumes the identity of the thread that runs it
	- If a fiber calls the `ExitThread` function, the thread that is running exits
	- The only state information maintained is its stack, a subset of its registers, and the fiber data provided during fiber creations
- User-mode scheduling (UMS) is a lightweight mechanism that applications can use to schedule there own threads
- Each UMS thread has its own thread context instead of sharing the thread context of a single thread
- Ability to switch modes make UMS more efficient than thread pools for short-durations
- UMS is used with high performance requirements that have many threads concurrently on multiprocessor or multicore systems
- UMS must implement a scheduler component that manages application threads

**Management of Background Tasks and Application Lifecycles**
- Developers are responsible for managing the state of their individual application
- Windows Live Tiles give the appearance of applications constantly running on the system, but actual receive push notifications and do not use system resources
- Application, such as new feeds, may look at the data stamp associated with previous execution to resume its state
- With foreground tasks occupying all the system resources, starvation of background apps is a reality in Windows
- A background task API is built to allow apps to perform small tasks while not in the foreground
- App receive push notifications from a server
- Push notifications are template XML string managed through a cloud service, Windows Notification Service (WNS)
- API will queue request and process them when it receives enough processor resources

**The Windows Process**
- Windows processes are implemented as objects
- A process can be created as a new process or as a copy of an existing process
- An executable process may contain on or more threads

- Each process is assigned a security access token, called the primary token of the process
- When a user logs in, Windows creates an access token with a security ID
- Window users token to validate user to access secured objects, or to perform restricted functions on the system and on secured objects
- The process defines a virtual address space
- The process includes an object tables, with handles to other objects known to this process
	- Process has access to file objects and to section object that defines a section of shared memory

![[Pasted image 20250130202942.png|600]]

**Process and Thread Objects**
- Object-oriented structure of Windows facilitates the development of a general-purpose process facility
- A process is an entity corresponding to a user job or application that owns resources, such as memory and open files
- A thread is a dispatchable unit of work that executes sequentially and it interruptible, so the processor can turn to another thread
- When Windows creates a new process, it uses the object class, or type, defined for the Windows process as a template to generate a new object instance
- At the time of creation, attribute values are assigned
- A Window process must obtain at least one thread to execute
	- Thread processor affinity is the set of processors in a multiprocessor system that may execute this thread; this set is equal to or a subset of the process processor affinity

Windows Process Object Attributes

| Windows Process Object Attributes | Description                                                                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Process ID                        | A unique value that identifies the process to the operating system                                                                     |
| Security descriptor               | Describes who created an object, who can gain access to or use the object, and who is denied access to the object                      |
| Base priority                     | A baseline execution priority for the process's threads                                                                                |
| Default processor affinity        | The default set of processors on which the process's threads can run                                                                   |
| Quota limits                      | The maximum amount of paged and nonpaged system memory, paging file space, and processor time a user's processes can use               |
| Execution time                    | The total amount of time of all threads in the process have executed                                                                   |
| I/O counters                      | Variables that record the number and type of I/O operation that the process's threads have performed                                   |
| VM operating counters             | Variables that record the number and types of virtual memory operations that the process's threads have performed                      |
| Exception/debugging ports         | Interprocess communication channels to which the process manager sends a message when one of the process's threads causes an exception |
| Exit status                       | The reason for a process's termination                                                                                                 |

Windows Thread Object Attributes

| Windows Thread Object Attributes | Description                                                                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Thread ID                        | A unique value that identifies a thread when it calls a server                                                                               |
| Thread context                   | The set of register values and other volatile data that defines the execution state of a thread, enables threads to be suspended and resumed |
| Dynamic priority                 | The thread's execution priority at any given moment                                                                                          |
| Base priority                    | The lower limit of the thread's dynamic priority                                                                                             |
| Thread processor affinity        | The set of processors on which the thread can run, which is a subset or all of the processor affinity of the thread's process                |
| Thread execution time            | The cumulative amount of time a thread has executed in user mode and in kernel mode                                                          |
| Alert status                     | A flag that indicated whether a waiting thread may execute an asynchronous procedure call                                                    |
| Suspension count                 | The number of times the thread's execution has been suspended without being resumed                                                          |
| Impersonation token              | A temporary access token allowing a thread to perform operation on behalf of another process (used by subsystems)                            |
| Termination port                 | An interprocess communication channel to which the process manager sends a message when the thread terminates (used by subsystem)            |
| Thread exit status               | The reason for a thread's termination                                                                                                        |

**Multithreading**
- Windows support concurrency because threads in different processes may execute concurrently
- Multiple threads within the same process may be allocated to separate processors and execute simultaneously
- A multithreaded process achieves concurrency without the overhead of using multiple processes
- An object-oriented multithreaded process is an efficient way to implement a server application

**Thread States**
An existing Windows thread is in 1 of 6 states:
1) Ready
	1) Kernel dispatcher keeps track of all the ready threads and schedules them in priority order
2) Standby
	1) Selected to run next, waits until the processor is available
	2) If high enough priority, running thread is preempted in favour of the standby thread
3) Running
	1) Standby enter running state and begins execution
	2) Continues execution until it is preempted by a higher-priority thread, exhausts its time slice, blocks, or terminates
4) Waiting
	1) It is blocked on an event (I/O)
	2) It voluntarily waits for synchronization purposes
	3) An environment subsystem directs the thread to suspend itself
5) Transition
	1) After waiting if it is ready to run, but the resources are not available
		1) Thread stack is pages out of memory
6) Terminated
	1) By itself, another thread, of parent process


![[Pasted image 20250130204149.png|500]]


**Support for OS Subsystems**
- Responsible of each OS subsystem to exploit the Windows process and thread features to emulate the process and thread facilities of its corresponding OS
- Process creation begins with a request for a new process from an application
- The application issues a create-process request to the protected subsystem, which passes the request to the Executive
- The Executive create a process object and returns a handle for that object to to the subsystem
- In Win32, a new process must always be created with an initial thread
- When a new process is created it inherit many of its attributes from the creating process
- In Win32, the process creation is done indirectly
- An application client process issues its process creation request, the subsystem issues a requires to the Windows executive
- The new process then inherits the parent's access token, quota limits, base priority, and default processor affinity
   
4.5. <font style="color:#A8A8A8; text-decoration: underline;">Solaris Thread and SMP Management</font>  
- Solaris implements multilevel thread support designed to provide considerable flexibility in exploiting processor resources

**Multithreaded Architecture**
- Solaris makes use of 4 separate thread-related concepts
1) Process
	1) This is the normal UNIX process and includes the user's address space, stack, and process control block
2) User-level threads
	1) Implemented through a threads library in the address space of a process, invisible to the OS
	2) A user-level thread (ULT) is a user-created unit of execution within a process
3) Lightweight processes
	1) A lightweight process (LWP) is a mapping between ULTs and kernel threads
	2) Each LWP supports ULT and maps to one kernel thread
	3) LWPs are scheduled by the kernel independently, and may execute in parallel on multiprocessors
4) Kernel threads
	1) Fundamental entities that can be scheduled and dispatched to run on one of the system processors

- LWP is visible within a process to the application
- LWP data structures exist within their respective process address space
- LWP is bound to a single dispatchable kernel threads, and the data structure for that kernel thread is maintained within the kernel's address space
- A process may consist of a single ULT bound to a single LWP
	- Single thread of execution, corresponding to a traditional UNIX process
- If an application requires concurrency, its process contains multiple threads, each bound to a single LWP

**Motivation**
- The 3-level thread structure (ULT, LWP, kernel thread) in Solaris is intended to facilitate thread management by the OS and to provide a clean interface to applications
- A LWP is bound to a kernel thread with a one-to-one correspondence in execution states
- Concurrency and execution are managed at the level of the kernel thread
- Applications have access to hardware through system calls
- The API allows the user to invoke kernel service to perform privileged tasks
	- Read/write to a file
	- Issue a control command to a device
	- Create a new process or thread
	- Allocate memory for the process to use


![[Pasted image 20250130204209.png|500]]

**Process Structure**
- Process structure on a typical UNIX implementation
	- Process ID
	- User ID
	- signal dispatch table
	- File descriptors
	- Memory map
	- Processor state structure
- Solaris retains this basic structure but replaces the processor state block with a list of structures containing one data block for each LWP
	- A LWP identifier
	- The priority of this LWP and and hence the kernel thread that supports it
	- A signal mask that tells the kernel which signal will be accepted
	- Saved values of user-level registers (when LWP is not running)
	- The kernel stack for this LWP
		- Includes system call arguments, results, and error codes for each call level
	- Resource usage and profiling data
	- Pointer to the corresponding kernel thread
	- Pointer to the process structure


![[Pasted image 20250130204229.png|600]]

**Thread Execution**
- Execution states reflect that status of both kernel thread and the LWP bound to it
- Some kernel threads are not associated with an LWP, the same execution diagram applies
States
- RUN: the thread is runnable; the thread is ready to execute
- ONPROC: thread is executing on a processor
- SLEEP: the thread is blocks
- STOP: thread is stopped
- ZOMBIE: thread has terminated
- FREE: thread resource have been released and the thread is awaiting removal from the OS thread data structure

- A thread moves from ONPROC to RUN if it is preempted by a higher-priority thread or because of time slicing
- A thread moves fro ONPROC to SLEEP if it is blocks and must await an event to return the RUN state
- Blocking occurs if the thread invokes a system call and waits
- A thread enters STOP if process stopped

![[Pasted image 20250130204244.png|600]]


**Interrupt as Threads**
- Most OS contain 2 fundamental forms of concurrent activity
	- Process
	- Interrupts
- Solaris unifies these two concepts into a single model, namely kernel threads, and the mechanisms for scheduling and executing kernel threads
- Interrupts are converted to kernel threads
	- Reduces overhead

Solaris
- Solaris employs a set of kernel threads to handle interrupts
- The kernel controls access to data structures and synchronizes among interrupts threads using mutual exclusion primitives
- Interrupt threads are assigned higher pirouettes than all other types of kernel threads

- When an interrupt occurs, it is delivered to a processor and the thread is pinned
- A pinned thread cannot move to another processor, and begins to execute

4.6. <font style="color:#A8A8A8; text-decoration: underline;">Linux Process and Thread Management</font>  
- A process, or task, in Linux is represented by a `task_struct` data structure
- The `task_struct` contain the following information:
	- State
		- The execution state of the process (executing, ready, suspended, stopped, zombie)
	- Scheduling information
		- Information needed by the Linux schedule processes
		- Process: Normal or real time
		- Real-time are scheduled before normal
		- A reference counter keeps track of the amount of time a process is allowed to execute
	- Identifiers
		- Each process has a unique process identifier (PID) and has user and group identifiers
		- A group identifier is used to assign resource access privileges to a group of processes
	- Interprocess communication
		- Linux support the IPC mechanisms found in UNIX SVR4
	- Links
		- Each process includes a link to its parent, sibling, and children processes
	- Times and timers
		- Includes process creation time and amount of processor time so far consumed
		- Have one or more interval timers and may be single or periodic
	- File system
		- Includes pointers to any files opened by this processes, as well as pointers to the current and the root directories for this process
	- Address space
		- Virtual address space
	- Processor-specific context
		- Registers and stack information that constitute the context of this process
	- Running
		- State value has two states (Running or Ready)
	- Interruptible
		- A blocked state, in which the process is waiting for an event
	- Uninterruptible
		- Another blocked state
		- A process is waiting directly on hardware conditions and therefore will not handle any signals
	- Stopped
		- A halted process, resumed by positive action from another process
			- Debugged process
	- Zombie
		- Terminates, but sill has its task structures in the process table

![[Pasted image 20250130205443.png|600]]

**Linux Threads**
- Tradition UNIX system support single thread of execution per process, while modern provide multiple kernel-level thread per process
- Older version of Linux kernel offer no support for multithreading
- Linux does not recognize between threads and processes
- ULTs are mapped into kernel-level processes
- Multiple user-level threads that make a single user-level are mapped into Linux kernel-level processes that share the same group ID
	- Shares resources and avoid the need for context switch
- A new process is created in Linux by copying the attributes of the current process
- A new process is _cloned_, although no separate type of data structure is defined
- In place of a `fork()`, processes are created in Linux using `clone()`
	- Includes a set of flags as arguments

Clone flags:
- `CLONE_NEWPID`: Creates new process ID namespace
- LONE_PARENT: Caller and new task share the same parent process
- `CLONE_SYSVSEM`: Shares System V `SEM_UNDO` semantics
- `CLONE_THREAD`: Inserts this process into the same thread group of the parent
	- If flag is true, it implicitly enforces `CLONE_PARENT`
- `CLONE_VM`: Shares the address space
- `CLONE_FS`: Shares the same filesystem information (current working directory, root of filesystem, and the unmask)
- `CLONE_FILES`: Shares the same file descriptor table
	- Changing the associated flags of a file descriptor using the `fcntl()` system call

- When a context switch is performed, is checks if the current page directory to the scheduled one
	- If they are the same, context switch is just a jump from one location of code to another location of code
	- `clone()` call create separate stack spaces for each process

**Linux Namespaces**
- A namespace enables a process to have different view of the system than other processes that have other associated namespaces
- Namespaces and cgroups are the basis of Linux lightweight virtualization
	- Pretends that a group of process are the only processes on the system
	- Used by Linux Containers
	- 6 namespaces: mnt, pid, net, pic, utc, and user

6 Namespaces:
`CLONE_NEWNS
`CLONE_NEWPID`
`CLONE_NEWIPC`
`CLONE_NEWUTS`
`CLONE_NEWUSER``

- Namespaces are created by the `CLONE()` system call
- A process can also create a namespace with the `unshare()` system call with one of the 6 flags
	- A new process is not created, only a new namespace is created, which is attached to the calling process

_MOUNT NAMESPACE_
- A mount namespace provides the process with a specific view of the filesystem hierarchy

_UTS NAMESPACE_
- UTS (UNIX timesharing) namespace is related to the uname Linux system call
	- Returns the name and information about the current kernel
		- Nodename
			- System name
		- Domainname
			- NIS domain name
- NIS (Network Information Service) is a standard scheme used on all major UNIX systems
- System administrator sets up MIS client systems with only minimal configuration data and add, remove, or modify configuration data from a single location

_IPC NAMESPACE_
- AN PIC namespace isolated certain interprocess communication resources
	- Semaphores
	- POSIX message queues
- Concurrency mechanisms can be employed by the programmer that enable IPC among processes that share the same IPC namespace

_PID NAMESPACE_
- PID namespace isolate the process ID space, so processes in different PID namespaces can have the same PID
- This feature is used for Checkpoint/Restore In Userspace (CRIU), a Linux software tool
- Freeze running application and checkpoint it to a hard drive as a collection of files
- Restore and run the application from the freeze point on that machine or on a different host
- CRIU is mainly implemented in the user space

_NETWORK NAMESPACE_
- Provides isolation of the system resources associated with networking
- Each networking namespace has its own network devices, IP addresses, IP routing tables, port numbers, etc
- Namespaces virtualize all access to network resources
- Each process or group has access to this network
- A network device belongs to only one network namespace

_USER NAMESPACE_
- Provides a container with its own set of UIDs, separate from the parents
- The cloned process can have access to and privileges of the parent

_THE LINUX CGROUP SUBSYSTEM_
- The Linux cgroup subsystem, together with namespace subsystem are the basis of lightweight process virtualization; they form the basis of Linux containers
- Almost every Linux contains project (Docker, LXC, Kubernetes, etc.) is based on both of them
- In Linux cgroup subsystems provide resource management and account
- It handles resources such as CPU, network, memory, and more
- Needed in both ends of the spectrum (embedded devices and servers, less in desktops)
- Development of cgroups was starts in 2006 by engineers at Google
- In order to use the cgroups filesystem (brose, attach tasks to cgroups, and so on), it first must be mounted, like when working with any other filesystem
- Mounting is the way an operating system connects a storage device to the overall file system, allowing users to read, write, and manage data on that device
- The cgroup filesystem can be mounted on any path, and many userspace applications and container projects use `/sys/fs/cgroup` as a mounting point
- After mounting, you can create subgroups, attach processes and tasks to these groups, set limitations on various system resources, and more
   
4.7. <font style="color:#A8A8A8; text-decoration: underline;">Android Process and Thread Management</font>  
   
**Android Applications**
- An Android application is the software that implements an app
- Each application consists of one or more instance of one or more of 4 types of application components
- Each component performs a distinct role in the overall application, and can be activated independently within the application and even by other applications

4 Types of Components
- Activities
	- An activity corresponds to a single screen visible as a user interface
		- E-mail application
			- New emails
			- Compose an e-mail
			- Reading e-mail
		- Each one is independent of the others
	- Android makes a distinction between internal and exported activities
- Services
	- Services are used to perform background operations that take a considerable amount of time to finish
	- This ensures faster responsiveness, for the main thread (UI)
	- Services that run for the entire lifetime of the Android system
		- Power Manager
		- Battery
		- Vibrator services
	- These system services create threads that are part of the System Server process
- Content providers
	- A content provider acts as an interface to application data that can be usd by the application
	- Category of managed data
		- Private data
		- Shared data
- Broadcast receivers
	- A broadcast receiver responds to system-wide broadcast announcements
	- A broadcast can originate from another application
		- Data has been downloaded
		- Batter from headphone is running low

- Each application runs on its own virtual machines and its own single process that encompasses the application and its virtual machine
- This approach is referred to as _sandboxing model_, that isolates each application
- One application cannot access the resources of the other without permission being granted

**Activities**
- An Activity is an application component that provides a screen with which users can interact in order to do something
	- Phone call
	- Take a phote
	- Send an e-mail
	- View a map
- Each activity is given a window in which to draw its user interface
- The window fills the screen, or may float on top of other windows
- An application may include multiple activities
- One activity is in the foreground, and interacts with the user
- Activities are arranges last-in-first-out stack, _back stack_, in the order in which each activity is opened
- If the user switches to another activity within the application, the new activity is created and pushed on to the top, while the preceding foreground activity becomes the second item
- The user can back up to the most recent foreground by pressing a 'back button'

![[Pasted image 20250130205500.png | 400]]


_ACTIVITY STATES_
- Below is a state transition diagram of an activity
- When a new activity is launches, the software performs a series of API calls to the Activity Manager: 
`onCreate()` does the static set of the activity, include any data structure initialization
`onStart()` makes the activity visible
`onResume()` passes control to the activity so the user input goes in to the activity
- The Resumed state is referred to as _foreground lifetime_
- Activity is in from of all other activities on screen and has user input focus
`onPause()` places the currently running activity on the stack, Paused state
`onStop(0)` stops the activity, invoked by a back button, or closing of a window
- The activity on top of the stack is resumed
- the Resumed and Paused state together are the _visible lifetime_
	- User can see the activity on-screen and interact with it
- When the user goes to the Home screen, the currently running activity is paused and then stopped
- When the user resumes this application, the stopped activity, on the top of the back stack restarts and becomes the foreground activity

![[Pasted image 20250130205520.png | 400]]

_KILLING AN APPLICATION_
- System will reclaim memory by killing one or more activities within an application. and also terminate the process, although the application itself still exists
- If the user returns to the application, it is necessary for the system to recreate any killed activities as they are invoked
- The system kills applications in a stack-oriented style
	- Kill recently used apps first
- Apps with foreground services are unlikely to be killed

**Processes and Thread**
- The default allocation of processes and threads to an application is a single process and single thread
- All of the components of the application run on the single thread of the single process for that application
- To avoid slowdown, multiple threads within a process or processes are within an application
- System may kill one or more processes to reclaim memory
- Every process exists at a particular level at a given time, and processes are killed beginning with the lowest precedence first

Level of hierarchy (descending in higher order or precedence):
- Foreground process
	- Process that is requires for the user in the current moment
	- More than one process
		- Host process of the activity
		- Host process of a service
- Visible process
	- A process that hosts a component that is not in the foreground, but still visible to the user
- Service process
	- A process running a service that does not fall into either of the higher categories
		- Playing music in the background
		- Downloading data on the network
- Background process
	- A process hosting an activity in the Stopped state
- Empty process
	- A process that does not hold any active application components
		- For caching purposes to improve startup time when needed next
   
   
4.8. <font style="color:#A8A8A8; text-decoration: underline;">Mac OS X Grand Central Dispatch</font>  
- Mac OS x Grand Central Dispatch (GCD) provides a pool of available threads
- Programmers designate portions of applications, called blocks, that can be dispatched independently and run concurrently
- OS will provide as much concurrency as possible based on cores available and the thread capacity of the system
- A block is a simple extension to C/C++
`x = ^{printf(hello world\n");}`

- Defines x as a way of calling the function, so that invoking the function `x()` will print `hello world`
- Blocks enable the programmer to encapsulate complex function with their arguments and data, to be easily references and passed like variables
- Blocks are scheduled and dispatched by queues
- GCD uses queues to describe concurrency, serialization, and callbacks
- Queues are lightweight user-space data structures, and are more efficient than manually managing threads and locks
- GCD treads blocks as potentially concurrent, or as serial activities
- The use of predefined threads saves the cost of creating a new thread for each request, reducing the latency with processing a block
- Thread pools are automatically sized by the system to maximize the performance of the application while minimizing the number of idle of competing threads
- The application can associate a single block and queue with an even source, such as a timer, network socket, or file descriptor
- This allows rapid response without he expense of polling of "parking a thread" on the event source

Ex. Document-based application with a button that, when clicked, will analyze the current document and display some interesting statistics

```swift
(Inaction)analyzeDocument:(NSButton *)sender
{ 
	NSDictionary *stats = [myDoc analyze];
	[myModel setDict:stats];
	[myStatsView setNeedsDisplay:YES];
	[stats release]; 
}
```

- First line analyzes document
- Second updates application internal state
- Third tells the application the statistics need to be updates
- If the document is large and complex, the analyze step will take to long, and response time will be slow
- All functions in GCD begin with `dispatch_`
- With GCD, the task is put on a global concurrent queue
	- Block can be assigned to a separate concurrent queue, off the main queue, and execute in parallel
- In the main thread, the inner `dispatch_async()` call is encountered
- This directed the OS to put the following block of code at the end of the main queue, to be executed when it reaches the head of the queue

```Swift
(IBAction)analyzeDocument:(NSButton *)sender 
	{dispatch_async(dispatch_get_global_queue(0, 0), ^{ 
		NSDictionary *stats = [myDoc analyze]; 
		dispatch_async(dispatch_get_main_queue(), ^{ 
			[myModel setDict:stats]; 
			[myStatsView setNeedsDisplay:YES]; 
			[stats release]; 
		}); 
	}); 
}
```

   
4.10. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| application<br>fiber<br>jacketing<br>job object<br>kernel-level thread<br>lightweight process<br>message<br>multithreading | namespaces<br>port<br>process<br>task<br>thread<br>thread pool<br>user-level thread<br>user-mode scheduling (UMS) |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |

![[Pasted image 20250130205617.png|500]]



## 5. <font style="color:#FCD12A">Concurrency: Mutual Exclusion and Synchronization</font>  
^e78bc7

Central themes of OS:
- Multiprogramming
	- The management of multiple processes within a uniprocessor system
- Multiprocessing
	- The management of multiple processes within a multiprocessor
- Distributed processing
	- The management of multiple processes executing on multiple, distributed computer systems
	- The recent proliferation of clusters is prime example of this type of system

Concurrency arises in 3 different context:
- Multiple applications
	- Multiprogramming was invested to allow processing time to be dynamically shared among a number of active applications
- Structured applications
	- As an extension of the principles of module design and structural programming, some applications can be effectively programmed as a set of concurrent processes
- Operating system structure
	- The same structuring advantages apply to systems programs





| Key Terms Related to Concurrency | Description                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Atomic operation                 | A function or action implemented as a sequence of one or more instructions that appears to be indivisible; no other process can see an intermediate state or interrupt the operation. The sequence of instruction is guaranteed to execute as a group, or not execute at all, having no visible effect on system state. Atomicity guarantees isolation form concurrent processes |
| Critical section                 | A section of code within a process that request access to shared resources, and that must not be executed while another process is in a corresponding section of code                                                                                                                                                                                                            |
| Deadlock                         | A situation in which two or more processes are unable to proceed because each is waiting for one of the others to do something                                                                                                                                                                                                                                                   |
| Livelock                         | A situation in which two or more processes continuously change their states in response to changes in the other process(es) without doing any useful work                                                                                                                                                                                                                        |
| Mutual exclution                 | The requirement that when one process is in a critical section that accesses shared resources, no other process may be in a critical section that accesses any of those shared resouces                                                                                                                                                                                          |
| Race condition                   | A situation in which multiple threads or processes read and write a shared data item, and the final result depends on the relative timing of their execution                                                                                                                                                                                                                     |
| Starvation                       | A situation in which a runnable process is overlooked indefinitely by the scheduler;  although it is able to proceed, it is never chosen                                                                                                                                                                                                                                         |



5.1 <font style="color:#A8A8A8; text-decoration: underline;">Mutual Exclusion: Software Approaches</font>  

Dekker's Algorithm

_FIRST ATTEMPT_

_SECOND ATTEMPT_

_THIRD ATTEMPT_

_FOURTH ATTEMPT_

Peterson's Algorithm


   
   
5.2 <font style="color:#A8A8A8; text-decoration: underline;">Principle of Concurrency</font>  
   
**A Simple Example**

**Race Condition**

**Operating System Concerns**

**Process Interaction**

_COMPETITION AMONG PROCESSES FOR RESOURCES_


![[Pasted image 20250130205758.png]]

_COOPERATION AMONG PROCESSES BY SHARING_

_COOPERATION AMONG PROCESSES BY COMMUNICATION_

**Requirements for Mutual Exclusion**
   
5.3 <font style="color:#A8A8A8; text-decoration: underline;">Mutual Exclusion: Hardware Support</font>  
   
**Interrupt Disabling**

**Special Machine Instructions**

_COMPARE&SWAP INSTRUCTION_

_EXCHANGE INSTRUCTION_

_PROPERTIES OF THE MACHINE-INSTRUCTION APPROACH_


   
5.4 <font style="color:#A8A8A8; text-decoration: underline;">Semaphores</font>  


| Common Concurrency Mechanisms | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Semaphore                     | An interger value used for signalling among processes. Only three operation may be performed on a semaphore, all of which are atomic: initialize, decrement, and increment. The decrement operation may result in the blocking of a process, and the increment operation may result in the unblocking of a process. Also known as a counting semaphore or a general semaphore                                 |
| Binary semaphore              | A semaphore that takes on only the values 0 and 1                                                                                                                                                                                                                                                                                                                                                             |
| Mutex                         | Similar to a binary semaphore. A key difference between the two is that the process that locks the mutex (sets the value to 0) must be the one to unlock it (sets the value to 1)                                                                                                                                                                                                                             |
| Condition Variable            | A data type that is used to block a process or thread until a particular condition is true                                                                                                                                                                                                                                                                                                                    |
| Monitor                       | A programming langauge construct that encapsulates variables, access procedures, and intialization code within an abstract data type. The monitor;s variable may only be accessed via its access procedures and only one process may be actively accessing the monitor at only one time. The access procedures are _critical sections_. A monitor may have a queue of processes that are waiting to access it |
| Event flag                    | A memory work used as a synchronization mechanism. Application code may associate a different event with each bit in a flag. A thread can wait for either a single event or a combination of event by checking one or multiple bits in the corresponding flag. The thread is blocked until all of the required bits are set (AND) or until at least one of the bits is set (OR)                               |
| Mailboxes/messages            | A means for two processes to exchange information and that may be used for synchronization                                                                                                                                                                                                                                                                                                                    |
| Spinlocks                     | Mutual exclusion mechanism in which a process executes in an infinite loop waiting for the value of a lock variable to indicate availability                                                                                                                                                                                                                                                                  |





![[Pasted image 20250130210001.png|500]]

**Mutual exclusion**

![[Pasted image 20250130210020.png|500]]




**The Producer/Consumer Problem**
![[Pasted image 20250130210052.png|400]]
![[Pasted image 20250130210109.png|500]]
![[Pasted image 20250130210129.png|400]]


**Implementation of Semaphores**


   
   
5.5 <font style="color:#A8A8A8; text-decoration: underline;">Monitors</font>  
   
**Monitor with Signal**





![[Pasted image 20250130210154.png|500]]
**Alternate Model of Monitors with Notify and Broadcast**


   
   
6. <font style="color:#A8A8A8; text-decoration: underline;">Message Passing</font>  

**Synchronization**

**Addressing**



![[Pasted image 20250130210219.png]]
**Message Format**
![[Pasted image 20250130210232.png|300]]
**Queuing Discipline**

**Mutual Exclusion**
   
   
5.7 <font style="color:#A8A8A8; text-decoration: underline;">Readers/Writers Problems</font>  
   
**Reader Have Priority**

**Writers Have Priority**

![[Pasted image 20250130210331.png|600]]


5.8 <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
   



   
9. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>
   

| Atomic<br>binary semaphore<br>blocking<br>busy waiting<br>concurrency<br>concurrent processes<br>condition variable<br>coroutine<br>counting semaphore<br> | critical resource<br>critical section<br>deadlock<br>direct addressing<br>general semaphore<br>indirect addressing<br>livelock<br>message passing<br>monitor | mutual exclusion<br>mutual exclusion lock (mutex)<br>nonblocking<br>race condition<br>semaphore<br>spin waiting<br>starvation<br>strong semaphore<br>weak semaphore |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |


   

## 6. <font style="color:#FCD12A">Concurrency: Deadlock and Starvation</font>  

^17da49

6.1 <font style="color:#A8A8A8; text-decoration: underline;">Principles of Deadlock</font>  

![[Pasted image 20250130210406.png|500]]

![[Pasted image 20250203144406.png|600]]
![[Pasted image 20250203144425.png|600]]

**Reusable Resources**

**Consumable Resources**


3 Approaches when dealing with deadlock:
- Deadlock prevention
	- Disallow one of the three necessary condition for deadlock occurrence, or prevent circular wait condition from happening
- Deadlock avoidance
	- Do not grant a resource request if this allocation might lead to deadlock
- Deadlock detection
	- Grant resource request when possible, but periodically check for the presence of deadlock and take action to recover

**Resource Allocation Graphs**



![[Pasted image 20250203144451.png|600]]
![[Pasted image 20250203144512.png|500]]

**The Conditions for Deadlock**

3 Conditions of policy must be present for a deadlock to be possible:
- Mutual exclusion
	- Only one process may use a resource at a time
	- No process may access a resource unit that has been allocated to another process
- Hold and wait
	- A process may hold allocated resources while awaiting assignment of other resources
- No preemption
	- No resource can be forcibly removed from a process holding it
   
6.2 <font style="color:#A8A8A8; text-decoration: underline;">Deadlock Prevention</font>  

**Mutual Exclusion**

**Hold and Wait**

**No Preemption**

**Circular Wait**


- Possible of Deadlock
	- Mutual exclution
	- No preemption
	- Hold and wait
- Existence of Deadlock
	- Mutual exclusion
	- No preemption
	- Hold and wait
	- Circular wait

   
6.3 <font style="color:#A8A8A8; text-decoration: underline;">Deadlock Avoidance</font>  
   
**Process Initiation Denial**

![[Pasted image 20250203144538.png|600]]

**Resource Allocation Denial**
![[Pasted image 20250203144603.png|600]]
![[Pasted image 20250203144619.png|600]]
   
6.4 <font style="color:#A8A8A8; text-decoration: underline;">Deadlock Detection</font>  
   
**Deadlock Detection Algorithm**

**Recovery**
![[Pasted image 20250203144640.png|600]]

   
6.5 <font style="color:#A8A8A8; text-decoration: underline;">An Integrated Deadlock Strategy</font>  
   

- Swappable space
	- Blocks of memory on secondary storage for use in swapping processes
- Process resource
	- Assignable devices, such as tape drivers, and files
- Main memory
	- Assignable to processes in pages or segments
- Internal resources
	- Such as I/O channels

- 
   
6.6 <font style="color:#A8A8A8; text-decoration: underline;">Dining Philosophers Problem</font>  


![[Pasted image 20250203144659.png|400]]

**Solution Using Semaphore**

**Solution Using a Monitor**


   
   
6.7 <font style="color:#A8A8A8; text-decoration: underline;">UNIX Concurrency Mechanisms</font>  
   
**Pipes**

**Messages**

**Shared Memory**

**Semaphores**

**Signals**




   
6.8 <font style="color:#A8A8A8; text-decoration: underline;">Linux Kernel Concurrency Mechanisms</font>  

**Atomic Operations**

**Spinlocks**


![[Screenshot 2025-02-03 at 2.47.20 PM.png|600]]
![[Pasted image 20250203144753.png|600]]
![[Pasted image 20250203144812.png|600]]


_BASIC SPINLOCKS_

_READER-WRITER SPINLOCK_

**Semaphores**

![[Pasted image 20250203144851.png|600]]



_BINARY AND COUNTING SEMAPHORES_

_READER-WRITER SEMAPHORES_

Barriers

![[Pasted image 20250203144910.png|600]]


_RCU (READ-COPY-UPDATE)_


   
6.9 <font style="color:#A8A8A8; text-decoration: underline;">Solaris Thread Synchronization Primitives</font> 

1) Mutual exclusion (mutex) locks
2) Semaphores
3) Multiple readers, single writer (reader/writer) locks
4) Condition variables

![[Pasted image 20250203144943.png|500]]


**Mutual Exclusion Lock**

**Semaphores**

**Readers/Writer Lock**

**Condition Variables**

 
6.10 <font style="color:#A8A8A8; text-decoration: underline;">Windows Concurrency Mechanisms</font>  

**Wait Functions**

**Dispatcher Objects**
![[Pasted image 20250203145001.png|600]]


**Critical Sections**

**Slim Reader-Writer Locks and Condition Variables**

**Lock-free Synchronization**




6.11 <font style="color:#A8A8A8; text-decoration: underline;">Android Interprocess Communication</font>  

![[Pasted image 20250203145022.png|500]]

6.12 <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  



6.13 <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| banker's algorithm<br>circular wait<br>consumable resource<br>deadlock<br>deadlock avoidance<br>deadlock detection<br>deadlock prevention | fatal region<br>hold and wait<br>joint process diagram<br>memory barrier<br>message mutual exclusion<br>pipe | preemption<br>resource allocation graph<br>reusable resource<br>safe state<br>spinlock<br>starvation<br>unsafe state |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |






## 7. <font style="color:#FCD12A">Memory Management</font>  
^17a4c6

1. <font style="color:#A8A8A8; text-decoration: underline;">Memory Management Requirements</font>  



| Memory Management Terms | Description                                                                                                                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frame                   | A fixed-length block in main memory                                                                                                                                                                                                                                                                 |
| Page                    | A fixed-length block of data that resides in secondary memory (disk). A page of data may temporarily be copied into a frame of main memory                                                                                                                                                          |
| Segment                 | A variable-length block of data that resides in secondary memory<br>An entire segment may temporarily be copies into an available region of main memory (segmentation) or the segment may be divided into pages, which can be individual copies into main memory (combined segmentation and paging) |
|                         |                                                                                                                                                                                                                                                                                                     |


Relocation
![[Pasted image 20250203145352.png|450]]



Protection

Sharing

Logical Organization

Physical Organization


   
2. <font style="color:#A8A8A8; text-decoration: underline;">Memory Partitioning</font>  

Fixed Partitioning

_PARTITION SIZES_

_PLACEMENT ALGORITHM_

Dynamic Partitioning
![[Pasted image 20250203145432.png|600]]

_PLACEMENT ALGORITHM_
![[Pasted image 20250203145449.png|400]]
![[Pasted image 20250203145529.png|500]]

![[Pasted image 20250203145549.png|500]]
_REPLACEMENT ALGORITHM_

Buddy System




![[Pasted image 20250203145607.png|600]]
Relocation

![[Pasted image 20250203145623.png|500]]   ![[Pasted image 20250203145636.png|500]]
   
3. <font style="color:#A8A8A8; text-decoration: underline;">Paging</font>  
   ![[Pasted image 20250203145707.png|400]]   
   ![[Pasted image 20250203145724.png|400]]
   ![[Pasted image 20250203145749.png|500]]
1. <font style="color:#A8A8A8; text-decoration: underline;">Segmentation</font>  
   
   
   
5. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
   
   
   
6. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>  

| absolute loading<br>buddy system<br>compaction<br>dynamic linking<br>dynamic partitioning<br>dynamic run-time loeading<br>external fragmentation<br>fixed partitioning<br>frame<br>internal fragmentation | linkage editor<br>linking<br>loading<br>logical address<br>logical organization<br>memory management<br>page<br>page table<br>paging | partitioning<br>physical address<br>physical organization<br>protection<br>relative address<br>relocatable loading<br>relocation<br>segment<br>segmentation<br>sharing |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   
7. <font style="color:#A8A8A8; text-decoration: underline;">Loading and Linking</font>
![[Pasted image 20250203150036.png|400]]
![[Pasted image 20250203150053.png|500]]

**Loading**



_ABSOLTE LOADING_


![[Pasted image 20250203150129.png|600]]

_RELOCATABLE LOADING_

![[Pasted image 20250203150159.png|600]]


_DYNAMIC RUN-TIME LOADING_

**Linking**


![[Pasted image 20250203150223.png|600]]

_LINKAGE EDITOR_

_DYNAMIC LINKER_



## 8. <font style="color:#FCD12A">Virtual Memory</font>  

^6ee6a5

8.1. <font style="color:#A8A8A8; text-decoration: underline;">Hardware and Control Structures</font>  


| Virtual Memory Terminology | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Virtual memory             | A storage allocation scheme in which secondary memory can be addressed as thought it were part of main memory. The addresses a program may use reference memory are distinguished from the addresses the memory system uses to identity physical storages sites, and program-generated addresses are translated automatically to the corresponding machines addresses. The size of virtual storage is limited by the addressing scheme of the computer system, and by the amount of secondary memory available and not by the actual number of main storage locations |
| Virtual address            | The address assigned to a location in virtual memory to allow that location to be accessed as though it were part of main memory                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Virtual address space      | A virtual storage assigned to a process                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Address space              | A range of memory addresses available to a process                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Read address               | The address of a storage location in main memory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |



**Locality and Virtual Memory**

![[Pasted image 20250204010837.png|700]]


**Paging![[Pasted image 20250204010945.png]]**
![[Pasted image 20250204010856.png|600]]
![[Pasted image 20250204010914.png|600]]






_PAGE TABLE STRUCTURE_

![[Pasted image 20250204010931.png|600]]


_INVERTED PAGE TABLE_
![[Pasted image 20250204010946.png|600]]
![[Pasted image 20250204011037.png|600]]


_TRANSLATION LOOKASIDE BUFFER_

![[Pasted image 20250204011056.png|600]]
![[Pasted image 20250204011119.png|600]]
![[Pasted image 20250204011139.png|600]]
_PAGE SIZE_




![[Pasted image 20250204011203.png|500]]
![[Pasted image 20250204011216.png|450]]


**Segmentation**

_VIRTUAL MEMORY IMPLICATIONS_

_ORGANIZATION_



![[Pasted image 20250204011304.png|600]]

**Combined Paging and Segmentation**




![[Pasted image 20250204011325.png|600]]
**Protection and Sharing**


   
   
8.2 <font style="color:#A8A8A8; text-decoration: underline;">Operating System Software</font>  

![[Pasted image 20250204011343.png|500]]
![[Pasted image 20250204011357.png|500]]

Fetch Policy

Placement Policy

Replacement Policy

_FRAME LOCKING_

_BASIC ALGORITHMS_

![[Pasted image 20250204011425.png|600]]





![[Pasted image 20250204011502.png|500]]

![[Pasted image 20250204011520.png|400]]


_REPLACEMENT POLICY AND CACHE SIZE_

Resident Set Management

_RESIDENT SET SIZE_




_REPLACEMENT SCOPE_



![[Pasted image 20250204011609.png|600]]
_FIXED ALLOCATION, LOCAL SCOPE_

_VARIABLE ALLOCATION, GLOBAL SCOPE_

_VARIABLE ALLOCATION, LOCAL SCOPE_

![[Pasted image 20250204011639.png|500]]

![[Pasted image 20250204011659.png|500]]


Cleaning Policy

Load Control

_MULTIPROGRAMMING LEVEL_
![[Pasted image 20250204011721.png|400]]



_PROCESS SUSPENSION_


   
8.3 <font style="color:#A8A8A8; text-decoration: underline;">UNIX and Solaris Memory Management</font>  

**Paging System**

_DATA STRUCTURES_

_PAGE REPLACEMENT_

Kernel Memory Allocator

   
8.4 <font style="color:#A8A8A8; text-decoration: underline;">Linux Memory Management</font>  
   
Linux Virtual Memory 
_VIRTUAL MEMORY ADDRESSING_

_PAGE ALLOCATION_

_PAGE REPLACEMENT ALGORITHM

**Kernel Memory Allocation**

   
8.5 <font style="color:#A8A8A8; text-decoration: underline;">Windows Memory Management</font>  
   
**Window Memory Management**
**Windows Paging**
**Windows Swapping**
   
8.6 <font style="color:#A8A8A8; text-decoration: underline;">Android Memory Management</font>  
   
   
   
7. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
   
   
   
8. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| associative mapping<br>demand paging<br>external fragmentation<br>fetch policy<br>hash table<br>hashing<br>internal fragmentation<br>locality | page<br>page fault<br>page placement policy<br>page table<br>paging<br>prepaging<br>real memory<br>resident set<br>resident set management | segment<br>segment table<br>segmentation<br>slab allocation<br>thrashing<br>translation lookaside bugger (TLB)<br>virtual memory<br>working set |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |



## 9. <font style="color:#FCD12A">Uniprocessor Scheduling</font>  

^f3a001

9.1 <font style="color:#A8A8A8; text-decoration: underline;">Types of Processor Scheduling</font>  

**Long-Term Scheduling**
**Medium-Term Scheduling**
**Short-Term Scheduling**

   
   
9.2 <font style="color:#A8A8A8; text-decoration: underline;">Scheduling Algorithms</font>  
   
**Short-Term Scheduling Criteria**
**The Use of Priorities**
**Alternative Scheduling Policies**
_ROUND ROBIN_

_SHORTEST PROCESS NEXT_

**Performance Comparison**

_QUEUEING ANALYSIS_

_SIMULATION MODELING_

**Fair-Share Scheduling**


   
9.3 <font style="color:#A8A8A8; text-decoration: underline;">Traditional UNIX Scheduling</font>  


   
   
9.4 <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
   
   
   
5. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

 
| arrival rate<br> dispatcher<br> exponential averaging<br> fair-share scheduling<br> fairness<br> first-come-first-served<br> first-in-first-out | long-term scheduler<br> medium-term scheduler<br> multilevel feedback<br> predictability<br> residence time<br> response time<br> round robin |  scheduling priority<br> service time<br> short-term scheduler<br> throughput<br> time slicing<br> turnaround time (TAT)<br> utilization<br> waiting time |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |


## 10. <font style="color:#FCD12A">Multiprocessor, Multicore, and Real-Time Scheduling</font>  

^a0c7e7

10.1 <font style="color:#A8A8A8; text-decoration: underline;">Multiprocessor and Multicore Scheduling</font>  
**Granularity**

_INDEPENDENT PARALLELISM_

_COARSE AND VERY COARSE-GRAINED PARALLELISM_

_MEDIUM-GRAINED PARALLELISM_

_FINE-GRAINED PARALLELISM_

**Design Issues**

_ASSIGNMENT OF PROCESSES TO PROCESSORS_

_THE USE OF MULTIPROGRAMMING OF INDIVIDUAL PROCESSORS_

_PROCESS DISPATCHING_


**Process Scheduling**

**Thread Scheduling**

_LOAD SHARING_


_GANG SCHEDULING_

_DEDICATED PROCESSOR ASSIGNMENT_

_DYNAMIC SCHEDULING_

**Multicore Thread Scheduling**


   
10.2 <font style="color:#A8A8A8; text-decoration: underline;">Real-Time Scheduling</font>  
   
**Background**

**Characteristics of Real-Time Operating Systems**

**Real-Time Scheduling**

**Deadline Scheduling**

**Rate Monotonic Scheduling**

**Priority Inversion**


   
10.3 <font style="color:#A8A8A8; text-decoration: underline;">Linux Scheduling</font>  
   
**Real-Time Scheduling**

**Non-Real-Time Scheduling**


   
10.4 <font style="color:#A8A8A8; text-decoration: underline;">UNIX SVR4 Scheduling</font>  
   
   
10.5 <font style="color:#A8A8A8; text-decoration: underline;">UNIX FreeBSD Scheduling</font>  
   
**Priority Classes**

**SMP and Multicore Support**

_QUEUE STRUCTURE_

_INTERACTIVITY SCORING_

_THREAD MIGRATION_


   
10.6 <font style="color:#A8A8A8; text-decoration: underline;">Windows Scheduling</font>  
   
**Process and Thread Priorities**

**Multiprocessor Scheduling**



10.7 <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
   
   
   
8. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| aperiodic task<br>deadline scheduling<br>deterministic<br>deterministic operating system<br>fail-soft operation<br>gang scheduling<br>granularity | hard affinity<br>hard real-time task<br>load sharing<br>periodic task<br>priority task<br>priority ceiling | priority inheritance<br>priority affinity<br>pull mechanism<br>push mechanism<br>rate monotonic scheduling<br>real-time operating system | real-time scheduling<br>responsiveness<br>soft affinity<br>soft real-time task<br>thread scheduling<br>unbound priority inversion |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |


## 11. <font style="color:#FCD12A">I/O Management and Disk Scheduling</font>  

^a2d488

11.1 <font style="color:#A8A8A8; text-decoration: underline;">I/O Devices</font>  
   
   
   
11.2 <font style="color:#A8A8A8; text-decoration: underline;">Organization of the I/O Function</font>  
   
**The Evolution of the I/O Function**

**Direct Memory Access**

   
11.3 <font style="color:#A8A8A8; text-decoration: underline;">Operating System Design Issues</font>  
   
**Design Objectives**

**Logical Structures of the I/O Function**
   
11.4 <font style="color:#A8A8A8; text-decoration: underline;">I/O Buffering</font>  
   
**Single Buffer**

**Double Buffer**

**Circular Buffer**

**The Utility of Buffering**
   
11.5 <font style="color:#A8A8A8; text-decoration: underline;">Disk Scheduling</font>  
   
**Disk Performance Parameters**

_SEEK TIME_

_TRANSFER TIME_

_A TIMING COMPARISON_

**Disk Scheduling Policies**

_FIRST-IN-FIRST-OUT_

_LAST-IN-FIRST-OUT_

_SHORTEST-SERVICE-TIME-FIRST_

_SCAN_

C-SCAN

_N-STEP-SCAN AND FSCAN_


11.6 <font style="color:#A8A8A8; text-decoration: underline;">RAID</font>  
   
**RAID Level 0**

_RAID 0 FOR HIGH DATA TRANSFER CAPACITY_


_RAID 0 FOR HIGH I/O REQUEST RATE_

**RAID Level 2**

**RAID Level 3**

_REDUNDANCY_

_PERFORMANCE_

**RAID Level 4**

**RAID Level 5**

**RAID Level 6**


   
11.7 <font style="color:#A8A8A8; text-decoration: underline;">DISK Cache</font>  
   
**Design Considerations**

**Performance Considerations**


   
11.8 <font style="color:#A8A8A8; text-decoration: underline;">UNIX SVR4 I/O</font>  

**Buffer Cache**

**Character Queue**

**Unbuffered I/O**

**UNIX Devices**


   
   
11.9 <font style="color:#A8A8A8; text-decoration: underline;">Linux I/O</font>  
   
**Disk Scheduling**

_THE ELEVATOR SCHEDULER_

_DEADLINE SCHEDULER_

_ANTICIPATORY I/O SCHEDULER_

_THE NOOP SCHEDULER_

_COMPLETELY FAIR QUEUING I/O SCHEDULER_

**Linux Page Cache**


   
11.10. <font style="color:#A8A8A8; text-decoration: underline;">Windows I/O</font>  

Basic I/O Facilities
1) Cache manager
2) File system drivers
3) Network drivers
4) Hardware device drivers

Asynchronous and Synchronous I/O
1) Signalling the file object
2) Signalling an event object
3) Asynchronous procedure call
4) I/O completion ports
5) Polling

Software RAID
1) Hardware RAID
2) Software RAID

**Volume Shadow Copies**

**Volume Encryption**



4. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  




5. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| block<br>block-oriented device<br>buffer swapping<br>circular buffer<br>device I/O<br>direct memory access<br><br> | disk access time<br>disk cache<br>double buffering<br>gap<br>interrupt-driven I/O<br>input/output (I/O)<br>least frequently used (LFU)<br><br> | I/O buffer<br>I/O buffer<br>I/O processor<br>logical I/O<br>magnetic disk<br>programmed I/O<br>read/write head<br> | redundant array of independent disks<br>rotational delay<br>sector<br>seek time<br>stream-oriented device<br>stripe<br>track<br>transfer time<br> |
| ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |



## 12. <font style="color:#FCD12A">File Management</font>  

^cf3bfc

1. <font style="color:#A8A8A8; text-decoration: underline;">Overview</font>  

**Files and File System**

- Long-term existence
- Sharable between processes
- Structure

**File Management Systems**

_FILE SYSTEM ARCHITECTURE_


   
   
2. <font style="color:#A8A8A8; text-decoration: underline;">File Organization and Access</font>  
   
**The Pile**

**The Sequential File**

**The Indexed Sequential File**

**The Indexed File**

**The Direct or Hashed File**
   
3. <font style="color:#A8A8A8; text-decoration: underline;">B-Trees</font>  
   
   
   
4. <font style="color:#A8A8A8; text-decoration: underline;">File Dictionaries</font>  
   
**Contents**

**Structure**

**Naming**
   
5. <font style="color:#A8A8A8; text-decoration: underline;">File Sharing</font>  

**Access Rights**

**Simultaneous Access**
   
   
6. <font style="color:#A8A8A8; text-decoration: underline;">Record Blocking</font>  
   
   
   
7. <font style="color:#A8A8A8; text-decoration: underline;">Secondary Storage Management</font>  

**File Allocation**

_PREALLOCATION VS. DYNAMIC ALLOCATION_

_PORTION SIZE_

_FILE ALLOCATION METHODS_

**Free Space Management**
   
_CHAINED FREE PORTIONS_

_INDEXING_

_FREE BLOCK LIST_

**Volumes**

**Reliability**




8. <font style="color:#A8A8A8; text-decoration: underline;">UNIX File Management</font>  
   
1) Regular, or ordinary
2) Directory
3) Special
4) Named pipes
5) Links
6) Symbolic links

**Inodes**

**File Allocation**

**Directories**

**Volume Structure**


   
9. <font style="color:#A8A8A8; text-decoration: underline;">Linux Virtual File System</font>  

**The Superblock Object**

**The Dentry Object**

**The File Object**

**Caches**
   
   
10. <font style="color:#A8A8A8; text-decoration: underline;">Windows File Management</font>  

**Key Features of NTFS**

**NTFS Volume and File Structure**

_NTF VOLUME LAYOUT_

_MASTER FILE TABLE_

**Recoverability**



11. <font style="color:#A8A8A8; text-decoration: underline;">Android File Management</font>  

**File System**

**SQLite**




12. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  



13. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| access method<br>basic file system<br>bit table<br>block<br>cache directory<br>chained file allocation<br>contiguous file<br>allocation<br>database<br>data directory<br>device driver | disk allocation table<br>field<br>file<br>file allocation<br>file allocation table (FAT)<br>file directory<br>file management system<br>file name<br>hashed file | indexed file<br>indexed file allocation<br>indexed sequential file<br>inode<br>key field<br>logical I/O<br>master file table (MFT)<br>mnt/sdcard | partition boot sector<br>pathname<br>physical I/O<br>pile<br>portion<br>record sequential file<br>system directory<br>system files<br>virtual file system (VFS)<br>working directory |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |



## 13. <font style="color:#FCD12A">Embedded Operating Systems</font>  

^b127fb

1. <font style="color:#A8A8A8; text-decoration: underline;">Embedded Systems</font>  
   
**Embedded System Concepts**

**Application Processors vs. Dedicated Processors**

**Microprocessors**

**Microcontrollers**

**Deeply Embedded Systems**



2. <font style="color:#A8A8A8; text-decoration: underline;">Characteristics of Embedded Operating Systems</font>  
   
**Host and Target Environments**

_BOOT LOADER_

_KERNEL_

_ROOT FILE SYSTEM_

**Development Approaches**

**Adapting an Existing Commercial Operation System**

**Purpose-Built Embedded Operating System**


   
3. <font style="color:#A8A8A8; text-decoration: underline;">Embedded Linux</font>  
   
**Characteristics of an Embedded Linux System**

_KERNEL SIZE_

_MEMORY SIZE_

_OTHER CHARACTERISTICS_

**Embedded Linux File Systems**

**Advantages of Embedded Linus**

**$µ$Clinux**

_COMPARISON WITH FULL LINUX_

**_$µ$CLIBC**_

**Android**


   
4. <font style="color:#A8A8A8; text-decoration: underline;">TinyOS</font>  
   
**Wireless Sensor Networks**

**TinyOS Goals**

**TinyOS Components**

**TinyOS Scheduler**

**Example of Configuration**

**TinyOS Resource Interface**


   
5. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| application processors<br>chip<br>commands<br>dedicated processor<br>eCos<br>embedded operating system<br>embedded system | events<br>deeply embedded system<br>integrated circuit<br>microcontroller<br>motherboard<br>printed circuit board<br>task<br>TinyOS |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |



## 14. <font style="color:#FCD12A">Virtual Machines</font>  

^9f0cf8

1. <font style="color:#A8A8A8; text-decoration: underline;">Virtual Machine Concepts</font>  
   
   
   
2. <font style="color:#A8A8A8; text-decoration: underline;">Hypervisors</font>  

**Hypervisors**

_HYPERVISOR FUNCTIONS_
1) Execution management of VMs
2) Devices emulation and access control
3) Execution of privileged operations by hypervisor for guest VMs
4) Management of VMs
5) Administration of hypervisor platform and hypervisor software


_TYPE 2 HYPERVISOR_

**Para-virtualization**

**Hardware-Assisted Virtualization**

**Virtual Appliance**



   
3. <font style="color:#A8A8A8; text-decoration: underline;">Container Virtualization</font>  
   
**Kernel Control Groups**

**Container Concepts**

**Container File System**

**Microservices**

**Docker**


   
4. <font style="color:#A8A8A8; text-decoration: underline;">Process Issues</font>  
   
   
   
5. <font style="color:#A8A8A8; text-decoration: underline;">Memory Management</font>  
   
   
   
6. <font style="color:#A8A8A8; text-decoration: underline;">I/O Management</font>  
   
   
   
7. <font style="color:#A8A8A8; text-decoration: underline;">VMware ESXi</font>  
   
   
   
8. <font style="color:#A8A8A8; text-decoration: underline;">Microsoft Hyper-V and Xen Variants</font>  
   
   
   
9. <font style="color:#A8A8A8; text-decoration: underline;">Java VM</font>  
   
   
   
10. <font style="color:#A8A8A8; text-decoration: underline;">Linux Vserver Virtual Machine Architecture</font>  

Architecture

Process Scheduling




11. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  



12. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| container<br>container virtualization<br>consolidation ratio<br>Docker<br>guest OS<br>hardware virtualization<br>hardware-assisted virtualization<br>host OS | hypervisor<br>Java Virtual Machine (JVM)<br>kernel control group<br>memory ballooning<br>memory overcommit<br>microservice<br>page sharing | para-virtualization<br>type-1 hypervisor<br>type-2 hypervisor<br>virtual appliance<br>virtualization container<br>virtualization<br>virtual machine (VM)<br>virtual machine monitor |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |



## 15. <font style="color:#FCD12A">Operating System Security</font>  

^34d101

1. <font style="color:#A8A8A8; text-decoration: underline;">Intruders and Malicious Software</font>  

**System Access Threats**

_INTRUDERS_

_MALICIOUS SOFTWARE_

**Countermeasures**

_INTRUSION DETECTION_

_AUTHENTICATION_

_ACCESS CONTROL_

_FIREWALLS_


   
2. <font style="color:#A8A8A8; text-decoration: underline;">Buffer Overflow</font>  

Buffer Overflow Attacks
- A condition at an interface under which more input can be placed into a buffer of data-holding area than the capacity allocated, overwriting other information
- Attackers exploit this condition to crash a system or to insert specially crafted code that allows them to gain control of the system

**Compile-Time Defenses**

_CHOICE OF PROGRAMMING LANGUAGE_

_SAFE CODING TECHNIQUES_

_LANGUAGE EXTENSIONS AND USE OF SAFE LIBRARIES_

_STACK PROTECTION MECHANISMS_

**Runtime Defenses**

_EXECUTABLE ADDRESS SPACE PROTECTION_

ADDRESS SPACE RANDOMIZATION

_GUARD PAGES_



1. <font style="color:#A8A8A8; text-decoration: underline;">Access Control</font>  
   
**File System Access Control**

**Access Control Policies**

_DISCRETIONARY ACCESS CONTROL_

_ROLE-BASED ACCESS CONTROL_


   
4. <font style="color:#A8A8A8; text-decoration: underline;">UNIX Access Control</font>  

**Traditional UNIX File Access Control**

**Access Control Lists in UNIX**

6. <font style="color:#A8A8A8; text-decoration: underline;">Operating System Hardening</font> 
   
**Operating System Installation: Initial Setup and Patching**

**Remove Unnecessary Services, Application, and Protocols**

**Configure Users, Groups, and Authentication**

**Configure Resource Controls**

**Install Additional Security Controls**

**Test the System Security**

 
7. <font style="color:#A8A8A8; text-decoration: underline;">Security Maintenance</font>  

**Logging**

**Data Backup and Archive**


   
8. <font style="color:#A8A8A8; text-decoration: underline;">Windows Security</font>  
   
**Access Control Scheme**

**Access Token**

**Security Descriptors**


   
9. <font style="color:#A8A8A8; text-decoration: underline;">Summary</font>  
   
   
   
10. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms, Review Questions, and Problems</font>

| access control<br>access control list (ACL)<br>access control policy<br>access matrix<br>address space randomization<br>authentication | buffer overrun<br>buffer overflow<br>capability ticket<br>discretionary access control (DAC)<br>file system access control<br>firewall<br>guard page | intruder<br>intrusion detection<br>logging<br>malicious software<br>malware<br>role-based access control (RBAC)<br>stack overflow |
| -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |



## 16. <font style="color:#FCD12A">Cloud and IoT Operating Systems</font>  

^dc25b5

1. <font style="color:#A8A8A8; text-decoration: underline;">Cloud Operating</font>  

**Cloud Computing Elements**

**Cloud Service Models**

_SOFTWARE AS A SERVICE_

_PLATFORM AS A SERVICE_

**Cloud Deployment Models**

_PUBLIC CLOUD_

_PRIVATE CLOUD_

_COMMUNITY CLOUD_

_HYBRID CLOUD_

**Cloud Computing Reference Architecture**


   
2. <font style="color:#A8A8A8; text-decoration: underline;">Cloud Operating Systems</font>  
   
**Infrastructure as a Service**

**Requirements for Cloud Operating Systems**

**General Architecture of a Cloud Operating System**

_VIRTUALIZATION_

_VIRTUAL COMPUTING_

_VIRTUAL STORAGE_

_VIRUTAL NETWORK_

_DATA STRUCTURE MANAGEMENT_

_MANAGEMENT AND ORCHESTRATION_


**OpenStack**

_IMAGE (GLACE)_

_NETWORK (NEUTRON)_

_BLOCK STORAGE (CINDER)_

_IDENTITY (KEYSTONE)_

_DASHBOARD (HORIZON)_

_MONITOR (CEILOMETER)_

_ORCHESTRATION (HEAT)_

_OTHER OPTIONAL SERVICES_

   
   
3. <font style="color:#A8A8A8; text-decoration: underline;">The Internet of Things</font>  

**Things on the Internet of Things**

Evolution
1) Information technology (IT)
2) Operation technology (OT)
3) Person technology
4) Sensor/actuator technology


**Components of IoT-Enabled Devices**

**IoT and Cloud Context**

_EDGE_

_FOG_

_CORE_

_CLOUD_




1. <font style="color:#A8A8A8; text-decoration: underline;">IoT Operating Systems</font>  
**Constrained Devices**

**Requirements for an IoT OS**

**IoT OS Architecture**

RIOT

_RIOT KERNAL_

_OTHER HARDWARE-INDEPENDENT MODULES_

_HARDWARE ABSTRACTION LAYER_

   
   
5. <font style="color:#A8A8A8; text-decoration: underline;">Key Terms and Review Questions</font>

| actuators<br>backbone network<br>block storage<br>cloud<br>cloud auditor<br>cloud broker<br>cloud carrier<br>cloud computing<br>cloud service consumer (CSC) | cloud service provider (CSP)<br>community cloud<br>Constrained Application Protocol (CoAP)<br>constrained device<br>direct attach storage (DAS)<br>file-based storage<br>file storage<br>gateways | hybrid cloud<br>infrastructure as a service (IaaS)<br>Internet of Things (IoT)<br>microcontroller<br>network attached storage (NAS)<br>object storage<br>OpenStack<br>platform as a service (Paas) | <br>private cloud<br>public cloud<br>radio-frequency identification (RFID)<br>sensors<br>service models<br>software as a service (SaaS)<br>storage area network (SAN)<br>transceiver |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |




# <font style="color:#FCD12A">Appendix A - Concurrency</font>

A.1 Race Conditions and Semaphores

# <font style="color:#FCD12A">Appendix B - Programming and Operating System Projects</font>