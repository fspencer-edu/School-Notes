
- Confidentiality policies emphasize the protection of confidentiality

# Goals of Confidentiality Policies

- Also called an information flow policy
- Prevent the unauthorized disclosure of information

# The Bell-LaPadula Model

- Correspond to military-style classification

## Information Description

- A set of security clearances arranged in linear ordering
- Represent sensitivity levels
- A subject has a security clearance
- An object has a security classification
- The goal of the Bell-LaPadula security model is to prevent read access to objects at a security classification higher than the subject's clearance
- Combines mandatory and discretionary access control

"S has discretionary read (write) access to O"

![[Pasted image 20251104115051.png]]

- Simple security condition, preliminary version
	- S can read O if and only if $l_o$ <= l$_s$ has S has discretionary read access to O

- Expand the model by adding a set of categories to each security classification
- Objects placed i multiple categories have the kinds of information in all of those categories
- Each security level and category from a security level
- Subjects have clearance at a security level and that objects are at the level of a security level


![[Pasted image 20251104115452.png]]


## Example: The Data General B2 UNIX System

- Data General B2 UNIX system provide mandatory access controls (MAC)
- MAC label is a label identifying a particular compartment

### Assigning MAC Labels

- When a process begins, it is assigned the MAC label of its parent
- Initial label (assigned at login time) is the label assigned to the user in a database called the Authorization and Authentication (A&A) Database
- 3 Regions of lattice

1. Administrative regions
	1. Reserved for data that users cannot access
	2. Logs
	3. MAC label definitions
	4. Servers
2. User regions
	1. Sanitize data sent to user region
3. Virus prevention region
	1. No user process can write/alter to them
	2. Users can execute programs


![[Pasted image 20251104115827.png]]

- To prevent leakage of information, only programs with the same MAC label as the directory can create files in that directory
- Multilevel directory is a directory with a set of sub-directories, one for each label
- A trusted computing base (TCB) is used to enforce security

```c
state(".", &stat_buffer)
dg_mstat(".", &stat_buffer)
```

- Translates the notion of current working directory to a multilevel directory when current is hidden
- Mounting unlabeled files requires the files to be labeled
- Symbolic links aggravate this problem
- DG/UX uses inherited labels

1. Roots of file system have explicit MAC labels
2. An object with an implicit MAC label inherits the label of its parent
3. When a hard link to an object is created, that object must have an explicit label
4. If the label of a directory changes, any immediate children with implicit labels have those labels
5. When the system resolved a symbolic link, the label of the object is the label of the target of symbolic link


Hard Links
- Another name for the same file
- Points directory to the same inode
```c
$ echo "Hello" > file1.txt
$ ln file1.txt file2.txt

12345 -rw-r--r-- 2 user user 6 Nov 4 12:00 file1.txt
12345 -rw-r--r-- 2 user user 6 Nov 4 12:00 file2.txt
```

Symbolic Links
- Shortcut or pointer to another path (alias)
- Not the same inode
- Cross filesystems

```c
$ ln -s file1.txt link_to_file1

12345 -rw-r--r-- 1 user user 6 Nov 4 12:00 file1.txt
12346 lrwxr-xr-x 1 user user 9 Nov 4 12:01 link_to_file1 -> file1.txt
```


### Using MAC Labels

- Use Bell-LaPadula Notation of dominance
- Reading down is permitted
- Writing up is not permitted
- MAC tuple,  object with a range of labels
- A range is a set of labels expressed by a lower bound and an upper bound
- A MAC tuple consists of up to three regions

```c
// TS and S for categorties (COMP, NUC, ASIA)
[( S, { COMP } ), ( TS, { COMP } )]
[( S, ∅ ), ( TS, { COMP, NUC, ASIA } )]
[( S, { ASIA } ), ( TS, { ASIA, NUC } )]
```

# Summary