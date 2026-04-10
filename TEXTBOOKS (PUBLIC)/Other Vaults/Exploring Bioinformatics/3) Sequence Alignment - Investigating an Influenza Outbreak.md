## Understanding the Problem: The 2008 H1N1 Influenza Pandemic

- Bird flue virus had caused severe infections in domestic fowl and humans in direct contact
- Next human pandemic result from a unknown strain of H1N1 that escaped detection
- 1918 spanish flu
	- 8000 deaths

## Bioinformatics Solution: Sequence Alignment and Sequence Comparison

 - Alignment of the sequences of two genes or proteins refers to matching them to determine genetic similarity
 - Common ancestor
 - Sequence alignment is used in phylogenetic trees based on
	 - Molecular data
	 - Assembling genome sequences
	 - Predicting protein structure and functions


![[Pasted image 20251030230312.png]]



![[Pasted image 20251030230330.png]]


- "Strains of dog": groups of species that have distinct, inheritable genetic characteristics
- CDC determine whether new viruses have arisen from minor variants of existing viruses
	- Anti-genic "drift"
- Or different from circulating viruses
	- "Anti-genic" shift


## Understanding the Algorithm: Global Alignment

- Algorithm for optimal, global alignment of pairs of genes
	- Sual Needleman and Christian Wunsch (1970)
- Algorithm used today
	- Google search uses a similar search algorithm

**Optimal Alignment and Scoring**

- A global alignment is the mismatches along the entire length
- Used to compare sequences in their entirety
- Uses pairwise alignment algorithm
	- Compares only one sequence to another at a time

![[Pasted image 20251030230856.png]]

- Hyphens -> gaps
	- Insertions/deletions
- Indels can create frameshift
- Scoring metric
	- Match bonus
	- Mismatch score
	- Gap penalty

Match -> 1
Mismatch -> 0
Gap -> -1

- Left alignment scores 3
- Centre alignment scores 4
- Right alignment sores -5

- Needleman-Wunsch uses dynamic programming
- Divides a problem into a series of smaller sub-problems
- Solves partial alignment scores
- Backtracking along a path to best possible alignment(s)

**Needleman-Wunsch**

CGA

CACGTAT

- Construct an N x M matrix
- N => length of first sequence + 1
- M => length of second sequence + 1
- Start with a zero in the first cell and add a gap penalty of (-1) to each successive cell

![[Pasted image 20251030231746.png]]

1) If match, then score = 1, mismatch = 0
2) Add match to score horizontal, diagonal, and horizontal cell
3) Choose the highest score

- Follow alignment form bottom-right cell to top-left cell
- Each backtrack is an alignment


**Generating the Alignment**

- Global alignment compares the entire length
- Change scoring parameters based on problem
- Compare 2 protein coding genes
	- Penalize gaps because of the frameshift
- RNA
	- Gaps may be no worse than mismatch
- To output more similar sequences
	- Add penalty to gaps and mismatches

## Chapter Project: Investigation of Influenza Virus Strains


### Part 1: Pairwise Global Alignment with the Needleman-Wunsch Algorithm

- Genomes of influenza virus are divided into 8 segments
	- Each representing the coding information for a single protein
- Segment 4 contains the gene for hemagglutinin (HA)
- Align sequences using EMBOSS, a suite of alignment tools produced by European Bioinformatics Institute
- Needle uses an affine gap penalty
	- Imposes a larger penalty when a new gap is added
	- Smaller when the gap is extended

![[Pasted image 20251031083301.png]]

- $H_1$ HA gene is thought to be the source of the HA genes found in all modern human and swine $H_1$ viruses


### Part 2: Local Alignment with the Smith-Waterman Algorithm

- A local alignment looks for optimal partial matches
- Set a gap open penalty of 10 and gap extension of 0.1


### Part 3: Using Alignment to Investigate Virulence

- $H_5N_1$ "bird flue"
- Sequences are from Influenza Research Database
	- Indexes sequences from information on influenza viruses of all types



**Dynamic Programming and the Needle-Wunsch Algorithm**
- First to implement dynamic programming to solve an alignment problem
- Ways to divide problem into smaller sub-problems

**Implementing the Needle-Wunsch Algorithm**

- Build scoring matrix
- Find paths through matrix
- Generate alignments from the paths

**Local Alignment Algorithm**

AAA GCT CCG ATC TCG

TAA AGC AAT TTG GTT TTT TTC CGA

- Global or semi-global alignment with find AAAGC, but fail to align TCCGA

- Semi-global
	- Do not penalize terminal gaps
- Local alignment
	- Removes negative scores
	- Does not need "ideal" diagonal path
- In global and semi-global alignment builds the path to the lower right
- For local alignment optimal score is anywhere in the matrix

**The Influenza Virus and Molecular Evolution**
- Outside the host cell, viruses are metabolically inert
	- Nucleic acid in a protein shell
	- Sometime membrane
- An influenza viruses can be classified based on the type of HA proteins is carries and second protein, neuraminidase (NA)
- HA and NA involved in releasing the viral progeny from host


![[Pasted image 20251031084817.png]]



- RNA genome of influenza viruses is synthesized by a virus-encoded polymerase that does not "proofread" to remove errors
- New strain related to the original one by similarity of their genes
- When a gene in one species or strain is similar to a different species/strain it is called orthologs
- Many or most genes in evolutionary species should be orthologs
- Similar genes within the same species are paralogs

- A mutation in HA allowing the protein to be processed by more common protease, increased host range
- A mutation in viral polymerase allows higher activity at lower temperatures in human respiratory tract
- 

![[Pasted image 20251031085115.png]]

![[Pasted image 20251031085134.png]]

![[Pasted image 20251031085144.png]]