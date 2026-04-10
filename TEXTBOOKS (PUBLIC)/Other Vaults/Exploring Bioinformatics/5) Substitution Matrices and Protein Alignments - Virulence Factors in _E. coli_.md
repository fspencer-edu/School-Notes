## Understanding the Problem: Virulence Factors in E. coli Outbreaks

- E. coli strain is prevalent in bovine
	- Gastrointestinal illness
- Spread from from the intestine to infect the kidneys (hemolytic uremic syndrome)
- Contaminated in meat, water, lettuce and other products
- Most strains of E. coli are harmless resident of the large intestines of humans and other mammals
- Key factors in deadly strain is the toxin Shiga (Stx)
	- Binds receptors found in human kidney tissues (not cattle)
	- Some strains of "tame" E. coli in the gut are likely to encode virulence factors

## Bioinformatics Solutions: Protein Alignment and Clues to Function

- Needle-Wunsch algorithm has no mechanisms for deciding which nucleotide changes are more or less likely to have occurred over evolutionary time
- Align amino-acid sequences of two proteins rather than nucleotides to get genotype mutations
- Arginine and lysine
	- Both positively charge and similar size
	- Will have less difference to the structure

- Align amino acids rather than nucleotides
- Take into account measure of how similar two amino acids are

- Once a gene is identified and the aa sequence is inferred, use alignment to find similar proteins
	- Orthologs
	- Paralogs


![[Pasted image 20251031102833.png]]


\* -> Identical
\*: -> similar


## Understanding the Algorithm: Protein Alignment and Substation Matrices

- More algorithms for local alignment, multiple alignment, and database searching are similarly flexible with any two strings
- Needle-Wunsch can be modified to score based on the degree of similarity between two amino acids

**Substitution Matrices**
- Substitution matrix
	- A table of scores for different pairs of amino acids
	- Aligning two similar aa should get a higher score than two dissimilar ones
- A substitution matrix could be based on the hydrophobicity of the amino acids
	- Lower score -> hydrophilic with hydrophobic
	- High score -> Similar hydrophobicity values
- Large numbers represent more hydrophobic amino acids

![[Pasted image 20251031103311.png]]


- Most substitution matrices are based on observed conservation of amino acids
	- PAM (point accepted mutation)
	- BLOSUM (block substitution matrix) matrices
- PAM was developed by examining very closely related sequences for amino acid changes
- Comparing the hypothesis that amino acid occurs die to evolutionary conservation of a substitution with the hypothesis that its occurrence is random
- If this odds ratio is 1, then the substitution is no more likely than the change of finding j randomly
- If ration increases, then conserved
- Against, ration decreases
- Log-odds ration is more positive for likely (conservative) substitutions and negative for non-conservative substitutions
- Multplied by 10 to allow rounding to the nearest integer

![[Pasted image 20251031111209.png]]


- Initial PAM matrix, PAM 1, was built from 71 known protein families and normalized to represent an average change of 1 amino acid in 100
- PAM matrix represents a unit of evolutionary time
	- Time required for change to occur in 1% of amino acids


![[Pasted image 20251031111324.png]]

- PAM matrix shows that cysteine has a higher probability of remaining cysteine than being replaced by any other amino acid
- Strong bias is due to unique role of its forming structurally important disulfide bonds
- BLOSUM matrix is similar
- Scoring is based on a log-offs ration
	- Log of the frequency with which two amino acids align in a multiple alignment divided by the expected frequency for random alignment
- PAM
	- More positive the score, more likely a substitution will occur
- Default matrix for a BLAST search is BLOSUM 62, which means 62% identical
- Prefer PAM because it is based on global alignment
- BLOSUM based on local alignment

**Protein Alignment Algorithm: Scoring with the Substitution Matrix**
- Substitution matrix values represent the likelihood that one amino acid will substitute for another
- Better alignment has fewer non-conservative substitutions
- Alignment of two protein sequences begins with an alignment of matrix
	- Not substitution matrix

ANFNNASWF
ANFNCFWS

- Compare the two amino-acid sequences using hydrophobicity scoring matrix
- Gap penalty of -1
- Result of the matrix can be populated by Needle-Wunsch
	- Gap penalty holds
	- Match/mismatch value use lookup table
- Most protein alignment programs allow the user a choice of substitution matrix


![[Pasted image 20251031112100.png]]

## Chapter Project: Using Protein Alignment to Investigate Functions of Virulence Factors

- Novel virulence genes envolved in or acquired by highly pathogenic strains are important in dealing with food-borne disease
- The degree of difference between the genomes MG1655 and EDL933
	- The former has more than 500,000 bp of sequences not found in EDL933
- Hundreds of distinct genes could be virulence factors fro EDL933
- Bacterial virulence
	- Toxins
	- Pili and other bacterial surface features for host attachment
	- Enzymes

![[Pasted image 20251031112525.png]]

### Part 1: using Protein Alignment to Explore Protein Function

- Increasing the quality of the search results by adding to the BLAST search
- NCBI Protein database
- High-scoring match is a similar sequence

Conserved Domains
- A domain is a functional region of a protein
	- Energy-requiring enzyme that might have an ATP-binding domain
		- Substrate-binding domain for catalytic function
- Two DNA-binding proteins that hav different function might have similarity in their DNA-binding domains
- BLAST finds patterns that resemble known functional domains

Substitution Matrices
- Some bacteria-specific proteins have no identifiable human orthologs
- Others have conserved across long span of evolutionary time


### Part 2: More Tools for Exploring Protein Function

- Inferring putative functions for proteins based on amino-acid sequence is a major role in bioinformatics
	- Discover new genes
	- New sequences

PSI-BLAST
- A variation of BLAST in which initial matches are usde to refine the substitution matrix to identify more distant matches

pfam
- Pfam is a database of protein families
	- Group of proteins shown to be similar in structure and function

MOTIF
- MOTIF looks for alignment between a query of amino-acid and functional domains and motifs
	- Motif - short sequence segments associated with some function

DAS (Dense alignment surface)
- The localization of a protein within the cell can provide clues to possible fnunction
- DAS deals with one aspect of protein localization
	- Potential transmembrane domain that is an integral membrane proteins

- MicrobesOnline
- MAUVE, a visually oriented program for multiple genome alignment

**Substitution Matrices and Protein Alignment Algorithm**
- A global protein alignment using a substitution matrix require a few changes to Needle-Wunsch
- There are several substitution matrix files
	- Hydrophobicity
	- PAM 250
	- BLOSUM 52
- Use hash table or data structure to hold look up table

![[Pasted image 20251031113632.png]]

- FASTA-style comments
- Each line has amino acid substitution values
- First line represented the substitution values for A with all other amino acids
- Second line is the substitution for R
	- And so on
- Score cannot be looked up in the scoring matrix is the gap score
- Nested hash table can store two dimensional structure

(\*) -> identical
(:) -> similar

![[Pasted image 20251031113951.png]]

![[Pasted image 20251031114003.png]]

## Building a Substitution Matrix

**Scoring Matrices Based on Observed Substations**
- Different kinds of protein or different organisms have biases in the amino acids that are used
- Develop a scoring matrix that would give higher scores for amino-acid substitutions that are strongly conserved within the E. coli bacterial group
- Set aligned sequences from E. coli strains
- Algorithm should identify the substitutions and develop a substitution model
- Initial set if training set
	- Training set should contain a significant number of sequences that are relevant to the known relationship
	- Error-free as possible

![[Pasted image 20251031114335.png]]

- Build substitution matrix with pairwise alignments
- Training set shows that alanine is frequently replaced by lysine

1) Gap handling
2) Unrepresented amino acids
	1) Pseudo-count of 1 as a default starting value
3) Limitations of using the matrix


**Obtaining the Training Set**
- Identify orthologs from MicrobesOnline
- Use Find Shared Genes to identify genes that have recognizable orthologs
- Use pairwise protein alignments of the orthologous sequences
- Obtain alignments from Needle-Wunsch or Emboss


**Solving the Problem**
- Use log-odds scoring system
- For each pairwise alignment, count how many times each amino acid occurs and number of times each pair aligns
- Gap positions are ignored
- Count from 1


![[Pasted image 20251031114939.png]]

- Not all amino acids are present
- Generate a matrix including only the amino acids present in training set

- 1 occurrence of Y-R alignment, therefore $q_{YR}= 2$
- There are 16 total aligned position
	- Alignments = 18 - 2 (gaps) = 16
- $q_{YR}= 2/16 = 0.125$
- The value $p_Y$ and $p_R$ represent the estimated background frequencies, or likelihood of Y or R in sequence
- 32 total ungapped position
	- Y -> 5 + 1
	- R -> 11 + 1
- $p_Y = 6/32 = 0.19$
- $p_R = 12/32 = 0.38$
- Use positions rather than 16, to find frequency of occurrence of one amino, rather than pairs
- $e_{YR}$ is the expected frequency of aligning these two amino acids
	- Product of individual frequencies
- $0.19 \times 0.38 = 0.070$
- $e_{YR} = 2 \times 0.070 = 0.14$ 
	- Multiple $e_{YR}$ because the alignment is equal to $e_{RY}$
- Determine odd ratios

$q_{YR} / e_{YR} = 0.125/0.14 = 0.89$

- This value is less than 1, so it indicates that the Y-R alignment occurred less frequently than would have predicted by changes
- Take log of odds ratio to get a number for scoring
- PAM uses base 10
- Use base 2, to that a substitution gets a score of 1 if it is twice as common as expected by change

$log_2(0.89) = -0.17$


**Exercises for Programming Courses**
![[Pasted image 20251031120030.png]]

**Annotating Genomes**
- Process of assigning functional information to genome sequences is called annotating the genome
- Need to be corrected/updated from new experiments
- Specialized substitution matrices are used to look for membrane proteins
- Substitution matrix to predict protein structure
	- Identification of structural similarities

**Amino Acids, Protein Sequences, and Protein Function**
- Codons specify the amino acids of a protein
- Amino acids are small molecules consisting of carboxyl group (COOH), amino group, $(NH_3)$, and hydrogen atom attached to a central carbon
- 20 amino acids in proteins with a distinct side chain and chemical properties


![[Pasted image 20251031120300.png]]


- Nucleotide sequence determined the linear sequence of amino acids
	- Primary structure
- Two proteins with similar functions, have similar structures
- DNA-binding proteins has a stretch of amino acids that curve into a relatively rigid helix that fits into the major group of the DNA molecule


![[Pasted image 20251031120308.png]]


- Protein alignment can score exact matches and conservative substitutions
	- Mutations that result in the substitution of a similar one
- Chemical properties
	- Hydrophilic or hydrophobic
	- Charge


![[Pasted image 20251031120319.png]]