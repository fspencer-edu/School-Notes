
## Understanding the Problem: Genetic Screening and Inheritance of Cystic Fibrosis

- Cystic fibrosis (CF) is the most common in Caucasians
	- 1 in 25 individuals carries the allele
- Only symptomatic treatment is available for victims with CF
- CF is a recessive, single-gene genetic disorder that affects the ability of epithelial cells to secrete fluids
	- Sweat is salty
- Secretion of digestive enzymes from the pancreas, and mucus in the respiratory system are affected
- Failure of the van deferens to develop properly
	- CF males sterile
- Inheritance of 2 mutant allies of a gene called CFTR
- CF transmembrane conductance regulator, allows regulated movement of chloride ions out of an epithelial cells, causing water to flow through osmosis producing watery secretion
- Without CTRF, mucus is thick and builds up in the lungs
	- Pseudomonas aeruginosa can cause chronic lung inflammation
- Treatment
	- Antibiotics
- Can determine if a person is a carrier of the gene
- Some effects of CFTR result in cells that are aberrant (normal), whereas others result in production that failed to fold correctly or insert into the cell membrane
- Kalydeco was approved by FDA for CF pateints with a change at the 551st amino acid in CFTR from glycerine to aspartate (G551D)
	- Mutation that does not affects synthesis of localization of the protein
	- Prevents opening of the channel

![[Pasted image 20251030221934.png]]



## Bioinformatics Solutions: Computational Approaches to Genes

- Analysis of genetic data
	- Finding genes
	- Identifying mutations
	- Predicting the sequence
	- Structure and function of proteins
- Most labs use a faster, cheaper method to detect common CFTR mutations
	- PCR-based methods
	- ASO arrays
- Can only detect a particular set of alleles

## Understanding the Algorithm: Decoding DNA

- A computer algorithm is a set of specific steps that describes how a problem can be solved

**DNA Manipulation Algorithm**

- Transcribing and translating DNA requires a template or non-template strand and its orientation
- Unlabeled, single-stranded sequences are assumed to be 5' to 3' prime
	- Output the template strand
	- 3' -> 5'


![[Pasted image 20251030222444.png]]

![Common misconceptions in biology: Making sense of the sense and antisense DNA  strands - IndiaBioscience](https://cdn.indiabioscience.org/media/articles/Maya-sense_fig-3.png)


DNA Manipulation Algorithm
1) Input a DNA sequence
	1) Template or non-template
	2) Find 5'
2) Convert to capital letters
3) Choose operation
	1) Template
	2) Inverse
	3) Inverse complement
	4) Complement

**Transcription Algorithm**
- Ensure the input for transcription is a template strand 3' -> 5' orientation
	- To get mRNA
		- Complete
		- T -> U
- Outputs a 5' -> 3' mRNA
- Use the non-template, and replace T -> U

**Translation Algorithm**
- Find start codon
- Determine amino acids from triplet codons from genetic code table
	- Start codon is the N-terminal

![[Pasted image 20251030223139.png]]

**Mutation Detection Algorithm**
- Searching for alterations in protein sequence uses comparison of amino acids

1) Input two amino acid sequences
	1) Wild type
	2) Patience sequence
2) Traverse each character
	1) G551D
		1) At position 551 G -> D
3) Output list of differences


## Chapter Project: Genetic Screening for Carries of CF Mutations

- Sequence Manipulation Suite (SMS)
	- Set of tools written in JavaScript

**Translating the CFTR mRNA**
- There are 3 reading frames
- The codon region is the open reading frame (ORF)
- The real start of the aa is the first M in the ORF, encoded by the first AUG
- Use the ORF amino sequence to get start and stop reading frame
- Then apply to the entire sequence


**Detecting Mutations**
- Use pairwise align codon tools in SMS and submit wild-type and mutant sequence
- Returns FASTA sequence
	- Line up base-for-base
	- Dashed lines representing a shift
- Black -> same nucleotide
- White -> Difference

- Translate mutant coding sequence
	- Missense
	- Non-sense
	- Silent

![[Pasted image 20251030224214.png]]



- CFTR is the best-studied single-gene disease

**Mutation Detection Algorithm**

- Identify the location of all differences (mutations) between two strings

Input
- 2 equal length DNA

Output
- Description and location of all mutations



![[Pasted image 20251030224507.png]]

## Understanding the Problem: Insertions and Deletions

- Mutations are often insertions and deletions (indels) of one or more nucleotides
- If sequences are different lenths, they must be aligned so the characters match up

Align ACFTTA and ACTA

![[Pasted image 20251030224703.png]]

- Assume that the first sequence is the reference or wild-type
	- Single mutation event
		- Substation
		- Insertion
		- Deletion
- Assume the deletions occurred together, and not separately such as A-C-TA

**Programming the Solution**
1) Input and manipulate DNA
2) Determine equal length
3) Determine number of nucleotides
	1) Gaps
4) Construct all possible alignments by placing the number of gaps at each possible position
	1) Find fewest mis-matches

**Future of Genetics**
- Gene therapy is the idea of curing a genetic disease by changing an individual's DNA
	- Changing only affects cells (somatic cell gene therapy)
	- Changing the sperm/egg cells (germ-line gene therapy)
- Evidence that carrying one CF allele can be beneficial
	- Reducing cholera

**Genetic Code and Decoding DNA**
- DNA is a double-stranded molecule, with anti-parallel orientation
- 5' to 3' prime, with terminating in phosphate group and hydroxyl group, respectively
- RNA is a single-stranded molecule
	- Base-pairing RNA from DNA template strand
- RNA is complementary to the template strand, where T is U
	- Read 3' to 5' prime
- Codons are translated to amino acids
- Amino acid represent 64 possible codons
- Ribosome identifies the start codon for transcription
- Codons are read 5' to 3'
- Mutation changes a DNA sequence
	- Substitution
	- Insertion
	- Deletion
	- Frameshift


- Codon result
	- Silent
	- Missense
	- Non-sense


![[Pasted image 20251030225635.png]]