## Understanding the Problem: Gene Discovery

- Pre-antibiotics days, risk of infection made surgery more dangerous than the condition
- Nosocomial (hospital acquire infections) caused 99,000 death of the patient
- Common agents of nosocomial infection
	- Enterococcus species
- Once a genome, chromosome, plasmid, or other large pice of DNA has been sequences, the processes of gene discovery (gene prediction) and genome annotation begin
 - Using gene prediction methods, we can identify potential resistance genes within this plasmid sequence by looking for conserved sequences
	 - Determining what resistances the bacteria have and how to treat the infection
 
## Bioinformatics Solutions: Gene Prediction

- The coding sequence is called an open reading frame (ORF)
- Long ORFs are usually considered more likely to be real genes
	- Do not want to miss short, but genuine genes
- Untranslated functional RNAs have no coding sequence
- Predicting which sequence serve as promoters can help recognize actual genes
- Introns introduce a higher amount of difficulty in eukaryotic genomes

Computational Approaches
1. Algorithm-based
2. Sequence
3. Content
4. Probability

_Alignment-Based Algorithms_
- Strong conservation of genome region over evolutionary time is strongly suggestive of its functional importance
- Alignment-based algorithms look for genes based on conserved sequences

_Sequence-based algorithms_
- Search for ORF is an example of sequence-based method of gene prediction
	- Look for start (AUG) then stop codon
- Functional regions of DNA would be identified based on the development of consensus sequences
- Do not require similarity to other organisms, but they can only find genes that include sequences matching known patterns
- Difficulty with sequence patterns that are relatively loose
	- Sequences at boundaries of exons and introns

_Content-Base Algorithms_
- Do not look for specific sequences but rather for patterns such as nucleotide or codon frequency that are characteristics of coding sequences
- Identify novel genes and find coding regions

_Probabilistic Algorithms_
- More sophisticated gene discovery methods may combine elements of both sequence-based and content-based gene prediction in algorithms that model the probability that a given sequence is part of a gene
	- Hidden Markov models
	- Neural network algorithms

## Understanding the Algorithm: Pattern Matching in Sequence-Based Gene Prediction

- Sequence-based methods of gene prediction examine DNA sequences for patterns (motifs) that provide clues about the existence of transcriptional or translational units
- Rely on pattern-matching algorithms
	- Given a string to search and a pattern to be matched
	- Identify
		- How often and where the pattern occurs
- Content-based and probabilistic method include elements of pattern matching
- ORF-finding program is pattern matching in gene prediction
- Traversing the searched text, examining start and top codons
- Parameters are values set when an algorithm starts that allow it to solve variations of problems

![[Pasted image 20251103125356.png]]

- ATG is not just a start codon, but is used every time the amino acid methionine occurs in a protein
- ORF-like sequence would occur by change in non-coding DNA

![[Pasted image 20251103125507.png]]

- A simple ORF-finding algorithm will not be a very reliable method of gene prediction
- Look for regulatory sequences
	- Bacteria
		- Preceded by promoter sequences
		- Start codon is preceded by a Shine-Dalgarno sequence

## Chapter Project: Gene Discovery in a Resistance Plasmid

- Consider only prokaryotic genes, because the lack of introns and more clearly defined expression signals make them easier from a practical standpoint

**Prokaryotic Gene Prediction and Annotation**
- Find genes within an Enterococcus resistance plasmid sequence

### Part 1: Sequence-Based ORF Identification Using the NCBI ORF Finder

- The ORF could occur in any of the six possible reading frames (three on each strand)
- Gene prediction is more valuable if we can also annotate the genes with putative functions based on sequence comparison

![[Pasted image 20251103130026.png]]


### Part 2: Sequence-Based ORF Identification Using NEBcutter

- Primary goal of this program is to identify restriction endonuclease cut sites
	- GEne cloning
- Identifies ORF and places then on a map of the DNA in relation to the restriction sites
- Use the BLAST results to find a putative function for each ORF

![[Pasted image 20251103130224.png]]

### Part 3: Shine-Dalgarno Prediction and Codon Usage Analysis with EasyGene

- The results of ORF Finder or NEBbutter, is a long list of potential genes
- Eliminate ORFs from the list by requiring that the ORF be at least 100 amino acids long
- Use EasyGene to add this element of sophistication to prokaryotic gene prediction
	- Looks for ORFs and examines the region just before the putative start codon for possible Shine-Dalgarno sequence
- Content-based method
	- Asks whether the codons used in the ORF match the typical codon usage for the organisms of interest

![[Pasted image 20251103130540.png]]




**Evolution of Antibiotic Resistance and a Resistance Plasmid**



**Pattern Matching for Sequence-Based Gene Prediction**
- Use parameters to limit the range of the search and whether to consider imperfect matches
- Write code to implement to ORF finder algorithm
- Limit scope to prokaryotic gene prediction

![[Pasted image 20251103130735.png]]

- Loop terminates as soon as a match is found
- Calling the pattern-matching subroutine

1. Search for a start codon
2. Search for a stop codon
	1. If ORF is large enough, end
3. Search for a Shine-Dalgarno sequence no less than three and no more than 7 bases upstream of the start codon
4. Search the 500 nucleotides upstream of the Shine-Dalgarno sequence for a promoter sequence

## Understanding the Problem: Sequence-Based Pattern Matching in Eukaryotes

- In eukaryotes, the start codon is almost always the first AUG from the 5' end of the mRNA
	- First one after the transcription start site
- About 75% of cases, the transcriptional start cam be identified by the presence of a core promoter pattern
- Core promoter can be recognized by a consensus sequence TATA box
- The transcriptional start site lies within an additional consensus sequence, the initiator sequence (Inr)

![[Pasted image 20251103131209.png]]

- ATG codons use as start site commonly occur within a sequence known as the Kozac sequence
	- 5' gccRccATGG
- Capital letters represent highly conserved bases
- Lower case letters represent bases that are common but not as highly conserved
- A short distance past the stop codon, eukaryotic gene have a polyadenylation site where the mRNA is cleaved and the poly(A) tail added
	- 5' AAUAAA


**Transcription Factor Binding Site
- Sequence base method remain important for exploring how predicted genes might be regulated by identifying binding sites for known transcription factors
	- Jaspar
	- TFSEARCH
	- MAST

**Ongoing Need for Gene Discovery**
- Human genome finished in 2003
- Next-generation sequencing offers more sequencing faster and cheaper
- Study of RNAs have become a key area of molecular genetics
	- Increasing recognition that short functional RNA molecule play important roles in the lives of cells
- tRNA
	- Component of ribosomes, spliceosome, and some enzymes


**ORFs, Consensus Sequences, and Gene Structure**
- A gene that covers most bases is a transcription unit
	- A segment of DNA that can be transcribed into RNA
- A transcription unit must have a promoer
	- DNA sequences allowing RNA polymerase to identify and transcribe the gene
	- If a protein coding gene, there must be an ORF
		- Start and stop codon
- FOr a protein coding gene, eukaryotic ribosomes beings translating at the first start codon of an mRNA
	- Only contain one ORF
	- ORF may occur in segments called exons broken up by non-coding regions called introns
- In prokaryotic cells
	- Ribosomes find the correct start codon by binding to sequence known as the Shine-Dalgarno sequence or ribosomes binding site
	- May contain multiple ORFs, each encoding a distinct protein
- Identify unbroken ORFs by looking for start and stop codons

![[Pasted image 20251103132614.png]]

- DNA sequence that are important in gene expression in prokaryotes and eukaryotes
	- Consensus sequences
- If any natural promoters contain exactly there two sequences
- Genes expressed at a high level tend to have closely matching promoters sequences, whereas weaker promoters are father from the consensus sequence
- A graphical representation called a sequence logo gives a better idea of the relative occurrences of the four nucleotides at each position

![[Pasted image 20251103132843.png]]

![[Pasted image 20251103132906.png]]
