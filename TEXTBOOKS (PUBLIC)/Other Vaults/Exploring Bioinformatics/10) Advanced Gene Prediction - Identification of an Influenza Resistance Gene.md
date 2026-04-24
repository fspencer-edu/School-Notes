## Understanding the Problem: Exon Prediction

- One of the surprises in the "rough draft" of the human genome was the small number of protein coding genes
	- Less than 30,000 genes
- Annotation of the genome, the identification of genome elements and their functions is an ongoing effort

<img src="/images/Pasted image 20251103133246.png" alt="image" width="500">

- A sequence-based methods of gene prediction are the most straight forward and reliable in prokaryotes
- In eukaryotes, problems arise
	- No Shine-Dalgarno sequence
	- Translation begins at the first start codon in the mRNA
	- Unambiguous identification of the transcription start site is difficult
	- Promoters are a collection of transcription factor binding sites
	- Very few unbroken ORF
- Gene prediction is used to identify genes within a newly sequences genome, but is also valuable in identifying genes when a particular genome region has been associated with a disease or phenotype of interest

<img src="/images/Pasted image 20251103133546.png" alt="image" width="500">

## Bioinformatics Solutions: Content- and Probability-Based Gene Prediction


- Cannot rely on ORFs and consensus binding sites to clearly define the set of genes in a eukaryotic genome
- Codon usage is an example of a content-based method of gene prediction
	- A putative sequence is examined to see if the frequency of usage of different codons matches that observed for the organisms as a whole
- A content-based method is not precise
- Find regions where codon usage matches the expected frequency well or poorly
- Combining two methods, codon usage and consensus boundary sequence, yields a better prediction
- Bidden Markov models (HMMs)

## Understanding the Algorithm: Codon Usage, Frequency Matching, HMMs, and Neural Networks

2 Content-base methods of gene prediction
1) Codon usage
2) Identification of GpG islands


**Using Codon Frequencies in Gene Prediction**
- Codons are not used with equal frequency
- Some amino acids are much more common in protein than others
	- Vertebrate protein sequence
		- Serine most common (8%)
		- Tryptophan least common (1%)
- Genetic code is redundant
- A exon-intron boundary would be expected to separate a region where the codon frequency closely matches the expected frequency for the organism from a region where the frequency matches poorly
	- Intron-exon boundary would do the reverse

<img src="/images/Pasted image 20251103134304.png" alt="image" width="500">

- Several codon usage measures are in common use
	- Codon bias index (CBI)
		- Compares the usage of preferred to the random occurrence of codon
			- 0 to 1
- Sequence with sliding window approach, find points at which the boundary between A and B correspond to a drop in CBI to near zero, or an increase in CBI from near zero to larger number
- Putative boundaries can be rejected if the conserved GT and AG pairs are not present

**Prediction of CpG Island**
- Identification of CpG island adds valuable corroboration and can be used in combination with sequence-based methods and exon prediction techniques to help identify the first exon of a gene
- Find CpG island with a frequency matching algorithm

<img src="/images/Pasted image 20251103134619.png" alt="image" width="500">

<img src="/images/Pasted image 20251103134627.png" alt="image" width="500">


**HMMs for Gene Prediction**
- More popular gene prediction programs are now based on implementations of hidden Markov modeling, probability-based algorithms that use sequence and content data to inform a calculation of the likelihood a given sequence is part of an intron or exon
- HMM seeks to draw a conclusion about a hidden value based on a set of observations and a set of known probabilities
- Nucleotides are input
- 3 states
	- Exon
	- Intron
	- Splice site
- For each state, nucleotide could be A, C, G, or T
- Use data about genes to determine emission probabilities ($e$)
	- Likelihood of each output
- Nucleotide bias at the splice-donor site
- A more realistic HMM could take into account all the data depicted in sequence logo
- The last parameter needed are the transition probabilities ($t$)
	- Likelihood of changing from one state to the next vs. the likelihood of remaining in the same state

pg 196

## Chapter Project: Identifying and Influenza Resistance Gene