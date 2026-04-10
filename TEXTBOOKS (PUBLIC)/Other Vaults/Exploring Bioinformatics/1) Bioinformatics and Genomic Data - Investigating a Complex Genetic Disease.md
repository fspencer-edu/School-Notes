
## Understanding the Problem: Parkinson Disease, A Complex Genetic Disorder

- At least 7 million people with Parkinson disease (PD)
- Incurable disorder of the central nervous system
	- Shaking, stuttering, difficulty walking, or involuntary muscle movement
- PD is a complex or multi-factorial disorder
	- Heritable
	- Not attributed to a single gene

Other Genetic Disorders
- Autism
- Type 2 diabetes
- Asthma
- Obesity
- PD symptoms result from the death of a population of brain cells (neurons) in substania nigra that produce neuro-transmitter dopamine
- Close relatives of Parkinson patients are to higher risk
	- No clear pattern of inheritance

Inheritance of alleles
- Dominant
	- Huntingtons disease
	- Hyper-cholresterolemia
- Recessive
	- Cystic fibrosis
	- Sickle-cell anemia
	- Phenylketonuria
	- Tay-Sachs disease
- 3 billion nucleotides accessible in GenBank
- Different genomes
- Bioinformatics is the new science at the interface of molecular biology and computer science
	- Explore
	- Analyze
	- And understand genomic data
- Computational biology focuses on storage, retrieval, manipulation, and analysis od DNA and protein sequences
	- Structure
	- Expression


![[Pasted image 20251030184522.png]]




## Bioinformatics Solution: Databases and Data Mining

- Analysis refers to data mining of existing information in genomic databases
- Entire human genome has been determined
- Not determined all the genes or functions
- Data
	- Phenotypes
	- Expression
	- Intron/exon
	- prediction
	- Transcription factor binding
- Only 15% of PD patients have a parent, sibling, or child with PD
- Number of regions have been correlated with PD through genome-wide association studies (GWAS)
- GWAS can only identify genome regions that may be associated with a disease, not specific genes or specific mutations

## Understanding Genomic Databases

3 Major Databases
1) GenBank
	1) Maintained by the National Center for Biotechnology Information (NCBI)
2) European Molecular Biology Laboratory (EMBL)
3) DNA Data Bank Japan
	1) National Institute of Genetic of Japan


- Primary databases have raw nucleotide or amino acid sequence information is deposited
- Annotated databases contain additional information
	- Location of protein-coding regions
	- Introns and exons
	- Other genetic features
	- References to scientific literature
- Gene expression experiments/polymorphisms on primary databases
- Genomic databases are divided into records (sequences) and then info fields
- Fields for the raw data itself
	- Locations of features
		- Coding sequences
		- Promoters
		- Introns
	- Annotations for references



**Metadatabases**
- Secondary databases or metadatabases select and combine data from other databases
- NCBI's Gene database pulls together the DNA sequences, the protein sequences, references and information on expression, alleles, and phenotypes, genomic location, and more for genes that have been studies
- OMIM (Online Mendelian Inheritance in Man) database combines human genetic diseases and the genes that contribute to them
- KEGG Pathways database focuses on metabolic pathways and the genes that encode metabolic proteins

**Database Searching**
- Searching requires a user interfaces
	- Web-based



![[Pasted image 20251030191847.png]]

![[Pasted image 20251030191903.png]]

![[Pasted image 20251030191914.png]]

**Genome Browsers**
- A genome browser acts like a meta-database that brings information from many genomic databases
- Genome browser as a GUI for genomic databases
- Graphical representation of chromosomal position of a specified gene or genome segment
- Viewed zoomed out to show more chromosomes
- Zoomed in to show regions of a gene or DNA sequence
- Tracks can be shown representing genes that are defined as transcription regions
	- Hypothesized transcripts
	- Database sources or types of evidence
- Introns and exons

**Syntax for the Entrez Search Interface to NCBI Databases**
- AND, OR, NOT
- Quotes
- Parentheses for grouping
- Asterisk for wildcard character
	- cys*
- Square brackets for limits
	- [Title]

![[Pasted image 20251030192324.png]]


- Additional tracks show binding sites for transcription factors, expression in specific tissues, and locations of known genetic variations (mutations or polymorphisms), methylation sites, repeated sequences, and comparisons with the genome of other organisms
- Genome browsers have tools for predicting the sizes of polymerase chain reactions (PCR)
- Most used is the USCS genome browser maintained by the Genome Bioinformatics group
- NCBI has a map viewing


## Chapter Project: Genome Regions Associated with Parkinson Disease

**Part 1: Genome Browsing to Identify Possible Disease-Associated Genes**

![[Pasted image 20251030193406.png]]

- GWAS requires a method to determine many genotypes for a large number of sites in the genome where genetic variation is known to occurs
- DNA micro-array using allele-specific oligo-nucleotides
	- Short, single-stranded DNA segments with sequences matching known polymorphic sites
- Simple nucleotide polymorphisms (SNPs) are placed in the genome where variation occurs among individuals in the form of a single nucleotide
- Identified 11 SNP sites where one allele was correlated with PD with a statistically significant frequency

- Genome browsing for SNPs is in the primary genomic database dbSNP
	- Uses accession numbers for identification
	- `rs11868035`
- Use UCSC genome browser
	- Assembly version
	- Accession code
- Ideogram, or schematic drawing of the chromosomes where the SNp occurs
- Dark and light regions represent bands that are visible when real chromosomes are stained
- Chromosome arms
	- p (short)
	- q (long)
- Chromosome position notation
	- 6q25.1
- Thick lines -> exons
- Thin lines -> introns
- Introns are shown with arrows to show direction of transcription and transcribed, but untranslated regions (UTRs)
- Medium lines -> Flanking sequences

- SNP is within an intron, not the coding sequence
- Mutations within introns, occurs near the boundary between intron and exon, and can effect the expression of a gene by impeding the slicing process
- SNP within a chromosome region important in PD, but no within the actual gene correlated with the disease
- Multiz Alignment track shows the alignment between human genome region and other sequenced vertebrates
- Genes expressed in the brain or in nervous system tissue are good candidates for PD
- Genes show in red in the GAD View track have been associated with some kind of disease
- GNF Expression Atlas track, red bars represent genes that were strongly expressed in the indicated tissues
	- Brighter red -> more expression
	- Black -> genes that is neither over- nor under expressed
- Genes in these region most likely to be involved in PD were SREBF1 and RAI1

**Part 2: Retrieving Sequences and Examining Genes in Detail**
- NCBI's map views, ENSEMBLE genome browser
- NCBI Nucleotide database contains downloadable sequences
	- Ex. SREBF1
- Some results are for nearby genes
- Use search terms to narrow resutls
- Nucleotide is stored in FASTA formate
	- Sequence with no spaces or numbers
	- One-line descriptive comment
- Features of sequence
	- Coding sequence (CDS)
	- mRNA
	- Exons
	- Protein

![[Pasted image 20251030194738.png]]





## Clues to a Genetic Disease
- An organism's genome is the complete set of all its genes
	- Encoded instructions for synthesizing one protein
- Every cell in an organism carries this same genome
- All members of a species have the same set of genes
- Each gene is a segment of DNA (deoxyribonucleic acid)
- Genes are joined together to make a long DNA molecule, called a chromosome that reside within the nucleus of very cell
- Individuals have the same set of genes, but different alleles
- An allele is a specific form of a gene
	- Nucleotides
	- A, T, C, G
- A person might have different gene encoding for hemoglobin protein, arising from mutations
- Genetic variations are known as polymorphisms
- Each individual carries 2 complete sets of genes
	- Mother
	- Father
- Allele type
	- Dominant
	- Recessive
- The allele expressed is the phenotype (observable characteristic)
- Genetic diseases result from alleles that encode an abnormal version of protein
	- Recessive
		- Sickle-cell anemia
		- Hemoglobin protein
	- Dominant
- Childs genotype can be probabilistically determined by genotype of parents
- Types of cells express different genes
- Gene expression is the process of making the protein
	- Transcription (RNA polymerase)
		- Copied nucleic acid information from one gene from DNA to messenger RNA (mRNA)
	- Translation (ribosome)
		- Decodes RNA to make protein


![[Pasted image 20251030195618.png]]



- DNA, RNA, and proteins are represented by simple strings of letters
- DNA
	- Nucleic acid molecule consisting of 2 long chains of nucleotides or "bases"
		- ~ 100 M longs
		- A, T, C, G
	- Base pairing
		- A <-> T
		- C <-> G
- Gene occur within the DNA sequence
	- Promoter indicates the site where RNA polymerase should begin transcription
	- RNA synthesizes single-stranded RNA using complementary base-pairing rules for DNA
		- Replaces T for U
	- Process continues until terminator
	- Produces mRNA transcript (coding sequence)
		- Start codon => AUG
		- Stop codon => UGA, UAA, UAG
- Ribosomes use these for translation into 3-nucleotide codons (amino acids)
- The resulting protein folds into a 3D structure

![[Pasted image 20251030200055.png]]

![[Pasted image 20251030200107.png]]

![[Pasted image 20251030200116.png]]