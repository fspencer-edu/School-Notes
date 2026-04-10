## Understanding the Problem: Mammalian Evolution

- All mammals are evolutionarily related
- Hippos are the closest living relative of whales
	- Molecular evolution
- Linnaeus proposed the first systematic classification of living things (1700s)
	- Plant and animal kingdoms
	- Phyla, classes, order, families, genera, and species
- Observable morphological characteristics were used
- Darwin (1859) proposed from common ancestors from natural selection
	- Phylogenetics
- Gene and genome sequencing expands the scope of evolutionary evidence to include variation of genes
	- Mutations that are conserver over time

## Bioinformatics Solutions: Molecular Phylogenetics and Distance Measurement

- All DNA and protein alignment techniques are based on evolutionary relatedness
- Evolution can be defined as descent with modification from a common ancestor
- A phylogenetic tree shows the evolutionary history of sub-species
- Phylogenetic tree is always a model of evolution
	- Based on anatomy
	- Fossil records
	- Genetic sequence

![[Pasted image 20251031130438.png]]

- Natural selection
	- Favourable alleles reduced others' contribution to the next generation
- Molecular evolution is a specific application of bioinformatics
- Use a gene as a molecular clock and directly measure the substitutions that have occurred over evolutionary time
- Molecular evolution can determine relationships in the absence of physical clues
- Molecular evolution can never establish with certainty
## Understanding the Algorithm: Measuring Distance

- Constructing a phylogenetic tree uses evolutionary distance between each pair of species within an orthologous gene
- Distance metrics
	- Quantitative measures of evolutionary time required to account for observed sequence divergence
- Group related species (clustering algorithms) and generate the tree
- A gene must be chosen whose sequence can represent the species
- Substitution rate, $r$
	- Measure the rate of change of two DNA sequences over the time since they shared a common ancestor
- Changes occurred $(K)$
- Amount of time passed $(T)$

$r = K/2T$

- 2T because each species has been evolving independently for time T
- Calibrate a particular molecular clock by using external data
	- Radiometric dating
	- Fossil record

![[Pasted image 20251031131203.png]]

**Jukes-Cantor Model**
- Issues of substitutions is the "hidden" mutations
- Jukes-Cantor model makes the assumption that a set of sequences with few overall variations is less likely to have undergone multiple substitutions at any particular site over evolutionary time
	- Than a set of sequences with a large number of variation
- $\alpha$ = rate of change per unit time
- $t$ = small unit of time

- Probability of one of the three possible substitutions occurring at a given nucleotide is $3\alpha \Delta t$
	- A -> C
	- A -> G
	- A -> T

- Model assumes all changes are equally likely
	- A mutation is just as likely to replace A with T as with G, or C
- Estimation of the number of substitutions (visible and hidden), between sequences $a$ and $b$

$K_{ab} = -3/4ln(1 - 4/3D_{ab})$

D = fraction of observed substitutions (total substitutions/total nucleotides)
- Calculate K under the model, and obtain an estimate of evolutionary distance


**Kimura's Two-Parameter Substitution Model**
- Extension of the Jukes-Cantor model (1980)
- Accounts for "hidden" substitutions in estimating K
- Recognizes that transitions occur more frequently than transversions and introduces additional parameters to account

$S$ = transitions
$V$ = transversion

$K_{ab} = 1/2ln(1/(1 - 2S - V) + 1/4ln(1/(1-2V))$

**Tamura's Three Parameter Model**
- The frequencies of nucleotides are not uniform across the thee of life
	- G + C nucleotide can vary widely
		- 40% in human
		- 60% in bacterial
- Accounts for G+C content bias and transititon/transversion bias

$K_{ab}= -C ln(1 - S/C - V) - 1.2(1 - C)ln(1-2V)$

$S$ = fraction of transitions
$V$ = fraction of transversion

$C = GC_{s1} + GC_{s2} - 2 * GC_{s1} * GC_{s2}$

$GC_{s1}$ = fraction G+C in sequence 1
$GC_{s2}$ = fraction G+C in sequence 2


**Other Models**
- Tamura-Nei model
	- Recognizes not all transitions occur with the same frequency
- Felsenstein model
	- Allows for differences in the frequency of mutation at different sites
- Hasegawa-Kishino-Yano (KHY85) model
	- Allows variable base frequencies and a separate transition and transversion rate
- Best model depend on choice if distance metric

## Chapter Project: Evolution of the Whale

### Part 1: Whales, Porpoises, and the Mammalian Phylogenetic Tree

- Evolution of marine mammals and their common ancestor with land mammals
- Many genes might be chosen for different purposes
- Hemoglobin subunits are well conserved among vertebrate animals and have been used to examine evolutionary relationships
- 16S rRNA is found in every living creature
- Use gene encoding casein, the protein in milk
- All mammals nurse their young and this gene is conserved among the species
	- $\beta$-casein, encode the major milk protein
- 3 major groups of mammals
	- Marsupials
	- Montremes
	- Placental mammals
- Gene name for $\beta$-casein is CSN2
	- Use rat to represent rodents
	- Dromedary camel for herbivores
	- Dog for carnivores
- For each gene, obtain the complete nucleotide sequence of only the coding region
- Add sequence for whale and porpoise

- Whales and porpoises are more closely related to each other than to any of the other mammals
- They are much more closely related to the camel than to the rodents or carnivores

![[Pasted image 20251031133051.png]]

- Phylogram
	- Branch length is proportional to calculated evolutionary distance
- Other diagrams
	- Cladogram
	- Unrooted radial format


## Part 2:# Evolutionary Distance in the Mammalian Phylogenetic Tree

- Phylogenetic tree data using a multiple alignment program called MUSCLE
- Gblocks identifies the portions of the alignment suitable for distance calculation and tree-building
- Eliminated gapped positions
- Use the Evolutionary Distance to calculate distance between pairs of sequences using the Jukes-Cantor, Kimura, and Tamura models


**Using Distance Metrics to Measure Similarity**
- Count mismatches between two aligned sequences and then use the count in the Jukes-Cantor formula
- Jukes-Cantor model ignores gaps


**Evolutionary Distance Algorithm Using the Jukes-Cantor Model**


![[Pasted image 20251031133707.png]]

## Alignment and Evolutionary Distance

**Understanding the Problem**
- Measuring evolutionary distance depends on good alignment of "molecular clock" sequences and data used


**Solving the Problem**
- Start by aligning sequences using Needle-Wunsch algorithm
- Semi-global alignment that does not penalize terminal gaps is the best choice
- Calculate the distance according to metric



**Measuring Evolution**
- Phylogram
	- Branch lengths are proportional to the calculated evolutionary distance between species
- Cladogram
	- All branches are brought out to the same point
	- Branches are connected by internal nodes
		- Common ancestor
	- Terminal nodes are modern species

![[Pasted image 20251031134128.png]]

- The two trees are rooted
	- There is an oldest ancestor common to all species
- A tree can be rooted only by comparison with an outgroup
	- A reference group related to the species being studies, but can is outside the group
- Unrooted tree does not represent a unique evolutionary pathway
- A node and all its branches form a clade
- The clade should be a monophyletic groups
	- All branches descend from a ancestor
- Polyphyletic groupings arise from similarities that do not correspond to ancestry
	- Warm-blooded animation
- Cladists
	- Every grouping in a phylogenetic tree is to be monophyletic
	- Character-based methods rooted in evolutionary theory
- Primary concern of phoneticists
	- Degree of similarity among species
	- More relaxed
	- Statistical, quantitative, distance-based methods to measure relatedness


**Mutations and the Molecular Clock**

- Trees are drawn based on differences in DNA
- Phylogenetics does not deal with small changes
	- Instead based on species population
- Evolutionary biologists refer to these mutations as substitutions
- If two species have the same gene (ortholog), that gene must have been present in the common ancestor
- Counting differences between DNA does now show the true picture of species evolution
- A molecular mutation should "tick" at a constant rate
- Substitutions should occur at a constant frequency regardless of the specific mutation or where it occurs
- Not every mutation is preserved over time
	- Advantageous
	- Disadvantageous (deleterious)
	- Neutral
- Substitutions rarely occurs in portions of genes that encode functionally important parts of a protein
- Most mutations are deleterious
	- Functionally constrained
- Silent mutations that lead to conservative amino-acid substitutions are more likely to be preserved
- Less critical regions of protein change faster, whereas introns, and regions between genes can change more rapidly


![[Pasted image 20251031135224.png]]



- Biochemical constraint on mutation
	- Transition mutation change on purine or pyrimidine to the other
- These preserved as substitutions far more frequently than transversion mutations
	- Chemical reaction or replication error that causes a mutation does not directly change both nucleotides in a base pair
	- Opportunity for the error to be detected and repaired before DNA replication
- Transversion
	- Mispair
	- Significantly altering DNA spacing
- Transition
	- Normal-width purine-pyrimidine mispair
	- More likely to escape detected


![[Pasted image 20251031135612.png]]