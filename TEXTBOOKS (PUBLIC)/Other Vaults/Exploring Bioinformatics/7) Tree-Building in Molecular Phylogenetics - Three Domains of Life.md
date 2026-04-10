## Understanding the Problem: Rooting the Tree

- Carl Woese initiated a revolution in how biologists think about the living world (1977)
- Based on 16S rRNA genes, Woese was able to recognize that prokaryotes were as evolutionary distant from the bacteria as the bacteria are from the eukaryotes
- Higher level of classification
	- Eukaryotes
	- Bacteria
	- Archaea
- Certain prokaryotes found in extreme environments were structurally different from bacteria in more temperate environments
- Prokaryotes split
	- Peptidoglycan in cell walls
	- Carbohydrate in bacteria
	- DNA is wrapped around histone proteins like the DNA of eukaryotic cells
	- Double-ended lipids
- Prokaryotes could represent the modern remnants of the ancestors of all living things

![[Pasted image 20251031140416.png]]


- Archaea represent the oldest evolutionary line
- Archaea have been around at lest as long as the oldest bacteria

## Bioinformatics Solutions: Tree-Building

- Align orthologous genes and produce phylogenetic trees
- Sequence diversity can be related to evolutionary distance
- Any gene shared by two groups could be used to determine evolutionary distance
- Different sequences have different functional constraints

- Tree-building is more complicated
- Draw 3 different unrooted trees to show relationships of 4 species
- Distance data can help show relation, but in each shows a unique evolutionary pathway
- Phylogeneticists call "tree space" intractably complex unless dataset is small


![[Pasted image 20251031140809.png]]

- All tree-building method depend on a multiple sequence alignment of the genes
- A multiple alignment editor such as Jalview or BioEdit can be used to make adjustments
- Gapped positions can be removed from the alignment using a program such as Gblocks
- Result is a multiple alignment where every mismatched nucleotide or amino acid should represent the result of a substitution over evolutionary time

Distance-Based Method
- Clustering algorithm to decide how species should be grouped
- UPGMA
- Neighbour-joining (NJ)

Character-Based Methods
- Probabilistic
- Attempt to find highest probability tree given that model and particular dataset
	- Parsimony
	- Evolutionary pathway that is the simplest
	- Bayesian statistics to find an optimal tree


## Understanding the Algorithm: Clustering Algorithms

- Goal of phylogenetic trees
	- Reveal evolutionary relationship
	- Classify
- Hierarchical clustering is use to related similar objects in largest clusters and larger ones
- Agglomerative clustering begins with individual objects and then merges the cluster into a single large group

![[Pasted image 20251031141326.png]]

- To find distance between individual sequences
	- Clustering requires linkage method, which determines how the distance metric is applied when two groups are compared
	- After computing distance there is a merge step
- Construct a tree for 6 species (A-F) that all diverge from a common ancestor
	- Find orthologous gene
	- Align sequences
	- Apply distance metric
	- Construct matrix
- Agglomerative clustering algorithm works by sequentially merging the most closely related elements into clusters
- Choose several linkage methods

Single Linkage
- Calculate the distances between each item in one cluster and each item in the other and chooses the smallest distance
- Elements that are not tightly groups

Complete Linkage
- Opposite to single linkage
- Largest individual distance value is chosen
- Items are tightly grouped

Centroid Linkage
- Uses the distance between the centres of the clusters


![[Pasted image 20251031141727.png]]

![[Pasted image 20251031141759.png]]


**Agglomerative Clustering Algorithm**
1) Determine distance between sequences by alignment and a distance metric
2) Ignore diagonal
3) Redraw the distance matrix with the merge cluster
	1) Use linkage method to determine distance between clusters
4) Repeat steps 2 and 3 until only one cluster remains

- Newick Format
	- Clustering process
- Merge A and B -> (A, B)
- (A, B) is merged with C -> ((A, B), C)
- ((A, B), C) -> (((A, B), C), D) -> ((((A, B), C), D), (E, F))
- Each cluster has a common ancestor
- Agglomerative clustering algorithm is used in many distance based methods for calculating phylogenetic groupings

UPGMA (Unweighted Pair-Group Method with Arithmetic Mean)
- Linkage method
- Calculate the distance between two clusters by averaging the distance
- Assumes constant rate of evolution


![[Pasted image 20251031142301.png]]


![[Pasted image 20251031142327.png]]

## Chapter Project: Placing the Archaea in the Tree of Life



### Molecular Clocks and the Archaea

**Developing the Dataset**
- A sequence alignment is used to serve as the bases for a phylogenetic tree
	- Gene conserver across all three domains
	- Accessor factor
		- Involved in translation
		- Brings amino acid-carrying tRNA into ribosome
	- EF1- in eukaryotes
	- EF-Tu in prokaryotes
- DNA sequences change faster than protein sequences
- Use protein rather than DNA

**A Distance-Based Tree Using UPGMA**

Phylogeny.fr

- Calculate UPGMA with EMBOSS
	- MUSCLE for alignment
	- Gblocks for cuuration
	- ProtDist/FastDist + BioNJ for tree construction
	- TreeDyn for tree visualization
- The result is distance format in Phylip format

emboss.bioinformatics.nl

- Upload distance matrix

**Neighbour-Joining Algorithm**
- NJ is a distance based method, but models evolution differently

**Character-Based Algorithms**
- Consider individual characters
	- Nucleotides
	- Amino acids
- Phylogeny.fr is PhyML, a character-based algorithm that uses maximum likelihood
- Likelihood model can be further extended to use Bayesian statistics
- Bayes' theorem involves an initial prior probability leading to the computation of a posterior distribution of trees with high likelihood given the dataset
- Iterative repetitively using the outcome of one computation as the prior distribution

**Phylogenetic Trees Using Agglomerative Clustering**
- Develop a program to perform agglomerative clustering using the single linkage method
- Final tree in Newick format
- Hierarchical clustering determined distances between clusters using a linkage method
- A distance matrix was used to represent cluster distances

![[Pasted image 20251102183320.png]]

- This structure would not change during the program
- Copy of the structure is made to represent merging clusters

![[Pasted image 20251102183409.png]]

- Reduce the size of the nested hash structure
- Left with two keys that represent the final two clusters to merge


**Agglomerative Clustering Algorithm to Determine Evolutionary Relatedness**

- Cluster a set of data items from a set of sequence distances in a phylip formatted file

![[Pasted image 20251102183548.png]]

### The Neighbour Joining Method

**Understanding the Problem: Determining Branch Lengths**
- Simple agglomerative clustering is ultrametric
	- Assumes a constant rate of evolution or a molecular clock
- Distances between sequences may not be ultrametric

**Solving the Problem**
- The NJ method is an alternative that does not require the assmuption of a constant rate of evolution
- Uses desired metric
- NJ calculates a tnrasformed distance value when calculating distances between the remaining clusters at each iteration
- Therefore, branch lengths correspond to the observed distance between species
- Each iteration of the clustering algorithm begins by calculating an r value for each cluster, representing the corrected net distance between it an all other clusters
- Average distance between a given cluster, x, and each other cluster (i), with n total clusters

![[Pasted image 20251102184510.png]]

$d_ix$ = the distance between cluster x and i from previous iteration


- r values are used to compute transition distances (td)

![[Pasted image 20251102184608.png]]

- The cluster pair that has the smallest transition distance is merges
- New distances are calculated after the formed cluster (K)

![[Pasted image 20251102184647.png]]

- Process repeats fro all clusters
- Branch lengths within the tree must be calculated
- Distance from cluster i to k must be calculated as two branch lengths, from each of the clusters to their shared ancestor K

![[Pasted image 20251102184807.png]]

Find transformed values, $r_x = \sum d_{ix}/(n-3)$

$r_A = 13$
$r_B = 12$
$r_C = 11.34$
$r_D = 11$
$r_E = 10$

- Compute transition distance values for first iteration, resulting the transition matrix
- Lowest value in the transition matrix is in the cell represented by clusters A and B
- New distance matrix is formed with AB

![[Pasted image 20251102185017.png]]

**Find Transition Matrix**

$td_{XX} = d_{ij} - r_i - r_j$

$d_{AB} = 5$

AB = -20
$td_{AB} = d_{AB} - r_A - r_B = 5 - 13 - 13 = -20$
$td_{AC} = 11 - 13 - 11.34 = -13.34$

**Recalculated Distance Matrix**

AB-C = 8

$d_{(AB), C} = (d_{AC} + d_{BC} - d_{AB}) /2 = (11 + 10 - 5)/2 = 8$

- Merge implies two species have a common ancestor (AB)
- To find the branch length apply

![[Pasted image 20251102185058.png]]

**Calculate Branch Lengths**

$d_{A(AB)} = (5 + 13 - 12)/2 = 3$
$d_{B(AB)} = (5 + 12 - 13)/2 = 2$

![[Pasted image 20251102185134.png]]

- Next iteration begins by re-calculating transformed r values

AB = 12.5
C = 10.5
D = 9.5
E = 8.5

- Arbitrarily choose to merch AB with C or cluster D with E
- New distance matrix is population with previous iteration distances, except for newly created cluster

![[Pasted image 20251102185320.png]]

- Now merge any of the remaining clusters
- Choose to merge ABC and D
- Calculate the two internal nodes by finding the distance between a species in cluster and a species in another, then subtracting the calculated branch lengths
- NJ method has produced an unrooted tree
- UPGMA produced rooted trees
- Trees match the original distances, demonstrating the additivity property of NJ method

![[Pasted image 20251102185627.png]]

**Programming the Solution**
- The NJ method recalculated distances at each iteration from the previous cluster distances
- Run NJ as linkage method

![[Pasted image 20251102185759.png]]

- ((C:1,B:0):0.5,(A:0.5,(D:0.5, E:0.5):1));, after merging D with E, DE with A, and C with B


**What Is a Species?**
- Two organisms are members of the same species if they re able to mate and have fertile offspring


