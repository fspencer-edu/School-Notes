
**4 Types of Datasets**
1. Tables
2. Networks
3. Fields
4. Geometry

- Other collections
	- Clusters
	- Sets
	- Lists

**5 Data Types**
- Items
- Attributes
- Links
- Positions
- Grids

- Dataset
	- Static or dynamic
	- Ordinal
	- Categorical
	- Quantitative

<img src="/images/Pasted image 20260401135219.png" alt="image" width="500">

- Semantics
	- Real-world meaning of data
- Type
	- Structural or mathematical interpretation
- Metadata
	- Structured information that describes, explains, located, or manages other data

## Data Types

- 5 data types
	- Items
		- An individual entity that is discrete
	- Attribute
		- A specific property that can be measured, observed or logged
		- Also called variable, data dimension
	- Links
		- Relationship between items, within a network
	- Positions
		- Spatial data, providing a location in two dimensional or 3D space
	- Grids
		- Strategy for sampling continuous data in terms of both geometric and topological relationships between its cells

## Dataset Types

- Dataset
	- Any collection of information that is the target of analysis
- Dataset types
	- Tables
	- Networks
	- Fields
	- Geometry
	- Clusters
	- Sets
	- Lists

**Tables**
- Rows and columns
- Items and attributes
- Flat table
	- Each cell in the table is fully specified by the combination of a row and a column
	- A cell contains a value for that pair
- Multi-dimensional table
	- More complex structure for indexing into a cell, with multiple keys

**Network and Trees**
- Networks
	- Used to represent relationship between two or more items
	- Items in a network is called a node
	- A link is a relationship between two items
	- Referred to as graphs
	- Node = vertex
	- Link = edge
- Nodes and links can have associated attributes

- Trees
	- Networks with hierarchical structure
	- Do not have cycles
	- Each child node has only one parent node pointing to it
	- Hierarchical charts

**Fields**
- The field dataset type also contains attribute values associated with cells
- Each cells in a field contains measurements of calculations from a continuous domain
- Temperature, pressure, speed, force, density
- Continuous data
	- How frequent sample measurements are taken
	- Interpolations
		- Values in between the sampled points
	- Interpolation allows reconstructing of data into a new view from an arbitrary viewpoint that is faithful to what is measured
		- Signal processing
		- Statistics
- Discrete data
	- A finite number of individual items exist


- Spatial fields
	- Continuous data
	- Cell structure of the field is based on sampling at spatial positions
- Non-spatial data = abstract data

- 2 specializations
	- Scientific visualization (scivis)
		- Spatial position is given with the dataset
		- Continuous data within the mathematical framework of signal processing
	- Information visualization (infovis)
		- Used of space in a visual encoding is chosen by the designer
		- Chosen idiom is suitable for the combination of data and task

- Grid types
	- A field that contains data created by sampling at completely regular intervals
	- Cells form a uniform grid
	- No need to store grid geometry in terms of location in space, or grid topology in terms of how each cell contains with it neighbouring cell
- Rectilinear grid
	- Non-uniform sampling
	- Efficient storage of information that has high complexity in some areas and lows complexity in others, at the cost of storing geometric location
- Structured grid
	- Curvilinear shapes
	- Geometric locations of each cell needs to be specified
- Unstructured grids
	- Flexible
	- Topological information is stored explicitly in addition to their spatial positions


**Geometry**
- Specifies information about the shape of items with explicit spatial positions
- Points, lines, curves, planes, or 3D volumes
- Hierarchical structure at multiple scales
- Contours
- Boundaries
- Combines computer graphics


**Other Combinations**
- Set
	- Unordered group of items
- List
	- A group of items with a specified ordering
	- Also called array
- Cluster
	- A grouping based on attribute similarity
- Path
	- An ordered set of segments formed by links connecting nodes
- Compound network
	- A network with an associated tree
		- All of the nodes in the network are the leaves of the tree
		- Interior nodes in the tree provide a hierarchical structure for the nodes that is different from network links between them

- Data abstraction

**Dataset Availability**
- Static = offline
- Dynamic = online

## Attribute Types


<img src="/images/Pasted image 20260401141753.png" alt="image" width="500">

**Categorical**

Ordered: Ordinal and Quantitative

Sequential vs. Diverging

Cyclic

Hierarchical Attributes

## Semantics


Key vs. Value Semantics

Flat Tables


Multi-dimensional tables

Fields

Scalar Fields

Vector Fields

Tensor Fields

Semantics


Temporal Semantics

Time-Varying Data

