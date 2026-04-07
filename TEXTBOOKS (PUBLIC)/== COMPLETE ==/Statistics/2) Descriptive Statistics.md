# Stem-and-Leaf Graphs (Stemplots), Line Graphs, and Bar Graphs

- Stem-and-leaf graph (stemplots)
	- A data visualization tool that organizes numerical data by splitting each number into a leading digit and final digit
- Outliers
	- An observation of data that does not fit the rest
	- Extreme value
- Side-by-side stemplots
	- Comparison of two data sets
- Line graph
	- x and y axis consist of data value
- Bar graphs
	- Consist of bars that are separated from each other
# Histograms, Frequency Polygons, and Time Series Graphs

- Histogram
	- Consists of contiguous boxes
	- Vertical bars are frequency or relative frequency


$RF = f/n$

RF = relative frequency
f = frequency
n = total number of data values

- Continuous data
	- Group data into equal-interval numerical bins
- Discrete data
	- Create a frequency table listing each unique value and its count

- A value is counted in a class interval if it falls on the left boundary, but no on the right boundary

## Frequency Polygons

- Frequency polygons are analogous to line graphs
- Skewed
	- An asymmetrical data distribution where the peak is off-centre, with a long tail pointing toward the direction of skewness
	- Right-skewed
		- Tail on the right
	- Left-skewed
		- Tail on the left

## Time Series Graph

- Paired dataset
	- Consists of two related data samples where each observation in one group corresponds directly to a specific observation in the other
	- Cartesian coordinate system

# Measures of the Location of the Data

- Quartiles divide an ordered data set into 4 equal parts

$Q_1$ => 1/4 of the data thats on or below the first quartiels
$Q_2$ => 1/2 of the data thats on or below the second quartile
$Q_3$ => 3/4 of the data falls on or below the first quartile

- Quartiles divided ordered data into quarters
- Percentiles divide ordered data into hundredths

- Median
	- Centre value

- Interquartile range
	- A number that indicates the spread of the middle half of the data
	- Difference between $Q_3$ and $Q_1$

IQR = $Q_3$ - $Q_1$
Outlier = $(1.5)(IQR)$

$Q_1 - (1.5)(IQR)$ 
$Q_1 + (1.5)(IQR)$ 

- IQR can help to determine potential outliers
- A value is suspected to be a potential outlier if it is less than 1.5 IQR below the first quartile or more than 1.5 above the third quartile


## Finding the kth Percentile

$k$ = the kth percentile
$i$ = index
$n$ = total number of data

- Order data

$i = k/100(n+1)$

- If i is an integer, then the kth percentile is the data value in the ith position in the ordered set of data
- If i is not an integer, then round i up and round i down to the nearest integer
- Average the two data values

## Finding the Percentile of a Value in a Data Set

- Order the data

$x$ = number of data values counting from bottom of the data up to the the percentile value (not including)
$y$ = number of data values in the percentile
$n$ = total number of data

$(x + 0.5y)/n(100)$

## Interpreting Percentiles, Quartiles, and Median

- A percentile indicates the relative standing of a data value when data are sorted into numerical order from smallest to largest
- Percentages of data values are less than or equal the pth percentile
	- Low percentiles always correspond to lower data values
	- High percentiles always correspond to higher data values

# Box Plots

- Box plots
	- Also called box-and-whisker plots
	- Box-whisker plots
- Graphical image of the concentration of the data
	- Minimum value
	- First quartile
	- Median
	- Third quartile
	- Max value

- The smallest and largest data values label the endpoints on the axis
- The first the third quartile marks the end of the box
- The middle 50% of the data falls inside the box

<img src="/images/Pasted image 20260406151820.png" alt="image" width="500">

# Measures of the Centre of the Data

- Mean
	- Arithmetic average
- Median
	- Middle value in an ordered dataset
- Mode
	- Most frequency value

$\bar{x}$ = sample mean
$\micro$ = population mean

- As the sample size becomes larger, then the mean of the sample gets closer to the population mean
- $\bar{x} \rightarrow \micro$

## Sampling Distribution and Statistic of a Sampling Distribution

- Sampling distribution is a relative frequency distribution
- A statistic is a number calculated from a sample

## Calculating the Mean of Grouped Frequency Tables

- When only grouped data is available, cannot compute the exact mean for the data set
- Frequency table
	- A data representation in which grouped data is displayed along with the corresponding frequencies

$mean = sum/n$

- Midpoint

$M = (lower + uppe)r / 2$

Mean of frequency table $= \sum fm / \sum f$

$f$ = frequency of the interval
$m$ = midpoint

# Skewness and the Mean, Median, and Mode

- Symmetrical distribution
	- Vertical line can be drawn at some point in the histogram such that the shape to the left and the right are mirror images of each other
- In a perfectly symmetrical distribution, the mean, median, and mode are the same

- Mean is affected by outliers that do not influence the median
- Mean is often less than the median if data is skewed to the left
- Mean is often more than the median if data is skewed to the right

# Measures of the Spread of the Data

- Standard deviation
	- Measure of variation or spread in the data
	- How far data values are from their mean

## The Standard Deviation

- Provides a numerical measure of the overall amount of variation in a dataset
- Can be used to determined whether a particular data value is close to or far from the mean

- The standard deviations is always positive of zero
- Small when data is concentrated close to the mean
- Larger when values have more deviation

### Calculating Standard Deviation

- Deviation is the difference of "x - mean"
	- $x - \bar{x}$
- Standard deviation
	- Sample
		- $s = x - \bar{x}$
	- Population
		- $\sigma = x - \micro$

$s$ = standard deviation of sample
$\sigma$ = standard deviation of population


- Variance
	- Average of the squares of the deviations
		- $s^2$
		- $\sigma^2$
- For a population, the average of the squared deviations is divided by $N$
- If the data are from a sample, divided by $n-1$, one less than the number if items in the sample

**Sample Standard Deviation**
<img src="/images/Pasted image 20260406153552.png" alt="image" width="500">

**Population Standard Deviation**
<img src="/images/Pasted image 20260406153603.png" alt="image" width="500">


## Sampling Variability of a Statistic

- Sampling variability of a statistic
	- How much the statistic varies from one sample to another
- Standard error of the mean
	- $\sigma/\sqrt{n}$

$s^2$ = sample variance

$s$ = sample standard deviation

- The sum of the deviations, is always zero
- The variance, is the average squared deviation
- The sample variance is an estimate of the population variance
	- Based on theoretical mathematics, and divided by $(n-1)$ for a better estimate of the population variance
- Variability in data depends on the method by which the outcomes are obtained

# Descriptive Statistics


---

- Box plot
- First quartile
- Frequency
- Frequency polygon
- Frequency table
- Histogram
- Interquartile range
- Interval
- Mean
- Median
- Midpoint
- Mode
- Outlier
- Paired data set
- Percentile
- Quartiles
- Relative frequency
- Skewed
- Standard deviation
- Variation
- 