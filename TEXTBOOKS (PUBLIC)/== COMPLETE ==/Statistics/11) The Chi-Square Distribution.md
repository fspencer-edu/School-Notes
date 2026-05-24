
- Applications of chi-square distribution
	- Goodness-of-fit test
		- Data fits a particular distribution
	- Test of independence
	- Test of a single variance
# Facts About the Chi-Square Distribution

$\chi \sim \chi^2_{df}$

df = degree of freedom
$\micro = df$
$\sigma = \sqrt{2(df)}$

- The random variable for a chi-square distribution with k df is the sum of k independent, squared standard normal variables

$\chi^2 = (Z_1)^2 + (Z_2)^2 + ... + (Z_k)^2$

1. Curve is non-symmetric and skewed to the right
2. Different chi-square for each df
3. Test statistic for any test is always greater than or equal to zero
4. df > 90, the chi-squire approximates the normal distribution
5. Mean is located right of the peak

# Goodness-of-Fit Test

- The null and alternative hypothesis for this test may be written in sentences or states as equations or inequalities

$\sum_k \frac{(O -E)^2}{E}$

$O$ = observed values
$E$ = expected values
$k$ = data cells or categories

- The observed and expected values are the values if the null hypothesis is true
- df = number of categories - 1

- The goodness-of-fit test is always always right-tailed

<img src="/images/Pasted image 20260407134949.png" alt="image" width="500">


# Test of Independence

- Contingency table
	- A matrix format used to summarize the relationship between two or more categorical variables by displaying their frequency distribution

$\sum_{i \cdot j} \frac{(O -E)^2}{E}$

$O$ = observed
$E$ = expected
$i$ = rows
$j$ = columns

- Test of independence determines whether two factors are independent or not

# Test for Homogeneity

- Test for homogeneity
	- Draw conclusions about whether two populations have the same distribution

$H_0$ => distributions of the two populations are the same
$H_a$ => distributions of the two populations are not the same

$\chi^2$ = chi test
df = number of cols - 1
- All values in the table must be greater than or equal to five

# Comparison of the Chi-Square Test

1) Goodness of fit
	1) Decide whether a population with an unknown distribution fits a known distribution
2) Independence
	1) Decide whether two variables are independent or dependent
3) Homogeneity
	1) Decide if two populations with unknown distributions have the same distributions

# Test of a Single Variance

- A test of a single variance assumes that the underlying distribution is normal

$\frac{(n-1)s^2}{\sigma^2}$

$n$ = total number of data
$s^2$ = sample variance
$\sigma^2$ = population variance


---
- Contingency table