
ANOVA = Analysis of Variance
- Hypothesis test comparing averages between more than two groups

# One-Way ANOVA

- Determine the existence of a statistically significant difference among several group means
- Uses variances

1. Each population from which a sample is taken is assumed to be normal
2. All samples are randomly selected and independent
3. The populations are assumed to have equal standard deviations (or variances)
4. The factor is a categorical variable
5. The response is a numerical variable

## The Null and Alternative Hypotheses

- The null hypothesis shows that all group population means are the same
- The alternative hypothesis is that at least one pair of means id different

<img src="/images/Pasted image 20260407150025.png" alt="image" width="500">

# The F Distribution and the F-Ratio

- F-distribution
	- Used to compare variances or means across multiple groups
	- Derived from Student's t-distribution
- 2 sets of df
	- one for the numerator and one for denominator
- F-ratio
	- Determine if group means differ significantly

$F = \frac{between}{within}$

$SS_{between}$
- Large => groups are different
- Small => groups are similar

$SS_{within}$
- Large => lots of randomness/noise
- Small => data points are tight

**Sum of Squares (SS)**
- Add up squared differences
- Large F value => between variance > within variance
	- Groups are different
- Small F value => between variance < within variance
	- Differences are likely random

**Total Variation ($SS_{total}$)**

$SS_{total} = \sum x^2 - \frac{(\sum x)^2}{n}$

**Between Group Variation ($SS_{between}$)**

$SS_{between} = \sum (\frac{s^2_j}{n_j}) - \frac{(\sum x)^2}{n}$


**Within Group Variation ($SS_{within}$)**

$SS_{within} = SS_{total} - SS_{between}$

**Degrees of Freedom**

$df_{between} = k-1$
$df_{within} = n-k$


**Mean Squares**

$MS_{between} = \frac{SS_{between}}{k-1}$

$MS_{within} = \frac{SS_{within}}{n-k}$


**F-ratio**

$F = \frac{MS_{between}}{MS_{within}}$


**$H_0$ is True**
- Groups are basically the same
- $MS_{between} \approx MS_{within}$
- $F \approx 1$

**$H_0$ is False**
- Group means vary
- $MS_{between} \gt MS_{within}$
- $F \gt 1$


**F-Ratio when groups are the same size**

$F = \frac{n \cdot s_{\bar{x}}^2}{s^2_{pooled}}$

$n$ = sample size per group
$s_{\bar{x}}^2$ = variance of group means (between)
$s^2_{pooled}$ = average variance within groups (within)

## Notation

- F-distribution

$F \sim F_{df(num), df(denom)}$

$\micro = \frac{df(denom)}{df(denom)-2}$

# Facts About the F Distribution

1. Curve is not symmetric but skewed to the right
	1. Values cannot be lower than zero
2. There is a difference curve for each set of dfs
3. F statistic is greater than or equal to zero
4. df for the numerator and denominator get larger, the curve approximates the normal
5. Used for comparing two variances and ANOVA

<img src="/images/Pasted image 20260407152053.png" alt="image" width="500">

# Test of Two Variances

- F test of two variances
	- The populations from which the two samples are drawn are normally distributed
	- The two populations are independent of each other
- Sensitive to deviations from normality

Test
- Two populations have equal variance

$H_0: \sigma_1^2 = \sigma_2^2$

$F = \frac{s^2_1}{s^2_2}$

Distribution

$F \sim F(n_1 - 1, n_2 - 1)$

- If variances are equal, $F \approx 1$
- If variances are difference, $F \gt 1$

- Right tailed
	- Variance 1 > variance 2
- Left tailed
	- Variance 1 < variance 2
- Two tailed
	- Variances are different

---

- Analysis of variance
- One-way ANOVA
- Variance