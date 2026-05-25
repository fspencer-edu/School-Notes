# Statistical Hypothesis Testing

- Null hypothesis, $H_0$
	- Represents a default position
- Alternative hypothesis, $H_1$
	- Represents the rejection of $H_0$

# Example: Flipping a Coin

$H_0$: $p = 0.5$
$H_1$: $p \neq 0.5$

- X is a $Binominal(n, p)$ random variable

```python
from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Returns mu and sigma corresponding to a Binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
    
from scratch.probability import normal_cdf

# The normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is greater than lo."""
    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is between lo and hi."""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """The probability that an N(mu, sigma) is not between lo and hi."""
    return 1 - normal_probability_between(lo, hi, mu, sigma)
    
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
```
- After $n=1000$ flips
- X should be distributed approximately normally with mean 500 and standard deviation 15.8
- Significance
	- Type 1 error => false positive
		- Reject $H_0$ even though it is true
	- Type 2 error => false negative
		- Accept $H_0$ even though it is false

```python
from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    Returns the symmetric (about the mean) bounds
    that contain the specified probability
    """
    tail_probability = (1 - probability) / 2

    # upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound
```

- Two-sided 95% bounds
	- `(469, 531)`
	- Therefore, if X is below or above, reject $H_0$
- One sided test
	- Has higher power because all the rejection probability is placed in the upper tail

```python
# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# a type 2 error means we fail to reject the null hypothesis,
# which will happen when X is still in our original interval
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.887
hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (< 531, since we need more probability in the upper tail)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.936
```

# p-Values

- p-values
	- Compute the probability, assuming $H_0$ is true
- If p-value is greater than the 5% significance, than we do not reject the null

```python
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    How likely are we to see a value at least as extreme as x (in either
    direction) if our values are from an N(mu, sigma)?
    """
    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)
        
two_sided_p_value(529.5, mu_0, sigma_0)   # 0.062
```

# Confidence Intervals

- The previous test where hypotheses about the value of the heads probability p
	- Parameter
- Confidence interval
	- Estimate the probability and the confidence of this estimate

$$  
\text{CI} = \text{estimate} \pm z_{\alpha/2} \cdot \text{SE}  
$$
CI for Mean
$$  
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}  
$$
# p-Hacking

- Erroneously rejects the null hypothesis only 5% of the time
- Manipulating your data analysis until you get a statistically significant result (usually $p<0.05$)

$$  
P(\text{at least one false positive}) = 1 - (1 - \alpha)^k  
$$

# Example: Running an A/B Test


# Bayesian Inference

- Bayesian inference is a way to update what you believe about an unknown parameter after seeing data
- Treating the unknown parameters as random variables
	- Prior distribution
	- Posterior distribution
- Make probability judgements about the parameters
- Beta distribution
	- Conjugate prior to the binomial distribution

```python
def B(alpha: float, beta: float) -> float:
    """A normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:          # no weight outside of [0, 1]
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
```
<img src="/images/Pasted image 20260428161939.png" alt="image" width="500">

# For Further Exploration