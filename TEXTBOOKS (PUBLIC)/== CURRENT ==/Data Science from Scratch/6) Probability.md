
- Probability
	- A way of quantifying the uncertainty associated with events chosen from some universe of events
- $P(E)$
	- Probability of the event E
- $P(A|B)$
	- Probability of the event A given B

# Dependence and Independence

- Independent
	- Two events do not effect each other

$P(A \cap B) = P(A)P(B)$

- Dependent
	- Two events if one changes the probability of the other

$P(A \cap B) = P(A)P(B|A)$

# Conditional Probability

- Conditional

$P(A | B) = \frac{P(A \cap B)}{P(B)}$

$P(A \cap B) = P(A|B)P(B)$

- When A and B are independent

$P(A|B) = P(A)$

# Bayes's Theorem

- Reversing conditional probabilities
- Want to know the probability of some event E conditional on some other event F occurring
- Only have information about F conditional on E occurring

$P(A)$ -> prior
$P(B|A)$ -> likelihood
$P(B)$ -> evidence
$P(A|B)$ -> posterior

$$  
P(E \mid F) = \frac{P(E, F)}{P(F)} = \frac{P(F \mid E)\,P(E)}{P(F)}  
$$
Splitting event $F$

$$  
P(F) = P(F, E) + P(F, \neg E)  
$$
Expanded Denominator

$$  
P(E \mid F) =  
\frac{P(F \mid E)\,P(E)}  
{P(F \mid E)\,P(E) + P(F \mid \neg E)\,P(\neg E)}  
$$

# Random Variables

- Random variables
	- A variable whose possible values have an associated probability distribution
- Expected values
	- The average of its values weighted by their probabilities
- Conditioned events
	- $X \mid A$
	- Looks at a random variable given than an event has already happened

# Continuous Distribution

- Discrete distribution
	- Also called PMF (Probability Mass Function)
	- One that associates a positive probability with discrete outcomes

$$  
\sum_x p(x) = 1  
$$
- Continuous distribution
	- Uniform distribution
	- Infinite numbers between 0 and 1
	- Probability over interval

$$  
P(a \le X \le b) = \int_a^b f_X(x)\,dx  
$$

- Probability density function (PDF)
	- How probability is distributed for a continuous random variable 

$$  
\int_{-\infty}^{\infty} f_X(x)\,dx = 1  
$$


- Cumulative distribution function (CDF)
	- Probability that a random variable is less than or equal to a certain value

$$  
F_X(x) = \sum_{t \le x} p(t)  
$$

```python
def uniform_pdf(x: float) -> float:
    return 1 if 0 <= x < 1 else 0
    
def uniform_cdf(x: float) -> float:
    """Returns the probability that a uniform random variable is <= x"""
    if x < 0:   return 0    # uniform random is never less than 0
    elif x < 1: return x    # e.g. P(X <= 0.4) = 0.4
    else:       return 1    # uniform random is always less than 1
```

# The Normal Distribution

- Normal distribution
	- Bell curve-shaped distribution
	- Determined by 2 parameters
		- $\micro$ (mean)
		- $\sigma$ (standard deviation)

PDF
$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \; e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

CDF
$$  
F_X(x) = \int_{-\infty}^{x} \frac{1}{\sigma \sqrt{2\pi}} \, e^{-\frac{(t - \mu)^2}{2\sigma^2}} \, dt  
$$

```python
import math
SQRT_TWO_PI = math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))
    
def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
```

- Standard normal distribution
	- $\micro = 0$
	- $\sigma = 1$

$Z = \frac{X - \micro}{\sigma}$


<img src="/images/Pasted image 20260428155752.png" alt="image" width="500">

<img src="/images/Pasted image 20260428155805.png" alt="image" width="500">

# The Central Limit Theorem

- A random variable defined as the average of a large number of independent and identically distributed random variables is itself approximately normally distributed
- Binomial random variables
	- $Binomial(n, p)$

```python
def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))
```

$\micro = np$
$\sigma = \sqrt{np(1-p)}$

# For Further Exploration

- Scipy.stats