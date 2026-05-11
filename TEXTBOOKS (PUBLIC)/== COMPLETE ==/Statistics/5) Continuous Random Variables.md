# Continuous Probability Functions

## Properties of Continuous Probability Distributions

- The graph of a continuous probability distribution is a curve
- Probability is represented by area under the curve

$f(x)$ = probability density function (pdf)

- Area under the curve
	- Cumulative distribution function (cdf)

Distribution types
- Uniform
- Exponential
- Normal

**Continuous PDF**

$P(a \leq X \leq n) = \int^{b}_{a}f(x)dx$

- Total area = 1

**Cumulative Distribution Function**
$PX \lt x = 1 - e^{-\lambda x}$


<img src="/images/Pasted image 20260407102844.png" alt="image" width="500">

- The maximum probability is one
- Probability = area

$P(x_1 \lt x \lt x_2) = (x_2 - x_1)f(x)$

- The probability at $x$ has no width, and therefore has a zero probability
- CDF is the area to the left of right

# The Uniform Distribution

$X \sim U(a, b)$

$a$ = lowest value of x
$b$ = highest value of x

$PDF = f(x) = \frac{1}{b-a}$

- Theoretical mean and standard deviation are close to the sample mean and standard deviation

$\micro = (a+b)/2$
$\sigma = \sqrt{(b-a)^2/12}$

# The Exponential Distribution

$X \sim Exp(\lambda)$

$f(x) = \lambda e^{-\lambda x}$

- Exponential distribution
	- Amount of time until some specific event occurs
	- Fewer large values
	- More small values
	- Used to calculate reliability, or length of time a product lasts

X = continuous random variable

$m = 1/\micro = 1/4$ => decay parameter
$\sigma = \micro$

$X \sim Exp(0.25)$
$f(x) = 0.25e^{-0.25x}$
$f(5) = 0.25e^{-0.25(5)}=0.072$

<img src="/images/Pasted image 20260407104625.png" alt="image" width="500">

- Max value is when $f(x)=0.25 = \lambda$

**Percentile (k) for an exponential distribution**

$k = \frac{ln(1-LeftArea)}{-m}$

$P(X \leq k) = 1 - e^{-mk}$
$1 Area = e^{-mk}$
$ln(1 Area) = -mk$

## Memorylessness of the Exponential Distribution

- Memoryless property
	- Knowledge of what has occurred in the past has no effect on future probabilities

$P(X \gt r+t | X \gt r) = P(X \gt t)$ for all $r \geq 0$ and $t \geq 0$

## Relationship Between the Poisson and the Exponential Distribution

- Poisson distribution
	- Counts number of events in a fixed time
	- $\lambda$ = average number of events per unit time
- Exponential distribution
	- Measures time between events
- Both
	- $\lambda$ = average time between events
	- $\micro$ = average number of events per time
	- $\lambda = \frac{1}{\micro}$


- The number of events per unit time follows a Poisson distribution with $\lambda = 1/\micro$
	- Two independent exponential distribution events occur
	- Events are random
	- Rate is constant over time

$P(X = k) = \frac{\lambda^ke^{-\lambda}}{k!}$

- If the number of events per unit time follows a Poisson distribution, than the amount of time between events follows the exponential distribution

---

- Conditional probability
- Decay parameter
- Exponential distribution
- Memoryless property
- Poisson distribution
- Uniform distribution