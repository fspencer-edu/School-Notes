
- Random variable
	- Describes the outcomes of a statistical experiment in words

## Random Variable Notation
- Random variable => X, Y
- Random variable value => x, y

# Probability Distribution Function (PDF) for a Discrete Random Variable

- A discrete probability distribution function
	- Each probability is between zero and 1, inclusive
	- The sum of probabilities is one

# Mean or Expected Value and Standard Deviation

- Expected value
	- Referred to as the "long-term" average or mean
- The law of large numbers
	- As the number of trials in a probability experiment increases, the differences between the theoretical probability of an event and the relative frequency approaches zero

$\micro = \sum(x \cdot P(x))$ = mean

$\sigma = \sqrt{[x - \micro]^2 \cdot P(x)}$ = standard deviation


# Binomial Distribution

- Binomial experiment characteristics
	- Fix number of trials
	- Only two possible outcomes
		- Success of failure
		- $p + q = 1$
	- n trials are independent and are repeated using identical conditions

- Binomial probability distribution
	- $\micro$ and $\sigma^2$, for the binomial probability are, $\micro = np$ and $\sigma = npq$


$x$ = number of occurrences of a specific outcome in n trials
$p$ = probability of success in a single trial
$n$ = number of trials
$\binom{n}{x} = n!/[x!(n-x)!]$

- Bernoulli Trial
	- A random experiment with exactly two mutually exclusive outcomes

## Notation for the Binomial, B = Binomial Probability Distribution Function

$X \sim B(n, p)$
- X is a random variable with a binomial distribution

$X$ = random variable with a binomial distribution
$p$ = success one each trial
$n$ = number of trials

$P(x;n,p) = \binom{n}{x}p^x(1 - p)^{n-x}$


**Examples**
$n = 5$
$k = 3$
$p = 0.5$

$X \sim B(5, 3)$

$P(X = 3) = \binom{5}{3}(0.5^3)(0.5)^2 = 0.3125$

# Geometric Distribution

- 4 main characteristics of a geometric experiment
	- Trial is repeated until a success occurs
		- Repeated until the first success
	- The repeated trials are independent of each other
	- Probability of success and failure
		- $p + q = 1$
		- Random variable X represents the number of trials in which the first success occurs
- Additional attributes
	- The random variable is discrete
	- Implicit in the random variable is that the probability of a success and failure is constant
	- Memoryless

$P(x = n+k | x \geq k + 1) = P(x = n)$, $k =$ number of previous failures

$P(X \gt n) = (1 - p)^n$
- The probability that there is not a success in the first n trials

$P(X \lt n) = 1 - (1 - p)^n$
- Probability of success before trial n

$(1 - p)^n$ => all failures
$1 - (1 - p)^n$ => at least one success

## Notation for the Geometric: G = Geometric Probability Distribution Function

$X \sim G(p)$

- X is a random variable with a geometric distribution

$p$ = probability of a success for each trial

**CASE 1: Random Variable X is the Event of First Success**

- Probability that the first occurrence of success requires x number of failure independent trials, each with probability $(1 - p)$
- $x$ = independent trials that are failures, until first success

$P(X = x)(1 - p)^{x-1}p$

$x = 1, 2, 3, ...$

$\micro = 1/p$

$\sigma = \sqrt{(1/p)(1/p - 1)}$

**CASE 2: Random Variable X is The Number of Failures before a Success**

$P(X = x)(1 - p)^{x}p$

$x = 0, 1, 2, 3, ...$

- The trial that is the success is not counted as a trial in the formula
- $x$ = number of failures
- Trial of success is not counted

$\micro = (1 - p)/p$

$\sigma = \sqrt{(1-p)/p}$

- Common ratio, $r$
	- Constant factor multiplied to each term to produce the next
	- Fixed, nonzero number throughout a sequence
	- Determined the pattern in a geometric progression
	- Calculated by value divide by previous value

$r = P(x=5)/P(x=4)=0.98$

- The common ratio multiplied by any other probability value will provide the next probability value in the sequence

# Hypergeometric Distribution

- 5 characteristics of a hypergeometric experiment
	- Samples from 2 groups
	- Group of interest (first group)
	- Sample without replacement from the combined groups
	- Each pick is not independent
	- Does no use Bernoulli Trials

$X$ = the number of items from the group of interest

- Calculates the probability of a specific number of successes in n trials from a finite population without replacement
- Used on small, finite populations
## Notation for the Hypergeometric: H = Hypergeometric Probability Distribution Function

$X \sim H(r, b, n)$
- X is a random variable with a hypergeometric distribution

$r$ = size of first group
$b$ = size of second group
$n$ = size of chosen sample

$P(X = k) = \frac{\binom{K}{k}{\binom{N -K}{n-k}}}{\binom{N}{n}}$

$N$ = total population size
$n$ = number of draws
$K$ = success items
$k$ = number of successes you want

$\binom{K}{k}$ => choose k successes
$\binom{N -K}{n-k}$ => choose failures

- Together get exactly k successes

$\binom{N}{n}$ => total ways to pick any sample

**Example**

$X \sim H(6, 5, 4)$

$P(X = 2) = \frac{\binom{6}{2}{\binom{5}{1}}}{\binom{11}{3}}$

$N$ = 11
$n$ = 3
$K$ = 6
$k$ = 2

$P(x = 2) = 0.4545$

- The probability that there are two men on the committee is about 0.45

$\micro = nr/(r+b)$

# Poisson Distribution

- 2 main characteristics of a Poisson experiment
	- The Poisson probability distribution gives the probability of a number of events occurring in a fixed interval
	- Used to approximate the binomial if the probability of success is "small" and number of rials is large

## Notation for the Poisson: P = Poisson Probability Distribution Function

$X \sim P(\micro)$
- X is a random variable with a Poisson distribution

$\micro/\lambda$ = mean for he interval of interest

$P(X = K) = \frac{e^{- \lambda}\lambda^k}{k!}$

$X$ = number of events
$k$ = specific number you want
$\lambda$ = average number of events
$e$ = 2.718 (euler's constant)
$k!$ = factorial

- Given an average rate, $\lambda$, what is the probability of seeing exactly k events

$\lambda = np$

**Example**
- A call center gets 2 calls/min
- What is the probability of 3 calls in a minute?

$\lambda = 2$

$P(X = K) = \frac{e^{- 2}2^3}{3!} = 0.180$


**Poisson Distribution with "Greater than" probability**

Find $P(X \gt 1)$

- Poisson is a discrete and infinite
	- $P(2) + P(3) + P(4) +...$
- Complete rule
	- $P(X \gt 1) = 1 - P(X \leq 1)$

$P(X \leq 1) = P(X = 0) + P(X = 1)$

$\therefore P(X \gt 1) = P(X = 0) + P(X = 1)$


# Discrete Distribution - Playing Card Experiment

# Discrete Distribution - Dice Experiment

---

- Bernoulli Trails
- Binomial experiments
- Binomial probability distribution
- Expected value
- Geometric distribution
- Geometric experiment
- Hypergeometric experiment
- Mean
- Mean of a probability distribution
- Poisson probability distribution
- Probability distribution function (PDF)
- Random variable (RV)
- Standard deviation of a probability distribution
- The law of large numbers