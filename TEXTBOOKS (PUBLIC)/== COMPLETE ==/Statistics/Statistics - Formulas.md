
# Formulas

## Factorial

$$  
n! = n(n-1)(n-2)\cdots(1)  
$$  
$$  
0! = 1  
$$

## Combinations

$$  
\binom{n}{r} = \frac{n!}{(n-r)! \, r!}  
$$

## Binomial Distribution

$$
X \sim B(n, p)
$$

$$
P(X = x) = \binom{n}{x} p^x q^{\,n-x}, \quad x = 0,1,2,\dots,n
$$

## Geometric Distribution

$$
X \sim G(p)
$$

$$
P(X = x) = q^{x-1} p, \quad x = 1,2,3,\dots
$$


## Hypergeometric Distribution

$$
X \sim H(r, b, n)
$$

$$
P(X = x) = \frac{\binom{r}{x} \binom{b}{n-x}}{\binom{r+b}{n}}
$$

## Poisson Distribution

$$
X \sim P(\mu)
$$

$$
P(X = x) = \frac{\mu^x e^{-\mu}}{x!}
$$

## Uniform Distribution

$$
X \sim U(a, b)
$$

$$
f(x) = \frac{1}{b-a}, \quad a < x < b
$$

## Exponential Distribution

$$  
X \sim \text{Exp}(m)  
$$  
  
$$  
f(x) = m e^{-mx}, \quad m > 0, \; x \ge 0  
$$
## Normal Distribution

$$
X \sim N(\mu, \sigma^2)
$$

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \; e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \quad -\infty < x < \infty
$$


## Student's t-distribution

$$
X \sim t_{df}
$$

$$
f(x) = \frac{\Gamma\left(\frac{n+1}{2}\right)}
{\sqrt{n\pi}\,\Gamma\left(\frac{n}{2}\right)}
\left(1 + \frac{x^2}{n}\right)^{-\frac{n+1}{2}}
$$

$$
X = \frac{Z}{\sqrt{Y/n}}
$$

$$
Z \sim N(0,1), \quad Y \sim \chi^2_{df}, \quad n = \text{degrees of freedom}
$$

## Chi-Square Distribution

$$
X \sim \chi^2_{df}
$$

$$
f(x) = \frac{1}{2^{n/2}\Gamma\left(\frac{n}{2}\right)} 
x^{\frac{n}{2}-1} e^{-x/2}, \quad x > 0
$$

$$
n = \text{degrees of freedom}
$$

## F Distribution

$$
X \sim F_{df(n), df(d)}
$$

$$
df(n) = \text{degrees of freedom (numerator)}, \quad
df(d) = \text{degrees of freedom (denominator)}
$$

$$
f(x) = \frac{\Gamma\left(\frac{u+v}{2}\right)}
{\Gamma\left(\frac{u}{2}\right)\Gamma\left(\frac{v}{2}\right)}
\left(\frac{u}{v}\right)^{u/2}
x^{\frac{u}{2}-1}
\left[1 + \frac{u}{v}x\right]^{-\frac{u+v}{2}}
$$

$$
X = \frac{Y_u}{W_v}, \quad Y, W \sim \chi^2
$$

---

# Symbols

## Descriptive Statistics

$Q_1$ = Quartile one
$Q_2$ = Quartile two
$Q_3$ = Quartile three
$IQR = Q_3 - Q_1$ = interquartile range

$\bar{x}$ = sample mean
$\micro$ = population mean

$s$ = sample standard deviation 
$\sigma$ = population standard deviation

$s^2$ = sample variance
$\sigma^2$ = population variance


## Probability

$S$ = sample space
$A$ = event A
$P(A)$ = probability of A
$P(A|B)$ = probability of A given B
$P(A \cap B)$ = probability of A and B
$P(A \cup B)$ = probability of A or B
$A'$ = complement of A
$P(A')$ = probability of complement of A
$G_1$ = green of first pick
$P(G_1)$ = probability of green of first pick

## Discrete Random Variables


$PDF$ = probability distribution function
$X$ = random variable
$X \sim$ distribution of X
$B$ = binomial distribution
$G$ = geometric distribution
$H$ = hypergeometric distribution
$P$ = Poisson distribution
$\lambda$ = average of Poisson distribution

## Continuous Random Variables

$pdf$ = probability density function
$U$ = uniform distribution
$Exp$ = exponential distribution
$k$ = critical value
$m = \lambda$ = decay rate for (exp dist.)

## Normal Distribution

$N$ = normal distribution
$z$ = z-score
$Z$ = standard normal distribution

## CLT

$CLT$ = central limit theorem
$\bar{X}$ = random variable

## Confidence Intervals

$CL$ = confidence level
$CI$ =critical interval
$EBM$ = error bound for a mean
$EBP$ = error bound for a proportion
$t$ = Student's t-distribution
$df$ = degrees of freedom
$\hat{p} = p'$ = sample proportion of success
$\hat{q} = q'$ = sample proportion of failure

## Hypothesis Testing

$H_0$ = null hypothesis
$H_a$ = alternative hypothesis
$H_1$ = alternative hypothesis
$\alpha$ = Type I error
$\beta$ = Type II error

## Chi-Square Distribution

$\chi^2$ = chi-square
$O$ = observed frequency
$E$ = expected frequency

## Linear Regression and Correlation

$y = a + bx$ = equation of a line
$\hat{y}$ = estimated value of y
$r$ = correlation coefficient
$\epsilon$ = error
$SSE$ = Sum of Squared errors

## F-Distribution and ANOVA

$F$ = F-ratio 