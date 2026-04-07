
- The sample mean, $\bar{x}$, is the point estimate for the population mean, $\micro$
- The sample standard deviation, $s$,is the point estimate of the population standard deviation, $\sigma$

$\bar{x}$ and $s$ are called a statistic

- Confidence interval
	- An interval of numbers that provides a range of reasonable values in which the population parameter is expected to be in

- Empirical Rule
	- Applies to bell-shaped distribution
	- Approx. 95% of the sample mean, will be within the two standard deviations of the population mean

$\frac{\sigma}{\sqrt{n}} = \frac{1}{\sqrt{100}} = 0.1$

$(2)(0.1) = 0.2$ => two standard deviations
$\bar{x}$ is likely to be within 0.2 units of $\micro$
$\micro$ is between $\bar{x} - 0.2$ and \bar{x} + 0.2 in 95% of all the samples

A sample with $\bar{x} = 2$, has a 95% confidence interval of (1.8, 2.2)

- Interval contains the true mean or the sample mean

**Confidence Interval**

$CI = \bar{X} \pm \cdot \frac{\sigma}{\sqrt{n}}$

# A Single Population Mean using the Normal Distribution

- A confidence interval for a population mean, when the population standard deviation is known, is based on the conclusion of the CLT
	- Sampling distribution of the sample means following an approximately normal distribution

## Calculating the Confidence Interval

- Confidence interval for a single unknown population mean
	- Population standard deviation is known
	- Estimate $\micro$ based on $\bar{x}$
	- Margin of error (EBM) = error bound for a population mean
- Confidence interval estimate

$(\bar{x} - EBM, \bar{x} + EBM)$

- The EBM depends on the confidence level (CL)
- Confidence level
	- Probability that the calculated confidence interval estimate will contain the true population parameter
	- Percent of confidence intervals that contain the true population parameter when repeated samples are taken

$\alpha$ = probability that the interval does not contain the unknown population parameter

$\alpha + CL = 1$

**Examples**

$\bar{x}$ = 7
$EBM = 2.5$

$CI = (7-2.5, 7+2.5) = (4.5, 9.5)$

- When the population standard deviation is known, us a normal distribution to calculate the error bound

- Increasing the confidence level increases the error bound, making the confidence interval wider
- Decreasing the confidence level decreases the error bound, making the confidence interval narrower

## Working Backwards to Find the EBM or Sample Mean

- Calculate the interval
	- Find the sample mean
	- Calculate error bound

- Find the error bound
	- From the upper value for the interval, subtract the sample mean
	- From the upper value for the interval, subtract the lower value
	- Divide the difference by two
- Find the sample mean
	- Subtract the error bound from the upper value of confidence interval
	- Average the upper and lower endpoints of the confidence interval

## Calculating the Sample Size n

- For a specific margin or error, we can use the error bound formula to calculate the required sample size

$EBM = (\frac{z_{\alpha}}{2})(\frac{\sigma}{\sqrt{n}})$

$n = \frac{z^2\sigma^2}{EBM^2}$

# A Single Population Mean using the Student t Distribution

- When sample sizes are small the confidence interval is inaccurate
- Normal distribution
	- Approximation for large samples sizes
- Student's t-distribution
	- Continuous probability distribution used to estimate population parameters when samples sizes are small
	- $n \lt 30$

- Student's t-distribution with n-1 df
	- t-score has the same interpretation as the z-score

$t = \frac{\bar{x} - \micro}{\frac{s}{\sqrt{n}}}$

- Degrees of freedom
	- Calculation of the sample standard deviation
	- Sum of the deviations is zero, find the last deviation once we know the other n-1 deviations

![t-Distribution and Degrees of Freedom | CFA Level 1](https://analystprep.com/cfa-level-1-exam/wp-content/uploads/2019/10/page-157.jpg)

**Properties of the Student's t-Distribution**
- Graph is similar to the standard normal curve
- Mean is zero and the distribution is symmetric about zero
- Has more probability in its tails than the standard normal distribution
	- Spread is greater
	- Shape depends on the df
		- As df increases, the graphs becomes more like the graph of the standard normal distribution
	- The underlying population of individual observations is assumed to be normally distributed

- Use the inverse probability to find the t when probability is known

**Student's t-distribution**
$T \sim t_{df}$, $df = n -1$

**Population standard deviation is not known**
$EBM = (\frac{t_{\alpha}}{2})(\frac{s}{\sqrt{n}})$

# A Population Proportion

- Confidence intervals can be calculated as percentages
- Distribution is a binomial distribution

$X \sim B(n, p)$
$P' = \frac{X}{n}$

- When n is large and p is not close to zero or one, use the normal distribution to approximate the binomial

$X \sim N(np, \sqrt{npq})$

<img src="/images/Pasted image 20260407121404.png" alt="image" width="500">

## "Plus Four" Confidence Interval for p

- There is a certain amount of error introduced into the process of calculating a confidence interval for a proportion
- Use point estimates to calculate the appropriate deviations of the sampling distribution
- Change the sample size to $n+4$
- Successes is$ x + 2$

## Calculating the Sample Size

- To calculate a sample size, from a specific margin of error

$EBP = (\frac{z_{\alpha}}{2})(\sqrt{\frac{p'q'}{n}})$

<img src="/images/Pasted image 20260407123029.png" alt="image" width="500">


---

- Binomial distribution
- Confidence interval (CI)
- Confidence level (CL)
- Degrees of freedom (Df)
- Error bound for a population mean (EMB)
- Inferential statistics
- Normal distribution
- Standard normal distribution
- Parameter
- Point estimate
- Standard deviation
- Student's t-distribution