
- Hypothesis testing
	- Collecting data from a sample and evaluating the data
	- Determine if the data accepts or rejects the null hypothesis

- Hypothesis test
	- Set up two contradictory hypotheses
	- Collect sample data
	- Determine the correct distribution
	- Analyze sample data to reject or decline to reject the null hypothesis

# Null and Alternative Hypotheses

- Null hypothesis, $H_0$
	- Assumes there is no effect, or change
- Alternative hypothesis, $H_a$
	- There is a meaningful effect or change

<img src="/images/Pasted image 20260407123421.png" alt="image" width="500">

$H_0$ always has a symbol with an equal in it

# Outcomes and the Type I and Type II Errors

<img src="/images/Pasted image 20260407123453.png" alt="image" width="500">

1. True positive
2. True negative (Power of the Test)
3. False positive (Type I Error)
4. False negative (Type II Error)

$\alpha$ = Type I error
$\beta$ = Type II error

$\alpha$ and $\beta$ should be as small as possible, since they are probabilities of errors
Power of the test is $1 - \beta$
- Ideally want high power that is close to one
- Increasing the sample size can increase the power of the test

# Probability Distribution Needed for Hypothesis Testing

<img src="/images/Pasted image 20260407124025.png" alt="image" width="500">


# Rare Events, the Sample, Decision and Conclusion

## Rare Events

- If something very unlikely happens under an assumption, the assumption may be incorrect

## Using the Sample to Test the Null Hypothesis

- p-value
	- Use the sample data to calculate the actual probability of getting the test result
	- Probability that, if the null hypothesis is true, the results from another randomly selected sample will be as extreme or more extreme as the results obtained form the given sample
- Large p-value => not reject the null
- Small p-value => stronger evidence against the null

# Additional Information and Full Hypothesis Test Examples

- Level of significance ($\alpha$)
	- Pre-determined threshold representing the max. probability a researcher is willing to accept for incorrectly rejecting a true null hypothesis (Type I error)
- p-value
	- Left tail
	- Right tail
	- Split evenly between the two tails


**Example**

Hypothesis Test => Test of a single population mean

$H_0: \micro = 16.43$
$H_a: \micro \lt 16.43$

- For Jeffrey to swim faster, his time will be less than 16.43 seconds

$\bar{X}$ => mean time to swim the 25-yard freestyle
- Normal distribution
- $\sigma = 0.8$

$\bar{X} \sim (\micro, \frac{\sigma_X}{\sqrt{n}})$
$\bar{X} \sim (16.43, \frac{0.8}{\sqrt{15}})$

$n = 15$
p-value = $P(\bar{x} \lt 16) = 0.0187$

<img src="/images/Pasted image 20260407125332.png" alt="image" width="500">

- There is a 1.87^ probability that the time to swim the 25-yard freestyle 16 sec or less
- The mean time of 16 sec or less is unlikely

$\alpha = 0.05 \gt 0.0187$ = p-value

- The null hypothesis is rejected

Conclusion
- At the 5% significance level, there is sufficient evidence that the mean time to swim 25-yard less than 16.43 seconds
- Therefore, Jeffery swims faster using the new googles

# Hypothesis Testing of a Single Mean and Single Proportion

---

- Binomial distribution
- Central limit theorem
- Confidence interval (CI)
- Hypothesis
- Level of significance of the test
- Normal distribution
- p-value
- Standard deviation
- Student's t-distribution
- Type 1 error
- Type 2 error