
- Bivariate
	- The analysis of two paired variables simultaneously to determine their empirical relationship
- Multivariate
	- Involves more than two variables, for interrelationships

# Linear Equations

$y = a + bx$

$a = b =$ constant
$x$ = independent variable
$y$ = dependent variable

## Slope and Y-intercept of a Linear Equations

- Slopes
	- Positive
	- 0
	- Negative


# Scatter Plots

- Scatter plot
	- Direction of a relationship
	- Strength
	- Overall pattern
	- Deviations
- Linear regression
	- Model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data

# The Regression Equation

**Least-Squares Regression Line**

$\hat{y} = b_0 + b_1x$

$\hat{y}$ = predicted value
$b_0$ = intercept
$b_1$ = slope

- Choose the line that minimizes the sum of squared residuals

**Residual ($\epsilon$)**

$\epsilon = y = \hat{y}$

$y$ = actual value
$\hat{y}$ = predicted value

**Absolute Value of Residual**

$|\epsilon|$

**SSE (Sum of Squared Errors)**

$SSE = \sum(y = \hat{y})^2$

- Removes negatives
- Penalizes larger errors


## Residual Plots

- A residual plot can be used to help determine if a set of (x, y) data is linearly correlated
- Diagnose if a linear model is appropriate
- A residual plot should appear random, with no pattern and not outliers
- Show constant error variance

## Least Squares Criteria for Best Fit

- Linear regression
	- Best fit line is where the SSE is minimized
- Least-squares regression line
	- A specific type of regression line that minimizes the sum of squared vertical distance between the points and the line

## Understanding Slope

- The slope of a line, $b$, describes how changes in the variables are related
- The slope of the best-fit is how the dependent variables changes for every one unit increase in the independent variable, on average

## Correlation Coefficient r

- Provides a measure of strength and direction of the linear association between the independent variable x and the dependent variable y

<img src="/images/Pasted image 20260407141430.png" alt="image" width="500">

$n$ = number of data points

- r is always between -1 and +1
- Size of the correlation indicates the strength
- If, $r=0$, there is likely no linear correlation
	- Although data that is curved, may have a correlation of 0
- $r=1$
	- Perfect positive correlation
- $r=-11$
	- Perfect negative correlation

- Correlation does not imply causation

<img src="/images/Pasted image 20260407141619.png" alt="image" width="500">


## The Coefficient of Determination

- Coefficient of determination, $r^2$
	- Square of the correlation coefficient
	- Expressed as a percent
	- Percent of variation in the dependent variable that can be explained by variation in the independent variable
	- $1-r^2$
		- Percent of variation in y that is not explained by variation in x using the regression line
# Testing the Significance of the Correlation Coefficient

- Correlation coefficient, r
	- Strength and direction of linear relationship
	- Reliability of the linear model depends on how many observed data points are in the sample
- Significance of the correlation coefficient

$\rho$ = population correlation coefficient (unknown)
$r$ = sample correlation coefficient

- The hypothesis test
	- Decides if $\rho$ is close to zero or significantly different from zero

## Hypothesis Test

$H_0: \rho=0$
- The population correlation coefficient is not significantly different from zero
$H_a: \rho \neq 0$
- The population correlation coefficient is significantly different from zero
- There is a significant linear relationship

Methods
- p-value
- Critical values table

## Critical Value Method

- Determines if a test statistic falls within a designated "rejection region/critical region"

## Assumptions in Testing the Significance of the Correlation Coefficient

- Assumptions underlying the test of significance are
	- Relationship between the variables being correlated should be linear
	- residual errors are mutually independent


# Prediction

- Prediction
	- Uses data analysis, modeling, and historical patterns to estimate unknown of future outcomes
- Interpolation
	- Predicting inside of observed x values observed in the data
- Extrapolation
	- Predicting outside of the observed x values observed in the data

# Outliers

- Outliers
	- Observed data points that are far from the least squares line
- Influential points
	- Observed data points that are far from the other observed data points in the horizontal direction
	- Effect slope of the regression line

## Identifying Outliers

- Points that are two standard deviations above or below the best-fit line is an outlier

## Numerical Identification of Outliers

- Standard deviation of the residuals

$s = \sqrt{\frac{SSE}{n-2}}$

- Regression model involves two estimates $n-2$

## How does the outliers affect best fit line


## Numerical identification of Outliers: Calculating s and Finding Outliers Manually


---

- Outlier