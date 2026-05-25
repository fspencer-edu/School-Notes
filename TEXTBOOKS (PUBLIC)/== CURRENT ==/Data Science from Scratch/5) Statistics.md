- Statistics
	- Math. and techniques to understand data

# Describing a Single Set of Data

## Central Tendencies

- Mean
- Median
- Mode

```python
def mean(xs: List[float]) -> float:
	return sum(xs) / len(xs)
	
def median(v: List[float]) -> float:
    """Finds the 'middle-most' value of v"""
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)
    
def mode(x: List[float]) -> List[float]:
    """Returns a list, since there might be more than one mode"""
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]
```
- Quantile
	- Represents the value under a certain percentile of the da data lies

```python
def quantile(xs: List[float], p: float) -> float:
    """Returns the pth-percentile value in x"""
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]
```

## Dispersion

- Refers to the measures of how spread out the data is
- Large spread -> large values
- Small spread -> values near zero
- Range
	- Difference between largest and smallest elements

```python
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)
```

- Population
	- Mean -> $\micro$
	- Variance -> $\sigma^2$
	- Standard deviation -> $\sigma$
	- Total -> $n$
- Sample
	- Mean -> $\bar{x}$
	- Variance -> $s^2$
	- Standard deviation -> $s$
	- Total -> $(n-1)$
```python
from scratch.linear_algebra import sum_of_squares

def de_mean(xs: List[float]) -> List[float]:
    """Translate xs by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(xs)
    return [x - x_bar for x in xs]
    
def variance(xs: List[float]) -> float:
    """Almost the average squared deviation from the mean"""
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)
    
import math

def standard_deviation(xs: List[float]) -> float:
    """The standard deviation is the square root of the variance"""
    return math.sqrt(variance(xs))
```

- Interquartile range
	- Difference between 75th and 25th percentile values
	- $IQR = Q3 - Q1$

```python
def interquartile_range(xs: List[float]) -> float:
    """Returns the difference between the 75%-ile and the 25%-ile"""
    return quantile(xs, 0.75) - quantile(xs, 0.25)
```

# Correlation

- Variance
	- Measures how a single variable deviates from its mean
- Covariance
	- Measures how two variables vary in tandem from their means

```python
from scratch.linear_algebra import dot

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)
    
def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0    # if no variation, correlation is zero
```

Covariance = $s_{XY} = \frac{1}{(n-1)}\sum^n_{i=1}(x_i - \bar{x})(y_i - \bar{y})$

Correlation = $\frac{s_{XY}}{s_{X}s_{Y}}$

- Correlation
	- Divides out the standard deviation of both variables from covariance
	- Units are between -1 and 1

# Simpson's Paradox

- Simpson's paradox
	- Correlation can be misleading when cofounding variables are ignored
- Correlation is measuring the relationship between two variables all else being equal


# Some Other Correlational Caveats

- A correlation of zero indicates that there is no linear relationships between two variables
- Other relationships
	- Non linear
	- Absolute values
# Correlation and Causation

- Conduct randomized trials
- split users into groups with similar demographics and compare outcomes

# For Further Exploration

- SciPy
- pandas
- StatsModels