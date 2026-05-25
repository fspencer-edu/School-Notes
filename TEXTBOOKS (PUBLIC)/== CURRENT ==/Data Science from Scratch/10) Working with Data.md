# Exploring Your Data

## Exploring 1D Data

- Collection of numbers

- Statistics summary
	- Bucketize
	- Histogram

## 2D

- Scatter plot
- Correlation

## Many Dimensions

- Correlation matrix
- Scatterplot matrix

# Using NamedTuples

```python
import datatime

stock_price = {'closing_price': 102.06,
               'date': datetime.date(2014, 8, 29),
               'symbol': 'AAPL'}
```

- Inefficient representation of a `dict`
- Use a `namedtuple` class
	- Slots
	- Immutable

```python
from collections import namedtuple

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)
```

# Dataclasses

- Mutable version of `NamedTuple`
- Dataclasses are regular python classes that generate a method automatically
- Instead of inheriting a base class, a decorator is used
- Modify a dataclass instance's values


```python
from dataclasses import dataclass

@dataclass
class StockPrice2:
	symbol: str
	data: datatime.date
	closing_price: float
	
	def is_high_tech(self) -> bool:
		return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
		

price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)
```

# Cleaning and Munging

- Real world data is dirty
- Parsing data types
- Delimiting
- Reading files

# Manipulating Data

# Rescaling

- Compute
	- Distance
	- Normalization
	- Standardization
	- Euclidean distance

# An Aside: tqdm

- Generates customer progress bars

```python
python -m pip install tqdm

import tqdm

for i in tqdm.tqdm(range(100)):
	_ = [random.random() for _ in range(100000)]
	

 56%|████████████████████              | 56/100 [00:08<00:06,
```

# Dimensionality Reduction

- Principal component analysis (PCA)
	- Extract one or more dimensions

```python
# translate data - each dim has a mean of 0

from scratch.linear_algebra import subtract

def de_mean(data: List[Vector]) -> List[Vector]:
	"""Recenters the data to have mean 0 in every dimension"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]

# normalize
from scratch.linear_algebra import magnitude

def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]
    
# compute variance
from scratch.linear_algebra import dot

def directional_variance(data: List[Vector], w: Vector) -> float:
    """
    Returns the variance of x in the direction of w
    """
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)
```

# For Further Exploration