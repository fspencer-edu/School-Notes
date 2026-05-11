
- Nearest neighbour classification

# The Model

- Simplest predictive models
- No mathematical assumptions
- Distance
- Similarity/dissimilarity metrics


# Example: The Iris Dataset

```python
import requests

data = requests.get(
	"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

with open('iris.dat', 'w') as f:
	f.write(data.text)
```

# The Curse of Dimensionality

- High-dimensional spaces are complex
- As the number of dimensionality increases, the average distance between points increases

# For Further Exploration