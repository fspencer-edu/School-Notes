
# Virtual Environments

- Virtual environments
	- Sandboxed Python env that maintain their own versions of Python libraries
- Anaconda Python distribution

# Whitespace Formatting

- Python uses indentation to delimit blocks of code

# Modules

- `import` third-party features

# Functions

- `def`
	- Taking zero or more inputs and returning a corresponding output
	- Functions are first-class
		- Assign them to variables and pass them into functions
- Anonymous functions (lambdas)
- Function parameters

# Strings

- Strings can be delimited by single or double quotation marks
- Special characters
	- `r"\n"`
- Multiline strings
- f-string
- Concatenation

# Exceptions

- Python raises an exception
- Unhandled exception will cause the program to crash
- `try` and `except`

# Lists

- Fundamental data structure in Python
- Ordered collection
- Heterogeneous
- Index, `x[i]`
- Square brackets to slice lists
- `in` operator to check for list membership
- `extend` to add items from another collection
- `append`, add one item at a time
- Unpack lists
	- `x, y = [1, 2]`
- Throwaway value
	- `_, y = [1, 2]`

# Tuples

- Tuples are immutable
- Parentheses

`my_tuple = (1, 2)`

- Tuples and lists can also be used for multiple assignment

`x, y = y, x`

# Dictionaries

- Associates values with keys
- Retrieve the value corresponding to a given key

```python
fruit = {"apple": 12, "banana": 10}

num_apples = fruit["apple"]
```

- `get` method that returns a default value
- Represent structured data
- Look for specific keys, values, or items

```python
tweet_keys   = tweet.keys()     # iterable for the keys
tweet_values = tweet.values()   # iterable for the values
tweet_items  = tweet.items()    # iterable for the (key, value) tuples
```
- Dictionary keys must be hashable
	- Cannot use lists as keys

## defaultdict

- Similar to a regular dictionary, except that when a key does not exist, it first adds a value for it using a zero-argument function
- Use to collect results of keys, without checking if key exists

```python
from collections import defaultdict

word_counts = defaultdict(int)
for word in document:
	word_counts[word] += 1
```

# Counters

- Turns a sequence of values into a `defaultdict(int)`-like object mapping keys to counts

```python
from collections import Counter
c = Counter([0, 1, 2, 0])  # c is (basically) {0: 2, 1: 1, 2: 1}

# print the 10 most common words and their counts
for word, count in word_counts.most_common(10):
    print(word, count)
```

# Sets

- Represents a collection of distinct elements

```python
primes = {2, 3, 5, 7}
```

# Control Flow

- Ternary

```python
if
while
for...in
```

# Truthiness

- Booleans in Python
- Capitalized
	- `True, False`
- Nonexistent value
	- `None`

```python
# falsy values
False
None
[]
{}
""
set()
0
0.0
```

- Logical operators
	- `and, or, not`

# Sorting

- Every Python list has a `sort` method, that sorts in place
- `sorted(x)` returns a new list
- Parameters
	- `key=abs`
	- `reverse=True`

# List Comprehensions

- Transform a list into another list
- Transform dictionaries or sets

# Automated Testing and Assert

- `assert`
	- Cause code to raise an `AssertionError` if condition is not truthy

```python
assert 1 + 1 == 2, "1 + 1 should equal 2 but didn't"

def smallest_item(xs):
    return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1

def smallest_item(xs):
    assert xs, "empty list has no smallest item"
    return min(xs)
```

# Object Oriented Programming

- Python allows you to define classes that encapsulate data and the functions the operate on them
- A class contains zero or more member functions
- Each takes a first parameter, `self`, that refers to the particular class instance
- A class has a constructor, `__init__`
	- Dunder methods (double underscore)
- Class methods that start with an underscore are considered private

```python
class CountingClicker:
    """A class can/should have a docstring, just like a function"""
	def __init__(self, count = 0):
		self.count = count
		
		
clicker1 = CountingClicker()           # initialized to 0
clicker2 = CountingClicker(100)
```

- `__repr__`
	- Produces the string representation of a class instance

```python
def __repr__(self):
        return f"CountingClicker(count={self.count})"
```

- Subclasses inherit come of their functionality from a parent class
- Create a non-reset-able clicker by using `CounterClicker` as the base class and overriding the `reset` method'

```python
class NoResetClicker(CountingClicker):
	
	def reset(self):
		pass
```

# Iterables and Generators

- Retrieve specific elements by their indices
- Generators
	- Iterated over like lists but generate their values lazily on demand

```python
def generate_range(n):
	i = 0
	while i < n:
		yield i
		i += 1
		
for i in generate_range(10):
	print(f"i: {i})
```

- `range` is a lazy function
- `yield`
	- Every call produces a value of the generator
- Can only iterate through a generator once
- Re-create generator each time or use a list to reuse it
- Generating is expensive
- Use `for` comprehensions wrapped in parantheses

```python
evens = (i for i in generate_range(20) if i % 2 == 0)
```

- `enumerate`
	- Provides indices and values
	- `(index, value)`

```python
for i, name in enumerate(names):
    print(f"name {i} is {name}")
```

# Randomness

- `random` module produces a pseudorandom (deterministic) numbers based on an internal state

```python
import random
random.seed(10)
random.randrange(3, 6)
random.shuffle([1, 2, 3, 4, 5])
random.choice()
```

# Regular Expressions

- Provide a way of searching text
- `re`
- `re.match`
	- Checks whether the beginning of a string matches
- `re.search`
	- Checks whether any part of a string matches a regular expression

# Functional Programming

- `partial`
- `map`
- `reduce`
- `filter`

# zip and Argument Unpacking

- `zip`
	- Combine two or more iterables together
	- Transforms multiple iterables into a single tuple

```python
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

[pair for pair in zip(list1, list2)]    # is [('a', 1), ('b', 2), ('c', 3)]
```

- Unzip a list or tuples

```python
pairs = [('a', 1), ('b', 2), ('c', 3)]
letters, numbers = zip(*pairs)
```

- `*` performs argument unpacking
	- Uses the elements of `pairs` as individual arguments

# args and kwargs

- Specify a function that takes arbitrary arguments

```python
def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")

#  unnamed args: (1, 2)
#  keyword args: {'key': 'word', 'key2': 'word2'}
```

- `args`
	- Tuple if its unnamed arguments
- `kwargs`
	- `dict` of its named arguments
- Use a `list` and `dict` to supply arguments to a function

```python
def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2 + 3 should be 6"
```

# Type Annotations

- Python is a dynamically type language
- Does not care about the types of objects used
- Statically type langauge
	- Functions and objects have specific types

```python
# dynamic
def add(a, b):
    return a + b
    
# static
def add(a: int, b: int) -> int:
    return a + b
```

- Type annotations
	- Form of documentation
	- External tools can inspect, and report errors
	- Cleaner code
	- Helps editor with autocomplete

## How to Write Type Annotations

- `typing` module
	- Provide a number of parameterized types

```python
from typing import List, Optional

def total(xs: List[float]) -> float:
	return sum(total)
	
	
x: int = 5

best: Optional[Float]
```

- `Optional`
	- Value can be either float or none
