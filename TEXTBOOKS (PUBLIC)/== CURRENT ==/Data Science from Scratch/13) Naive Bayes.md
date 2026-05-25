# The Really Dump Spam Filter

- Bayes theorem
	- Probability that the message is span conditional on containing a certain word

- Numerator
	- Message is span and contain a word
- Denominator
	- Message contain a word
- Calculation is the proportion of messages that are spam

$P(S|B) = \frac{[P(B|s)P(S)]}{[P(B|S)P(S)]} + P(B|\neg S)P(\neg S)$


# A More Sophisticated Spam Filter

- Naive Bayes
	- Compute each of the probabilities on the right by multiplying the individual probability estimates for each word
- Underflow
	- Dealing with floating-point numbers that are too close to 0
- Compute the exp(log)

# Implementation

# Testing Our Model

# Using Our Model

# For Further Exploration