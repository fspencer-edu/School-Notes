
- RNNs
	- A class of nets that can predict the future
	- Analyze time series data
	- Work on sequences of arbitrary length, rather than on fixed-sized inputs
- 2 main challenges
	- Unstable gradients
		- Recurrent dropout and recurrent layer normalization
	- Limited short-term memory
		- LSTM and GRU cells

# Recurrent Neurons and Layers

- A recurrent neural network looks like a feedforward neural network, except it also has connections pointing backwards
- At each time step (frame), this recurrent neuron receives the inputs $X_{(t)}$, as well as its own output from the previous time step, $\hat{y}_{(t-1)}$
- The initial time step is set to 0
- Unrolling the network through time

<img src="/images/Pasted image 20260204110433.png" alt="image" width="500">
<img src="/images/Pasted image 20260204110444.png" alt="image" width="500">
- Each recurrent neuron has 2 sets of weights
	- One for inputs, $w_X$
	- Other for outputs of the previous time step, $w_{\hat{y}}$
- Place all the weight vectors in two weight matrices
 - $b$ is the bias vector, and $\theta$ is the activation function

- Output of a recurrent layer for a single instance

![[Pasted image 20260302172159.png]]

- Output of a layer of recurrent neurons for all instance in a pass (mini-batch)
![[Pasted image 20260302172252.png]]

## Memory Cells

- Since the output of a recurrent neuron at time step t is a function of all the inputs from previous time steps, it has a form of memory
- A part of neural network that preserved some state is called a memory cell
- A single recurrent neuron, or a layer of recurrent neurons, is a basic cell
	- Learning only short patterns
- $h_{(t)} = f(x_{(t)}, h_{(t-1)})$ = cell's state at time step t

<img src="/images/Pasted image 20260204110453.png" alt="image" width="500">

## Input and Output Sequences

- Sequence-to-sequence network
	- An RN can simultaneously take a sequence of inputs and produce a sequence of outputs
	- Time series data
- Sequence-to-vector network
	- Feed the network a sequence of inputs and ignore all outputs except for the last one
- Vector-to-sequence network
	- Input vector over and over again, and output a sequence
	- Image to caption
- Sequence-to-vector network (encoder), followed by a vector-to-sequence (decoder)
	- Translating a sentence from one language to another
	- Encoder-decoder
		- Encoder converts this sentence into a single vector representation, then the decoder decodes this vector into a sequence in another language
	- The last words of a sentence can affect the first worlds of the translation
		- Wait until the entire sentence has been encoded

<img src="/images/Pasted image 20260204110504.png" alt="image" width="500">
# Training RNNs

- Backpropagation through time (BPTT)
	- Unroll it through time, and use regular backpropagation
- There is a first forward pass through the unrolled network
- The output sequence is evaluated using a loss function
- The gradients of that loss function are then propagated backward through the unrolled network
- BPTT can perform a gradient descent step to update the parameters

<img src="/images/Pasted image 20260204110519.png" alt="image" width="500">
# Forecasting a Time Series

- Build a model capable of forecasting the number of passengers that will ride on bus and rail the next day

```python
import pandas as pd
from pathlib import Path

path = Path("datasets/ridership/CTA_-_Ridership_-_Daily_Boarding_Totals.csv")
df = pf.read_csv(path, parse_dates=["service_date"])
df.columns = ["date", "day_type", "bus", "rail", "total"]
df = df.sort_values("date").set_indext("date")
df = df.drop("total", axis=1)
df = df.drop_duplicates()

df.head()
           day_type     bus    rail
date
2001-01-01        U  297192  126455
2001-01-02        W  780827  501952
2001-01-03        W  824923  536432
2001-01-04        W  870021  550011
2001-01-05        W  890426  557917

import matplotlib.pyplot as plt
df["2019-03":"2019-05"].plot(grid=True, marker=".", figsize=(8, 3.5))
plt.show()
```
<img src="/images/Pasted image 20260204110529.png" alt="image" width="500">
- Multivariate time series
- The time series shows weekly seasonality
- Naive forecasting
	- Simply copying a past value to make our forecast
- Overlay two time series, and the same series lagged by one week
	- Plot the different between the two (differencing)

```python
diff_7 = df[["bus", "rail"]].diff(7)["2019-03":"2019-05"]
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
df.plot(ax=axs[0], legend=False, marker=".")
df.shift(7).plot(ax=axs[0], grid=True, legend=False, linestyle=":")
diff_7.plot(ax=axs[1], grid=True, marker=".")
plt.show()
```
- When a time series is correlated with a lagged version, is is autocorrelated

```python
list(df.loc["2019-05-25":"2019-05-27"]["day_type"])
['A', 'U', 'U']
```
<img src="/images/Pasted image 20260204110539.png" alt="image" width="500">
```python
diff_7.abs().mean()
bus     43915.608696
rail    42143.271739
dtype: float64

targets = df[["bus", "rail"]]["2019-03":"2019-05"]
(diff_7 / targets).abs().mean()
bus     0.082938
rail    0.089948
dtype: float64
```
- Mean absolute percentage error (MAPE)
	- Naive forecasts give us a MAPE of ~8.3^ 


<img src="/images/Pasted image 20260204110548.png" alt="image" width="500">



<img src="/images/Pasted image 20260204110601.png" alt="image" width="500">

## The ARMA Model Family
## Preparing the Data for Machine Learning Models
## Forecasting Using a Linear Model
## Forecasting Using a Simple RNN
## Forecasting Using a Deep RNN

<img src="/images/Pasted image 20260204110615.png" alt="image" width="500">

## Forecasting Multivariate Time Series
## Forecasting Several Time Steps Ahead

<img src="/images/Pasted image 20260204110632.png" alt="image" width="500">

## Forecasting Using a Sequence to Sequence Model


# Handling Long Sequences

<img src="/images/Pasted image 20260204110649.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110704.png" alt="image" width="500">

## Fighting the Unstable Gradients Problem
## Tackling the Short-Term Memory Problem
