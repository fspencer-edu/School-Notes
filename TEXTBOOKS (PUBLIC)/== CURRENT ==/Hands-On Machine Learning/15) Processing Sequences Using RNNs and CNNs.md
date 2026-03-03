
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
	- Naive forecasts give us a MAPE of ~8.3% for bus and 9.0% for rail
- Metrics to evaluate forecasts
	- MAE
	- MAPE
	- MSE
- Find yearly seasonality

```python
period = slice("2001", "2019")
df_monthy = df.resample('M').mean()
rolling_average_12_months = df_monthly[period].rolling(window=12).mean()

fig, ax = plt.subplots(figzie=(8, 4))
df_monthly[period].plot(ax=ax, marker=".")
rolling_average_month.plot(ax=ax, grid=True, legend=False)
plt.show()
```
<img src="/images/Pasted image 20260204110548.png" alt="image" width="500">
```python
df_monthly.diff(12)[period].plot(grid=True, marker=".", figsize=(8, 3))
plt.show()
```

<img src="/images/Pasted image 20260204110601.png" alt="image" width="500">
- Differencing removes yearly seasonality, and long term trends
- Easier to study a stationary time series
	- Statistical properties remain constant over time, without any seasonality or trends

## The ARMA Model Family

- Autoregressive moving average (ARMA) model
	- Computes its forecasts using a simple weighted sum of lagged values and corrects these forecasts by adding a moving average

- Forecasting using an ARMA model

![[Pasted image 20260302174748.png]]

First sum
$\alpha_i$ = learned weights 
$p$ = hyperparameter

Second sum
$\epsilon_{(t)}$ = forecast errors
$\theta_i$ = learned weights

- Model assumes that the time series is stationary
- Running 2 rounds of differencing will eliminate quadratic trends
- Order of integration
	- Running $d$ consecutive rounds of differencing computes an approximation of the $d^{th}$ order derivative of the time series
- Seasonal ARIMA (SARIMA)
	- It models the time series in the same way as ARMIA
	- Models a seasonal component for a given frequency

```python
from statsmodels.tsa.arima.model import ARMIA

origin, today = "2019-01-01", "2019-05-31"
rail_series = df.loc[origin:today]["rail"].asfreq("D")
model = ARIMA(rail_series,
			  order=(1, 0, 0),
			  seasonal_order=(0, 1, 1, 7))
model = model.fit()
y_pred = model.forecast() # 427,758.6
```

- Take the rail ridership from the start up to today, and use daily frequency
- Create an ARIMA instance, passing it all the data until today, and set hyperparameters, and seasonal order
- Fit the model, and use it to make a forecast for tomorrow
	- Predicts => 427,758.6
	- Actual =>  379,044
- 12.9% error
- Make forecasts for every day in March, April, and May, and compute the MAE

```python
origin, start_date, end_date = "2019-01-01", "2019-03-01", "2019-05-31"
time_period = pd.date_range(start_date, end_date)
rail_series = df.loc[origin:end_date]["rail"].asfreq("D")
y_preds = []
for today in time_period.shift(-1):
    model = ARIMA(rail_series[origin:today],  # train on data up to "today"
                  order=(1, 0, 0),
                  seasonal_order=(0, 1, 1, 7))
    model = model.fit()  # note that we retrain the model every day!
    y_pred = model.forecast()[0]
    y_preds.append(y_pred)

y_preds = pd.Series(y_preds, index=time_period)
mae = (y_preds - rail_series[time_period]).abs().mean() # 32,040.7
```
- Choosing hyperparameters
	- Brute-force approach
	- Run search grid

## Preparing the Data for Machine Learning Models

- Use ML models to forecast this time series
- Forecast tomorrow's ridership based on the ridership of the past 8 weeks of data
- The inputs of out models will be sequences
- Use 56-day window from the past as training data, and target for each window will be the value immediately following it

```python
import tensorflow as tf
my_series = [0, 1, 2, 3, 4, 5]
my_dataset = tf.keras.utils.timeseries_dataset_from_array(
	my_series,
	targets=my_series[3:],
	sequence_length=3,
	batch_size=2
)
list(my_dataset)
[(<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
  array([[0, 1, 2],
         [1, 2, 3]], dtype=int32)>,
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], dtype=int32)>),
 (<tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[2, 3, 4]], dtype=int32)>,
  <tf.Tensor: shape=(1,), dtype=int32, numpy=array([5], dtype=int32)>)]
```

- Each sample in the dataset is a window of length 3, along with its corresponding target
	- `[0, 1, 2], [1, 2, 3], [2, 3, 4]`
		- Targets are 3, 4, 5
- Another way to get the same result is to use the `window()` method

```python
for window_dataset in tf.data.Dataset.range(6).window(4, shift=1):
	for element in window_dataset:
		print(f"{element}", end=" ")
	print()
0 1 2 3
1 2 3 4
2 3 4 5
3 4 5
4 5
5
```

- The dataset contains 6 windows, each shifted by one step compared to the previous one
- The last 3 windows are smaller because they have reached the end
	- Remove these smaller windows
		- `drop_remainder=True`
- Returns a nested dataset
- Cannot use a nested dataset directly for training (expect tensors)
- Call `flat_map()`
	- Converts a nested dataset into a flat dataset
	- Takes a function, to transform each dataset in the nested dataset before flatting

```python
dataset = tf.data.Dataset.range(6).window(4, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window_dataset: window_dataset.batch(4)
for window_tensor in datasetL
	print(f"{window_tensor}")
[0 1 2 3]
[1 2 3 4]
[2 3 4 5]

# helper function: extract windows from a dataset
def to_windows(dataset, length):
	dataset = dataset.window(length, shift=1, drop_remainder=True)
	return dataset.flat_map(lambda window_df: window_ds.batch(length))
	
# split each window into inputs and targets
dataset = to_windos(tf.data.Dataset.range(6), 4)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
list(dataset.batch(2))
[(<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[0, 1, 2],
         [1, 2, 3]])>,
  <tf.Tensor: shape=(2,), dtype=int64, numpy=array([3, 4])>),
 (<tf.Tensor: shape=(1, 3), dtype=int64, numpy=array([[2, 3, 4]])>,
  <tf.Tensor: shape=(1,), dtype=int64, numpy=array([5])>)]
```

- Before training, split data into a training period, a validation period, and a test period

```python
# scaled down rail data
rail_train = df["rail"]["2016-01":"2018-12"] / 1e6
rail_valid = df["rail"]["2019-01":"2019-05"] / 1e6
rail_test = df["rail"]["2019-06":] / 1e6
```
- Since gradient descent expects the instances in the training set to be independent and identically distributed (IDD)
	- Shuffle training windows

```python
seq_length = 56
train_ds = tf.keras.utils.timeseries_dataset_from_array(
	rail_train.to_numpy(),
	targets=rail_train[seq_length:],
	sequence_length=seq_length,
	batch_size=32,
	shuffle=True,
	seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
	rail_valid.to_numpy(),
	targets=rail_valid[seq_length:],
	sequence_length=seq_length,
	batch_size=32
)
```

## Forecasting Using a Linear Model

- Use a linear model
	- Huber loss

```python
tf.random.set_seed(42)
model = tf.keras.Sequential([
	tf.keras.layers.Dense(1, input_shape=[seq_length])
])
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
	monitor="val_mae", patience=50, restore_best_weights=True)
opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
history = model.fit(train_ds, validation_data=valida_ds, epochs=500m
		callbacks=[early_stopping_cb])
```

- Model reaches a validation MAE of 37,866
	- Better than naive forecast, worse than SARIMA

## Forecasting Using a Simple RNN

- Run a basic RNN
- Containing a single recurrent layer with one recurrent neuron

```python
model = tf.keras.Sequential([
	tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
```

- All recurrent layers in Keras expect 3D inputs of shape `[batch size, time steps, dimensionality]`, where dimensionality is 1 for univariate time series and and more for multivariate
- The initial state is set to 0, and is passed to a single recurrent neuron, with the value of the first time step
- Neuron computes a weighed sum plus bias term, and applies the activation function to the result
- The new state is passed to the same recurrent neuron with the next input value, and the process is repeated until the last time step

- Model issues
	- Model only has a single recurrent neuron
	- The time series contains values from 0 to about 1.4
- Create a model with a larger recurrent layer, containing 32 recurrent neurons, add a dense output layer on top of it with a single output neuron and no activation function

```python
univar_model = tf.keras.Sequential([
	tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
	tf.keras.layers.Dense(1)
])
```

- This produces a better model, and beats SARIMA
	- Without removing trend and seasonality
	- Improve with stationary

## Forecasting Using a Deep RNN

- Stack multiple layers of cells for a deep RNN

<img src="/images/Pasted image 20260204110615.png" alt="image" width="500">
- Stack recurrent layers
- Use `LSTM` or a `GRU` layer
- The first two are sequences-to-sequence layers
- The last one is a sequence-to-vector layer
- Add dense layer to produce the model's forecast

```python
deep_model = tf.keras.Sequential([
	tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
	tf.keras.layers.SimpleRNN(32, return_sequences=True),
	tf.keras.layers.SimpleRNN(32),
	tf.keras.layers.Dense(1)
])
```
- Reaches a better MAE, but not shallow RNN

## Forecasting Multivariate Time Series

- A great quality of NN is their flexibility
	- Deal with multivariate time series
- Forecast the rail time series using both the bus and rail data as input
- Shift the day type series one day into the future, so that the model is given tomorrow's data type as input

```python
df_mulvar = df[["bus", "rail"]] / 1e6
df_mulvar["next_day_type"] = df["day_type"].shift(-1)
df_mulvar = pd.get_dummies(df_mulvar)

# split data: training, valid, test
mulvar_train = df_mulvar["2016-01":"2018-12"]
mulvar_valid = df_mulvar["2019-01":"2019-05"]
mulvar_test = df_mulvar["2019-06":]

# create the datasets
train_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
	mulvar_train.to_numpy(),
	targets=mulvar_train["rail"][seq_length:],
	[...]
)
valid_mulvar_ds = tf.keras.utils.timeseries_dataset_from_array(
	mulvar_valid.to_numpy(),
	targets=mulvar_valid["rail"][seq_length:],
	[...]
)

# create RNN
mulvar_model = tf.keras.Sequential([
	tf.keras.layers.SimpleRNN(32, input_shape=[None, 5]),
	tf.keras.layers.Dense(1)
])
```

- The only difference form the univariate mode to multivariate is the input shape
- Using a single model for multiple related tasks often results in better performance than using a separate model for each task

## Forecasting Several Time Steps Ahead

```python
import numpy as np
X = rail_valid.to_numpy()[np.newaxis, :seq_length, np.newaxis]
for step_ahead in range(14):
	y_pred_one = univar_model.predict(X)
	X = np.concatenate([X, y_pred_one.reshape(1, 1, 1)], axis=1)
```

- Take the rail ridership of the first 56 days of the validation period, and convert the data to a numpy array
- Repeatidely use the model to f


<img src="/images/Pasted image 20260204110632.png" alt="image" width="500">

## Forecasting Using a Sequence to Sequence Model


# Handling Long Sequences

<img src="/images/Pasted image 20260204110649.png" alt="image" width="500">

<img src="/images/Pasted image 20260204110704.png" alt="image" width="500">

## Fighting the Unstable Gradients Problem
## Tackling the Short-Term Memory Problem
