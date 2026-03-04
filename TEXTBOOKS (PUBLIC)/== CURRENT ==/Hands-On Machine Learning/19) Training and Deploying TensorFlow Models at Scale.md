- As time passes, you will need to retrain the model on fresh data and push the updated version to production
- Handle model versioning
	- Transitioning from one model to the next
	- Rollback if there are any issues
	- Run different models in parallel for A/B experiments
- Queries per second (QPS)
	- Scale up and support the load
- Use TensorFlow Serving, on a local machine or cloud service
	- Google Vertex AI
- Deploying models to mobile apps, embedded devices, and web apps

# Serving a TensorFlow Model

- A trained model can be used in any Python code
- As infrastructure grows, there comes a point where it is preferable to wrap the model in a small service whose sole role is to make predictions and have the rest of the infrastructure query it
	- Decouples model and infrastructure
	- A/B experiments
	- Create microservice using another technology

## Using TensorFlow Serving

- Writing in C++
- Sustain a high load, serve multiple version, and watch a model repository to automatically deploy version

![[Pasted image 20260304161908.png]]


### Exporing SavedModels

 - `model.save()`
 - Create a subdirectory for eaach model version

```python
from pathlib import Path
import tensorflow as tf
X_train, x_valid, X_test = [...]
mode =
```


## Creating a Prediction Service on Vertex AI
## Running Batch Prediction Jobs on Vertex AI

# Deploying a Model to a Mobile or Embedded Device

# Running a Model in a Web Page

# Using GPUs to Speed Up Computations

## Getting Your Own GPU
## Managing the GPU RAM
## Placing Operations and Variables on Devices
## Parallel Execution Across Multiple Devices

# Training Models Across Multiple Devices

## Model Parallelism
## Data Parallelism
## Training at Scale Using the Distribution Strategies API
## Training a Model on a TensorFlow Cluster
## Running Larger Training Jobs on Vertex AI
## Hyperparameter Tuning on Vertex AI