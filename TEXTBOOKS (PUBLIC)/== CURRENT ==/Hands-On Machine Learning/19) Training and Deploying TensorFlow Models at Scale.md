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
model = [...]

model_name = "my_mnist_model"
model_version = "0001"
model_path = Path(model_name) / model_version
model.save(model_path, save_format="tf")
```

- Include all the preprocessing layers in the final model
- The model can only be used with models that are based on TF operations
- `saved_model_cli`

```python
saved_model_cli show --dir my_mnist_model/0001
The given SavedModel contains the following tag-sets:
'serve'
```

- SavedModel contains one or more metagraphs
	- A computation graph plus some function signature definitions, including input and output names, types, and shaped
- Each metagraph is identified by a set of tags
	- "train"
	- "serve"
	- "gpu"

```python
saved_model_cli show --dir 0001/my_mnist_model --tag_set serve
The given SavedModel MetaGraphDef contains SignatureDefs with these keys:
SignatureDef key: "__saved_model_init_op"
SignatureDef key: "serving_default"
```

- This metagraph contains 2 signature definitions
	- Initialization function
	- Default serving function

```python
 saved_model_cli show --dir 0001/my_mnist_model --tag_set serve \
                       --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['flatten_input'] tensor_info:
      dtype: DT_UINT8
      shape: (-1, 28, 28)
      name: serving_default_flatten_input:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['dense_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 10)
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

- This function's input is named "flatten_input"
- Output is `dense_1`
- These correspond to the model's input and output layer name

### Installing and Starting TensorFlow Serving

- System's package manager
- Docker image
- Installing from source

```python
url = "https://storage.googleapis.com/tensorflow-serving-apt"
src = "stable tensorflow-model-server tensorflow-model-server-universal"
!echo 'deb {url} {src}' > /etc/apt/sources.list.d/tensorflow-serving.list
!curl '{url}/tensorflow-serving.release.pub.gpg' | apt-key add -
!apt update -q && apt-get install -y tensorflow-model-server
%pip install -q -U tensorflow-serving-api
```

- This adds the TF package repository to Ubuntu's list of package sources
- Downloads TF public GPG key and adds it to the package manager's key list to verify TF package signatures
- Uses `apt` to install packages and libraries
- Start the server, with the absolute path of the base model directory

```python
import os
os.environ["MODEL_DIR"] = str(model_path.parent.absolute())

# start server
%%bash --bg
tensorflow_model_server \
     --port=8500 \
     --rest_api_port=8501 \
     --model_name=my_mnist_model \
     --model_base_path="${MODEL_DIR}" >my_server.log 2>&1
```

- This redirects the standard output and standard error to the `my_server.log` file
- TF Serving is running in the background, and its logs are saved to a file
- `docker pull tensorflow/serving`
- Start server inside a Docker container

```python
docker run -it --rm -v "/path/to/my_mnist_model:/models/my_mnist_model" \
    -p 8500:8500 -p 8501:8501 -e MODEL_NAME=my_mnist_model tensorflow/serving
```

- `it`
	-  Makes the container interactive and displays the server's output
- `--rm`
	- Deletes the container when you stop is
	- Does not delete image
- `-v "/path/to/my_mnist_model:/models/my_mnist_model"`
	- Makes the host's directory available to the container at the path
- `-p 8500:8500`
	- Makes the Docker engine forward the host's TCP port 8500 to container's TCP port 8500
- `-p 8500:8501`
	- Forwards the host's TCP port 8501 to container's TCP port 8501
- `-e MODEL_NAME=my_mnist_model`
	- Sets the container's environment variable
- `tensorflow/serving`
	- Name of image to run


### Querying TF Serving Through the REST API

- Create the query
- Contain the name of the function signature to call, and input data
- Request uses JSON format, convert the input image from a NumPy array to a Python list

```python
import json

X_new = X_test[:3]
request_json = json.dumps({
	"signature_name": "serving_default",
	"instance": X_new.tolist(),
})

>>> request_json
'{"signature_name": "serving_default", "instances": [[[0, 0, 0, 0, ... ]]]}'

# send this request to TF Serving via an HTPP POST request
import requests
server_url = "http://localhost:8501/v1/models/my_mnist_model:predict"
response = requests.post(server_url, data=request_json)
response.raise_for_status()
response = response.json()

# response should be a dictionary containing a single pred. key
>>> import numpy as np
>>> y_proba = np.array(response["predictions"])
>>> y_proba.round(2)
array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ],
       [0.  , 0.  , 0.99, 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.97, 0.01, 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  ]])
```

- Any client application can makes REST queries without additional dependencies
- Other protocols are not always readily available
- Use gRPC, to transmit more efficiently
	- Compact binary format, based on HTTP/2 framing


### Querying TF Serving through the gRPC API

- gPRC API expects a serialized `PredictRequest` protocol buffer as input, and it outputs a serialized `PredictResponse`

```python
from tensorflow_serving.apis.predict_pb2 import PredictRequest

request = PredictRequest()
request.model_spec.name = model_nae
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))
```
- This code creates a `PredictRequest` protocol buffer and fills in the required fields
	- Model name
	- Signature
	- Input data
- Creates a `Tensor` protocol buffer based on the given tensor of NumPy array
- Send the request to the server and get its response

```python
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)
```

- After the imports, create a gRPC communication channel to localhost on TCP post 8500
- Create a service over this channel and use it to send a request, with a 10-sec timeout
	- Synchronous and insecure

```python
output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)
```

- If you run this code and print `y_proba.round(2)`, you will get the exact same estimated class probabilities as earlier

### Deploying a new model version

- Create a new model version and export a SavedModel

```python
model = [...]

model_version = "0002"
model_path = Path(model_name) / model_version
model.save(model_path, save_format="tf")
```

- At regular intervals, TF Serving checks the model directory for new model version
- Automatically handles the transition gracefully
- Answers pending requests
- When every request has been answered, the previous model version is unloaded

```python
[...]
Reading SavedModel from: /models/my_mnist_model/0002
Reading meta graph with tags { serve }
[...]
Successfully loaded servable version {name: my_mnist_model version: 2}
Quiescing servable version {name: my_mnist_model version: 1}
Done quiescing servable version {name: my_mnist_model version: 1}
Unloading servable version {name: my_mnist_model version: 1}
```

 - Model warmup
	- If the SavedModel contains some examples instances in the assets/extra directory, you can configure TF Serving to run the new model on these instances before starting to use it to serve request
- This approach offers a smooth transition, but uses too much RAM
	- Configure TF Serving so it handles all pending requests with the previous model versions and unloads it before loading and using the new model version
	- Service will be unavailable for a short period
- Automatic batching capabilities
	- `--enable_batching`
- When TF Serving receives multiple requests within a short period of time, it will automatically batch them together before using the model


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