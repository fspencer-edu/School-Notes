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

<img src="/images/Pasted image 20260304161908.png" alt="image" width="500">


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
- TF Serving dispatches each prediction to the right client
- If there are many QPS, use multiple servers and load-balance the queries
	- Kubernetes
		- An open source system for simplifying container orchestration across many servers
	- Amazon AWS
	- Microsoft Azure
	- Google Cloud Platform
	- IBM Cloud
	- Alibaba Cloud
	- Oracle Cloud
	- PaaS
- Vertex AI
	- Only platform with TPU
	- Supports TensorFlow 2
	- Scikit-Leran
	- XGBoost
	- AWS SageMaker and Microsoft AI Platform

<img src="/images/Pasted image 20260304165237.png" alt="image" width="500">


## Creating a Prediction Service on Vertex AI

- Vertex AI is a platform within Google Cloud Platform (GCP) that offers a wide range of AI-related tools and services
- AutoML
	- Architecture search
- Manage trained models, use them to make batch predictions on large amount of data, schedule multiple jobs for data workflows, serve your models via REST or gPRC as scale, and experiment with your data and models within a hosted Jupyter environment (Workbench)
- Matching Engine
- GCP also includes other AI services
	- Vision
	- Translation
	- Speech-to-text

1. Log in to your Google account
2. Every resource in GCP belongs to a project

- Write a script to automate the GCP
- Google Cloud's CLI, `gcloud`
- `gsutil`
	- Interact with Google Cloud Storage

- Authenticate GCP

```python
from google.colab import auth

auth.authentication_user()
```

- The authentication process is based on OAuth 2.0
- When an application needs to access a service on GCP on its own behalf, not with a user, then is should use a service account
- Google's Workload Identity
	- Map the right service account to each Kubernetes service account
- Create a Google Cloud Storage bucket to store SavedModels

```python
from google.cloud import storage

project_id = "my_project"
bucket_name = "my_bucket"
location = "us-centrall"

storage_client = storage.Client(project=project_id)
bucket = storage_client.create_bucket(bucket_name, location=location)
```

- GCS uses a single worldwide namespace for buckets
	- DNS naming conventions
	- Public
	- Use domain name or project ID
- Update the model directory to the new bucket
- Files in GCS are called blobs or objects
- Blob names can be arbitrary Unicode strings, and can contain forward slashes

```python
def upload_directory(bucket, dirpath):
	dirpath = Path(dirpath)
	for filepath in dirpath.glob("**/*")
		if filepath.is_file():
			blob = bucket.blob(filepath.relative_to(dirpath.parent).as_posix())
			blob.upload_from_filepath(filepath)
			
upload_directory(bucket, "my_mnist_model")
```

- Speed up the file upload with multithreading

```python
!gsutil -m cp -r my_mnist_model gs://{bucket_name}/
```

- To communicate with Vertex AI, use `google-cloud-aiplatform` library
- Create an endpoint
	- Client application

```python
from google.cloud import aiplatform

server_image = "gcr.io/cloud-aiplatform/prediction/tf2-gpu.2-8:latest"

aiplatform.init(project=project_id, location=location)
mnist_model = aiplatform.Model.upload(
    display_name="mnist",
    artifact_uri=f"gs://{bucket_name}/my_mnist_model/0001",
    serving_container_image_uri=server_image,
)

# deploy model to this endpoint
endpoint = aiplatform.Endpoint.create(display_name="mnist-endpoint")

endpoint.deploy(
    mnist_model,
    min_replica_count=1,
    max_replica_count=5,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1
)
```
- This code take a few minutes to run, because Vertex AI needs to set up a VM
- GCP enforces various GPU quotas
- Vertex AI will initially spawn the min number of compute nodes
- If QPS rate goes down, it removes node
- Cost is linked to load, machine, and accelerator types

```python
response = endpoint.predict(instances=X_new.tolist())
```
- Convert the images to a Python list

```python
>>> import numpy as np
>>> np.round(response.predictions, 2)
array([[0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  ],
       [0.  , 0.  , 0.99, 0.01, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
       [0.  , 0.97, 0.01, 0.  , 0.  , 0.  , 0.  , 0.01, 0.  , 0.  ]])
       
endpoint.undeploy_all()  # undeploy all models from the endpoint
endpoint.delete()
```

- The prediction is the same on the cloud

## Running Batch Prediction Jobs on Vertex AI

- Instead of calling the prediction service repeatedly, ask Vertex AI to run a prediction job
	- Does not require an endpoint, only a model
- Prepare a batch, upload it to GCS
	- Create a file containing one instance per line, with JSON (JSON lines)
	- Pass this to Vertex AI

```python
batch_path = Path("my_mnist_batch")
batch_path.mkdir(exist_ok=True)
with open(batch_path / "my_mnist_batch.jsonl", "w") as jsonl_file:
	for image in X_test[:100].tolist():
		jsonl_file.write(json.dumps(image))
		jsonl_file.write("\n")
		
upload_directory(bucket, batch_path)
```

- To launch the prediction job, specify the name, and type and number of machines, and accelerators to use

```python
batch_prediction_job = mnist_model.batch_predict(
    job_display_name="my_batch_prediction_job",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=5,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=1,
    gcs_source=[f"gs://{bucket_name}/{batch_path.name}/my_mnist_batch.jsonl"],
    gcs_destination_prefix=f"gs://{bucket_name}/my_mnist_predictions/",
    sync=True  # set to False if you don't want to wait for completion
)

```
- For large batches, split the input into multiple JSON lines file and list them all via the `gcs_source`
- Each value is a directory containing an instance and its corresponding prediction
- The instances are listed in the same order as the input
	- Also outputs prediction-errors file

```python
y_probas = []
for blob in batch_prediction_job.iter_outputs():
    if "prediction.results" in blob.name:
        for line in blob.download_as_text().splitlines():
            y_proba = json.loads(line)["prediction"]
            y_probas.append(y_proba)
            
>>> y_pred = np.argmax(y_probas, axis=1)
>>> accuracy = np.sum(y_pred == y_test[:100]) / 100
0.98
```

- Delete model, bucket, or batch prediction job

```python
for prefix in ["my_mnist_model/", "my_mnist_batch/", "my_mnist_predictions/"]:
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        blob.delete()

bucket.delete()  # if the bucket is empty
batch_prediction_job.delete()
```

# Deploying a Model to a Mobile or Embedded Device

- ML models are not limited to running on big centralized servers with multiple GPUs
- They can also run closer to source of data (edge computing)
	- Without internet
	- Reduces latency
	- Improve pricacy
- A large model may not fit in the device
- TFLite
	- Reduce the model size, and RAM usage
	- Reduce the computations for each prediction
	- Reduce latency, battery usage, and heating
	- Adapt the model to device-specific constraints

- TFLite's model converter can take a SavedModel and compress it to a lighter format based on FlatBuffer
- Load FlatBuffers straight to RAM without preprocessing
- Interpreter will execute it to make predictions

```python
converted = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
tflite_model = converter.convert()
with open("my_converted_savedmodel.tflite", "wb") as f:
	f.write(tflite_nodel)
```

- The converted optimizes the model
- Prunes all the operations that are not needed to make predictions
	- Training operations
- Batch normalization layers are folded into the previous layer's addition and multiplication operations
- Smaller bit-widths
	- Half-floats (16 bits)
- Quantizing the model weights down to fixed-point, 8-bit integers
- Post-training quantization
	- Quantizes the weights after training
	- Finds max absolute weight value, m, then maps the floating-point range -m to +m to the fixed point (integer)

<img src="/images/Pasted image 20260304172312.png" alt="image" width="500">

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

- At runtime the quantized weights get converted back to floats before they are used
- Accuracy loss is acceptable (float point errors)
- Quantizes the activations so that computations can be done with integers
- Quantization-aware training
	- Adding fake quantization operations to the model so it can learn to ignore the quantization noise during training

# Running a Model in a Web Page

- Running the ML model on the client side, rather that server side can be used
	- Web application in situations where the user's connectivity is slow
	- When you need the model's responses to be fas
	- Web service makes predictions based on some private user data
- TensorFlow.js
	- Load TFLite model
- The following JS script imports the TFJS library, download pretrained MobileNet Model, and uses the model to classify an image and log the prediction

```python
import "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest";
import "https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@1.0.0";

const image = document.getElementById("image");

mobilenet.load().then(model => {
	model.classify(image).then(predictions => {
		for (var i = 0; i < predictions.length; i++) {
			let className = predictions[i].className
			let proba = (predictions[i].probability * 100).toFixed(1)
			console.log(className + " : " + proba + "%");
		}
	});
});
```

- Turn a website into a progressive web app (PWA)
	- Site that respects a number of criteria that allow it to be views in any browser, and installed as a standalone app on a mobile device
	- Use a service worker to work offline
- TFJS also supports training a model directly in the web browser
	- Federated learning


# Using GPUs to Speed Up Computations

- Speed up training
	- Better weight initialization
	- Sophisticated optimizers
- Training a  large NN on a single machine is computationally expensive

<img src="/images/Pasted image 20260304174043.png" alt="image" width="500">


## Getting Your Own GPU

- Train your models locally to avoid uploading data to cloud
- GPU characteristics
	- > 10 GM of RAM
	- Bandwidth
	- Cores
	- Cooling system
- Compute Unified Device Architecture library (CUDA) toolkit
	- Allows developers to use CUDA-enabled GPUs for computations
	- Deep Neural Network library (cuDNN)
		- GPU-acclerlerated library
			- Activation layers
			- Normalization
			- Forward and backward convolutions
			- Pooling

<img src="/images/Pasted image 20260304174343.png" alt="image" width="500">

- `nvidia-smi`
	- Checks that drivers and libraries are installed

<img src="/images/Pasted image 20260304174435.png" alt="image" width="500">

```python
physical_gpus = tf.config.list_physical_device("GPU")
physical_gpus
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Managing the GPU RAM

- By default TF uses almost all the RAM available
- Does this to limit GPU RAM fragmentation
- Split the GPU between multiple processes, to run more than one

- Split programs per GPUs

```python
$ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 program_1.py
# and in another terminal:
$ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,2 python3 program_2.py
```

<img src="/images/Pasted image 20260304174634.png" alt="image" width="500">

- Tell TF to use a specific amount of GPU RAM
	- Logical GPU device (virtual GPU device) for each physical GPU device and set the memory to 2 GiB

```python
for gpu in physical_gpus:
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
    )
```
<img src="/images/Pasted image 20260304174752.png" alt="image" width="500">

- Another option is to tell TF to grab memory only when needed

```python
for gpu in physical_gpus:
	tf.config.experiental.set_memory_growth(gpu, True)
```

- Split a GPU into two or more logical devices
	- Only have one physical GPU

```python
tf.config.set_logical_device_configuration(
    physical_gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048),
     tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
)

>>> logical_gpus = tf.config.list_logical_devices("GPU")
>>> logical_gpus
[LogicalDevice(name='/device:GPU:0', device_type='GPU'),
 LogicalDevice(name='/device:GPU:1', device_type='GPU')]
```

## Placing Operations and Variables on Devices

- Keras and tf.data do a good job of placing operations and variables where they belong
	- Place the data preprocessing operation on the CPU
	- Place NN operations on GPUs
	- GPUs have limited communication bandwidth, avoid unnecessary data transfer into and out of the GPUs
	- Adding more CPU RAM to a machine is cheap
	- GPU RAM is baked into the GPU, and is expensive


- All variables and operations will be placed on the first GPU, except for variables and operations that do not have GPU kernel
	- Placed on the CPU

```python
>>> a = tf.Variable([1., 2., 3.])  # float32 variable goes to the GPU
>>> a.device
'/job:localhost/replica:0/task:0/device:GPU:0'
>>> b = tf.Variable([1, 2, 3])  # int32 variable goes to the CPU
>>> b.device
'/job:localhost/replica:0/task:0/device:CPU:0'
```

- The second variable is placed on the CPU because there are no GPU kernels for integer variables, or for operations involving integer tensors

```python
>>> with tf.device("/cpu:0"):
...     c = tf.Variable([1., 2., 3.])
...
>>> c.device
'/job:localhost/replica:0/task:0/device:CPU:0'
```

- CPU is always treated as a single device

## Parallel Execution Across Multiple Devices

- When a TF runs a function, it starts by analyzing its graph to find the list of operations that need to be evaluates, and counts how many dependencies each of them has
- TF adds each operation with zero dependencies to the evaluation queue of this operation's device
- Dependency counter of each operation that depends on it is decremented
- Once the dependency counter reaches zero, it is pushed to the evaluation queue of its device
- Returned when computed

<img src="/images/Pasted image 20260304175441.png" alt="image" width="500">

- Operations in the CPU's evaluation queue are dispatches to a thread pool called the inter-op thread pool
	- If there are multiple cores, then operations are parallel
	- Kernels split their tasks into multiple sub-operations, placed on another evaluation queue, and dispatched to intra-op thread pool
- For GPU
	- Operations are evaluated sequentially
	- More operations have multi-threaded GPU kernels
		- CUDA
		- cuDNN
	- No need for inter-op thread pools
- A, B, and C are source ops
	- Immediately evaluated
- A and B are placed on the CPU (inter-op) and processed in parallel
	- A is split into 3 parts in an intra-op thread pool
- C goes to GPU #0
	- cuDNN, manages its own intra-op thread pool
- If C finishes first, the dependency counters of D and E are decremented and they reach 0
- TF function modified a stateful resource, such as a variable
	- Ensures that the order of execution matches the order in the code
- Control threads

- Exploit GPU operations
	- Train models in parallel
	- Train model on a single GPU and perform all the preprocessing in parallel on the CPU
	- Model takes 2 images as input and processes them using 2 CNNs before joining their outputs
	- Create an efficient ensemble

# Training Models Across Multiple Devices

2 approaches to training a single model across multiple devices
1. Model parallelism
2. Data parallelism

## Model Parallelism

- To train on multiple devices, split model into separate chunks and run each chuck on a different device
- Fully connected networks are compromised when split
- Slice the model vertically
	- But this requires cross-device communication

<img src="/images/Pasted image 20260304180258.png" alt="image" width="500">

- Some NN architectures, such as convolutional NN, contain layers that are only partially connected to the lower layers, so it is easier to chunk

<img src="/images/Pasted image 20260304180334.png" alt="image" width="500">

- Deep recurrent NN can be split more efficiently
- Split horizontally by placing each layer on a different device, and feed the network with an input sequence to process, then at the first time step only one device will be active
- At the second step two will be active, and the signal propagates to the output layer
- All devices are active simultaneously

<img src="/images/Pasted image 20260304180454.png" alt="image" width="500">

- Model parallelism may speed up running or training some types of NN, but not al


## Data Parallelism

- Data parallelism or single program, multiple data (SPMD)
	- Another way to parallelize the training of NN is to replicate it on every device and run each training step on all replicas, using a different mini-batch
	- Gradients are averaged, and the result is used to update the model parameters

### Data Parallelism Using the Mirrored Strategy

- Simplest approach is to completely mirror all the model parameters across all the GPUs
- Expects the same parameter updates
- AllReduce algorithm
	- Multiple nodes collaborate to efficiently perform a reduce operation, while ensuring all nodes obtain the same final result

<img src="/images/Pasted image 20260304180906.png" alt="image" width="500">


### Data Parallelism with Centralized Parameters

- Store the model parameters outside the GPU device in workers on the CPU
- Place all the parameters on one or more CPU-only servers called parameter services, who host and update the parameters
	- Allows synchronous or asynchronous updates

<img src="/images/Pasted image 20260304181007.png" alt="image" width="500">


**Synchronous Updates**
- The aggregator waits until all gradients are available before it computes the average gradients and passes then to the optimizer, which updates the model parameters
- Some devices are slower, so fast devices need to wait
- Parameters are copied to every device (uses bandwidth)
- Spare replicas
	- Ignore gradients from the slowest few replicas


**Asynchronous Updates**

- When a replica has finished computing the gradients, the gradients are immediately used to update the model parameters
- No aggregation
- Work independently


- Data parallelism with asynchronous updates is attractive because of its simplicity, no delay, ad better use of bandwidth
- Stale gradients
	- Slow down convergence, introducing noise, and wobble effects


<img src="/images/Pasted image 20260304181419.png" alt="image" width="500">

- Reduce state gradients
	- Reduce the learning rate
	- Drop state gradients of scale them down
	- Adjust the mini-batch size
	- Start the few epochs using one replica (warmup phase)


### Bandwidth Saturation

- Saturation is more severe for large dense models
- Pipeline parallelism
	- Combines model parallelism and data parallelism
		- Model is chopped into consecutive parts (stages)
		- Each of which is trained on a different machine
		- Asynchronous pipeline in which all machines work in parallel with little idle time
		- Each stage alternates one round of forward and back propagation
	- Pulls a mini-batch from its input queue, processes it, and sends the output to the next stage's input queue, then pulls one mini-batch of gradients from its gradient queue, back propagates these gradients and updates its model parameters

<img src="/images/Pasted image 20260304181843.png" alt="image" width="500">

- Mini batch #5
	- When through stage 1 during forward pass, gradients from #4 have not been back propagated, but in #5 gradients flow back to stage 1
- Weight stashing
	- Each stage saves weights during forward propagation and restores then during backpropagation
- Pathways
	- Uses automated model parallelism
	- Asynchronous gang scheduling
- Use a few powerful GPUs rather than many weak GPUs
- Group GPUs on a well interconnected server
- Drop float precision from 32 to 16 bit
- When using centralized parameters, shard (split) the parameters across multiple parameter servers


## Training at Scale Using the Distribution Strategies API

- Distribution strategies API
- Create a `MirroredStrategy` object
	- Uses NVIDIA Collective Communications Library (NCCL)
- Wrap the creation and complication of the model inside that context

```python
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential([...])  # create a Keras model normally
    model.compile([...])  # compile the model normally

batch_size = 100  # preferably divisible by the number of replicas
model.fit(X_train, y_train, epochs=10,
          validation_data=(X_valid, y_valid), batch_size=batch_size)
          
>>> type(model.weights[0])
tensorflow.python.distribute.values.MirroredVariable

with strategy.scope():
    model = tf.keras.models.load_model("my_mirrored_model")
    
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

strategy = tf.distribute.experimental.CentralStorageStrategy()
```

- `fit()` will automatically split each training batch across all the replicas
- Make predictions efficiently

## Training a Model on a TensorFlow Cluster

- A TensorFlow cluster is a group of TensorFlow processes running in parallel, on different machines
	- Each task or TF server has an IP address, port, and a type (role/job)
		- Worker, chief, ps, evaluator

- Each worker performs computations
- The chief performs computations, and handles extra work
	- Writing logs or saving checkpoints
- A parameter server keeps track of variable values
	- CPU
- An evaluator takes care of evaluation


- Cluster specification
	- Define each task's IP address, TCP port, and type

```python
cluster_spec = {
    "worker": [
        "machine-a.example.com:2222",     # /job:worker/task:0
        "machine-b.example.com:2222"      # /job:worker/task:1
    ],
    "ps": ["machine-a.example.com:2221"]  # /job:ps/task:0
}
```

- In general, there is a single task per machine
- Every task in the cluster may communicate with every other task
	- Configure firewall to authorize all communications between these machines on these ports


<img src="/images/Pasted image 20260304182827.png" alt="image" width="500">

- When a task is started, give it a cluster spec, type, and index
	- `TF_CONFIG` environment variable
	- JSON dictionary
	- Define environment variable output of Python

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec,
    "task": {"type": "worker", "index": 0}
})
```

- Train a model on a cluster

```python
import tempfile
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy()  # at the start!
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
print(f"Starting task {resolver.task_type} #{resolver.task_id}")
[...] # load and split the MNIST dataset

with strategy.scope():
    model = tf.keras.Sequential([...])  # build the Keras model
    model.compile([...])  # compile the model

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10)

if resolver.task_id == 0:  # the chief saves the model to the right location
    model.save("my_mnist_multiworker_model", save_format="tf")
else:
    tmpdir = tempfile.mkdtemp()  # other workers save to a temporary directory
    model.save(tmpdir, save_format="tf")
    tf.io.gfile.rmtree(tmpdir)  # and we can delete this directory at the end!
```
- Start the script on the first workers
- 2 All Reduce implementations
	- A ring AllReduce algorithm based on gRPC for network communications
	- NCLL's implementation

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CollectiveCommunication.NCCL))
        
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
```
## Running Larger Training Jobs on Vertex AI

- Vertex AI allows you to create custom training jobs with your own training code
- Change where the chief should save the model, checkpoints, and logs
- Save to GCS, instead of local directory
	- `AIP_MODEL_DIR`

```python
import os
[...]  # other imports, create MultiWorkerMirroredStrategy, and resolver

if resolver.task_type == "chief":
    model_dir = os.getenv("AIP_MODEL_DIR")  # paths provided by Vertex AI
    tensorboard_log_dir = os.getenv("AIP_TENSORBOARD_LOG_DIR")
    checkpoint_dir = os.getenv("AIP_CHECKPOINT_DIR")
else:
    tmp_dir = Path(tempfile.mkdtemp())  # other workers use temporary dirs
    model_dir = tmp_dir / "model"
    tensorboard_log_dir = tmp_dir / "logs"
    checkpoint_dir = tmp_dir / "ckpt"

callbacks = [tf.keras.callbacks.TensorBoard(tensorboard_log_dir),
             tf.keras.callbacks.ModelCheckpoint(checkpoint_dir)]
[...]  # build and  compile using the strategy scope, just like earlier
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10,
          callbacks=callbacks)
model.save(model_dir, save_format="tf")
```

- Create a custom training job on Vertex AI, based on this script
- Specify job name, path, Docker image, predictions, additional Python libraries, and bucket used for staging

```python
custom_training_job = aiplatform.CustomTrainingJob(
    display_name="my_custom_training_job",
    script_path="my_vertex_ai_training_task.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    model_serving_container_image_uri=server_image,
    requirements=["gcsfs==2022.3.0"],  # not needed, this is just an example
    staging_bucket=f"gs://{bucket_name}/staging"
)

# 2 workers
mnist_model2 = custom_training_job.run(
    machine_type="n1-standard-4",
    replica_count=2,
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,
)
```

- Vertex AI will return the training script and the trained model
- Deploy is to an endpoint, or use it to make batch predictions


## Hyperparameter Tuning on Vertex AI

- Vertex AI's hyperparameter tuning service is based on a Bayesian optimization algorithm
	- Find optimal combinations of hyperparameters

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_hidden", type=int, default=2)
parser.add_argument("--n_neurons", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1e-2)
parser.add_argument("--optimizer", default="adam")
args = parser.parse_args()
```
- Hyperparameters call the script multiple times, with different hyperparameter values
	- Each run is called a trial, and the set of trials is called a study
- Training script uses the HP to build and compile a model
- Use mirrored distribution, if needed


```python
import tensorflow as tf

def build_model(args):
    with tf.distribute.MirroredStrategy().scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8))
        for _ in range(args.n_hidden):
            model.add(tf.keras.layers.Dense(args.n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        opt = tf.keras.optimizers.get(args.optimizer)
        opt.learning_rate = args.learning_rate
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        return model

[...]  # load the dataset
model = build_model(args)
history = model.fit([...])
```

- Script reports the model's performance back to HP

```python
import hypertune

hypertune = hypertune.HyperTune()
hypertune.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="accuracy",  # name of the reported metric
    metric_value=max(history.history["val_accuracy"]),  # metric value
    global_step=model.optimizer.iterations.numpy(),
)

# define a custom job
trial_job = aiplatform.CustomJob.from_local_script(
    display_name="my_search_trial_job",
    script_path="my_vertex_ai_trial.py",  # path to your training script
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    staging_bucket=f"gs://{bucket_name}/staging",
    accelerator_type="NVIDIA_TESLA_K80",
    accelerator_count=2,  # in this example, each trial will have 2 GPUs
)

# create and run HP tuning job
from google.cloud.aiplatform import hyperparameter_tuning as hpt

hp_job = aiplatform.HyperparameterTuningJob(
    display_name="my_hp_search_job",
    custom_job=trial_job,
    metric_spec={"accuracy": "maximize"},
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-3, max=10, scale="log"),
        "n_neurons": hpt.IntegerParameterSpec(min=1, max=300, scale="linear"),
        "n_hidden": hpt.IntegerParameterSpec(min=1, max=10, scale="linear"),
        "optimizer": hpt.CategoricalParameterSpec(["sgd", "adam"]),
    },
    max_trial_count=100,
    parallel_trial_count=20,
)
hp_job.run()
```

- We tall Vertex AI to max. the metric "accuracy"
- Define the search space, using a log scale for the learning rate and a linear scale for other hyperparameters
- Max trials to 100
- Max trials running in parallel is 20

- Fetch the trail results
- Each trial result is represented as a protobuf objects

```python
def get_final_metric(trial, metric_id):
    for metric in trial.final_measurement.metrics:
        if metric.metric_id == metric_id:
            return metric.value

trials = hp_job.trials
trial_accuracies = [get_final_metric(trial, "accuracy") for trial in trials]
best_trial = trials[np.argmax(trial_accuracies)]

>>> max(trial_accuracies)
0.977400004863739
>>> best_trial.id
'98'
>>> best_trial.parameters
[parameter_id: "learning_rate" value { number_value: 0.001 },
 parameter_id: "n_hidden" value { number_value: 8.0 },
 parameter_id: "n_neurons" value { number_value: 216.0 },
 parameter_id: "optimizer" value { string_value: "adam" }
]
```

- Get this trial's SavedModel, train it more, deploy it to production
- Create an AutoML training job, pointing to the dataset and specifying max number of computes hours
- Keras Tuner
	- `KERASTUNER_TUNER_ID` = chief
	- `KERASTUNER_ORACLE_IP`
	- `KERASTUNER_ORACLE_PORT`