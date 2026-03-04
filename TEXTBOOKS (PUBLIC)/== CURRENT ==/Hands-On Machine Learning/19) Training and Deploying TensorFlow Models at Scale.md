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

![[Pasted image 20260304165237.png]]


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

![[Pasted image 20260304172312.png]]

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

![[Pasted image 20260304174043.png]]


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

![[Pasted image 20260304174343.png]]

- `nvidia-smi`
	- Checks that drivers and libraries are installed

![[Pasted image 20260304174435.png]]

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

![[Pasted image 20260304174634.png]]

- Tell TF to use a specific amount of GPU RAM
	- Logical GPU device (virtual GPU device) for each physical GPU device and set the memory to 2 GiB

```python
for gpu in physical_gpus:
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
    )
```
![[Pasted image 20260304174752.png]]

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
	- GPUs have limited communicati

## Parallel Execution Across Multiple Devices

# Training Models Across Multiple Devices

## Model Parallelism
## Data Parallelism
## Training at Scale Using the Distribution Strategies API
## Training a Model on a TensorFlow Cluster
## Running Larger Training Jobs on Vertex AI
## Hyperparameter Tuning on Vertex AI