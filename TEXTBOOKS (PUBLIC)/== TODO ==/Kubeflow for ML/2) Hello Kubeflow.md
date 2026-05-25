
# Getting Set up with Kubeflow

- Work on cloud providers or on-premises Kubernetes clusters
- Google Cloud Platform (GCP)
	- Used for deploying models

## Installing Kubeflow and Its Dependencies

- Install Kubernetes with `kubectl`

```python
brew install kubernetets-cli
brew install kubectl kind kustomize

# create k cluster
kind create cluster --name kubeflow
kubectl get nodes
# install Kubeflow
git clone https://github.com/kubeflow/manifests.git
cd manifests
# apply kubeflow
while ! kustomize build example | kubectl apply -f -; do echo "Retrying..."; sleep 10; done
# Run pod
kubectl get pods -A
# Access KF UI
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

## Setting Up Local Kubernetes


- Minikube
	- Local version of Kubernetes
	- Local computer can simulate a cluster
- `micros8s`
	- Linux platforms
- `MiniKF`
	- Vagrant to launch a VM to run Kubernetes with Kubeflow

## Setting up Your Kubeflow Development Environment

**Setting up the Pipeline SDK**
- Install python and create a venv

```python
conda create --name kfenv
conda activate kfenv

# install KF pipeline SDK
URL=https://storage.googleapis.com/ml-pipeline/release/latest/kfp.tar.gz
pip install "${URL}" --upgrade

# clone the KF pipelines repo
  git clone --single-branch --branch 0.3.0 https://github.com/kubeflow/pipelines.git
```

**Setting Up Docker**

- Install Docker
- Create a container registry that will be accessed by the Kubeflow cluster
- Docker Hub and RedHad offers Quay
	- A cloud neutral platform
	- Use your own cloud provider's container registry
- When a push a container, specify the `tag` which determines the image name, version, and where it is stored

```python
# specify the new container is build on top of KF container
FROM gcr.io/kubeflow-images-public/tensorflow-2.1.0-notebook-cpu:1.0.0
# build the new contaienr and push to registry
IMAGE="${CONTAINER_REGISTRY}/kubeflow/test:v1"
docker build  -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"
```

**Editing YAML**
- Most of Kubernetes configuration is represented in YAML

## Creating Our First Kubeflow Project

- Specify a manifest file that configures what is build
- Build an end-to-end pipeline for MNIST


# Training and Deploying a Model

## Training and Monitoring Progress

- Use a pre-created training container that downloads the training data and trains the model
	- `train_pipeline.py`
```python
dsl-compile --py train_pipeline.py --output job.yaml

# Kubeflow UI
kubectl get ingress -n istio-system
```
- Upload the pipeline to create a run of the pipeline

## Test Query

- Query the model and monitor the results
- Model returns a JSON of the 10 digits and the probability of whether the submitted vector represents a specific digit

```python
# model query
import requests
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

def download_mnist():
    return input_data.read_data_sets("MNIST_data/", one_hot=True)


def gen_image(arr):
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, cmap=plt.cm.gray_r, interpolation='nearest')
    return plt
mnist = download_mnist()
batch_xs, batch_ys = mnist.train.next_batch(1)
chosen = 0
gen_image(batch_xs[chosen]).show()
data = batch_xs[chosen].reshape((1, 784))
features = ["X" + str(i + 1) for i in range(0, 784)]
request = {"data": {"names": features, "ndarray": data.tolist()}}
deploymentName = "mnist-classifier"
uri = "http://" + AMBASSADOR_API_IP + "/seldon/" + \
    deploymentName + "/api/v0.1/predictions"

response = requests.post(uri, json=request)
```

# Going Beyond a Local Deployment

- Kubernetes can run on a single machine or many computers

# Conclusion