
![[Pasted image 20260317154001.png]]


# Getting Around the Central Dashboard

- The main interface is the central dashboard
	- Access the majority of components

## Notebooks (JupyterHub)

- JupyterHub
	- Prototyping
	- Experimentation
	- A multi-user hub that spawns, manages, and proxies multiple instances of a single-user Jupyter notebook
- Connect to existing servers or create a new one
- Specify the server name, namespace, image, and specify resource requirements
	- CPU/memory
	- Workspace
	- Data volumes
	- Custom configurations
- Bind the service account `default-editor` to limit/extend permissions of the notebook server
- To execute all of the commands without leaving the notebook environment

```python
# create a new juypter notebook
!kubectl create -f myspec.yaml
```

## Training Operations

- Kubeflow provides several training components to automate the execution of ml
	- Chainer training
	- MPI training
	- Apache MXNet training
	- PyTorch training
	- TensorFlow training

- Distributed training jobs are managed by application-specific controllers, known as operators
- Extend the Kubernetes API to create, manage, and manipulate the state of resources
- Run a distributed TF training job
	- User just needs to provide a specification that describes the desired state (workers and parameters servers)

## Kubeflow Pipelines

- Pipelines
	- Allow you to orchestrate the execution of ML application
	- Based on Argo Workflows
	- Container-native workflow engine for Kubernetes
- Python SDK
- DSL compiler
- Pipeline Service
- Kubernetes resources
- Orchestration controllers
- Artifact storage
	- Metadata
	- Artifacts

- Artifact store
	- MiniIO server
	- Google Cloud Storage (GCS)
	- Amazon S3

- Kubeflow can track data and metadata
- Pipelines can expose the parameters of the underlying ML

## Hyperparameter Tuning

- Traditional methods for tuning
	- Grid search
- Katlib
	- Allows users to perform hyperparameter optimizations on Kubernetes clusters
	- Inspired by Google vizier
		- Black box optimization framework
		- Leverages advanced searching algorithms to find the optimal hyperparameter configuration
			- Bayesian optimization
- 4 main concepts
	- Experiment
		- Runs over a feasible space
		- Contains config describing the space, a set of trials
	- Trial
		- A list of parameter values, x, that will lead to a single evaluation of f(x)
	- Job
		- A process responsible for evaluating a pending trial and calculating its objective value
	- Suggestion
		- An algorithm to construct a parameter set
			- Random
			- Grid
			- Hyperband
			- Bayesian optimization

## Model Inference

- Deploy ML models in production environments at scale
	- TFServing
	- Seldon serving
	- PyTorch serving
	- TensorRT
- KFServing
	- Generalizes the model inference concerns of autoscaling, networking, health checking, and server configuration
- Implementation is based on leveraging Istio and Knative serving
	- Knative serving
		- Serverless containers on Kubernetes
		- Provides middleware primitives
			- Rapid deployment of serverless containers
			- Automatic scaling up and down to zero
			- Routing and network programming for Istio components
- KFServing
	- Preprocessor
	- Predictor
	- Postprocessor

## Metadata

- Metadata management
	- Provides capabilities to capture and track information about a model's creation
	- Difficult to manage model's related information
	- ML Metadata is both the infrastructure and a library for recording and retrieving metadata associated with an ML project
		- Data source used for the model's creation
		- The artifacts generated through the components/steps of the pipeline
		- The executions of these components
		- Pipeline and associated lineage information


- Examples of ML Metadata operations
	- List all artifacts of a specific type
	- Compare two artifacts of the same time
	- Show a DAG of all related executions and their input and output artifacts
	- Display how an artifact was created
	- Identify all artifacts that were created using a given artifact
	- Determine if an execution has been ru non the same inputs before
	- Record the query context of workflow runs


![[Pasted image 20260317160304.png]]
## Component Summary

- Add custom components

# Support Components

## MinIO

- The foundation of the pipeline architecture is shared storage
- Keep data in external storage
- Different cloud providers
	- Amazon S3
	- Azure Data Storage
	- Google Cloud Storage
- Kubeflow ships with MinIO
	- A high performance distributed object storage server
	- Large-scale private cloud infrastructure
	- Consistent gateway to public APIs
- Deployed in different configuration
	- Default is single container mode
- Distributed MinIO
	- Pool multiple failures and ensure full data protection
- Provides S3 API on top of Azure Blob storage, GCS, Gluster, or NAS storage
- Kubeflow installation hardcodes MinIO credential, which you used in the application
	- Better to use a secrete, if switching to S3


```python
# port-forwarding to minio user
kubectl port-forward -n kubeflow svc/minio-service 9000:9000 &

# install MinIO on Mac
brew install minio/stable/minio

# config MinIO Client to take to Kubeflow's MinIO
mc config host add minio http://localhost:9000 minio minio123

# create a bucket
mc mb minio/kf-book-examples
```

## Istio

- A service mesh providing vital features
	- Service discovery
	- Load balancing
	- Failure recovery
	- Metrics
	- Monitoring
	- Rate limiting
	- Access control
	- End-to-end authentication
- Integrates into any logging platform, or telemetry or policy systems
- Secure, connect, and monitor microservices
- Co-locates each service instance with a sidecar network proxy
- All network traffic from an individual service instance flows via its local sidecar proxy to the appropriate destination
- Data plane
	- Proxies
	- Mediate and control all network communication between pods
- Control plane
	- Manages and configures the proxies to route traffic

- Envoy
	- Failure handling
	- Dynamic service discovery
	- Load balancing
- Mixer
	- Enforces access control
- Pilot
- Galley
	- Validation, ingestion, processing, and distribution component
- Citadel
	- Enables strong service to service and end-user authentication by providing identity and credential management
- Kubeflow uses Istio to provide a proxy to the Kubeflow UI and to route requests securely

![[Pasted image 20260317165856.png]]

## Knative

- Built on Kubernetes and Istio
- Knative Serving
	- Support the deployment and serving of serverless applications
	- Middleware primitives
		- Rapid deployment of serverless containers
		- Automatic scaling up and down to zero
		- Routing and network programming for Istio components
		- Point-in-time snapshots of deployed code and configurations
- Control behaviour objects
	- Service
	- Route
	- Configuration
	- Revision

![[Pasted image 20260318083138.png]]



## Apache Spark

- Kubeflow has a built-in Spark operator for running Spark jobs
- Integration for using Google's Dataproc and Amazon's Elastic Map Reduce (EMR)
- Handle larger datasets and scale problems that cannot fit on a single machine

## Kubeflow Multiuser Isolation

- Allows sharing the same pool of resources across different teams and users
	- Administrator
	- User
	- Profile

# Conclusion

- 