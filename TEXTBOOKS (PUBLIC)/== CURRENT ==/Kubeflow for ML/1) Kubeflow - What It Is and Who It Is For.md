
# Model Development Life Cycle

- ML development
	- Data
	- Information
	- Knowledge
	- Insight
- Model development life cycle (MDLC)

![[Pasted image 20260317144928.png]]

# Where Does Kubeflow Fit In?

- Kubeflow is a collection of cloud native tools for all of the stages of MDLC
	- Data exploration
	- Feature preparation
	- Model training/tuning
	- Model serving
	- Model testing
	- Model versioning

# Why Containerize?

- Isolation allows ML stages to be portable and reproducible

# Why Kubernetes?

- Kubernetes is an open source system for automating the deployment, scaling, and management of containerized applications
- Different stages can request different amounts or types of resources

# Kubeflow's Design and Core Components

- Composability
- Portability
- Scalability

## Data Exploration with Notebooks

- Exploration
	- Plotting
	- Segmenting
	- Manipulating data
- Jupyter
	- An open source web application to create and share data, code snippets, and experiments
## Data/Feature Preparation

- Extract, transform, and load data
- Filters, normalizes, and prepares input data
	- Apache Spark
	- TensorFlow Transform

## Training

- Training frameworks
	- TensorFlow
	- PyTorch
	- Apache MXNet
	- XGBoost
	- Chainer
	- Caffe2
	- Message passing interface (MPI)
## Hyperparameter Tuning

- Hyperparameters are variables that govern the training process
## Model Validation

- Perform cross validation of model validation
- A/B testing
- Multi-armed bandit

## Inference/Prediction

- KFServing
- TenserFlow Serving
- Seldon Core

- Once the model is served, it needs to be monitored for performance and possibly updated

## Pipelines

- Each node is a stage in a workflow
- Kubeflow Pipelines is a component that allows uses to compose reusable workflows
	- An orchestration engine for multistep workflows
	- An SDK to interact with pipelines components
	- A user interface that allows users to visualize and track experiments, and to share results with collaborators
## Component Overview

- Built-in components
	- Data preparation
	- Feature preparation
	- Model training
	- Data exploration
	- Hyperparameter tuning
	- Model inference
	- Pipelines

# Alternatives to Kubeflow

- Model development, and training with improvements in infrastructure, theory, and systems
- Prediction and model serving, have received relatively less attention
- Kubeflow can be used a architectural abstraction tools

## Clipper (RiseLabs)

- Clipper
	- A general purpose low latency prediction serving system developed by RiseLabs
	- Simplifies deployment, optimization, and inference
	- Layered architecture system
	- Low latency and high-throughput predictions
- 2 abstraction layers
	- Model selection
		- Uses adaptive online model selection policy and various ensemble techniques
		- Self-calibrates failed models without needing to interact with policy layer
	- Model abstraction
- Better caching and batching mechanisms

## MLflow (Databricks)

- Developed by Databricks as an open source ML development platform
- Framework-agnostic nature
	- Tracking
	- Projects
	- Models
- Functions as an API with a complementing UI for logging parameters, code versions, metrics, and output files
- Provide standard format for packaging reusable code, defined by a YAML file
- Anaconda dependency management

## Others

- Internal platforms
	- Bloomberg - Data Science Platform
	- Facebook - FBLearner Flow
	- Google - TensorFlow Extended
	- Uber - Michelangelo
	- IBM - Watson Studio

# Introducing Our Case Studies

## Modified Natural Institute of Standards and Technology

- MNIST
	- Dataset of handwriting digits for classification

## Mailing List Data

- Apache Software Foundation mailing list

## Product Recommender

- Recommendation systems
- Rating based models cannot be standardized for non-scaled target values

## CT Scans

- Methods and techniques to assist medical providers with understanding the disease