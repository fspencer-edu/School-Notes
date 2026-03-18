
# Getting Started with Pipelines

- Kubeflow pipelines
	- UI for managing and tracking pipelines and their execution
	- An engine for scheduling a pipeline's execution
	- An SDK for defining, building, and deploying pipelines in Python
	- Notebook support for using the SDK and pipeline execution

## Exploring the Prepackaged Sample Pipelines


## Building a Simple Pipeline in Python

- Kubeflow pipelines are stored as YAML files executed by a program called Argo
- Kubeflow exposes a Python DSL (domain specific language) for authorizing pipelines
- A pipeline is a graph of container execution
- For each container
	- Create the container
		- Python or Docker container
	- Create an operation that references that container
	- Sequence the operations
	- Compile the pipelines, defined in Python, into a YAML file




## Storing Data Between Steps
# Introduction to Kubeflow Pipelines Components

## Argo: The Foundation of Pipelnies
## What Kubeflow Pipelines add to Argo Workflow
## Building a Pipeline Using Existing Images

## Kubeflow Pipeline Components

# Advanced Topics in Pipelines

## Conditional Execution of Pipelines
## Running Pipelines on Schedule

# Conclusion