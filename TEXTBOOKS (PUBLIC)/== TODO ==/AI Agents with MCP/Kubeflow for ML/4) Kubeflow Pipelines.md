
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


- Light weight Python functions
	- Define a Python function and then let Kubeflow take care of packaging that function into a container and creating an operation

```python
import kfp
def simple_echo(i: int) -> int:
	return i
```

- Wrap function into a Kubeflow Pipeline operation
- The factor function will construct a ContainerOp, which will run the original function in a container

```python
simpleStronglyTypedFunction =
	kfp.components.func_to_container_op(deadSimpleIntEchoFn)
	
foo = simpleStronglyTypedFunction(1)
type(foo)
Out[5]: kfp.dsl._container_op.ContainerOp
```

- Sequence the ContainerOp into a pipeline
- Compile pipeline into a zipped YAML file

```python
@kfp.dsl.pipeline(
	name='Simple Echo',
	description="Echoes numbers'
)

def echo_pipeline(param_1: kfp.dsl.PipelineParam):
	my_step = simpleStronglyTypedFunction(i = param_1)
	
kfp.compiler.Compiler().compile(echo_pipeline,
	'echo-pipeline.zip')
```

 - Create a new pipeline that installs and imports additional Python libraries
	 - Builds from a specified base image
	 - Passes output between containers
- Create a pipeline that divides a number by another number, and adds a third

```python
def add(a: float, b: float) -> float:
	'''Calc. sum of two args.'''
	return a + b
	
add_op = comp.func_to_container_op(add)
```

- Global imports from the notebook will no be packaged into the containers created

```python
from typing import NamedTuple

def my_divmod(dividend: float, divisor: float) -> \
	NamedTuple('MyDivmodOutput', [('quotient', float), ('remainder', float)]):
    '''Divides two numbers and calculate  the quotient and remainder'''
    # imports inside a component function
    import numpy as np
    
    # component function
    def divmod_helper(dividend, divisor):
	    return np.divmod(dividend, divisor)
	    
	    (quotient, remainder) = divmod_helper(dividend, dividor)
	    
	    from collections import namedtuple
	    divmod_output = namedtuple('MyDivmodOutput', ['quotient', 'remainder'])
	    return divmod_output(quotient, remainder)

divmod_op = comp.func_to_container_op(
                my_divmod, base_image='tensorflow/tensorflow:1.14.0-py3')
```

- The pipeline uses the functions defined previously, `my_divmod` and `add`

```python
@dsl.pipeline(
   name='Calculation pipeline',
   description='A toy pipeline that performs arithmetic calculations.'
)
def calc_pipeline(
   a='a',
   b='7',
   c='17',
):
    #Passing pipeline parameter and a constant value as operation arguments
    add_task = add_op(a, 4) #Returns a dsl.ContainerOp class instance.

    #Passing a task output reference as operation arguments
    #For an operation with a single return value, the output
    # reference can be accessed using `task.output`
    # or `task.outputs['output_name']` syntax
    divmod_task = divmod_op(add_task.output, b)

    #For an operation with multiple return values, the output references
    # can be accessed using `task.outputs['output_name']` syntax
    result_task = add_op(divmod_task.outputs['quotient'], c)
```

- Use the client to submit the pipeline for execution, returns the links to execution and experiment

```python
client = kfp.Client()
#Specify pipeline argument values
# arguments = {'a': '7', 'b': '8'} #whatever makes sense for new version
#Submit a pipeline run
client.create_run_from_pipeline_func(calc_pipeline, arguments=arguments)
```

- Create a function that returns a `kfp.dsl.ContainerOp`
- `@kfp.dsl.component` for static type checking


```python
@kfp.dsl.component
def my_component(my_param):
  ...
  return kfp.dsl.ContainerOp(
    name='My component name',
    image='gcr.io/path/to/container/image'
  )
  
@kfp.dsl.pipeline(
  name='My pipeline',
  description='My machine learning pipeline'
)
def my_pipeline(param_1: PipelineParam, param_2: PipelineParam):
  my_step = my_component(my_param='a')
```

## Storing Data Between Steps

- 2 methods of passing large data
	- Persistent volumes inside the Kubernetes cluster
	- Cloud storage (S3)

**Persistent Volumes**
- Abstract the storage layer
- Can be slow with provisioning and have IO limits
	- ReadWriteOnce
	- ReadOnlyMany
	- ReadWriteMany
- `VolumeOp`
	- Create an automatically managed persistent volume

```python
# mailing list data prep
dvop = dsl.VolumeOp(name="create_pvc",
                    resource_name="my-pvc-2",
                    size="5Gi",
                    modes=dsl.VOLUME_MODE_RWO)
```

- Using an object storage
- MinIO provides cloud native object storage by working as a gateway to an existing object storage engine or on its won
- `file_output`
	- Automatically transfers the specified local file into MinIO between pipeline step

```python
# File output
fetch = kfp.dsl.ContainerOp(name='download',
                                image='busybox',
                                command=['sh', '-c'],
                                arguments=[
                                    'sleep 1;'
                                    'mkdir -p /tmp/data;'
                                    'wget ' + data_url +
                                    ' -O /tmp/data/results.csv'
                                ],
                                file_outputs={'downloaded': '/tmp/data'})
```

# Introduction to Kubeflow Pipelines Components

## Argo: The Foundation of Pipelines

- Kubeflow installs all of the Argo components
- 

## What Kubeflow Pipelines add to Argo Workflow
## Building a Pipeline Using Existing Images

## Kubeflow Pipeline Components

# Advanced Topics in Pipelines

## Conditional Execution of Pipelines
## Running Pipelines on Schedule

# Conclusion