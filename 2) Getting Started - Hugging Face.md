
- Python is the main programming language
- IDE
	- Jupyter Notebook

## Downloading Anaconda

- Anaconda
	- Distribution of Python and R programming languages
- `conda`
	- Package manager

### Virtual Environments

- Virtual envronment
	- A self-contained environment that allows you to install and manage Python packages separately from systemwide Python

## Installing the Transformers Library

- Transformers library
	- Open source library developed by Hugging Face
	- Interface for working with pretrained models

`!pip install transformers`

### Support for GPU

- Library is primarily build in PyTorch
	- Facebook
- Also supports TensorFlow
	- Google
- PyTorch has GPU support
	- Integration with Nvidia's Compute Unified Device Architecture (CUDA)
- Speed training with PyTorch wheels compatible with CUDA

```python
pip install torch torchvision torchaudio
--index-url https://download.pytorch.org/whl/cu121 -U

# test packages
import torch
print(torch.cuda.is_available())

# find GPU details
import torch
use_cuda = torch.cuda.is_available()

if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:', torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',
          torch.cuda.get_device_properties(0).total_memory/1e9)
```

- `GPUtil`
	- Find details about GPU
		- Total GPU
		- Utilization load
		- Temperature
		- Memory


```python
!pip install GPUtil

import GPUtil

gpus = GPUtil.getGPUs()

for gpu in gpus:
    print("GPU ID:", gpu.id)
    print("GPU Name:", gpu.name)
    print("GPU Utilization:", gpu.load * 100, "%")
    print("GPU Memory Utilization:", gpu.memoryUtil * 100, "%")
    print("GPU Temperature:", gpu.temperature, "C")
    print("GPU Total Memory:", gpu.memoryTotal, "MB")
```

### Using GPU in the pipeline object

- `pipeline`
	- Specify the `device` parameter

```python
from transformer import pipeline
question_clasisfier = pipeline("text-classification",
                               model="huaen/question_detection",
                               device = 0)
```

**Transformers Pipeline**

- Pipeline
	- Simplifies the process of building and using complex NLP workflows

```python
question_classifier = pipeline("text-classification",
                               model="huaen/question_detection",
                               device = "cuda:0")
```

- Mac
	- Accelerate pipelines with Apple's Metal Performance Shaders (MPS)

```python
question_classifier = pipeline("text-classification",
                               model="huaen/question_detection",
                               device = "mps:0")  
```

**Device Parameters**

![[Pasted image 20260514205616.png]]

**Autodetecting CUDA, MPS, or CPU for PyTorch Inference**

```python
from transformers import pipeline
import torch

if torch.cuda.is_available():
	device = "cuda"
elif torch.backends.mps.is_available();
	device = "mps"
else:
	device = "cpu"
	
question_classifier = pipeline("text-classification",
                               model="huaen/question_detection",
                               device=device)
print(f"Using device: {device}")
```


## Installing the Hugging Face Hub Package

- `huggingface_hub` CLI package
	- Managing project repositories
	- Uploading and downloading files
	- Fetching models

```python
!pip install huggingface_hub
```

### Downloading Files

- 

### 
### 