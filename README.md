
# Reducing Communication Rounds in Federated Learning

## Cloud Computing EN 601.619 - Instructor Prof. Soudeh Ghorbani

### Purpose
One of the main challenges in Federated Learning is communications as often the client devices have unreliable connection to the cloud. Our project aims to reduce the number of communication rounds while not compromising on the accuracy. Our current code snippet tries to replicate the ideas from *CMFL: Mitigating Communication Overhead for Federated Learning* and shows that their idea of incorporating a update relevance check at the client can help reduce communication rounds. Our next steps in this project would be use this and incorporate our ideas to reduce communication rounds by smart client selection at the cloud. The code is designed to closely replicate the Algorithm 1 in the paper. Some hyperparameters were not mentioned in the paper and we tried out some values for those. 


### Files:
- `cmfl_main.py` : Main training and test script
- `dataset.py` : This is a modified version of the original PyTorch MNIST loader to divide the dataset among clients

### Virtual Environment and Installation:
- To use the environment that we have already created run  `source /home/ank_roy4/cloud/bin/activate`
- You can create a virtualenvironment from scratch by running the following commands:
	+ `virtualenv -p python3 venv`
	+ `source venv/bin/activate`
	+ `pip install torch torchvision tqdm matplotlib tqdm`

### Reproducing results:
- The main code is cmfl_main.py. It supports various commandline arguments. You can run `python cmfl_main.py --help` to get a description of the commands supported
- We have already set the default parameters to accurately reproduce the result. To run the training  `python cmfl_main.py`
- The GCP instance does not use a GPU and hence the training takes about 2.5 hours to complete
- You can alternatively evaluate our models by runnnig


### Acknowledgements
- Prof. Ghorbani for discussions on the idea
- Sougol for help with GCP setup and GPU questions
- There is an unofficial code for CMFL available online *HyperionZhou/CMFL* but we found that it did not correctly replicate the paper and had some big issues with its implementation. Hence, we implemented our current version from scratch 















