
# Reducing Communication Rounds in Federated Learning

## Cloud Computing EN 601.619 - Instructor Prof. Soudeh Ghorbani

### Purpose
One of the main challenges in Federated Learning is communications as often the client devices have unreliable connection to the cloud. Our project aims to reduce the number of communication rounds while not compromising on the accuracy. Our current code snippet tries to replicate the ideas from *CMFL: Mitigating Communication Overhead for Federated Learning* and shows that their idea of incorporating a update relevance check at the client can help reduce communication rounds. Our next steps in this project would be use this and incorporate our ideas to reduce communication rounds by smart client selection at the cloud. The code is designed to closely replicate the Algorithm 1 in the paper. Some hyperparameters were not mentioned in the paper and we tried out some values for those. It also implements our proposed CMFL++ approach to reduce communication rounds.


### Reproducing Results:
There are two ways to reproduce our results.

1. Google Colab (Preferred)
2. Running on your system

### 1) Using Google Colab
Run the cells on the following two python notebooks to reproduce the results. The notebooks display will display the number of communication rounds required to reach 60% at the end along with a plot for Test accuracy vs Cumulative communication rounds. Make sure to select GPU as hardware accelerator in Runtime --> Change Runtime type
- CMFL++ --> https://colab.research.google.com/drive/18j-IIpIQK9ABOZUhKsZ7wu2zlSl7emcT?usp=sharing
- CMFL   --> https://colab.research.google.com/drive/1jOMNRNESXrADbygqWvW518OJJMSCffZp?usp=sharing
- Vanilla --> https://colab.research.google.com/drive/16g6o0Mh2J4ILu19oQQc-lB6dEK_XarMX?usp=sharing

### 2) Running on your system
#### Files:
- `cmfl_main_type6.py` : Main training and test script for CMFL++
- `cmfl_main_type3.py` : Main training and test script for CMFL
- `dataset.py` : This is a modified version of the original PyTorch MNIST loader to divide the dataset among clients

#### Virtual Environment and Installation:
- You can create a virtualenvironment from scratch by running the following commands:
	+ `virtualenv -p python3 venv`
	+ `source venv/bin/activate`
	+ `pip install torch torchvision tqdm matplotlib wandb seaborn`

#### Reproducing results:
- The main code is cmfl_main_type6.py. It supports various commandline arguments. You can run `python3 cmfl_main_type6.py --help` to get a description of the commands supported
- To run CMFL++, execute the following code : `cmfl_main_type6.py --batch-size=64 --cli-ite-num=3 --force_client_train=0 --lr=0.01 --start_threshold=0.5 --topk=0.3`
- To run CMFL, execute the following code : `cmfl_main_type3.py --batch-size=64 --cli-ite-num=3 --lr=0.01 --start_threshold=0.6 --formarkce_client_train 20 --topk 0.00`
- To run Vanilla, execute the following code : `cmfl_main_type3.py --batch-size=64 --cli-ite-num=3 --lr=0.01 --start_threshold=0.0 --formarkce_client_train 20 --topk 0.00`
- If you do not have GPUs on your system, it would take ~2.5 hours for the code to execute. On a GPU it should take ~15-20 minutes.
- The code will display the number of communication rounds required to reach 60% at the end. 


### Acknowledgements
- Prof. Ghorbani for discussions on the idea
- Sougol for help with GCP setup and GPU questions
- There is an unofficial code for CMFL available online *HyperionZhou/CMFL* but we found that it did not correctly replicate the paper and had some big issues with its implementation. Hence, we implemented our current version from scratch 


### Issues ?
Contact us at {ashah95,pdhar1,aroy28}@jhu.edu














