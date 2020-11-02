import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import copy
import numpy as np
from tqdm import tqdm
import dataset
import wandb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)

def combine_updates(local_updates,relevances):
    where_relevant = np.where(relevances == 1)[0]
    num_relevant = where_relevant.shape[0]

    effective_gradients = {}
    for rel_ind in where_relevant:
        for k in local_updates[rel_ind].keys():
            if k in effective_gradients:
                effective_gradients[k] = effective_gradients[k] + local_updates[rel_ind][k]
            else:
                effective_gradients[k] = local_updates[rel_ind][k]

    for k in effective_gradients.keys():
        effective_gradients[k]/=num_relevant

    return effective_gradients


def apply_update(model,update):
    current_model_dict = model.state_dict()
    updated_model_dict = copy.deepcopy(model.state_dict())

    for k,v in current_model_dict.items():
        updated_model_dict[k] = v + update[k]

    model.load_state_dict(updated_model_dict)

def compute_local_update(model,local_model):
    update = {}
    model_dict = model.state_dict()
    local_model_dict = local_model.state_dict()
    for k in model_dict.keys():
        update[k] = local_model_dict[k] - model_dict[k]
    return update

def check_relevance(local_model_update,last_updates,threshold):
    sign_sum = 0
    sign_size = 0
    rel_threshold = threshold
    model_para_list = {}
    signs = []
        
    if last_updates is None:
        return True,1.0

    for k in local_model_update.keys():
        sign_k = torch.sign(local_model_update[k])
        sign_k_last = torch.sign(last_updates[k])
        effective_sign = sign_k*sign_k_last
        effective_sign[effective_sign != 1] = 0.0 
        signs.extend(effective_sign.view(-1).cpu().numpy())

    signs = np.asarray(signs)

    avg_sign = np.sum(signs)/(signs.shape[0] +1E-6)

    return avg_sign>threshold,avg_sign

def client_train(args,ck,model,last_update,data_loader,device,threshold):
    local_model = copy.deepcopy(model)
    optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
    local_model.train()
    crtiterion = torch.nn.CrossEntropyLoss()
    for ep in range(args.cli_ite_num):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            data, target = data.type('torch.FloatTensor').to(device), target.to(device, dtype=torch.int64)
            output = local_model(data)
            loss = crtiterion(output, target)
            loss.backward()
            optimizer.step()

    local_model_update = compute_local_update(model,local_model)
    rel,avg_sign = check_relevance(local_model_update,last_update,threshold)
    # print(avg_sign)
    return rel,local_model_update,avg_sign


def global_train(args, model, device, train_loaders, it, last_update):
    local_updates = []
    relevances = []
    average_signs = []
    threshold = args.start_threshold/np.sqrt(it)
    print('Threshold : {}'.format(threshold))
    for ck in range(len(train_loaders)):
        rel,loc_up,avg_sign = client_train(args,ck,model,last_update,train_loaders[ck],device,threshold)
        local_updates.append(loc_up)
        relevances.append(rel)
        average_signs.append(avg_sign)

    print('Average sign: {}'.format(np.mean(np.asarray(average_signs))))

    relevances = np.asarray(relevances)
    effective_update = combine_updates(local_updates,relevances)
    c_rounds = np.where(relevances == 1)[0].shape[0]

    if len(list(effective_update.keys())) == 0:
        print('No relevant model found!')
        return -1,-1
    else:
        apply_update(model,effective_update)    
    return c_rounds,effective_update, np.mean(np.asarray(average_signs)),threshold

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    crtiterion = torch.nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += crtiterion(output, target) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, 100. * correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max_iterations', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--client-num', type=int, default=100)
    parser.add_argument('--cli-ite-num', type=int, default=4)
    parser.add_argument('--start_threshold', type=float, default=0.8)
    args = parser.parse_args()
    wandb.init(project="cloud-federated",entity="cloud")
    wandb.config.update(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # wandb.init(project="cloud")
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    train_loaders = []

    for i in range(args.client_num):
        train_loaders.append(torch.utils.data.DataLoader(
                            dataset.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]),client_id=i,num_clients=args.client_num),
                            batch_size=args.batch_size, shuffle=True, **kwargs))

    
    test_loader = torch.utils.data.DataLoader(
                    dataset.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]),client_id=-1,num_clients=1),
                    batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = Net().to(device)
    # model = Net2().to(device)

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    communication_rounds = []
    last_update = None
    for it in tqdm(range(1, args.max_iterations)):
        c_round,last_update,avg_sign,thresh = global_train(args, model, device, train_loaders, it,last_update)
        if c_round == -1:
            print('exiting!')
            return
        communication_rounds.append(c_round)
        test_acc,test_loss = test(args, model, device, test_loader)
        print('Cululative Communication Rounds : {}'.format(sum(communication_rounds)))
        wand_dict = {
            'Communication Rounds':sum(communication_rounds),
            'Test loss': test_loss,
            'Test Acc': test_acc,
            'Average sign':avg_sign,
            'Threshold' :thresh
        }   
        wandb.log(wand_dict)


    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")


if __name__ == '__main__':
    main()