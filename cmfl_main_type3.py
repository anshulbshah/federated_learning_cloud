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
import matplotlib.pyplot as plt
import random
import os

def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def draw_heatmap(inputdata):
    """Return a heatmap generated from te input data along witha colormap"""
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(inputdata)
    cbar = ax.figure.colorbar(im, ax=ax,fraction=0.02, pad=0.04)
    ax.set_xticks(np.arange(inputdata.shape[1]))
    fig.tight_layout()
    return fig


def combine_updates(local_updates,relevances,topk):
    """
    This function computes the effective update for each parameter by
    going over each parameter and averaging the update from all client models.
    This function also implements a methodology to combine select good clients on the cloud
    """
    where_relevant = np.where(relevances == 1)[0]
    num_relevant = where_relevant.shape[0]
    # use 10% of the total number of clients
    topk = int(num_relevant * topk)

    effective_gradients = {}
    for rel_ind in where_relevant:
        for k in local_updates[rel_ind].keys():
            if k in effective_gradients:
                effective_gradients[k] = effective_gradients[k] + local_updates[rel_ind][k]
            else:
                effective_gradients[k] = local_updates[rel_ind][k]

    for k in effective_gradients.keys():
        effective_gradients[k]/=num_relevant

    deviation_rel = []
    avg_dev_list = []
    # only 20 are rel

    for indx in range(len(relevances)):
        if indx in where_relevant:
            deviation = {}
            deviations_list = []
            for k in local_updates[rel_ind].keys():
                # if 'fc2' in k:
                deviation[k] = torch.sqrt((effective_gradients[k]  - local_updates[rel_ind][k])**2 / num_relevant)
                deviations_list.append(deviation[k].mean())
                # deviation[k] -> var size

            avg_of_deviations = sum(deviations_list)/len(deviations_list)
            # dev list - size 8
            avg_dev_list.append(avg_of_deviations.item())
        else:
            # if not relevant, we dont care.
            avg_dev_list.append(9999)
    sorted_ind = np.argsort(avg_dev_list)[:topk]
    # sorted_ind = np.argsort(avg_dev_list)[::-1][:topk]


    relevances[sorted_ind] = 0


    return effective_gradients, relevances  


def apply_update(model,update):
    """
    Applies the gradients to update the global model
    """
    current_model_dict = model.state_dict()
    updated_model_dict = copy.deepcopy(model.state_dict())

    for k,v in current_model_dict.items():
        updated_model_dict[k] = v + update[k]

    model.load_state_dict(updated_model_dict)

def compute_local_update(model,local_model):
    """
    Computes the local updates for each parameter of the local model wrt the last global model
    """
    update = {}
    model_dict = model.state_dict()
    local_model_dict = local_model.state_dict()
    for k in model_dict.keys():
        update[k] = local_model_dict[k] - model_dict[k]
    return update

def check_relevance(local_model_update,last_updates,threshold):
    """
    This function computes the relevance of the local update as defined in 
    CMFL. It effectively compares the sign of previous update and the current update and compare the average value of this
    sign across parameters to a threshold. An update is considered relevant if it exceeds a threshold
    """
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
    """
    This function makes a copy of the incoming global model and runs a training loop on the local dataset. It then returns 
    the effective local updates, relevances and the average sign 
    """
    local_model = copy.deepcopy(model)
    optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
    local_model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(args.cli_ite_num):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            data, target = data.type('torch.FloatTensor').to(device), target.to(device, dtype=torch.int64)
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    local_model_update = compute_local_update(model,local_model)
    rel,avg_sign = check_relevance(local_model_update,last_update,threshold)
    # print(avg_sign)
    return rel,local_model_update,avg_sign


def global_train(args, model, device, train_loaders, it, last_update, previous_relevances, previous_local_updates):
    """
    This function trains the global model by sending the current model to all local clients, getting their updates and 
    using the updates to get the new global model
    """
    local_updates = []
    relevances = []
    average_signs = []
    threshold = args.start_threshold/np.sqrt(it)
    print('Threshold : {}'.format(threshold))

    for ck in range(len(train_loaders)):
        if previous_relevances[ck]:
            rel,loc_up,avg_sign = client_train(args,ck,model,last_update,train_loaders[ck],device,threshold)
        else:            
            loc_up = previous_local_updates[ck]
            rel,avg_sign = check_relevance(loc_up,last_update,threshold)
        # rel,loc_up,avg_sign = client_train(args,ck,model,last_update,train_loaders[ck],device,threshold)
        local_updates.append(loc_up)
        relevances.append(rel)
        average_signs.append(avg_sign)

    print('Average sign: {}'.format(np.mean(np.asarray(average_signs))))

    relevances = np.asarray(relevances)

    c_rounds_cmfl = np.where(relevances == 1)[0].shape[0]

    effective_update,relevances = combine_updates(local_updates,relevances,args.topk)

    c_rounds = np.where(relevances == 1)[0].shape[0]

    print(c_rounds_cmfl,c_rounds)

    c_rounds += c_rounds_cmfl

    if len(list(effective_update.keys())) == 0:
        print('No relevant model found!')
        return 0,None,np.mean(np.asarray(average_signs)),threshold,relevances, local_updates
    else:
        apply_update(model,effective_update)    
    
    return [c_rounds_cmfl,c_rounds],effective_update, np.mean(np.asarray(average_signs)),threshold, relevances, local_updates

def test(args, model, device, test_loader):
    """
    A simple testing, that happens on the cloud(global)
    """
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

def plot_data(data_dict,save_location):
    num_its_log = len(data_dict['communication_rounds'])
    cum_comm_rounds = np.cumsum(data_dict['communication_rounds'])
    plt.figure('1')
    plt.plot(range(num_its_log),cum_comm_rounds)
    plt.title('Cumulative Communication Rounds')
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Communication Rounds')
    plt.savefig(save_location+'cumulative_comm_rounds.png')

    plt.figure('2')
    plt.plot(cum_comm_rounds,data_dict['test_loss'])
    plt.title('Test loss vs Cumulative communication rounds')
    plt.xlabel('Iterations')
    plt.ylabel('Test Loss')
    plt.savefig(save_location+'test_loss.png')

    plt.figure('3')
    plt.plot(cum_comm_rounds,data_dict['test_acc'])
    plt.title('Test Acc vs Cum communication rounds')
    plt.xlabel('Iterations')
    plt.ylabel('Test Acc')
    plt.savefig(save_location+'test_acc.png')

    plt.figure('4')
    plt.plot(range(num_its_log),data_dict['avg_sign'])
    plt.title('Average sign over iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Average Sign')
    plt.savefig(save_location+'avg_sign.png')

    plt.figure('5')
    plt.plot(range(num_its_log),data_dict['threshold'])
    plt.title('Threshold vs iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Threshold')
    plt.savefig(save_location+'threshold.png')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Reducing Communication Rounds in Federated Learning')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--max_iterations', type=int, default=20, metavar='N',
                        help='number of iterations for global training (default:20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--client-num', type=int, default=100,
                        help='Number of clients(locals) to use')
    parser.add_argument('--cli-ite-num', type=int, default=4,
                        help='Number of epochs to train the local clients for')
    parser.add_argument('--start_threshold', type=float, default=0.8,
                        help='Starting threshold for Check relevance function')
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Use wandb for logging')
    parser.add_argument('--anonymous_mode', action='store_true', default=False,
                        help='Use anonymous mode for visualizing results')
    parser.add_argument('--force_client_train', type=int, default=5,
                        help='After these many epochs, clients are selectively trained')
    parser.add_argument('--topk', type=float, default=0.1,
                        help='topk')

    args = parser.parse_args()

    #We use wandb for code logging. First run would require you to set it up on wandb.ai. 
    #To use wandb, set the flab --use_wandb. If not, matlplotlib plots are stored 

    if args.use_wandb:
        if args.anonymous_mode:
            wandb.login()
            wandb.init(project="cloud-federated",anonymous="must")
        else:
            wandb.init(project="cloud-federated",entity="cloud")
            
        wandb.config.update(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    set_random_seeds(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

    #All trainloders to be appended to this list
    train_loaders = []

    #We modify the original MNIST dataloader to work for our setting as different clients have different subsets of the data
    #Refer to dataset.py for changes
    for i in range(args.client_num):
        train_loaders.append(torch.utils.data.DataLoader(
                            dataset.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ]),client_id=i,num_clients=args.client_num),
                            batch_size=args.batch_size, shuffle=True, **kwargs))

    
    #Test uses the standard dataloader used for MNIST
    test_loader = torch.utils.data.DataLoader(
                    dataset.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]),client_id=-1,num_clients=1),
                    batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = Net().to(device)


    if not os.path.exists('generated_data'):
        os.mkdir('generated_data')
    if not os.path.exists('generated_data/thresh'+str(args.start_threshold)):
        os.mkdir('generated_data/thresh'+str(args.start_threshold))

    communication_rounds = []
    last_update = None # Last update is initialized to None --> implies that for the first run, all updates are relevant
    previous_relevances = [True]*args.client_num # Last update is initialized to None --> implies that for the first run, all updates are relevant
    prev_local_updates = None # Last update is initialized to None --> implies that for the first run, all updates are relevant

    all_relevances = [] 
    test_losses = []
    test_accuracies = []
    thresholds =[]
    average_signs =[]
    when_above_60 = -1
    for it in tqdm(range(1, args.max_iterations)):
        (c_round_cmfl,c_round_after_sel),last_update,avg_sign,thresh,rel_it, prev_local_updates = global_train(args, model, device, train_loaders, it,last_update, previous_relevances,prev_local_updates)
        if it>args.force_client_train:
            previous_relevances = rel_it
            print('Not training all! ')
            c_round = c_round_after_sel
        else:
            c_round = c_round_cmfl + args.client_num
        communication_rounds.append(c_round)
        test_loss,test_acc = test(args, model, device, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print('Cumulative Communication Rounds : {}'.format(sum(communication_rounds)))
        thresholds.append(thresh)
        average_signs.append(avg_sign)
        all_relevances.append(rel_it)
        if args.use_wandb:
            heatmap = draw_heatmap(np.stack(all_relevances,1).astype(np.float))
            wand_dict = {
                'Communication Rounds':sum(communication_rounds),
                'Test loss': test_loss,
                'Test Acc': test_acc,
                'Average sign':avg_sign,
                'Threshold' :thresh,
                'relevance_it': wandb.Image(heatmap)
            }   
            wandb.log(wand_dict)
        if last_update == None:
            print('exiting!')
            break
        if (args.save_model):
            torch.save(model.state_dict(),"model_"+ str(it) +".pt")
        if when_above_60 == -1 and test_acc > 60:
            when_above_60 = sum(communication_rounds)

    all_stats = {
        'communication_rounds':communication_rounds,
        'test_loss':test_losses,
        'test_acc':test_accuracies,
        'avg_sign':average_signs,
        'threshold':thresholds,
        #'heatmap':np.stack(all_relevances,1).astype(np.float)
    }
    print('Training done, the model reached above 60 percent accuracy after {} communication rounds'.format(when_above_60))
    with open('generated_data/thresh'+str(args.start_threshold)+'/' + 'all_stats.pkl','wb') as f:
        pickle.dump(all_stats,f)
    plot_data(all_stats,'generated_data/thresh'+str(args.start_threshold)+'/')



if __name__ == '__main__':
    main()
