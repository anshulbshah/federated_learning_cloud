from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pickle
import copy
import numpy as np
DATA_LEN = 60000
from tqdm import tqdm
import dataset
# import wandb
# class custom_MNIST_dset(Dataset):
#     def __init__(self,
#                  image_path,
#                  label_path,
#                  img_transform = None):
#         with open(image_path, 'rb') as image_file:
#             self.image_list = pickle.load(image_file)
#         with open(label_path, 'rb') as label_file:
#             self.label_list = pickle.load(label_file)
        
#         self.img_transform = img_transform
    
#     def __getitem__(self, index):
#         image = self.image_list[index]
#         label = self.label_list[index]
        
#         if self.img_transform is not None:
#             image = self.img_transform(image)
        
#         return image, label
    
#     def __len__(self):
#         return len(self.image_list)

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
        return F.log_softmax(x, dim=1)
    
def cli_train(args, old_model, device, train_loader, epoch, last_updates,iteration):
    model = copy.deepcopy(old_model)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    cli_ite_num = args.cli_ite_num
    # print(cli_ite_num)
    for batch_idx, (data, target) in enumerate(train_loader):
        if cli_ite_num == 0:
            break
        cli_ite_num -= 1
        optimizer.zero_grad()
        data, target = data.type('torch.FloatTensor').to(device), target.to(device, dtype=torch.int64)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


    threshold = args.start_threshold/np.sqrt((epoch-1)*(DATA_LEN // (args.client_num * args.batch_size)) + iteration+1)
    # print(epoch,threshold)
    return check_relevance(model, old_model, last_updates,threshold)

def glo_train(args, model, device, train_loaders, optimizer, epoch, commu, flag):
    model.train()
    print('DATALEN:{}, product: {}'.format(DATA_LEN,args.client_num * args.batch_size))

    client_signs = np.zeros(args.client_num)

    # for i in tqdm(range(DATA_LEN // (args.client_num * args.batch_size))):
    i = 0
    new_model_list = []
    relevances = []
    
    if flag:
        tmp_flag = True
        
        for j in range(args.client_num):
            # print('client {}'.format(j))
            _, new_model,e = cli_train(args, model, device, train_loaders[j], epoch, None,i)
            client_signs[j] = e
            new_model_list.append(new_model)
            relevances.append(e)
            
        cur_commu = args.client_num
        flag = False
    else:
        print('here!')
        cur_commu = 0

        for j in range(args.client_num):
            relv, new_model,e = cli_train(args, model, device, train_loaders[j], epoch, last_updates,i)
            client_signs[j] = (client_signs[j]*(i) + e)/i+1
            if relv:
                cur_commu += 1
                new_model_list.append(new_model)
            relevances.append(e)

    

    # Merge model grad
    print('ep,len',epoch,len(new_model_list))
    # wandb.log({'client_nums':client_signs})

    # wandb.log{'':client_signs}
    # print(model.fc2.weight)
    model_before_update = copy.deepcopy(model)
    last_updates = merge(model, new_model_list)

    # print(model.fc2.weight)

    
    commu.append(cur_commu)

        
        # print('Train Epoch: {}/{}'.format(i,epoch))
            
def test(args, model, device, test_loader, commu):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # Draw picture acc vs commu
    # Use sum matrix
    pass
    #
    
def check_relevance(model, old_model, last_updates,threshold):
    sign_sum = 0
    sign_size = 0
    rel_threshold = 0.0#0.8#threshold
    model_para_list = []
    cur_updates = []
    
    if last_updates is None:
        for cur_para, old_para in zip(model.parameters(), old_model.parameters()):
            cur_updates.append(cur_para.data - old_para.data)
        return True, cur_updates, None
    
    for cur_para, old_para in zip(model.parameters(), old_model.parameters()):
        cur_updates.append(cur_para.data - old_para.data)
        
        # Collect model parameters into lists
        model_para_list.append(cur_para)
        
    for i in range(len(cur_updates)):
        cur_sign = torch.sign(cur_updates[i])
        old_sign = torch.sign(last_updates[i])
        
        sign = cur_sign * old_sign
        sign[sign < 0] = 0
        sign_sum += torch.sum(sign)
        sign_size += sign.numel()
    
    # 0.000001 is given in case of dividing by 0
    # if last_updates is not None:

    e = sign_sum / (sign_size + 0.000001)

    # print(e)
    return e >= rel_threshold, model_para_list, e
    # return True, model_para_list, e


def merge(model, new_model_list):
    # print(len(new_model_list))
    if len(new_model_list) == 0:
        # print("No model's revelence is higher than threshold.")
        return
    
    old_model = copy.deepcopy(model)
    
    para_ind = 0

    for name,param in model.named_parameters():
        #print(param.data)        
        # if 'fc2.weight' in name:

        param.data = new_model_list[0][para_ind].data
        for i in range(1, len(new_model_list)):
            # cannot use "+=" cause torch doesn't allow in-place
            # operations on variables you create directly
            param.data = param.data + new_model_list[i][para_ind].data
        
        para_ind += 1
        param.data /= len(new_model_list)
    
    # Record updates
    last_updates = []
    for old_param, new_param in zip(old_model.parameters(), model.parameters()):
        last_updates.append(new_param.data - old_param.data)
    
    return last_updates
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--client-num', type=int, default=10)
    parser.add_argument('--cli-ite-num', type=int, default=4)
    parser.add_argument('--start_threshold', type=float, default=0.5)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # wandb.init(project="cloud")
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    train_loaders = []
    # for i in range(args.client_num):
    #     train_loaders.append(torch.utils.data.DataLoader(
    #                 dataset.MNIST('../data', train=True, download=True, transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,), (0.3081,))
    #                                ]),client_id=i,num_clients=args.client_num


    #                 ),
    #                 batch_size=args.batch_size, shuffle=True, **kwargs))
    for i in range(args.client_num):
        train_loaders.append(torch.utils.data.DataLoader(
                            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ])),
                            batch_size=args.batch_size, shuffle=True, **kwargs))

    test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
                    batch_size=args.test_batch_size, shuffle=False, **kwargs)

    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    commu = []
    flag = True
    for epoch in range(1, args.epochs + 1):
        glo_train(args, model, device, train_loaders, optimizer, epoch, commu, flag)
        test(args, model, device, test_loader, commu)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()