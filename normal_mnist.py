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
class custom_MNIST_dset(Dataset):
    def __init__(self,
                 image_path,
                 label_path,
                 img_transform = None):

        self.image_list = []
        self.label_list = []
        for i in range(100):
            with open(image_path + str(i), 'rb') as image_file:
                self.image_list.extend(pickle.load(image_file))
            with open(label_path + str(i), 'rb') as label_file:
                self.label_list.extend(pickle.load(label_file))
        
        self.img_transform = img_transform
        print(len(self.image_list))
    
    def __getitem__(self, index):
        image = self.image_list[index]
        label = self.label_list[index]
        
        if self.img_transform is not None:
            image = self.img_transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.image_list)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        import pdb; pdb.set_trace()  # breakpoint 7d5622bb //
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        import pdb; pdb.set_trace()  # breakpoint 11797506 //

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    model.zero_grad()
    crtiterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        optimizer.zero_grad()

        data, target = data.type('torch.FloatTensor').to(device), target.to(device, dtype=torch.int64)
        output = model(data)
        loss = crtiterion(output, target)
        loss.backward()
        optimizer.step()

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

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # wandb.init(project="cloud")
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}


    train_loader = torch.utils.data.DataLoader(
                            custom_MNIST_dset('MNIST_data/train_data-', 'MNIST_data/train_label-',
                                           img_transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ])),
                            batch_size=args.batch_size, shuffle=True, **kwargs)
    # train_loader = torch.utils.data.DataLoader(
    #                 datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Normalize((0.1307,), (0.3081,))
    #                                ])),
    #                 batch_size=args.batch_size, shuffle=True, **kwargs)


    
    test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
                    batch_size=args.test_batch_size, shuffle=False, **kwargs)


    
    model = Net().to(device)
    # model = Net2().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()