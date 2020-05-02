import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dlc_practical_prologue import load_data
from models import CNN
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

#Global constants shared among all clients that we vary in testing
EPOCHS = 5
BATCH_SIZE = 50

class Client:
    def __init__(self,data_loader):
        self.data_loader = data_loader
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.lr = 0.001
        self.log_interval = 5
        self.seed = 42
        self.save_model = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.gradient_compression = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_name = "mnist_cnn"


def train(args, model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        args.gradient_compression(model)
        optimizer.step()
        model.get()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size,
                       len(federated_train_loader) * args.batch_size,
                       100. * batch_idx / len(federated_train_loader),
                loss.item())
            )


def train(client, epoch, logging=True):
    
    # put model in train mode, we need gradients
    client.model.train()
    train_loader = client.data_loader
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        client.optimizer.zero_grad()
        output = client.model(data)
        # get the basic loss for our main task
        total_loss = client.criterion(output, target)
        total_loss.backward()
        train_loss += total_loss.item()
        client.optimizer.step()
    _, train_accuracy = test(client, logging=False)
    if logging:
        print(f'Train Epoch: {epoch} Loss: {total_loss.item():.6f}, Train accuracy: {train_accuracy}')
    return train_loss, train_accuracy


def test(client, logging=True):
    # put model in eval mode, disable dropout etc.
    client.model.eval()
    test_loss = 0
    correct = 0
    test_loader = client.data_loader
    # disable grad to perform testing quicker
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(client.device), target.to(client.device)
            output = client.model(data)
            test_loss += client.criterion(output, target).item()
            # prediction is an output with maximal probability
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    if logging:
        print(f'Test set: Average loss: {test_loss:.4f}, '
              f'Test accuracy: {correct} / {len(test_loader.dataset)} '
              f'({test_accuracy:.0f}%)\n')
    return test_loss, test_accuracy


def get_data_loaders(batch_size, num_clients, data_split_type='iid', percentage_val=0.2):
    val_loader = None
    train_input, train_target, test_input, test_target = load_data(flatten = False)
    train_dataset = TensorDataset(train_input, train_target)
 
    # if validation set is needed randomly split training set
    if percentage_val:
        val_dataset, train_dataset = torch.utils.data.random_split(train_dataset,
                                                               (int(percentage_val*len(train_dataset)),
                                                                int((1-percentage_val)*len(train_dataset)))
                                                               )
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    #Split data for each client
    if data_split_type=='iid':
        # Random IID data split
        client_datasets = torch.utils.data.random_split(train_dataset,np.tile(int(len(train_dataset)/num_clients),num_clients).tolist())
    #else:
        # TODO: code non-iid data split
        
    train_loaders = []
    for train_dataset in client_datasets:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        train_loaders.append(train_loader)
        
    test_loader = DataLoader(dataset=TensorDataset(test_input, test_target),
                             batch_size=batch_size)
    return train_loaders, val_loader, test_loader

def averageClientModels(clients):
    "Input list of clients, output a dictionary with their averaged parameters"
    dicts = []
    for client in clients:
        #TO-DO: Sparsify client paramters
        client_dict = dict(client.model.named_parameters())
        dicts.append(client_dict)
    dict_keys = dicts[0].keys()
    final_dict = dict.fromkeys(dict_keys)
    for key in dict_keys:
        #Average model parameters
        final_dict[key] = torch.cat([dictionary[key].unsqueeze(0) for dictionary in dicts],dim=0).sum(0).div(len(dicts))
    return final_dict