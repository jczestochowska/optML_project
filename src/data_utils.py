import os
import random
import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from torchvision import datasets

from definitions import PYTORCH_DATA_DIR


def get_data_loaders(batch_size, num_clients, non_iid_mix_p, iid_split=True, percentage_val=0.2, full=False, non_iid_mix = False):
    val_loader = None
    train_input, train_target, test_input, test_target = load_data(flatten=False, full=full)
    train_dataset = TensorDataset(train_input, train_target)

    # If validation set is needed randomly split training set
    if percentage_val:
        val_dataset, train_dataset = torch.utils.data.random_split(train_dataset,
                                                                   (int(percentage_val * len(train_dataset)),
                                                                    int((1 - percentage_val) * len(train_dataset)))
                                                                   )
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    # Split data for each client
    if iid_split:
        # Random IID data split
        client_datasets = torch.utils.data.random_split(train_dataset, np.tile(int(len(train_dataset) / num_clients),
                                                                               num_clients).tolist())
    else:
        
        #### Arman ####
        if non_iid_mix:
            print('Creating mixed non-iid client data set with {0} non-iid'.format(non_iid_mix_p))

            non_iid_part, iid_part = get_non_iid_split(train_dataset,non_iid_mix_p)
            client_datasets = get_non_iid_datasets(num_clients, non_iid_part)       # make client_datasets with non_iid_part

            for client_nr, client_dataset in enumerate(client_datasets):            # append client_datasets with iid_part                                                       

                #print('before: '+str(len(client_dataset)))
                chunk_size = int(len(iid_part)/num_clients)
                client_dataset.indices = torch.cat((torch.as_tensor(client_dataset.indices), torch.as_tensor(iid_part.indices[chunk_size*client_nr : chunk_size*(1+client_nr)])))                                                                       
                #print('after: '+ str(len(client_dataset)))

            random.shuffle(client_datasets)
        ### Arman ###    

        else:
        	# Each client has different set of non overlapping digits
            client_datasets = get_non_iid_datasets(num_clients, train_dataset)
            random.shuffle(client_datasets)

    train_loaders = []
    for train_dataset in client_datasets:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        train_loaders.append(train_loader)

    test_loader = DataLoader(dataset=TensorDataset(test_input, test_target),
                             batch_size=batch_size)

    return train_loaders, val_loader, test_loader

### Arman ###
def get_non_iid_split(train_dataset,non_iid_mix_p):
    #split train_dataset into a non-iid and iid part

    p = non_iid_mix_p
    non_iid_part , iid_part = torch.utils.data.random_split(train_dataset, [round(p*len(train_dataset)),round((1-p)*len(train_dataset))])


    return non_iid_part, iid_part

### Arman ###


def load_data(cifar=False, one_hot_labels=False, normalize=False, flatten=False, full=False):
    data_dir = PYTORCH_DATA_DIR
    if data_dir is None:
        data_dir = './data'

    if cifar:
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train=True, download=True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train=False, download=True)

        train_input = torch.from_numpy(cifar_train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.targets, dtype=torch.int64)

        test_input = torch.from_numpy(cifar_test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.targets, dtype=torch.int64)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True)

        train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.targets
        test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.targets

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if not full:
        print('** Reducing the data-set, (use --full for the full thing)')
        train_input = train_input.narrow(0, 0, 5000)
        train_target = train_target.narrow(0, 0, 5000)
        test_input = test_input.narrow(0, 0, 5000)
        test_target = test_target.narrow(0, 0, 5000)

    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target


def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


def get_non_iid_datasets(num_clients, train_dataset):
    """
    This function divides samples in a way that
    each client has non-overlapping classes,
    e.g client 1 has only digits 0 and 1 while client 2 has only digits 2 and 3.
    To achieve this we perform binary search on labels tensor
    to divide initial dataset
    """

    client_datasets = []
    # if we have validation set then train is a Subset type
    if isinstance(train_dataset, Subset):
        labels = train_dataset.dataset.tensors[1][train_dataset.indices]
    else:
        labels = train_dataset.tensors[1]
    labels, sorted_indices = torch.sort(labels)
    digits_per_client = 10 // num_clients
    digit = 0
    for client in range(num_clients):
        first_idx = first_index(labels, 0, len(labels), digit)
        if client == num_clients - 1:
            last_idx = len(labels) - 1
        else:
            last_idx = last_index(labels, 0, len(labels), digit + (digits_per_client - 1))
        if isinstance(train_dataset, Subset):
            client_dataset = Subset(train_dataset.dataset, sorted_indices[first_idx: last_idx + 1])
        
        else:
            client_dataset = Subset(train_dataset, sorted_indices[first_idx: last_idx + 1])
        client_datasets.append(client_dataset)

        digit += digits_per_client
    return client_datasets


# binary search functions to retrieve first and last index of label in sorted labels array
def first_index(array, low, high, item):
    if high >= low:
        mid = low + (high - low) // 2
        if (mid == 0 or item > array[mid - 1]) and array[mid] == item:
            return mid
        elif item > array[mid]:
            return first_index(array, (mid + 1), high, item)
        else:
            return first_index(array, low, (mid - 1), item)
    print(f"This label {item} was not found")
    return -1


def last_index(array, low, high, item):
    if high >= low:
        mid = low + (high - low) // 2
        if (mid == len(array) - 1 or item < array[mid + 1]) and array[mid] == item:
            return mid
        elif item < array[mid]:
            return last_index(array, low, (mid - 1), item)
        else:
            return last_index(array, (mid + 1), high, item)
    print(f"This label {item} was not found")
    return -1


def get_model_bits(state_dict):
    """
    :param state_dict: model object for which we want to get size in bits
    :return: model_size - number of bites for all model's parameters
    """
    torch.save(state_dict, "temp.p")
    # Multiply by 8 to go from bytes to bits
    model_size = os.path.getsize("temp.p") * 8
    os.remove('temp.p')
    return model_size
