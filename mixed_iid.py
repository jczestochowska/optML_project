import torch
import numpy as np
from src.data_utils import get_data_loaders, get_model_bits


##############################
# Configure here to get a specific experiment
batch_size = 32
num_clients = 5
target_accuracy = 93
iid_split = True
# default setup is 5 epochs per client,
# here we have five clients therefore  we need [5, 5, 5, 5, 5]
# change the list accordingly to get variable
# number of epochs for different clients
epochs_per_client = 5 * [5]
##############################

# Load data
train_loaders, _, test_loader, client_datasets = get_data_loaders(batch_size, num_clients, percentage_val=0, iid_split=iid_split)

print(client_datasets.dataset.tensors[1])
