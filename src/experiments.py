import os
import pickle
from functools import reduce

from definitions import ROOT_DIR
from src.data_utils import get_data_loaders, get_model_bits
from src.weight_quantization import quantize_float16
from src.training import *

##############################
# Configure here to get a specific experiment
batch_size = 25
num_clients = 5
target_accuracy = 93
iid_split = False
non_iid_mix = 0
# validation set
percentage_val = 0
# default setup is 5 epochs per client,
# here we have five clients therefore  we need [5, 5, 5, 5, 5]
# change the list accordingly to get variable
# number of epochs for different clients
epochs_per_client = 5 * [5]
quantization = quantize_float16
##############################

# Load data
train_loaders, _, test_loader = get_data_loaders(batch_size, num_clients, non_iid_mix=non_iid_mix,
                                                 percentage_val=percentage_val, iid_split=iid_split)

# Initialize all clients
clients = [Client(train_loader, epochs) for train_loader, epochs in zip(train_loaders, epochs_per_client)]

# Set seed for the script
torch.manual_seed(clients[0].seed)

testing_accuracy = 0
num_rounds = 0

central_server = Client(test_loader)

experiment_state = {"num_rounds": 0,
                    "test_accuracies": [],
                    "conserved_bits_from_server": [],
                    "conserved_bits_from_clients": [],
                    "transferred_bits_from_server": [],
                    "transferred_bits_from_clients": [],
                    "original_bits_from_server": [],
                    "original_bits_from_clients": []
                    }

while testing_accuracy < target_accuracy:
    print("Communication Round {0}".format(num_rounds+1))

    if num_rounds > 1:
        # Load server weights onto clients
        for client in clients:
            with torch.no_grad():
                # Calculate number of bits in full server model
                float_model_bits = get_model_bits(central_server.model.state_dict())
                # Quantize server's model
                quantized_model = quantization(central_server.model.state_dict())
                bits_transferred = get_model_bits(quantized_model)
                # Calculate how many bits we saved
                bits_conserved = float_model_bits - bits_transferred
                # Add to our summary
                experiment_state["conserved_bits_from_server"].append(bits_conserved)
                experiment_state["transferred_bits_from_server"].append(bits_transferred)
                experiment_state["original_bits_from_server"].append(float_model_bits)
                # Distribute quantized model on clients
                client.model.load_state_dict(quantized_model)

    # Perform E local training steps for each client
    for client_idx, client in enumerate(clients):
        print("Training client {0}".format(client_idx))
        for epoch in range(1, client.epochs + 1):
            train(client, epoch)

    with torch.no_grad():
        # Get number of bits in all clients' models before quantization
        clients_bits = reduce((lambda x, y: x * y), [get_model_bits(client.model.state_dict()) for client in clients])
        # Quantize clients models
        quantized_clients_models = [quantization(client.model.state_dict()) for client in clients]
        quantized_clients_bits = reduce((lambda x, y: x * y),
                                        [get_model_bits(client) for client in quantized_clients_models])
        bits_conserved = clients_bits - quantized_clients_bits
        # Add to summary
        experiment_state["conserved_bits_from_clients"].append(bits_conserved)
        experiment_state["transferred_bits_from_clients"].append(quantized_clients_bits)
        experiment_state["original_bits_from_clients"].append(clients_bits)
        # Send quantized models to server and average them
        averaged_model = average_client_models(quantized_clients_models)
        central_server.model.load_state_dict(averaged_model)
    # We have to convert back to float32 otherwise there is mismatch with input dtype
    central_server.model.to(torch.float32)
    # Test the aggregated model
    test_loss, testing_accuracy = test(central_server)
    experiment_state['test_accuracies'].append(testing_accuracy)
    experiment_state['num_rounds'] = num_rounds + 1

# Save model
if central_server.save_model:
    torch.save(central_server.model.state_dict(), f"{central_server.model_name}.pt")

# Save experiment states
filename = f"iid_split_{iid_split}_quantization_{quantization.__name__}.pkl"
with open(os.path.join(ROOT_DIR, "outputs", filename), "wb") as f:
    pickle.dump(experiment_state, f)
