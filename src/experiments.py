import os
import pickle
from functools import reduce

from definitions import ROOT_DIR
from src.data_utils import get_data_loaders, get_model_bits
from src.training import *
from src.weight_quantization import quantize_float16, quantize_int8, decode_quantized_model_int8

##############################
# Configure here to get a specific experiment
batch_size = 25
num_clients = 5
target_accuracy = 93
iid_split = True
non_iid_mix = 0
# validation set
percentage_val = 0
# if use full MNIST with 60000 train and 10000 test
full = False
# default setup is 5 epochs per client,
# here we have five clients therefore  we need [5, 5, 5, 5, 5]
# change the list accordingly to get variable
# number of epochs for different clients
client_epochs = 5
epochs_per_client = num_clients * [client_epochs]
quantization = quantize_float16
##############################


# Load data
train_loaders, _, test_loader = get_data_loaders(batch_size, num_clients, non_iid_mix=non_iid_mix,
                                                 percentage_val=percentage_val, iid_split=iid_split, full=full)

# Initialize all clients
clients = [Client(train_loader, epochs) for train_loader, epochs in zip(train_loaders, epochs_per_client)]

# Set seed for the script
torch.manual_seed(clients[0].seed)

testing_accuracy = 0
num_rounds = 1

central_server = Client(test_loader)

filename = f"iid_split_{iid_split}_quantization_{quantization.__name__}_num_epochs_{client_epochs}_mix_{non_iid_mix}.pkl"
experiment_state = {"num_rounds": 0,
                    "test_accuracies": [],
                    "conserved_bits_from_server": [],
                    "conserved_bits_from_clients": [],
                    "transferred_bits_from_server": [],
                    "transferred_bits_from_clients": [],
                    "original_bits_from_server": [],
                    "original_bits_from_clients": []
                    }

# Multiplier for int8 quantization
multiplier = 0

while testing_accuracy < target_accuracy:
    print("Communication Round {0}".format(num_rounds))

    if num_rounds > 1:
        # Load server weights onto clients
        total_bits_conserved = 0
        total_bits_transferred = 0
        total_float_model_bits = 0
        for client in clients:
            with torch.no_grad():
                # Calculate number of bits in full server model
                float_model_bits = get_model_bits(central_server.model.state_dict())
                # Quantize server's model
                if quantization == quantize_int8:
                    quantized_model, multiplier = quantization(central_server.model.state_dict())
                else:
                    quantized_model = quantization(central_server.model.state_dict())
                bits_transferred = get_model_bits(quantized_model)
                # Calculate how many bits we saved
                bits_conserved = (float_model_bits - bits_transferred)
                # If quantization method is int8, decode the weights
                if quantization == quantize_int8:
                    quantized_model = decode_quantized_model_int8(quantized_model, multiplier)
                # Distribute quantized model on clients
                client.model.load_state_dict(quantized_model)
                # Update summary values
                total_bits_conserved += bits_conserved
                total_bits_transferred += bits_transferred
                total_float_model_bits += float_model_bits

        # Add to our summary
        experiment_state["conserved_bits_from_server"].append(total_bits_conserved // num_clients)
        experiment_state["transferred_bits_from_server"].append(total_bits_transferred // num_clients)
        experiment_state["original_bits_from_server"].append(total_float_model_bits // num_clients)

    # Perform E local training steps for each client
    for client_idx, client in enumerate(clients):
        print("Training client {0}".format(client_idx))
        for epoch in range(1, client.epochs + 1):
            train(client, epoch)

    with torch.no_grad():
        # Get number of bits in all clients' models before quantization
        clients_bits = reduce((lambda x, y: x + y), [get_model_bits(client.model.state_dict()) for client in clients])
        # Quantize clients models
        if quantization == quantize_int8:
            quantized_clients_models = []
            multipliers = []
            for client in clients:
                client_model, multiplier = quantization(client.model.state_dict())
                quantized_clients_models.append(client_model)
                multipliers.append(multiplier)
        else:
            quantized_clients_models = [quantization(client.model.state_dict()) for client in clients]
        quantized_clients_bits = reduce((lambda x, y: x + y),
                                        [get_model_bits(client) for client in quantized_clients_models])
        bits_conserved = (clients_bits - quantized_clients_bits) // num_clients
        # Add to summary
        experiment_state["conserved_bits_from_clients"].append(bits_conserved)
        experiment_state["transferred_bits_from_clients"].append(quantized_clients_bits // num_clients)
        experiment_state["original_bits_from_clients"].append(clients_bits // num_clients)
        # Decode bits on central server side:
        if quantization == quantize_int8:
            new_client_models = []
            for client, multiplier in zip(quantized_clients_models, multipliers):
                new_client = decode_quantized_model_int8(client, multiplier)
                new_client_models.append(new_client)
            quantized_clients_models = new_client_models
        # Send quantized models to server and average them
        averaged_model = average_client_models(quantized_clients_models)
        central_server.model.load_state_dict(averaged_model)
    # We have to convert back to float32 otherwise there is a mismatch with input dtype
    central_server.model.to(torch.float32)
    # Test the aggregated model
    test_loss, testing_accuracy = test(central_server)
    experiment_state['test_accuracies'].append(testing_accuracy)
    experiment_state['num_rounds'] = num_rounds + 1

    # Save experiment states
    with open(os.path.join(ROOT_DIR, "outputs", filename), "wb") as f:
        pickle.dump(experiment_state, f)

    num_rounds += 1

# Save model
if central_server.save_model:
    torch.save(central_server.model.state_dict(), f"{central_server.model_name}.pt")
