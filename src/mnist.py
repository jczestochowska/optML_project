import syft as sy
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms

from gradient_compression import sparsify_gradient_topk
from models import CNN


class Configuration:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.seed = 42
        self.log_interval = 30
        self.save_model = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.num_workers = 5
        self.gradient_compression = sparsify_gradient_topk
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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Test accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )


def get_virtual_workers(ids):
    hook = sy.TorchHook(torch)
    return [sy.VirtualWorker(hook, id="worker_" + str(worker_id)) for worker_id in ids]


def get_data_loaders():
    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
            .federate(virtual_workers),
        batch_size=config.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=config.test_batch_size, shuffle=True)
    return federated_train_loader, test_loader


if __name__ == '__main__':
    config = Configuration()
    virtual_workers = get_virtual_workers(ids=range(config.num_workers))
    torch.manual_seed(config.seed)
    train_loader, test_loader = get_data_loaders()

    print(summary(config.model, (1, 28, 28)))

    for epoch in range(1, config.epochs + 1):
        train(config, config.model, config.device, train_loader, config.optimizer, epoch)
        test(config.model, config.device, test_loader)

    if config.save_model:
        torch.save(config.model.state_dict(), f"{config.model_name}.pt")
