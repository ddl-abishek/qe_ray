import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
import ray.train as train
from ray.air import session
import argparse
import os

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_epoch(dataloader, model, loss_fn, optimizer, device='cuda'):
    size = len(dataloader.dataset) // session.get_world_size()  # Divide by word size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # We don't need this anymore! Ray Train does this automatically:
        X, y = X.to(device), y.to(device)  

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_epoch(dataloader, model, loss_fn, device='cuda'):
    size = len(dataloader.dataset) // session.get_world_size()  # Divide by word size
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def load_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, test_data


def train_func(config: dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    
    batch_size_per_worker = batch_size // session.get_world_size()
    
    training_data, test_data = load_data()  # <- this is new!
    
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size_per_worker)
    test_dataloader = DataLoader(test_data, batch_size=batch_size_per_worker)
    
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)
    
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for t in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_epoch(test_dataloader, model, loss_fn)
        checkpoint = Checkpoint.from_dict(
            dict(epoch=t, model=model.state_dict())
        )
        session.report(dict(loss=test_loss), checkpoint=checkpoint)

    print("Done!")

def train_fashion_mnist(num_workers=3, use_gpu=True):
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )
    result = trainer.fit()
    print(f"Last result: {result.metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers","-n", type=int, default=3, help="Sets number of workers for training.",)
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Enables GPU training")
    args, _ = parser.parse_known_args()

    import ray
    if ray.is_initialized() == False:
        print("Connecting to Ray cluster...")
        service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
        service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
        ray.util.connect(f"{service_host}:{service_port}")

    train_fashion_mnist(num_workers=3, use_gpu=True)
