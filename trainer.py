import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from opacus import PrivacyEngine
import matplotlib.pyplot as plt

class PrivacyTrainer:
    def __init__(self, model, train_dataset, test_dataset, batch_size=1024, lr=0.05, noise_multiplier=1.1, max_grad_norm=1.0):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.optimizer = SGD(model.parameters(), lr=lr)
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.train_losses = []
        self.test_accuracies = []

    def make_private(self):
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        print("Model has been made private with Opacus PrivacyEngine.")

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Average Training Loss: {avg_loss}")
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        self.test_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy}%")

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.show()

# Example usage with dummy data and model for demonstration purposes
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

# Load datasets
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

# Initialize the model and trainer
model = resnet18(num_classes=10)  # Adjust the model output to match CIFAR10 classes
trainer = PrivacyTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset)
trainer.make_private()

# Run training
trainer.train(epochs=10)

# Plot
trainer.plot_metrics()
