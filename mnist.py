#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

def plot_combined_results(train_results, sigma, batch_size):
    train_losses_dp_static, accuracy_per_epoch_dp_static = train_results['DP-SGD(static)']
    train_losses_dp_dynamic, accuracy_per_epoch_dp_dynamic = train_results['DP-SGD(dynamic)']

    epochs = range(1, len(next(iter(train_results.values()))[0]) + 1)
    fig, axs = plt.subplots(1, figsize=(10, 8))

    # Plot training losses
    axs[0].plot(epochs, train_losses_dp_static, label='DP-SGD(static)', color='red')
    axs[0].plot(epochs, train_losses_dp_dynamic, label='DP-SGD(dynamic)', color='green')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Training Loss')
    axs[0].legend(loc='upper right')

    # Plot testing accuracies
    axs[1].plot(epochs, accuracy_per_epoch_dp_static, label='DP-SGD(static)', color='red')
    axs[1].plot(epochs, accuracy_per_epoch_dp_dynamic, label='DP-SGD(dynamic)', color='green')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Testing Accuracy')
    axs[1].legend(loc='upper left')
 
    # fig.suptitle(f'CNN on MNIST dataset sigma_{sigma}_batch_{batch_size}')
    plt.savefig(f'/home/zahid/work/d2p2sgd/log/CNN_mnist/dp-sgd_sigma_{sigma}_batch_{batch_size}.png')


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    return losses

def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)

def update_privacy_engine(privacy_engine, model, optimizer, train_loader, args, current_epoch, max_grad_norm): 
    
    model.train()
    if not privacy_engine: 
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)

    if hasattr(optimizer, 'privacy_engine'):
        print("Detach previous privacy engine settings")
        optimizer.privacy_engine.detach()
        
    dynamic_sigma = args.sigma / current_epoch ** 0.25 # dynamic_sigma = sigma/sqrt(k)
    print("dynamic sigma:", dynamic_sigma)
    clipping = "per_layer" if args.clip_per_layer else "flat"
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier= dynamic_sigma,
        max_grad_norm=max_grad_norm,
        clipping=clipping,
        )
    return privacy_engine, optimizer


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    train_results = {} 

    for dp_mode in ['static', 'dynamic']:  # [SGD, DP-SGD(static), DP-SGD(dynamic)]
        args.disable_dp = (dp_mode is None)
        dp_label = 'SGD' if dp_mode is None else f'DP-SGD({dp_mode})'
        print(f"Running  for {dp_label}")

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.data_root,
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                    ]
                ),
            ),
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.data_root,
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                    ]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        run_results = []
        for _ in range(args.n_runs):
            model = SampleConvNet().to(device)
            

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
            privacy_engine = None

            if not args.disable_dp:
                
                if args.clip_per_layer:
                # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
                    n_layers = len([(n, p) for n, p in model.named_parameters() if p.requires_grad])
                    max_grad_norm = [args.max_per_sample_grad_norm / np.sqrt(n_layers)] * n_layers
                else:
                    max_grad_norm = args.max_per_sample_grad_norm

                privacy_engine = PrivacyEngine(
                    secure_mode=args.secure_rng,
                )
                clipping = "per_layer" if args.clip_per_layer else "flat"
            
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=args.sigma,
                    max_grad_norm=args.max_per_sample_grad_norm,
                )
            # Store logs
            accuracy_per_epoch = []
            train_losses =[]

            for epoch in range(1, args.epochs + 1):

                if not args.disable_dp and dp_mode == 'dynamic':  # Update sigma each epoch for dynamic DP-SGD
                    privacy_engine, optimizer = update_privacy_engine(privacy_engine, model, optimizer, train_loader, args, epoch, max_grad_norm)
                losses = train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
                top1_acc = test(model, device, test_loader)
                train_loss = np.mean(losses)
                accuracy_per_epoch.append(float(top1_acc))
                train_losses.append(train_loss)
            run_results.append(top1_acc)
            train_results[dp_label] = (train_losses, accuracy_per_epoch)

    plot_combined_results(train_results, args.sigma, args.batch_size)

    # if len(run_results) > 1:
    #     print(
    #         "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
    #             len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
    #         )
    #     )

    # repro_str = (
    #     f"mnist_{args.lr}_{args.sigma}_"
    #     f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    # )
    # torch.save(run_results, f"log/CNN_mnist/run_results_{repro_str}.pt")

    # if args.save_model:
    #     torch.save(model.state_dict(), f"log/CNN_mnist/mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()