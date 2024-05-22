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
Runs SVHN training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from opacus import PrivacyEngine
from privacy_engine import PrivacyEngine
import json
from datetime import datetime
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time



# Precomputed characteristics of the MNIST dataset
SVHN_MEAN = 0.5
SVHN_STD = 0.5

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

def plot_combined_results(train_results, sigma, batch_size, red_rate, seed):

    fig, axs = plt.subplots(2, figsize=(10, 10), dpi=400)

    # Plot training losses
    for optim, result in train_results.items():
        epochs = range(1, len(result['loss']) + 1)
        axs[0].plot(epochs, result['loss'], label=f'{optim}', linewidth=2)
    
    axs[0].set_xlabel('Epoch', fontsize=16)
    axs[0].set_ylabel('Training Loss', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].legend(loc='upper right', fontsize=12)
    
    # Plot testing accuracies
    for optim, result in train_results.items():
        epochs = range(1, len(result['acc']) + 1)
        axs[1].plot(epochs, result['acc'], label=f'{optim}', linewidth=2)
    
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].set_ylabel('Testing Accuracy', fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    filename = f'rrate/CNN_svhn/{current_time}_sigma_{sigma}_batch_{batch_size}_seed_{seed}_rrate_{red_rate}'
    fig.suptitle(f'CNN_SVHN_sigma_{sigma}_batch_{batch_size}_rrate{red_rate}', fontsize=16)
    plt.savefig(f"{filename}.png")
    
    with open(f"{filename}.json", "w") as file:
        json.dump(train_results, file, indent=4)


def train(args, model, device, train_loader, optimizer,dp_mode, privacy_engine, epoch):
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

        if _batch_idx % args.print_freq == 0:
            if not args.disable_dp:
                epsilon =0 
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"(ε = {epsilon:.5f}, δ = {args.delta})"
                )
            else:
                epsilon=0
                print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    if dp_mode =="dynamic" or dp_mode =="d2p2":
        return losses, optimizer.noise_multiplier, epsilon
    else:
        return losses, args.sigma, epsilon

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




def main():
    
    args = parse_args()
    # device = torch.device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_results = {} 

    for dp_mode in [None, "static","dynamic", "RP", "d2p2"]:
        args.disable_dp = (dp_mode is None)
        dp_label = 'SGD' if dp_mode is None else f'DP-SGD({dp_mode})'

        random_projection = False
        if dp_mode == 'RP' or dp_mode == 'd2p2':
            random_projection = True 
        print("random_projection=", random_projection)

        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                args.data_root,
                split='train',
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((SVHN_MEAN,), (SVHN_STD,)),
                    ]
                ),
            ),
            batch_size=args.batch_size,
            num_workers=20,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                args.data_root,
                split='test',
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((SVHN_MEAN,), (SVHN_STD,)),
                    ]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
        )
        run_results = []
        for _ in range(args.n_runs):
            model = SampleConvNet().to(device)

            print(f"-->>Training mode: {dp_label}, Device: {device}, args.disable_dp:", args.disable_dp)

            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
            
            privacy_engine = None
            if not args.disable_dp:
                # if args.clip_per_layer:
                # # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
                #     n_layers = len([(n, p) for n, p in model.named_parameters() if p.requires_grad])
                #     max_grad_norm = [args.max_per_sample_grad_norm / np.sqrt(n_layers)] * n_layers
                # else:
                #     max_grad_norm = args.max_per_sample_grad_norm

                privacy_engine = PrivacyEngine(
                    secure_mode=args.secure_rng,
                )
                # clipping = "per_layer" if args.clip_per_layer else "flat"
            
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=args.sigma,
                    max_grad_norm=args.max_per_sample_grad_norm,
                    random_projection=random_projection,
                    seed=args.seed,
                )
            # Store logs
            accuracy_per_epoch = []
            train_losses =[]
            epsilon_per_epoch = []
            sigma_per_epoch =[]


            for epoch in range(1, args.epochs + 1):
                if not args.disable_dp and dp_mode == 'dynamic':  # dynamic DP-SGD
                    new_noise_multiplier = args.sigma / (epoch ** 0.25)
                    optimizer.noise_multiplier = new_noise_multiplier
                    
                    print(f"Epoch {epoch}: Updated dynamic sigma to {new_noise_multiplier:.4f}")
            
                elif not args.disable_dp and dp_mode == 'RP':  # RP DP-SGD
                    optimizer.noise_multiplier = args.sigma
                    optimizer.red_rate = args.red_rate
                
                elif not args.disable_dp and dp_mode == 'd2p2':  # D2P2 DP-SGD
                    new_noise_multiplier = args.sigma / (epoch ** 0.25)
                    optimizer.noise_multiplier = new_noise_multiplier
                    optimizer.red_rate = args.red_rate

                    print(f"Epoch {epoch}: Updated d2p2 sigma to {new_noise_multiplier:.4f}")
                
                losses,sigma, epsilon = train(args, model, device, train_loader, optimizer,dp_mode, privacy_engine, epoch)
                top1_acc = test(model, device, test_loader)
                train_loss = np.mean(losses)
                accuracy_per_epoch.append(float(top1_acc))
                train_losses.append(train_loss)
                epsilon_per_epoch.append(epsilon)
                sigma_per_epoch.append(sigma)
            # run_results.append(top1_acc)
            train_results[dp_label] = {
                'loss': train_losses,
                'acc': accuracy_per_epoch,
                'ep': epsilon_per_epoch,
                'sigma':sigma_per_epoch
            }
            print(train_results)
        
    plot_combined_results(train_results, args.sigma, args.batch_size, args.red_rate, args.seed)

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=512,
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
        default=50,
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
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cuda" if torch.cuda.is_available() else "cpu",
    #     help="GPU ID for this process",
    # )
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
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument(
        "--red_rate",
        type=float,
        default=0.3,
        help="random proj rate",
    )

    return parser.parse_args()

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
