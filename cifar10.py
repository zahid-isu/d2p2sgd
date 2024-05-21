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
Runs CIFAR10 training with differential privacy.
"""
import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
import time
import json

import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
from privacy_engine import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.functorch import make_functional
from torch.func import grad_and_value, vmap
# from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.profiler as profiler
from optimizer import DPOptimizer



logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("ddp")
logger.setLevel(level=logging.INFO)


def plot_combined_results(train_results, sigma, batch_size, seed, red_rate, epochs):

    epochs = range(1, int(len(next(iter(train_results.values()))[0])) + 1)
    fig, axs = plt.subplots(2, figsize=(10, 10), dpi=400)

    # Plot training losses
    for optim, (train_losses, accuracy_per_epoch, epsilons, train_duration) in train_results.items():
        epochs = range(1, len(train_losses) + 1)
        axs[0].plot(epochs, train_losses, label=f'{optim}', linewidth=2)
    
    axs[0].set_xlabel('Epoch', fontsize=16)
    axs[0].set_ylabel('Training Loss', fontsize=16)
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].legend(loc='upper right', fontsize=12)
    
    # Plot testing accuracies
    for optim, (train_losses, accuracy_per_epoch, epsilons, train_duration) in train_results.items():
        epochs = range(1, len(accuracy_per_epoch) + 1)
        axs[1].plot(epochs, accuracy_per_epoch, label=f'{optim}', linewidth=2)
    
    axs[1].set_xlabel('Epoch', fontsize=16)
    axs[1].set_ylabel('Testing Accuracy', fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_dir = os.path.join('log','CNN_cifar', f'sigma_{sigma}', f'batch_{batch_size}', f'rr_{red_rate}',f'epo_{epochs}', f'seed_{seed}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # filename = f'log/CNN_cifar/{current_time}_sigma_{sigma}_batch_{batch_size}_seed_{seed}_red_rate_{red_rate}'

    # fig.suptitle(f'CNN_CIFAR10_sigma_{sigma}_batch_{batch_size}_red_rate_{red_rate}', fontsize=16)
    file_name = "plot.png"
    plt.savefig(save_dir, "plot.png")

    total_times_to_reach_accuracy = {}
    target_accuracy=0.35

    for dp_label, (train_losses, accuracy_per_epoch, epsilon, time_per_epoch) in train_results.items():
        total_time = 0
        for acc, epoch_time in zip(accuracy_per_epoch, time_per_epoch):
            if acc >= target_accuracy:
                total_times_to_reach_accuracy[dp_label] = total_time
                break
            total_time += epoch_time
        else:
            total_times_to_reach_accuracy[dp_label] = "t > train_time"
    json_file_path = os.path.join(save_dir, "time_comp.json")
    with open(json_file_path, "w") as files:
        json.dump(total_times_to_reach_accuracy, files, indent=4)
    json_file_path = os.path.join(save_dir, "acc_loss.json")
    with open(json_file_path, "w") as file:
        json.dump(train_results, file, indent=4)
    

def setup(args):
    if not torch.cuda.is_available():
        raise NotImplementedError(
            "DistributedDataParallel device_ids and output_device arguments \
            only work with single-device GPU modules"
        )

    if sys.platform == "win32":
        raise NotImplementedError("Windows version of multi-GPU is not supported yet.")

    # Initialize the process group on a Slurm cluster
    if os.environ.get("SLURM_NTASKS") is not None:
        rank = int(os.environ.get("SLURM_PROCID"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        world_size = int(os.environ.get("SLURM_NTASKS"))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "7440"

        torch.distributed.init_process_group(
            args.dist_backend, rank=rank, world_size=world_size
        )

        logger.debug(
            f"Setup on Slurm: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        return (rank, local_rank, world_size)

    # Initialize the process group through the environment variables
    elif args.local_rank >= 0:
        torch.distributed.init_process_group(
            init_method="env://",
            backend=args.dist_backend,
        )
        rank = torch.distributed.get_rank()
        local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()

        logger.debug(
            f"Setup with 'env://': rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        return (rank, local_rank, world_size)

    else:
        logger.debug(f"Running on a single GPU.")
        return (0, 0, 1)


def cleanup():
    torch.distributed.destroy_process_group()


def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(64, num_classes, bias=True),
    )


def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    start_time = time.time()

    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    if args.grad_sample_mode == "no_op":
        # Functorch prepare
        fmodel, _fparams = make_functional(model)

        def compute_loss_stateless_model(params, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = fmodel(params, batch)
            loss = criterion(predictions, targets)
            return loss

        ft_compute_grad = grad_and_value(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        # Using model.parameters() instead of fparams
        # as fparams seems to not point to the dynamically updated parameters
        params = list(model.parameters())
        
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            if args.grad_sample_mode == "no_op":
                per_sample_grads, per_sample_losses = ft_compute_sample_grad(
                    params, images, target
                )
                per_sample_grads = [g.detach() for g in per_sample_grads]
                loss = torch.mean(per_sample_losses)
                for p, g in zip(params, per_sample_grads):
                    p.grad_sample = g
            else:
                loss = criterion(output, target)
                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                # measure accuracy and record loss
                acc1 = accuracy(preds, labels)
                top1_acc.append(acc1)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

            # make sure we take a step after processing the last mini-batch in the
            # epoch to ensure we start the next epoch with a clean state
            
            

            
            if i % args.print_freq == 0:
                if not args.disable_dp:
                    epsilon =0 
                    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                    pbar.write(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc):.6f} "
                        f"(ε = {epsilon:.5f}, δ = {args.delta})"
                    )
                else:
                    epsilon =0 
                    pbar.write(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc):.6f} "
                    )
            pbar.update()

    train_duration = time.time()-start_time
    return losses, train_duration, epsilon


def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad(), tqdm(total=len(test_loader), desc="Testing") as pbar:
        for images, target in (test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)
            pbar.update(1)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)

# def get_time_to_reach_accuracy(train_results, target_accuracy=0.97):
#     total_times_to_reach_accuracy = {}

#     for dp_label, (train_losses, accuracy_per_epoch, epsilon, time_per_epoch) in train_results.items():
#         total_time = 0
#         for acc, epoch_time in zip(accuracy_per_epoch, time_per_epoch):
#             if acc >= target_accuracy:
#                 total_times_to_reach_accuracy[dp_label] = total_time
#                 break
#             total_time += epoch_time
#         else:
#             total_times_to_reach_accuracy[dp_label] = "t > train_time"
    
#     return total_times_to_reach_accuracy


# flake8: noqa: C901
def main():
    args = parse_args()
    if args.debug >= 1:
        logger.setLevel(level=logging.DEBUG)

    train_results = {}  #store training results

    # for dp_mode in [ None, 'static', 'dynamic', 'RP', 'd2p2']:  # [SGD, DP-SGD, D2P-SGD, DP2-SGD]
    dp_modes = [None if mode == "None" else mode for mode in args.dp_modes]
    print(args.dp_modes)
    for dp_mode in dp_modes:
        args.disable_dp = (dp_mode is None)
        dp_label = 'SGD' if dp_mode is None else f'DP-SGD ({dp_mode})'

        random_projection = False
        if dp_mode == 'RP' or dp_mode == 'd2p2':
            random_projection = True 
        print("random_projection=", random_projection)

        # Sets `world_size = 1` if you run on a single GPU with `args.local_rank = -1`
        if args.local_rank != -1:
            rank, local_rank, world_size = setup(args)
            device = 0
        else:
            # device = "cpu"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rank = 0
            world_size = 1

        print(f"----------- Training mode: {dp_label}, Device: {device}, args.disable_dp:", args.disable_dp)

        if args.secure_rng:
            try:
                import torchcsprng as prng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            generator = prng.create_random_device_generator("/dev/urandom")

        else:
            generator = None

        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        train_transform = transforms.Compose(
            augmentations + normalize if args.disable_dp else normalize
        )

        test_transform = transforms.Compose(normalize)

        train_dataset = CIFAR10(
            root=args.data_root, train=True, download=True, transform=train_transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            generator=generator,
            num_workers=args.workers,
            pin_memory=True,
        )

        test_dataset = CIFAR10(
            root=args.data_root, train=False, download=True, transform=test_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=args.workers,
        )

        best_acc1 = 0

        model = convnet(num_classes=10)
        model = model.to(device)

        # Use the right distributed module wrapper if distributed training is enabled
        if world_size > 1:
            if not args.disable_dp:
                if args.clip_per_layer:
                    model = DDP(model, device_ids=[device])
                else:
                    model = DPDDP(model)
            else:
                model = DDP(model, device_ids=[device])

        if args.optim == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        elif args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError("Optimizer not recognized. Please check spelling")

        privacy_engine = None
        if not args.disable_dp: # DP-SGD both static and dynamic
            if args.clip_per_layer:
                # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.
                n_layers = len([(n, p) for n, p in model.named_parameters() if p.requires_grad])
                max_grad_norm = [args.max_per_sample_grad_norm / np.sqrt(n_layers)] * n_layers
            else:
                max_grad_norm = args.max_per_sample_grad_norm

            privacy_engine = PrivacyEngine(
                secure_mode=args.secure_rng,red_rate=args.red_rate,
            )
            clipping = "per_layer" if args.clip_per_layer else "flat"

            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma, # starting sigma for DP-dynamic = args.sigma/1
                max_grad_norm=max_grad_norm,
                clipping=clipping,
                grad_sample_mode=args.grad_sample_mode,
                random_projection=random_projection,
                seed=args.seed,
            )

        # Store logs
        accuracy_per_epoch = []
        time_per_epoch = []
        train_losses =[]
        epsilon_per_epoch = []

        for epoch in range(args.start_epoch, args.epochs + 1):  #training loop
            if args.lr_schedule == "cos":
                lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            
            if not args.disable_dp and dp_mode == 'dynamic':  # dynamic DP-SGD
                new_noise_multiplier = args.sigma / (epoch ** 0.25)
                optimizer.noise_multiplier = new_noise_multiplier
                

                print(f"Epoch {epoch}: Updated dynamic sigma to {new_noise_multiplier:.4f}")
            
            elif not args.disable_dp and dp_mode == 'RP':  # RP DP-SGD
                optimizer.noise_multiplier = args.sigma
            
            elif not args.disable_dp and dp_mode == 'd2p2':  # D2P2 DP-SGD
                new_noise_multiplier = args.sigma / (epoch ** 0.25)
                optimizer.noise_multiplier = new_noise_multiplier

                print(f"Epoch {epoch}: Updated d2p2 sigma to {new_noise_multiplier:.4f}")
            
            # else:  # static DP-SGD
            #     privacy_engine.noise_multiplier = args.sigma

            losses, train_duration, epsilon = train(args, model, train_loader, optimizer, privacy_engine, epoch, device)
            top1_acc = test(args, model, test_loader, device)
            train_loss = np.mean(losses)
            train_losses.append(train_loss)

            # remember best acc@1 and save checkpoint
            is_best = top1_acc > best_acc1
            best_acc1 = max(top1_acc, best_acc1)

            time_per_epoch.append(train_duration)
            accuracy_per_epoch.append(float(top1_acc))
            epsilon_per_epoch.append(epsilon)

            # save_checkpoint(
            #     {
            #         "epoch": epoch + 1,
            #         "arch": "Convnet",
            #         "state_dict": model.state_dict(),
            #         "best_acc1": best_acc1,
            #         "optimizer": optimizer.state_dict(),
            #     },
            #     is_best,
            #     filename=args.checkpoint_file + ".tar",
            # )

        train_results[dp_label] = (train_losses, accuracy_per_epoch, epsilon_per_epoch, time_per_epoch)
        print(train_results)
    
    plot_combined_results(train_results, args.sigma, args.batch_size, args.seed, args.red_rate, args.epochs)

    # if rank == 0:
    #     time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
    #     avg_time_per_epoch = sum(time_per_epoch_seconds) / len(time_per_epoch_seconds)
    #     metrics = {
    #         "accuracy": best_acc1,
    #         "accuracy_per_epoch": accuracy_per_epoch,
    #         "avg_time_per_epoch_str": str(timedelta(seconds=int(avg_time_per_epoch))),
    #         "time_per_epoch": time_per_epoch_seconds,
    #     }

    #     logger.info(
    #         "\nNote:\n- 'total_time' includes the data loading time, training time and testing time.\n- 'time_per_epoch' measures the training time only.\n"
    #     )
    #     logger.info(metrics)

    # if world_size > 1:
    #     cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument("--grad_sample_mode", type=str, default="hooks")
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )

    parser.add_argument(
        "--rp",
        action="store_true",
        default=False,
        help="Random projection for DP",
    )

    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size-test",
        default=512,
        type=int,
        metavar="N",
        help="mini-batch size for test dataset (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--batch-size",
        default=512,
        type=int,
        metavar="N",
        help="approximate bacth size",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
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
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
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
        help="Enable Secure RNG to have trustworthy privacy guarantees."
        "Comes at a performance cost. Opacus will emit a warning if secure rng is off,"
        "indicating that for production use it's recommender to turn it on.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--red_rate",
        type=float,
        default=0.3,
        help="random proj rate",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="path to save check points",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/stat/tensorboard",
        help="Where Tensorboard log will be stored",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device on which to run the code."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank if multi-GPU training, -1 for single GPU training. Will be overriden by the environment variables if running on a Slurm cluster.",
    )

    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="Choose the backend for torch distributed from: gloo, nccl, mpi",
    )

    parser.add_argument(
        "--clip_per_layer",
        action="store_true",
        default=False,
        help="Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="debug level (default: 0)",
    )
    parser.add_argument('--dp_modes', nargs='+', help='DP mode', default=["None", "static", "dynamic", "RP", "d2p2"])


    return parser.parse_args()


if __name__ == "__main__":
    main()
