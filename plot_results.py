import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json


# Command formatting is a little different than for the shell script, an example is below
# python plot_results.py --nums_models 10 20 --data_dists iid --datasets "cifar100" --archs resnet20 --nums_epochs 2 --matching_algs nonseq_match_permute --merg_itrs 30 --name num_models

# Add argparse for input arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--root_dir', type=str, default='log')
parser.add_argument('--sigmas', nargs='+', type=int, default=[1.0, 1.5, 2.0, 4.0])
parser.add_argument('--red_rates', nargs='+', type=int, default=[0.3, 0.5, 0.7])
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_sizes', nargs='+', type=int, default=[128, 256, 512, 1024])
parser.add_argument('--datasets', nargs='+', type=str, default=["cifar10"])
parser.add_argument('--dp_modes', nargs='+', type=str, default=["None", "static", "dynamic", "RP", "d2p2"])
parser.add_argument('--name', type=str, default='test')
args = parser.parse_args()

# Extract input variables from argparse
dp_modes= args.dp_modes
root_dir = args.root_dir
sigmas = args.sigmas
datasets = args.datasets
red_rates = args.red_rates
batch_sizes = args.batch_sizes
epochs=args.epochs
plot_name = args.name
# Define the single-value variables based on input arguments
single_value_vars = [arg for arg in vars(args) if isinstance(getattr(args, arg), list) and len(getattr(args, arg)) == 1]
#print(single_value_vars)

# Mapping for exp and matching_alg names
dp_modes_names = {
    "None": "SGD",
    "static": "DPSGD",
    "dynamic": "D2P-SGD",
    "RP": "DP2-SGD",
    "D2P2": "D2P2-SGD"
}

dataset_names = {'cifar10': "CIFAR-10", 'cifar100': "CIFAR-100", 'tinyimgnt': 'Tiny ImageNet'}

configuration = {
    'DP Mode': [dp_modes_names[dp_mode] for dp_mode in dp_modes],
    'Dataset': [dataset_names[dataset] for dataset in datasets],
    'Sigma': sigmas,
    'Reduction Rate': red_rates,
    'Batch Size': batch_sizes,
    'Epochs': epochs
}

# Create initial part of the title
title_parts = []
for key, val in configuration.items():
    if len(val) == 1:
        if isinstance(val[0], int):
            title_parts.append(f"{val[0]} {key}")
        else:
            title_parts.append(f"{val[0]}")
shared_title = ", ".join(title_parts)


# Figure output dir
figures_dir = os.path.join(root_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Function to collect result CSV files
def collect_results(root_dir, sigma, batch_size, seed, red_rate, epochs):
    file_path = os.path.join('log','CNN_cifar', f'sigma_{sigma}', f'batch_{batch_size}', f'rr_{red_rate}',f'epo_{epochs}')

    #print(file_path)
    file_list = glob.glob(os.path.join(file_path, 'seed_*', '*.json'))
    return file_list


# Function to process data
def process_data(file_list):
    all_accuracies = []
    all_losses = []
    for file in file_list:
        df = pd.read_csv(file)
        # print(df)
        last_row = df.tail(1)  # Get the last row of the DataFrame
        # print(last_row['All Classes Accuracy'].values[-1])
        all_accuracies.append(last_row['All Classes Accuracy'].values[-1])
        all_losses.append(last_row['All Classes Loss'].values[-1])
    # print('accuracies', all_accuracies)
    # print('losses', all_losses)
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    return mean_accuracy, std_accuracy, mean_loss, std_loss


def plot_data(average_data, std_data, label, ylabel, x_values, plot_init, average_init_data, std_init_data, color):
    # Add more matching_algs here
    
    plt.plot(x_values, average_data, '-', label=label, color=color)
    plt.fill_between(x_values, average_data - std_data, average_data + std_data, alpha=0.5, color=color)
    plt.xlabel('Number of Iterations', fontsize=15)
    if ylabel == "Loss":
        plt.ylabel(f"{ylabel}", fontsize=15)
    else:
        plt.ylabel(f"{ylabel} (%)", fontsize=15)
    # plt.title(f'{shared_title}: Average {ylabel}')
    #plt.title(f'{title_topology}: Average {ylabel}', fontsize=16)
    print(f"{ylabel}, {label}: ",average_data[-1], std_data[-1])
    if plot_init:
        plt.axhline(y=average_init_data, linestyle='--', label=f"Initial {ylabel}", color = 'tab:blue')
        plt.fill_between(x_values, average_init_data - std_init_data, average_init_data + std_init_data, alpha=0.5, color = 'tab:orange')
    # Set custom x-axis tick values
    custom_xticks = np.arange(0, x_values[-1]+1, 10)  # Replace with your desired tick values
    plt.xticks(custom_xticks)
    # Replace with your desired tick values
    if ylabel == "Loss":
        plt.yscale('log')
        custom_yticks = np.logspace(0, 2, num=3)
    else:
        plt.yscale('linear')
        custom_yticks = np.arange(0, 101, 10)    
    plt.yticks(custom_yticks)
    plt.legend(fontsize=15)
    # plt.show()


all_accuracies = []
all_losses = []
labels = []
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# colors = ['#004488', '#DDAA33', '#BB5566', '#000000', '#999933', '#332288', '#88CCEE']
# colors = ['#004488', '#BB5566', '#DDAA33', '#88CCEE', '#CC6677', '#882255', '#AA4499']
colors = ['#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99']
colors_idx = []
label_idx = 0
# Iterate through input combinations
for dataset in datasets:
    for sigma in sigmas:
        for wsize in nums_models:
            for data_dist in data_dists:
                current_init_accuracies = []
                current_init_losses = []
                if init_path == 'trianedinit':
                    init_file_lists = [collect_init_results(root_init_dir, dataset, arch, init_model, wsize, data_dist, init_epochs)]
                    #print(init_file_lists)
                    for init_file_list in init_file_lists:
                        avg_acc, std_acc, avg_loss, std_loss = process_init_data(init_file_list)
                        current_init_accuracies.append(avg_acc)
                        current_init_losses.append(avg_loss)
                        current_init_accuracies.append(std_acc)
                        current_init_losses.append(std_loss)
                    all_init_accuracies.append((current_init_accuracies))
                    all_init_losses.append((current_init_losses))
                else:
                    plot_init = None
                    current_init_accuracies.append(0)
                    current_init_losses.append(None)
                    current_init_accuracies.append(0)
                    current_init_losses.append(None)
                    all_init_accuracies.append((current_init_accuracies))
                    all_init_losses.append((current_init_losses))
                for num_epochs in num_epochs_list:
                    for matching_alg in matching_algs_list:
                        for exp in sigmas:
                            current_accuracies = []
                            current_losses = []
                            for merg_i in merg_itrs_list:
                                # Collect result CSV files
                                file_lists = [collect_results(root_dir, init_path, dataset, arch, f"with_training_epo{num_epochs}", wsize, data_dist, matching_alg, merg_i, exp)]

                                # print(file_lists)
                                # Process data
                                for file_list in file_lists:
                                    avg_acc, std_acc, avg_loss, std_loss = process_data(file_list)
                                    current_accuracies.append((avg_acc, std_acc))
                                    current_losses.append((avg_loss, std_loss))
                            all_accuracies.append((current_accuracies))
                            all_losses.append((current_losses))
                            if exp in exp_names:
                                topo = exp_names[exp]
                            else:
                                raise ValueError("Invalid value for 'exp': {}".format(exp))
                            if matching_alg in matching_alg_names:
                                matching_alg_name = matching_alg_names[matching_alg]
                                color_idx = list(matching_alg_names.keys()).index(matching_alg)
                            # Update label for specific experiment being plotted
                            legend_label_parts = []
                            for key, val in configuration.items():
                                if len(val) > 1:
                                    # print(val)
                                    if isinstance(val[label_idx], int):
                                        legend_label_parts.append(f"{val[label_idx]} {key}")
                                        # color_idx = list(matching_alg_names.keys()).index(val)
                                    else:
                                        legend_label_parts.append(f"{val[label_idx]}")
                                        # color_idx = list(matching_alg_names.keys()).index(val)

                            legend_label = ", ".join(legend_label_parts)
                            labels.append(legend_label)


                            # legend_label = ", ".join(f"{key}: {configuration[key][label_idx]}" for key in configuration if len(configuration[key]) > 1)
                            # # labels.append(f'{wsize} Models, {matching_alg_name}, {topo}')
                            # labels.append(legend_label)
                    
                            colors_idx.append(color_idx)
                            label_idx += 1
                            # print(label_idx)

#print('accuracies', all_accuracies)
#print('losses', all_losses)

all_accuracies = np.array(all_accuracies)
#print(all_accuracies)
all_losses = np.array(all_losses)

if len(matching_algs_list) == 1:
    colors_idx = list(range(len(labels)))


# Plot data for accuracy
plt.figure(figsize=(8, 5), dpi=1200)
for i, label in enumerate(labels):
    current_accuracies = np.array(all_accuracies[i])
    # print()
    # print(colors_idx[i])
    # print()
    plot_data(current_accuracies[:, 0], current_accuracies[:, 1], label, 'Accuracy', merg_itrs_list, plot_init, current_init_accuracies[0], current_init_accuracies[1], colors[colors_idx[i]])
    plot_init = None

# Replace spaces with underscores and remove special characters
file_name =  "_".join(title_parts)
#print("file_name",file_name)
#print("title_parts",title_parts)
file_name = file_name.replace(" ", "_")
#file_name = ''.join(char for char in file_name if char.isalnum() or char == ',')
file_name = file_name.replace("_Agents", "M")
file_name = file_name.replace("_Epochs", "E")
file_name = file_name.replace("Pre-Trained", "PT")
file_name = file_name.replace("Random Initialization", "RI")
file_name = file_name.replace("Same Random Initialization", "SRI")
file_name = file_name.replace("Fully_Connected", "FC")
file_name = file_name.replace("ResNet", "RN")
file_name = file_name.replace("VGG", "V")
file_name = file_name.replace("CIFAR", "Ci")
print(file_name)
plt.savefig(figures_dir+"/"+file_name+"_acc.png", dpi=300)
plt.show()


if init_path == 'trianedinit':
    plot_init = args.plot_init
# Plot data for loss
plt.figure(figsize=(8, 5), dpi=1200)
for i, label in enumerate(labels):
    current_losses = np.array(all_losses[i])
    plot_data(current_losses[:, 0], current_losses[:, 1], label, 'Loss', merg_itrs_list, plot_init, current_init_losses[0], current_init_losses[1], colors[colors_idx[i]])
    plot_init = None
        #plot_data(current_init_losses[0], current_init_losses[1], "Initial Loss", 'Loss', merg_itrs_list)
plt.yscale('log')
custom_yticks = np.logspace(0, 2, num=3)

min_val = np.nanmin([item[0] if isinstance(item, np.ndarray) else item for sublist in all_losses for item in sublist])
# print(min_val)
max_val = np.nanmax([item[0] if isinstance(item, np.ndarray) else item for sublist in all_losses for item in sublist])
# print(max_val)

# Calculate the range of exponents
if min_val == 0:
    min_exponent = 0
else:
    min_exponent = np.floor(np.log10(min_val))
# print(min_exponent)
max_exponent = np.ceil(np.log10(max_val))
# print(max_exponent)

# Have min/max exponents
if min_exponent > 0:
    min_exponent = 0
if max_exponent < 2:
    max_exponent = 2

# Calculate the number of ticks based on the range
num_ticks = int(max_exponent - min_exponent) + 1

# Generate equally spaced ticks
custom_yticks = np.logspace(min_exponent, max_exponent, num=num_ticks)

# Set custom yticks
plt.yticks(custom_yticks) 
plt.savefig(figures_dir+"/"+file_name+"_loss.png")
plt.show()

print("shared_title",shared_title)
