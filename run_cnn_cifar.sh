#!/bin/bash

# Define arrays of sigma values, batch sizes, and seeds
sigmas=(0.5 1.0 1.5 2.0)
batch_sizes=(128 256 512)
seeds=(42 123 456)

epochs=50
checkpoint_base="/home/zahid/work/d2p2sgd/ckpt/CNN_cifar"
log_base="/home/zahid/work/d2p2sgd/log/CNN_cifar"

for sigma in "${sigmas[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do
            checkpoint_file="${checkpoint_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"
            log_dir="${log_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"

            echo "Running with sigma: $sigma, batch size: $batch_size, seed: $seed"

            python opacus/examples/cifar10.py \
                --epochs ${epochs} \
                --sigma ${sigma} \
                --checkpoint-file "${checkpoint_file}" \
                --log-dir "${log_dir}" \
                --local_rank -1 \
                --device gpu \
                --batch-size ${batch_size} \
                --seed ${seed}

            # sleep 10s
        done
    done
done
