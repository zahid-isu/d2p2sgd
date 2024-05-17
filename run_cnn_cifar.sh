#!/bin/bash

# Define arrays of sigma values, batch sizes, and seeds
sigmas=(1.0)  #2.0 4.0 6.0 8.0 10.0
batch_sizes=(512) # 128 256
seeds=(12) # 123 456

epochs=2
checkpoint_base="/home/zahid/work/d2p2/d2p2sgd/ckpt/CNN_cifar"
log_base="/home/zahid/work/d2p2/d2p2sgd/log/CNN_cifar"

for sigma in "${sigmas[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do
            checkpoint_file="${checkpoint_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"
            log_dir="${log_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"

            echo "---------------------------------------------------------"
            echo "  Sigma: $sigma, Batch size: $batch_size, Seed: $seed"
            echo "---------------------------------------------------------"

            python cifar10.py \
                --epochs ${epochs} \
                --sigma ${sigma} \
                --checkpoint-file "${checkpoint_file}" \
                --log-dir "${log_dir}" \
                --local_rank -1 \
                --device gpu \
                --batch-size ${batch_size} \
                --workers 1 \
                --seed ${seed}
        

            # sleep 10s
        done
    done
done
