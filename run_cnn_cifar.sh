#!/bin/bash
# Terminate child processes of the current script's process ID if the shell script itself is terminated
# If you don't have permission to execute the shell script use the following line in the terminal
#chmod +x run_cnn_cifar.sh
function kill_background_processes() {
    pkill -P $$
}
trap "kill_background_processes; exit 1" SIGINT SIGTERM

# Define arrays of sigma values, batch sizes, and seeds
sigmas=(1.0 2.0)  #2.0 4.0 6.0 8.0 10.0
batch_sizes=(1024) # 128 256
seeds=(33) # 123 456


epochs=50

for sigma in "${sigmas[@]}"s
do
    for batch_size in "${batch_sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do

            for red_rate in "${red_rates[@]}"
            do
                # checkpoint_file="${checkpoint_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"
                # log_dir="${log_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"

                echo "---------------------------------------------------------"
                echo "  Sigma: $sigma, Batch size: $batch_size, Seed: $seed", 
                echo "---------------------------------------------------------"

            python cifar10.py \
                --epochs ${epochs} \
                --sigma ${sigma} \
                # --checkpoint-file "${checkpoint_file}" \
                # --log-dir "${log_dir}" \
                --local_rank -1 \
                --device gpu \
                --batch-size ${batch_size} \
                --workers 16 \
                --seed ${seed}
        

            # sleep 10s
        done
    done
done

wait

