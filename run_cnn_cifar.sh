#!/bin/bash
# Terminate child processes of the current script's process ID if the shell script itself is terminated
# If you don't have permission to execute the shell script use the following line in the terminal
#chmod +x run_cnn_cifar.sh
function kill_background_processes() {
    pkill -P $$
}
trap "kill_background_processes; exit 1" SIGINT SIGTERM

if [ ! -d "runs/train" ]; then
    mkdir -p runs/train
fi
# Define arrays of sigma values, batch sizes, and seeds
sigmas=(1.0 1.5 2.0 4.0 6.0) 
red_rates=(0.1 0.3 0.5 0.7 0.9)  #
batch_sizes=(128 256 512 1024) # 
seeds=(0 1 2 3 4) # 123 456
dp_modes=("None" "static" "dynamic" "RP" "d2p2")
i=0

epochs=50
# checkpoint_base="/home/zahid/work/d2p2/d2p2sgd/ckpt/CNN_cifar"
# log_base="/home/zahid/work/d2p2/d2p2sgd/log/CNN_cifar"

# Set the maximum number of concurrent scripts
num_parallel=20
run_python=true
# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sigmas)
            sigmas=($2)
            shift 2
            ;;
        --red_rates)
            red_rates=($2)
            shift 2
            ;;
        --batch_sizes)
            batch_sizes=($2)
            shift 2
            ;;
        --seeds)
            seeds=($2)
            shift 2
            ;;
        --epochs)
            epochs=$2
            shift 2
            ;;
        --num_parallel)
            num_parallel=$2
            shift 2
            ;;
        --run_python)
            run_python=$2
            shift 2
            ;;
        --dp_modes)
            dp_modes=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print processed input arguments
echo "Input Arguments:"
echo "sigmas: ${sigmas[@]}"
echo "red_rates: ${red_rates[@]}"
echo "batch_sizes: ${batch_sizes[@]}"
echo "epochs: ${epochs[@]}"
echo "seed: ${seeds[@]}"
echo "num_parallel: $num_parallel"

for sigma in "${sigmas[@]}"
do
    for red_rate in "${red_rates[@]}" 
    do
        for batch_size in "${batch_sizes[@]}"
        do
            for seed in "${seeds[@]}"
            do
                # checkpoint_file="${checkpoint_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"
                # log_dir="${log_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"

                echo "---------------------------------------------------------"
                echo "  Sigma: $sigma, Batch size: $batch_size, Seed: $seed Reduction Rate: $red_rate DO Mode: $dp_modes"
                echo "---------------------------------------------------------"

                log_file="runs/train/log_sigma_${sigma}_batch_${batch_size}_seed_${seed}_rr_${red_rate}.txt"

                if [ "$run_python" = true ]; then
                python -u -m cifar10 --epochs="$epochs" --sigma="$sigma"  --red_rate="$red_rate" --local_rank=-1 --device gpu --batch-size="$batch_size" --workers=16 --seed "$seed" --dp_modes ${dp_modes[@]} 2>&1 | tee "$log_file" &
                fi
                
                ((i++))
                # Check if the maximum number of concurrent scripts has been reached
                if [ $i -ge $num_parallel ]; then
                    # Wait for the background processes to finish
                    echo "Wait for the background processes to finish, check progress with the text file outputs"
                    wait

                    # Reset the counter
                    i=0
                fi

                # Uncomment the following line if you want to pause between each batch of parallel executions
                # sleep 10s
            done
        done
    done
done

# Wait for any remaining background processes to finish
wait

