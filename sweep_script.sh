#!/bin/bash

CONFIG_PATH="$(pwd)/configs/eneryield/ft_sweep"
SCRIPT_PATH="$(pwd)/scripts/launch_finetuning_wandb_hp_sweep.py"

# List of config names (suffixes)
names=("interruption_in_seq" "interruption_5_day" "interruption_7_day" "interruption_1_day")

for name in "${names[@]}"; do
    CONFIG_NAME="FT_hp_sweep_$name"
    echo "Running sweep with config: $CONFIG_NAME"

    # Run the script with a 3-hour timeout
    timeout 3h python "$SCRIPT_PATH" --config-path="$CONFIG_PATH" --config-name="$CONFIG_NAME"
    
    # Check exit status
    status=$?
    if [ $status -eq 124 ]; then
        echo "Process timed out for config: $CONFIG_NAME"
    elif [ $status -ne 0 ]; then
        echo "Process failed for config: $CONFIG_NAME with exit code $status"
        exit 1
    fi
done
