#!/bin/bash

CONFIG_PATH="$(pwd)/configs/eneryield"
SCRIPT_PATH="$(pwd)/scripts/launch_finetuning_wandb_hp_sweep.py"

# List of config names (suffixes)
names=("interruption_in_seq" "interruption_next_week")

for name in "${names[@]}"; do
    CONFIG_NAME="FT_hp_sweep_$name"
    echo "Running sweep with config: $CONFIG_NAME"

    python "$SCRIPT_PATH" --config-path="$CONFIG_PATH" --config-name="$CONFIG_NAME" || exit 1
done