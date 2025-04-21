#!/bin/bash

# filepath: /home/filip-marcus/run_finetuning.sh

# Define an array of configuration file paths
CONFIG_FILES=(
    "finetune_interruption_1_day2"
    "finetune_interruption_3_day2"
    "finetune_interruption_5_day2"
    "finetune_interruption_7_day2"
    "finetune_event_label2"
    "finetune_class_dist2"
    "finetune_interruption_in_seq2"
    "finetune_interruption_next_week2"
    # Add more config files as needed
)

# Loop through each configuration file and run the finetuning process
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "Starting finetuning with config: $CONFIG_FILE"
    
    # Run the finetuning command (replace with the actual command for your setup)
    python /home/filip-marcus/ESGPT_new/EventStreamGPT/scripts/finetune.py --config-path=/home/filip-marcus/ESGPT_new/EventStreamGPT/configs/eneryield --config-name=$CONFIG_FILE
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Finetuning failed for config: $CONFIG_FILE"
        exit 1
    fi
    
    echo "Finished finetuning with config: $CONFIG_FILE"
done

echo "All finetuning runs completed successfully."