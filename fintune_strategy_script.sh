#!/bin/bash

# Define the names of the configurations
names=("event_label" "class_dist" "interruption_in_seq")
# names=("event_label2")
# Define interruption x day runs
interruption_x_day_runs=("interruption_3_day" "interruption_5_day")
#interruption_x_day_runs=("interruption_1_day2" "interruption_3_day2")
# Define the additional config to run after each order
final_config="interruption_7_day"

# Base directory for model and weights
base_model_dir="$(pwd)/pretrain/eneryield1"
base_weights_fp="$base_model_dir/pretrained_weights"

# Temporary file to store the updated config
temp_config_dir="$(pwd)/configs/eneryield/finetune/temp_configs"
mkdir -p $temp_config_dir

# Log file to track success and failure
log_file="$(pwd)/finetune_run_log.txt"
echo "Finetune Run Log - $(date)" > $log_file

# Generate all unique permutations of the names array
generate_permutations() {
    local arr=("$@")
    local n=${#arr[@]}
    if (( n == 1 )); then
        echo "${arr[0]}"
        return
    fi

    for i in "${!arr[@]}"; do
        local current="${arr[i]}"
        local rest=("${arr[@]:0:$i}" "${arr[@]:$((i + 1))}")
        while IFS= read -r perm; do
            echo "$current $perm"
        done < <(generate_permutations "${rest[@]}")
    done
}

# Generate permutations
permutations=$(generate_permutations "${names[@]}" | sort | uniq)
echo "Generated permutations: $permutations" >> $log_file

# Track completed configurations for each permutation
declare -A completed_in_order

# Iterate over each permutation and run the script
echo "$permutations" | while read -r permutation; do
    echo "Running finetune.py with configs: $permutation"
    echo "Permutation: $permutation" >> $log_file
    current_model_dir="$base_model_dir"
    current_weights_fp="$base_weights_fp"
    success=true
    is_first_run=true

    for config in $permutation; do
        # Check if this configuration has already been completed in the current order
        if [[ ${completed_in_order["$current_model_dir:$config"]} ]]; then
            echo "Skipping already completed config in order: $config" >> $log_file
            current_model_dir="$current_model_dir/finetuning/task_df_eneryield_${config}"
            current_weights_fp="$current_model_dir/finetune_weights"
            continue
        fi

        # Update the config file with the current model and weights paths
        original_config="$(pwd)/configs/eneryield/finetune/finetune_$config.yaml"
        temp_config="$temp_config_dir/finetune_$config.yaml"

        # Set the "strategy" variable dynamically
        if $is_first_run; then
            grep -v "strategy:" $original_config > $temp_config
            is_first_run=false
        else
            sed "s|strategy:.*|strategy: true|" $original_config > $temp_config
        fi

        sed -i "s|load_from_model_dir:.*|load_from_model_dir: $current_model_dir|" $temp_config
        sed -i "s|pretrained_weights_fp:.*|pretrained_weights_fp: $current_weights_fp|" $temp_config

        # Log the current model and weights paths
        echo "Running config: $config" >> $log_file
        echo "  current_model_dir: $current_model_dir" >> $log_file
        echo "  current_weights_fp: $current_weights_fp" >> $log_file

        # Run the script with the updated config
        python $(pwd)/scripts/finetune.py --config-path=$(pwd)/configs/eneryield/finetune/temp_configs --config-name=finetune_$config
        if [ $? -ne 0 ]; then
            echo "Error: finetune.py failed for config $config. Terminating this permutation."
            echo "Config $config: FAILED" >> $log_file
            success=false
            break
        else
            echo "Config $config: SUCCESS" >> $log_file
            completed_in_order["$current_model_dir:$config"]=1
        fi

        # Update the paths for the next run
        current_model_dir="$current_model_dir/finetuning/task_df_eneryield_${config}"
        current_weights_fp="$current_model_dir/finetune_weights"
    done
    # Run the interruption x day runs if the permutation was successful

    if $success; then
        for day_run in "${interruption_x_day_runs[@]}"; do
            echo "Running finetune.py with config: $day_run"
            original_day_run_config="$(pwd)/configs/eneryield/finetune/finetune_$day_run.yaml"
            temp_day_run_config="$temp_config_dir/finetune_$day_run.yaml"

            sed "s|load_from_model_dir:.*|load_from_model_dir: $current_model_dir|" $original_day_run_config > $temp_day_run_config
            sed -i "s|pretrained_weights_fp:.*|pretrained_weights_fp: $current_weights_fp|" $temp_day_run_config
            sed -i "s|strategy:.*|strategy: true|" $temp_day_run_config
            # Log the current model and weights paths for the day run
            echo "Running day run config: $day_run" >> $log_file
            echo "  current_model_dir: $current_model_dir" >> $log_file
            echo "  current_weights_fp: $current_weights_fp" >> $log_file

            python $(pwd)/scripts/finetune.py --config-path=$(pwd)/configs/eneryield/finetune/temp_configs --config-name=finetune_$day_run
            if [ $? -ne 0 ]; then
                echo "Error: finetune.py failed for day run config $day_run."
                echo "Day Run Config $day_run: FAILED" >> $log_file
                success=false
                break
            else
                echo "Day Run Config $day_run: SUCCESS" >> $log_file
                completed_in_order["$current_model_dir:$day_run"]=1
            fi

            # Update the paths for the next run
            current_model_dir="$current_model_dir/finetuning/task_df_eneryield_${day_run}"
            current_weights_fp="$current_model_dir/finetune_weights"
        done
    fi
    # If the permutation was successful, run the final config

    # Run the final config after each permutation if all previous runs succeeded
    if $success; then
        echo "Running finetune.py with final config: $final_config"
        original_final_config="$(pwd)/configs/eneryield/finetune/finetune_$final_config.yaml"
        temp_final_config="$temp_config_dir/finetune_$final_config.yaml"

        sed "s|load_from_model_dir:.*|load_from_model_dir: $current_model_dir|" $original_final_config > $temp_final_config
        sed -i "s|pretrained_weights_fp:.*|pretrained_weights_fp: $current_weights_fp|" $temp_final_config

        # Log the current model and weights paths for the final config
        echo "Running final config: $final_config" >> $log_file
        echo "  current_model_dir: $current_model_dir" >> $log_file
        echo "  current_weights_fp: $current_weights_fp" >> $log_file

        python $(pwd)/scripts/finetune.py --config-path=$(pwd)/configs/eneryield/finetune/temp_configs --config-name=finetune_$final_config
        if [ $? -ne 0 ]; then
            echo "Error: finetune.py failed for final config $final_config."
            echo "Final Config $final_config: FAILED" >> $log_file
        else
            echo "Final Config $final_config: SUCCESS" >> $log_file
        fi
    else
        echo "Skipping final config $final_config due to earlier failure." >> $log_file
    fi
done

# Clean up temporary config files
rm -rf $temp_config_dir
echo "Log file created at: $log_file"