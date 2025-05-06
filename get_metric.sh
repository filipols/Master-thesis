#!/bin/bash
METRIC_NAME="precision"
# Path to config
CONFIG_PATH="$(pwd)/configs/eneryield_final/inference"
CONFIG_NAME="inference_get_metric"
CONFIG_FILE="${CONFIG_PATH}/${CONFIG_NAME}.yaml"

# Output CSV file
OUTPUT_FILE="precision_recall_eneryield_final_interruption_3_day.csv"

# Initialize arrays to store row values
row1=()
row2=()

# Loop over thresholds from 0.01 to 0.99
for i in {1..19}; do
    threshold=$(LC_NUMERIC=C printf "%.2f" "$(echo "$i * 0.05" | bc -l)")

    echo "Setting threshold to $threshold"

    # Use Python to update the YAML
    python - <<EOF
import yaml

config_path = "$CONFIG_FILE"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["threshold"] = float("$threshold")  # Modify this if threshold is nested

with open(config_path, "w") as f:
    yaml.dump(config, f)
EOF

    # Run inference
    echo "Running inference TUNING with threshold=$threshold..."
    output=$(python inference_tuning.py --config-path="$CONFIG_PATH" --config-name="$CONFIG_NAME")

    # Get second to last and last line
    second_last_line=$(echo "$output" | tail -n 2 | head -n 1)
    last_line=$(echo "$output" | tail -n 1)

    # Extract last 6 characters and trim
    val1=$(echo "$second_last_line" | tail -c 7 | xargs)
    val2=$(echo "$last_line" | tail -c 7 | xargs)

    # Append to arrays
    row1+=("$val1")
    row2+=("$val2")
done

# Write to CSV file
{
    IFS=','; echo "${row1[*]}"
    IFS=','; echo "${row2[*]}"
} > "$OUTPUT_FILE"

echo "Finished. Results  saved to $OUTPUT_FILE"