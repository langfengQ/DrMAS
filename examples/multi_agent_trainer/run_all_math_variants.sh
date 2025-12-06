#!/bin/bash

# Script to run all four math agent variants in sequence
set -e  # Exit on any error

echo "Starting sequential execution of all math agent variants..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Working directory: $SCRIPT_DIR"

# Function to run a script with logging
run_script() {
    local script_name="$1"
    local script_path="$SCRIPT_DIR/$script_name"

    echo "=========================================="
    echo "Running $script_name..."
    echo "Started at: $(date)"
    echo "=========================================="

    if [ ! -f "$script_path" ]; then
        echo "Error: Script $script_path not found!"
        return 1
    fi

    # Run the script and capture exit code
    if bash "$script_path" "$@"; then
        echo "=========================================="
        echo "$script_name completed successfully at: $(date)"
        echo "=========================================="
        return 0
    else
        local exit_code=$?
        echo "=========================================="
        echo "ERROR: $script_name failed with exit code $exit_code at: $(date)"
        echo "=========================================="
        return $exit_code
    fi
}

# List of scripts to run in order
scripts=(
    "run_math_group_by_agent_id_True_model_sharing_True.sh"
    "run_math_group_by_agent_id_True_model_sharing_False.sh"
    "run_math_group_by_agent_id_False_model_sharing_True.sh"
    "run_math_group_by_agent_id_False_model_sharing_False.sh"
)

# Run all scripts in sequence
for script in "${scripts[@]}"; do
    if ! run_script "$script"; then
        echo "Script $script failed. Stopping execution."
        exit 1
    fi
    echo ""  # Add blank line between runs
done

echo "=========================================="
echo "All math agent variants completed successfully!"
echo "Finished at: $(date)"
echo "=========================================="
