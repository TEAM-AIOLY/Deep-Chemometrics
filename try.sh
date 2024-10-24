CONFIG_PATH="./data/dataset/ossl/config/ViT1D/config_batch.json/"
SCRIPT_PATH="./batch_train.py"
CONFIG_COUNT=$(jq '. | length' "$CONFIG_PATH")

for ((i=0; i<CONFIG_COUNT; i++))
do
    echo "Running configuration $i"
    
    # Call the Python script with the current index
    python "$SCRIPT_PATH" --config_path "$CONFIG_PATH" --config_index "$i"
    
    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run configuration $i"
        exit 1  # Exit the loop if any configuration fails
    fi
done

echo "All configurations ran successfully."