CONFIG_PATH="./data/dataset/ossl/config/ViT1D/config_batch.json"
SCRIPT_PATH="./batch_train.py"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found at $CONFIG_PATH"
    exit 1
fi
CONFIG_COUNT=$(jq '. | length' "$CONFIG_PATH")



for ((i=0; i<1; i++)) #CONFIG_COUNT
do
    echo "Running configuration $i"
    
    # Call the Python script with the current index
    python3 "$SCRIPT_PATH" --config_path "$CONFIG_PATH" --config_index "$i"
    
    # Check if the Python script ran successfully
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run configuration $i"
        exit 1  # Exit the loop if any configuration fails
    fi
done

echo "All configurations ran successfully."