# Activate the virtual environment
source ./indo_fashion_classifier_venv/bin/activate

# Run the program
python3 ./src/cnn_fashion.py # --train_subset 8000 --val_subset 2000 --test_subset 2000

# deactivate
deactivate