# Activate the virtual environment
source ./indo_fashion_classifier_venv/Scripts/activate

# Run the program
python ./src/cnn_fashion.py # --train_subset 8000 --val_subset 2000 --test_subset 2000

# deactivate
deactivate