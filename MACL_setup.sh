# Create a virtual environment
python3 -m venv indo_fashion_classifier_venv

# Activate the virtual environment
source ./indo_fashion_classifier_venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf indo_fashion_classifier_venv