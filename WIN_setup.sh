# Create a virtual environment
python -m venv indo_fashion_classifier_venv

# Activate the virtual environment
source ./indo_fashion_classifier_venv/Scripts/activate

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf indo_fashion_classifier_venv