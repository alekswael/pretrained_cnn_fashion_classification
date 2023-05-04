<!-- PROJECT LOGO -->
<br />
  <h1 align="center">Visual Analytics Portfolio</h1> 
  <h2 align="center">Assignment 3: Using pretrained CNNs for image classification</h2> 
  <h3 align="center">Cultural Data Science, 2023</h3> 
  <p align="center">
  Auther: Aleksander Moeslund Wael <br>
  Student no. 202005192
  </p>
</p>

## About the project
This repo contains code for finetuning a pretrained convolutional neural network, and conducting a 15-label image classification task.

### Data
The data used in the project is the [Indo fashion dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) available on Kaggle. The dataset consists of ~106K images displaying 15 unique cloth categories. The data is pre-split into a training, test and validation set.

### Model
The `EfficientNetB0` model was used, being a light-weight alternative to other pretrained CNNs (like VGG16). The model has approx 5.3M parameters and is trained on the ImageNet dataset. [(source)](https://arxiv.org/abs/1905.11946).

### Pipeline
The `cnn_fashion.py` script in the `src` folder contains the main code pipeline. The structure is as follows:
1. Import and preprocess images and metadata
2. Augment image data
3. Load the `EfficientNetB0` model
4. Train the model on the Indo fashion training set.
5. Save the learning curve plots from training.
6. Predict labels of the test set.
7. Print and save a classification report.

## Requirements

The code is tested on Python 3.11.2. Futhermore, a bash-compatible terminal is required for running shell scripts (such as Git for Windows).

## Usage

The repo was setup to work with Windows (the WIN_ files), MacOS and Linux (the MACL_ files).

### 1. Clone repository to desired directory

```bash
git clone https://github.com/AU-CDS/assignment3-pretrained-cnns-alekswael
cd assignment3-pretrained-cnns-alekswael
```
### 2. Run setup script 
**NOTE:** Depending on your OS, run either `WIN_setup.sh` or `MACL_setup.sh`.

The setup script does the following:
1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the correct versions of the packages required
5. Deactivates the virtual environment

```bash
bash WIN_setup.sh
```

### 3. Run pipeline
**NOTE:** Depending on your OS, run either `WIN_run.sh` or `MACL_run.sh`.

Run the script in a bash terminal.

The script does the following:
1. Activates the virtual environment
2. Runs `cnn_fashion.py` located in the `src` folder
3. Deactivates the virtual environment
```bash
bash WIN_run.sh
```

## Repository structure
This repository has the following structure:
```
│   MACL_run.sh
│   MACL_setup.sh
│   README.md
│   requirements.txt
│   WIN_run.sh
│   WIN_setup.sh
│
├───images
│   ├───metadata
│   │       test_data.json
│   │       train_data.json
│   │       val_data.json
│   │
│   ├───test
│   │       7500.jpeg
│   │       
│   ├───train
│   │      91166.jpeg
│   │
│   └───val
│           7500.jpeg
│
├───out
└───src
        cnn_fashion.py
```

## Remarks on findings
When using a subset of the data (train, test, val = 8000, 2000, 2000), the model 