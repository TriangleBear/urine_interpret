# Urine Interpretation Project

This project is part of the thesis titled "Colorimetric Analysis of Urine Test Strips Using U-Net and Support Vector Machine" by Buenaventura, Ubando, and Zacarias. The goal of this project is to develop a deep learning model using a UNet architecture for segmenting urine test strips and reagent pads, and to classify the segmented regions based on LAB color features using an SVM classifier.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/TriangleBear/urine_interpret.git
    cd urine_interpret
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Compute Dataset Statistics**:
    ```bash
    python -m Train.main
    ```

2. **Train the UNet Model**:
    ```bash
    python -m Train.main
    ```

3. **Evaluate the Model**:
    ```bash
    python -m Train.main
    ```

## Project Structure

```
urine_interpret/
│
├── Datasets/
│   ├── Test test/
│   │   ├── images/
│   │   ├── labels/
│   │   └── test/
│   └── ...
├── Train/
│   ├── config.py
│   ├── datasets.py
│   ├── losses.py
│   ├── main.py
│   ├── models.py
│   ├── train_unet.py
│   ├── utils.py
│   └── test_pass.py
├── Old/
│   ├── test_modelUnet.py
│   └── TrainUNetSVMrbf.py
├── models/
│   └── ... (saved models)
├── README.md
└── requirements.txt
```

## Training

To train the UNet model, run the following command:
```bash
python -m Train.main
```
This will train the model and save the best model based on validation loss.

## Evaluation

To evaluate the trained model, run the following command:
```bash
python -m Train.main
```
This will evaluate the model on the test dataset and print the accuracy.

## Thesis Reference

For more detailed information about the project, please refer to the thesis document:
**"Colorimetric Analysis of Urine Test Strips Using U-Net and Support Vector Machine"** by Buenaventura, Ubando, and Zacarias.
