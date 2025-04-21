# Fruits and Vegetables Disease Detection 🍎🥦 (As an example)

A deep learning project to classify diseases in fruits and vegetables using CNNs and traditional ML models (SVM, KNN, Random Forest).

**Note**: This project is designed to be adaptable for any CNN-based classification dataset, making it a versatile tool for various image classification tasks.

![Demo](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-orange)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [Contributing](#contributing)

---

## Project Overview
This project aims to:
1. **Detect Diseases**: Classify fruits/vegetables as healthy or diseased (e.g., `Apple_Healthy` vs. `Apple_Rotten`).
2. **Compare Models**: Evaluate CNN performance against traditional ML models (SVM, KNN, Random Forest).

**Key Features**:
- Data preprocessing and augmentation.
- CNN model training with TensorFlow/Keras.
- Traditional ML pipelines with scikit-learn.
- Model accuracy comparison and visualization.
- Confusion matrix and classification report generation.

---

## Dataset
The dataset is downloaded from Kaggle:  
[**Fruit and Vegetable Disease Dataset**](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten?resource=download)

**Structure**:
- Images of fruits and vegetables categorized as `Healthy` or `Diseased`.
- Split into training, validation, and test sets during preprocessing.

---

## Installation
Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SecurDrgorP/Model_Crafter_Project.git
   cd Fruits-and-Vegetables-Disease-Detection
   ```

---

## Usage
To run the project, follow these steps:

1. **Prepare the Dataset**:
   - Ensure the dataset is downloaded and placed in the `data/raw` directory.
   - Run the main script to clean and preprocess the dataset:
     ```bash
     python main.py
     ```

2. **Train the CNN Model**:
   - During training, you will be prompted to choose whether to use the custom checkpoint logic:
     ```
     Do you want to use the custom checkpoint logic? (y/n):
     ```
   - Type `y` to enable saving the model based on the lowest difference between training and validation accuracy and the lowest validation loss.

3. **Evaluate Models**:
   - The pipeline will automatically evaluate both CNN and traditional ML models and save the results in the `results/` directory.

4. **View Results**:
   - Check the classification reports, confusion matrix, and model comparison CSV in the `results/` directory.

---

## Features
- **Custom Checkpoint Logic**: Save the CNN model based on the lowest difference between training and validation accuracy and the lowest validation loss.
- **Traditional ML Models**: Compare CNN performance with SVM, KNN, and Random Forest.
- **Visualization**: Generate confusion matrices and classification reports for better insights.
- **Data Augmentation**: Automatically applies augmentation to improve model generalization.

---

## Results
- **CNN Model**: Achieved high accuracy in detecting diseases in fruits and vegetables.
- **Traditional ML Models**: Performance varies depending on the dataset and preprocessing.
- **Comparison**: Results are saved in `results/f1_comparison.png` for easy analysis.

---

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
