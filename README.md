# 🧬 Breast Cancer Detection using CNN

This project uses a Convolutional Neural Network (CNN) to detect breast cancer based on the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to classify tumors as *Benign* or *Malignant* using structured data and image-based prediction.

---

## 📁 Project Structure



BreastCancerDetection/
├── DetectionCode.py         # Main script with CNN model
├── \[Image File]             # Input image for testing (grayscale, e.g. .jpeg)
├── requirements.txt         # (optional) Python packages used

`

---

## 🧪 Dataset Information

- Dataset: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Features: 30 numerical attributes derived from digitized images of a breast mass.
- Classes:
  - *M* = Malignant (1)
  - *B* = Benign (0)

---

## ⚙ Workflow Overview

1. Load data from UCI repository
2. Preprocess:
   - Drop id column
   - Encode labels (M = 1, B = 0)
   - Normalize features
   - Reshape data for CNN (6×5×1)
3. Build CNN model with:
   - Conv2D + ReLU
   - MaxPooling
   - Flatten + Dense layers
4. Train for 10 epochs
5. Evaluate accuracy
6. Load a test image and make prediction

---

## 🧠 Model Architecture

text
Input Shape: (6, 5, 1)
↓
Conv2D(32 filters, 3x3) + ReLU
↓
MaxPooling2D(2x2)
↓
Flatten
↓
Dense(64 units) + ReLU
↓
Dense(1 unit) + Sigmoid
`

---

## 📦 Requirements

Install necessary libraries using:

bash
pip install pandas numpy scikit-learn tensorflow pillow


---

## 🚀 How to Run

### 1. Clone the repository

bash
git clone https://github.com/your-username/BreastCancerDetection.git
cd BreastCancerDetection


### 2. Run the script

bash
python DetectionCode.py


> The script will:
>
> * Train the CNN on the dataset
> * Evaluate model accuracy
> * Load and predict on a sample grayscale image

### 🖼 Image Format for Prediction

* Must be grayscale (L mode)
* Resized to 5x6 pixels
* Example input image path:
  /content/WhatsApp Image 2023-10-24 at 9.39.52 PM.jpeg

---

## 🧾 Sample Output


Epoch 1/10 ...
...
Test Accuracy: 0.96
Prediction: Benign


---

## 📂 Dataset Source

* [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)

---


## 📃 License

This project is licensed under the MIT License.

---
