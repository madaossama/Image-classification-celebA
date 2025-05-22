# CelebA Gender & Smile Classification

This repository demonstrates two complementary methods for classifying **gender** and **smile** attributes from the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html):

1. **Feature-based Supervised Learning** (scikit-learn classifiers)
2. **End-to-End Feedforward Neural Network** (PyTorch)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Dependencies](#setup--dependencies)
- [Data Preparation](#data-preparation)
- [1. Supervised Learning Classification](#1-supervised-learning-classification)
  - [Notebook](#notebook)
  - [Workflow](#workflow)
  - [Usage](#usage)
- [2. Feedforward Neural Network (FFNN)](#2-feedforward-neural-network-ffnn)
  - [Notebook](#notebook-1)
  - [Workflow](#workflow-1)
  - [Usage](#usage-1)
- [Results & Analysis](#results--analysis)
- [License](#license)

---

## Project Structure

```
README.md
celeba_image_classification.ipynb   # Feature extraction + scikit-learn
feedforward_neural_network.ipynb     # PyTorch FFNN implementation
landmarks.py                         # Landmark extraction utility
shape_predictor_68_face_landmarks.dat # dlib pre-trained model
celeba_training.zip, celeba_testing.zip# Raw image archives
... other resources
```

## Setup & Dependencies

1. Clone this repository:
   ```bash
   git clone https://github.com/madaossama/Image-classification-celebA.git
   cd celeba-classification
   ```

2. Install Python dependencies (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* For large files (image archives, shape predictor), consider using [Git LFS](https://git-lfs.github.com/).

## Data Preparation

1. Unzip the image archives:
   ```bash
   unzip celeba_training.zip    # creates celeba_training/img/
   unzip celeba_testing.zip     # creates celeba_testing/img/
   ```

2. Place the dlib model file `shape_predictor_68_face_landmarks.dat` in the root.

---

## 1. Supervised Learning Classification

### Notebook
- **File:** `celeba_image_classification.ipynb`

### Workflow
1. **Feature Extraction**: Uses `landmarks.py` with dlib/OpenCV to extract 68 facial landmark coordinates.
2. **Data Flattening**: Reshapes features into 1D arrays for classifier input.
3. **Model Training**: Trains multiple classifiers (SVM, Logistic Regression, Decision Tree, KNN, LDA, QDA, Nearest Centroid) with optional hyperparameter tuning (GridSearchCV).
4. **Evaluation**: Prints training/testing accuracy, classification reports, and confusion matrices for gender and smile tasks.

### Usage
```bash
# Open the notebook in Jupyter or VS Code
jupyter notebook celeba_image_classification.ipynb
``` 

---

## 2. Feedforward Neural Network (FFNN)

### Notebook
- **File:** `feedforward_neural_network.ipynb`

### Workflow
1. **Reproducibility & Device Setup**: Seeds for numpy, torch; auto-detects GPU.
2. **Data Loading & Preprocessing**: Reads `celeba_training/labels.csv`/`celeba_testing/labels.csv`, encodes labels, resizes images to 64×64, normalizes pixel values.
3. **Dataset & DataLoader**: Constructs PyTorch `TensorDataset` and `DataLoader` for train/val/test splits.
4. **Model Definition**: `MultiTaskFFNN` with one shared hidden layer and two output heads (gender, smile).
5. **Training**: Uses `CrossEntropyLoss` and `Adam` optimizer; trains for specified epochs.
6. **Evaluation**: Reports loss and accuracy on validation and test splits.

### Usage
```bash
# Open the FFNN notebook
jupyter notebook feedforward_neural_network.ipynb
``` 

---

## Results & Analysis
- Both approaches achieve competitive accuracy on gender and smile classification.
- The feature-based pipeline is fast and interpretable; the FFNN can learn directly from raw pixels and benefits from GPU acceleration.
- Consult each notebook’s final cells for detailed metrics and confusion matrices.

---

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.