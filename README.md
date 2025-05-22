# CelebA Image Classification

This project demonstrates image classification on the CelebA dataset using facial landmarks and various machine learning classifiers. The main tasks are gender and smile classification based on features extracted from facial images.

## Features
- Feature extraction using dlib and OpenCV
- Multiple classifiers: SVM, Logistic Regression, Decision Tree, KNN, LDA, QDA, Nearest Centroid
- Hyperparameter tuning with GridSearchCV
- Evaluation with accuracy, classification report, and confusion matrix

## Files
- `celeba_image_classification.ipynb`: Main notebook for data loading, feature extraction, model training, and evaluation
- `landmarks.py`: Helper module for extracting facial landmarks and features
- `celeba_training.zip` / `celeba_testing.zip`: Training and testing image datasets
- `shape_predictor_68_face_landmarks.dat`: Pre-trained dlib model for facial landmark detection

## Usage
1. **Extract the datasets**: Unzip `celeba_training.zip` and `celeba_testing.zip` in the project directory.
2. **Run the notebook**: Open `celeba_image_classification.ipynb` in VS Code or Jupyter and run all cells.
3. **Results**: The notebook will print training/testing accuracy, classification reports, and confusion matrices for each classifier and task.

## Requirements
- Python 3.7+
- numpy
- scikit-learn
- dlib
- opencv-python

Install dependencies with:
```bash
pip install numpy scikit-learn dlib opencv-python
```

## Notes
- The `landmarks.py` script must be present in the same directory as the notebook.
- The dlib shape predictor file (`shape_predictor_68_face_landmarks.dat`) is required for feature extraction.


## Output Discussion & Analysis

After running the notebook, you will see printed outputs for each classifier and task, including:
- **Training Accuracy**: How well the model fits the training data.
- **Testing Accuracy**: Generalization performance on unseen data.
- **Classification Report**: Precision, recall, f1-score, and support for each class.
- **Confusion Matrix**: Breakdown of true/false positives/negatives.

### Example Results
Below is a sample output for SVM (results will vary depending on random state and data split):

```
Classifier: SVM
Best Parameters: {'C': 0.1, 'kernel': 'linear'}
Training Accuracy: 0.95
Testing Accuracy: 0.92
Classification Report:
              precision    recall  f1-score   support
           0       0.93      0.91      0.92       500
           1       0.91      0.93      0.92       500
    accuracy                           0.92      1000
   macro avg       0.92      0.92      0.92      1000
weighted avg       0.92      0.92      0.92      1000
Confusion Matrix:
[[455  45]
 [ 36 464]]
```

### Analysis
- **High training accuracy** with slightly lower testing accuracy may indicate mild overfitting, but good generalization.
- **Precision and recall** are balanced, showing the model is not biased toward one class.
- **Confusion matrix** helps identify if the model is confusing certain classes (e.g., false positives/negatives).
- **Comparing classifiers**: SVM and Logistic Regression often perform best for this feature set, while tree-based models may overfit if not tuned.

You can use these metrics to select the best model for your application and further tune hyperparameters or features for improved results.

## License
This project is for educational purposes.
