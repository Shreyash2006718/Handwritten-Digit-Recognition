# Handwritten Digit Recognition — Artificial Neural Network (ANN)

## Objective
A postal service organization wants to automate the recognition of handwritten digits on postal codes. This project builds an Artificial Neural Network (ANN) using TensorFlow/Keras to classify handwritten digits (0-9) from the MNIST dataset.

## Dataset
MNIST Handwritten Digits Dataset (CSV format) (Kaggle):
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

> Note: The dataset is **not** included in this repository. Download it from the Kaggle link above (files are named `mnist_train.csv` and `mnist_test.csv`) and place both in the project root before running the notebook.

## Libraries Used
- pandas
- numpy
- matplotlib
- tensorflow / keras
- scikit-learn (`sklearn`)

## Methodology
1. **Data Understanding** — Loaded the training and testing CSVs, inspected the first five records, identified the 784 pixel columns as input features and `label` as the target variable, reviewed dataset dimensions, and displayed a sample handwritten digit image.
2. **Data Preprocessing** — Checked for missing values, separated features and target, normalized pixel values to the 0-1 range, and one-hot encoded the target labels for the 10 digit classes.
3. **Model Development** — Built and compiled an ANN (see architecture below), then trained it for 10 epochs and generated predictions on the test set.
4. **Model Evaluation** — Evaluated the model using Test Accuracy, a Confusion Matrix, and a Classification Report, and plotted Accuracy vs Epoch and Loss vs Epoch curves.
5. **Conclusion** — Summarized key findings, the importance of hidden layers in ANN, an advantage of Deep Learning over traditional ML, and a limitation of ANN.

## Model Architecture
| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| Input | Input | 784 | — |
| Hidden Layer 1 | Dense | 128 | ReLU |
| Hidden Layer 2 | Dense | 64 | ReLU |
| Output Layer | Dense | 10 | Softmax |

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metric:** Accuracy
- **Epochs:** 10

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 0.9785 (97.85%) |
| Test Loss | ~0.089 |

**Key finding:** The model correctly classified 9,785 of the 10,000 test images. The most common misclassifications, per the confusion matrix, were digit 3 predicted as 5 (16 cases), digit 4 predicted as 9 (14 cases), and digit 8 predicted as 5 (11 cases) — pairs that share visually similar curved strokes. The Accuracy/Loss vs Epoch graphs show mild overfitting: training accuracy and loss kept improving through all 10 epochs (reaching >99% accuracy and ~0.02 loss), while validation accuracy plateaued around 97-98% and validation loss began rising slightly after epoch 3-4.

## Conclusion
This project used an Artificial Neural Network (128 → 64 → 10 neurons, ReLU/Softmax activations) to classify handwritten digits from the MNIST dataset. After normalizing pixel values and one-hot encoding the labels, the model was trained for 10 epochs using the Adam optimizer and categorical crossentropy loss, achieving a final test accuracy of 97.85% and a test loss of approximately 0.089. The confusion matrix showed the most common errors involved visually similar digit pairs, such as 3/5 and 4/9.

Hidden layers are essential in an ANN because they allow the network to learn increasingly abstract, non-linear combinations of the raw pixel inputs — early layers can capture simple stroke and edge patterns, while later layers combine these into more digit-like representations, something a single-layer model could not achieve. A key advantage of Deep Learning over traditional Machine Learning is that it can learn relevant features directly from raw pixel data without manual feature engineering, which is especially valuable for high-dimensional inputs like images.

A key limitation observed here is overfitting: training accuracy and loss kept improving throughout training while validation performance plateaued and validation loss began rising, indicating the network was starting to memorize the training data rather than continuing to generalize.

## How to Run
```bash
pip install pandas numpy matplotlib tensorflow scikit-learn jupyter
jupyter notebook Assignment-8.ipynb
```
