# federated_learning
Introduction: 
This repository contains a Federated Learning-based approach to predicting diseases from symptom data. The model utilizes TensorFlow Federated (TFF) to train a decentralized neural network across multiple simulated clients while maintaining data privacy. Additionally, it includes comparisons with traditional machine learning algorithms such as Random Forest, Decision Tree, K-Nearest Neighbors, and Naive Bayes to highlight the advantages of Federated Learning.

Key Features:
Federated Learning with TensorFlow Federated (TFF):

Implements a federated averaging algorithm to train a deep neural network.
Simulates distributed clients to demonstrate decentralized learning.
Traditional Machine Learning Algorithms:

Includes Random Forest, Decision Tree, K-Nearest Neighbors, and Naive Bayes classifiers for comparison.
Dimensionality Reduction:

Applies Principal Component Analysis (PCA) to reduce the feature space and improve computational efficiency.
Metrics for Evaluation:

Accuracy, F1 Score, Precision, Sensitivity, Specificity, Geometric Mean, and Matthews Correlation Coefficient (MCC).
Detailed Confusion Matrices and Classification Reports for performance analysis.
Interactive Symptom Input:

Allows users to input symptoms interactively and predict the disease using the trained model.
Comprehensive Visualizations:

Bar charts comparing the performance of classifiers.
Heatmaps for confusion matrices.
Technologies Used
Programming Language: Python
Libraries:
TensorFlow and TensorFlow Federated (TFF)
Scikit-learn
Imbalanced-learn
Matplotlib and Seaborn for visualizations
Pandas and NumPy for data processing
Model Architecture
Federated Learning Neural Network
Input Layer: Matches the number of PCA components.
Hidden Layers:
Dense layers with ReLU activation.
Dropout layers to prevent overfitting.
Output Layer: Softmax activation to predict disease probabilities.
Loss Function: Sparse Categorical Crossentropy.
Optimizers:
SGD with Momentum for clients.
Adam Optimizer for the server.
Machine Learning Classifiers
Random Forest
Decision Tree
K-Nearest Neighbors
Naive Bayes
Installation
Prerequisites
Python 3.7 or above
Google Colab (for running the code in an interactive environment)
Dependencies
Install the required Python packages using pip:

bash:- 
Copy code:
pip install tensorflow tensorflow-federated scikit-learn matplotlib seaborn imbalanced-learn
Usage
Step 1: Upload Dataset
Ensure the dataset is a .csv file with symptoms as features and the target disease as the label.
Columns unrelated to symptoms and diseases should be dropped during preprocessing.
Step 2: Training and Evaluation
Run the code to preprocess the dataset.
Train traditional machine learning models and compare their performance.
Train the Federated Learning model using TensorFlow Federated.
Step 3: Interactive Prediction
Input symptom data interactively to predict a disease using the Federated Learning model.
Step 4: Visualize Results
Review the evaluation metrics and performance comparison charts.
Evaluation Metrics: 
The following metrics are calculated for all models:
Accuracy
F1 Score
Precision
Sensitivity (Recall)
Specificity
Geometric Mean
Matthews Correlation Coefficient (MCC)
Confusion Matrix
