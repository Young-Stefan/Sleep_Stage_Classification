# File: Untrained kNN
# Description: This program allows the user to train and save a new kNN.
# Institution: University of Texas at Austin, Department of Biomedical Engineering
# Developer: Shao-Po (Shawn) Huang
# Team Members: Bryce Carr, Ajay Gadwal, Ethan Muyskens, Christian Schonhoeft

# Date Last Modified: 05/11/20

# This program uses sklearn and joblib.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Set up kNN
# Parameters can be adjusted using the following documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# Training data: trainX, trainy
# Validation data: testX, testy

kNNclassifier = KNeighborsClassifier(n_neighbors=9)
kNNclassifier.fit(trainX, trainy)

# Save trained kNN by providing destination
joblib.dump(kNNclassifier,PATH_kNN)
