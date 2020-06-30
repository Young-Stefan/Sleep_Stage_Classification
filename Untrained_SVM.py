# File: Untrained SVM
# Description: This program allows the user to train and save a new SVM.
# Institution: University of Texas at Austin, Department of Biomedical Engineering
# Developer: Shao-Po (Shawn) Huang
# Team Members: Bryce Carr, Ajay Gadwal, Ethan Muyskens, Christian Schonhoeft

# Date Last Modified: 05/11/20

# This program uses sklearn and joblib.

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Set up SVM
# Parameters can be adjusted using the following documentation:
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# Training data: trainX, trainy
# Validation data: testX, testy

SVMclassifier = SVC(gamma='auto',kernel='rbf')
SVMclassifier.fit(trainX,trainy)
SVC(gamma='auto',kernel='rbf')

# Save trained SVM by providing destination
joblib.dump(SVMclassifier,PATH_SVM)
