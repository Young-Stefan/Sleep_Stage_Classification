# File: Training, Validation, and Testing
# Description: This program processes and formats EEG and EOG data into numpy arrays that can be used by the machine learning algorithms.
# Institution: University of Texas at Austin, Department of Biomedical Engineering
# Developer: Shao-Po (Shawn) Huang
# Team Members: Bryce Carr, Ajay Gadwal, Ethan Muyskens, Christian Schonhoeft

# Date Last Modified: 05/11/20


# This program uses/can use os, pandas, numpy, keras, sklearn, joblib and matplotlib
import os
import pandas as pd
import numpy as np
import joblib
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix

# Load raw data by providing paths to data and label (categories) files
data_directory = PATH_DATA
label_directory = PATH_LABEL


# initialize arrays for storing EEG Fpz-Cz, EEG Pz-Oz, and EOG horizontal data based on sleep stage
wakeEEG_f, s1EEG_f, s2EEG_f, s3EEG_f, s4EEG_f, remEEG_f, \
    wakeEEG_p, s1EEG_p, s2EEG_p, s3EEG_p, s4EEG_p, remEEG_p, wakeEOG, s1EOG, s2EOG, s3EOG, s4EOG, remEOG = \
    ([] for i in range(18))

# store names of label files
labelfiles = []

for labelfile in os.listdir(label_directory):
    labelfiles.append(labelfile)

# initialize counter for iterating through label files
counter = 0

# iterate through data files and sort data based on type of signal and sleep stage
for datafile in os.listdir(data_directory):

    # load data and labels into dataframes and convert to lists
    data = pd.read_csv(data_directory+'\\'+datafile)
    labels = pd.read_csv(label_directory+'\\'+labelfiles[counter])
    df = pd.DataFrame(data)
    dfEEG_f = df['# EEG Fpz-Cz'].values.tolist()
    dfEEG_p = df['EEG Pz-Oz'].values.tolist()
    dfEOG = df['EOG horizontal'].values.tolist()
    lf = pd.DataFrame(labels)
    lf = lf['Sleep Stage'].values.tolist()

    # sort and store EEG Fpz-Cz data
    start = 0
    stop = labels.iloc[0]['Points']

    for m in range(len(lf)+1):
        if stop > len(dfEEG_f):
            break
        for point in range(start,stop):
            if lf[m] == 'Sleep stage W':
                wakeEEG_f.append(dfEEG_f[point])
            elif lf[m] == 'Sleep stage 1':
                s1EEG_f.append(dfEEG_f[point])
            elif lf[m] == 'Sleep stage 2':
                s2EEG_f.append(dfEEG_f[point])
            elif lf[m] == 'Sleep stage 3':
                s3EEG_f.append(dfEEG_f[point])
            elif lf[m] == 'Sleep stage 4':
                s4EEG_f.append(dfEEG_f[point])
            else:
                remEEG_f.append(dfEEG_f[point])
        start = stop
        stop = stop + labels.iloc[m+1]['Points']

    # sort and store EEG Pz-Oz data
    start = 0
    stop = labels.iloc[0]['Points']

    for n in range(len(lf)+1):
        if stop > len(dfEEG_p):
            break
        for point in range(start,stop):
            if lf[n] == 'Sleep stage W':
                wakeEEG_p.append(dfEEG_p[point])
            elif lf[n] == 'Sleep stage 1':
                s1EEG_p.append(dfEEG_p[point])
            elif lf[n] == 'Sleep stage 2':
                s2EEG_p.append(dfEEG_p[point])
            elif lf[n] == 'Sleep stage 3':
                s3EEG_p.append(dfEEG_p[point])
            elif lf[n] == 'Sleep stage 4':
                s4EEG_p.append(dfEEG_p[point])
            else:
                remEEG_p.append(dfEEG_p[point])
        start = stop
        stop = stop + labels.iloc[n+1]['Points']

    # sort and store EOG horizontal data
    start = 0
    stop = labels.iloc[0]['Points']

    for p in range(len(lf)+1):
        if stop > len(dfEEG_p):
            break
        for point in range(start,stop):
            if lf[p] == 'Sleep stage W':
                wakeEOG.append(dfEOG[point])
            elif lf[p] == 'Sleep stage 1':
                s1EOG.append(dfEOG[point])
            elif lf[p] == 'Sleep stage 2':
                s2EOG.append(dfEOG[point])
            elif lf[p] == 'Sleep stage 3':
                s3EOG.append(dfEOG[point])
            elif lf[p] == 'Sleep stage 4':
                s4EOG.append(dfEOG[point])
            else:
                remEOG.append(dfEOG[point])
        start = stop
        stop = stop + labels.iloc[p+1]['Points']
        
    counter += 1

# Normalize data for kNN and SVM only. Do not use for CNN.
# MIN_f is the minimum value in the EEG Fpz-Cz dataset.
# MIN_p is the minimum value in the EEG Pz-Oz dataset.
# MIN_EOG is the minimum value in the EOG horizontal dataset.
# RANGE_f is the range of the EEG Fpz-Cz dataset.
# RANGE_p is the range of the EEG Pz-Oz dataset.
# RANGE_EOG is the range of the EOG horizontal dataset.

wakeEEG_f = [(x-(MIN_f))/RANGE_f for x in wakeEEG_f]
s1EEG_f = [(x-(MIN_f))/RANGE_f for x in s1EEG_f]
s2EEG_f = [(x-(MIN_f))/RANGE_f for x in s2EEG_f]
s3EEG_f = [(x-(MIN_f))/RANGE_f for x in s3EEG_f]
s4EEG_f = [(x-(MIN_f))/RANGE_f for x in s4EEG_f]
remEEG_f = [(x-(MIN_f))/RANGE_f for x in remEEG_f]
wakeEEG_p = [(x-(MIN_p))/RANGE_p for x in wakeEEG_p]
s1EEG_p = [(x-(MIN_p))/RANGE_p for x in s1EEG_p]
s2EEG_p = [(x-(MIN_p))/RANGE_p for x in s2EEG_p]
s3EEG_p = [(x-(MIN_p))/RANGE_p for x in s3EEG_p] 
s4EEG_p = [(x-(MIN_p))/RANGE_p for x in s4EEG_p]
remEEG_p = [(x-(MIN_p))/RANGE_p for x in remEEG_p]
wakeEOG = [(x-(MIN_EOG))/RANGE_EOG for x in wakeEOG]
s1EOG = [(x-(MIN_EOG))/RANGE_EOG for x in s1EOG]
s2EOG = [(x-(MIN_EOG))/RANGE_EOG for x in s2EOG]
s3EOG = [(x-(MIN_EOG))/RANGE_EOG for x in s3EOG]
s4EOG = [(x-(MIN_EOG))/RANGE_EOG for x in s4EOG]
remEOG = [(x-(MIN_EOG))/RANGE_EOG for x in remEOG]


# Normalize data for CNN only. Do not use for kNN or SVM.
wakeEEG_f = [x/0.00556 for x in wakeEEG_f]
s1EEG_f = [x/0.00556 for x in s1EEG_f]
s2EEG_f = [x/0.00556 for x in s2EEG_f]
s3EEG_f = [x/0.00556 for x in s3EEG_f]
s4EEG_f = [x/0.00556 for x in s4EEG_f]
remEEG_f = [x/0.00556 for x in remEEG_f]
wakeEEG_p = [x/0.0051 for x in wakeEEG_p]
s1EEG_p = [x/0.0051 for x in s1EEG_p]
s2EEG_p = [x/0.0051 for x in s2EEG_p]
s3EEG_p = [x/0.0051 for x in s3EEG_p] 
s4EEG_p = [x/0.0051 for x in s4EEG_p]
remEEG_p = [x/0.0051 for x in remEEG_p]
wakeEOG = [x/0.00591 for x in wakeEOG]
s1EOG = [x/0.00591 for x in s1EOG]
s2EOG = [x/0.00591 for x in s2EOG]
s3EOG = [x/0.00591 for x in s3EOG]
s4EOG = [x/0.00591 for x in s4EOG]
remEOG = [x/0.00591 for x in remEOG]


# initialize arrays for 6-category classification only     
wake,s1,s2,s3,s4,rem = ([] for i in range(6))

# initialize arrays for 3-category classification only
wake, nonrem, rem = ([] for i in range(3))

# combine S1, S2, S3, and S4 sleep stages into non-REM category
nonrem_EEG_f = s1EEG_f + s2EEG_f + s3EEG_f + s4EEG_f
nonrem_EEG_p = s1EEG_p + s2EEG_p + s3EEG_p + s4EEG_p
nonrem_EOG = s1EOG + s2EOG + s3EOG + s4EOG

# Divide data into 30-second intervals or "samples". n is the number of data points in each sample.
# Use nonrem in 3-category classification only
# use S1, S2, S3, S4 in 6-category classification only

wakeEEG_f = [wakeEEG_f[k * n:(k + 1) * n] for k in range((len(wakeEEG_f) + n - 1) // n)]
nonrem_EEG_f = [nonrem_EEG_f[k * n:(k + 1) * n] for k in range((len(nonrem_EEG_f) + n - 1) // n)]
s1EEG_f = [s1EEG_f[k * n:(k + 1) * n] for k in range((len(s1EEG_f) + n - 1) // n)]
s2EEG_f = [s2EEG_f[k * n:(k + 1) * n] for k in range((len(s2EEG_f) + n - 1) // n)]
s3EEG_f = [s3EEG_f[k * n:(k + 1) * n] for k in range((len(s3EEG_f) + n - 1) // n)]
s4EEG_f = [s4EEG_f[k * n:(k + 1) * n] for k in range((len(s4EEG_f) + n - 1) // n)]
remEEG_f = [remEEG_f[k * n:(k + 1) * n] for k in range((len(remEEG_f) + n - 1) // n)]

wakeEEG_p = [wakeEEG_p[k * n:(k + 1) * n] for k in range((len(wakeEEG_p) + n - 1) // n)]
nonrem_EEG_p = [nonrem_EEG_p[k * n:(k + 1) * n] for k in range((len(nonrem_EEG_p) + n - 1) // n)]
s1EEG_p = [s1EEG_p[k * n:(k + 1) * n] for k in range((len(s1EEG_p) + n - 1) // n)]
s2EEG_p = [s2EEG_p[k * n:(k + 1) * n] for k in range((len(s2EEG_p) + n - 1) // n)]
s3EEG_p = [s3EEG_p[k * n:(k + 1) * n] for k in range((len(s3EEG_p) + n - 1) // n)]
s4EEG_p = [s4EEG_p[k * n:(k + 1) * n] for k in range((len(s4EEG_p) + n - 1) // n)]
remEEG_p = [remEEG_p[k * n:(k + 1) * n] for k in range((len(remEEG_p) + n - 1) // n)]

wakeEOG = [wakeEOG[k * n:(k + 1) * n] for k in range((len(wakeEOG) + n - 1) // n)]
nonrem_EOG = [nonrem_EOG[k * n:(k + 1) * n] for k in range((len(nonrem_EOG) + n - 1) // n)]
s1EOG = [s1EOG[k * n:(k + 1) * n] for k in range((len(s1EOG) + n - 1) // n)]
s2EOG = [s2EOG[k * n:(k + 1) * n] for k in range((len(s2EOG) + n - 1) // n)]
s3EOG = [s3EOG[k * n:(k + 1) * n] for k in range((len(s3EOG) + n - 1) // n)]
s4EOG = [s4EOG[k * n:(k + 1) * n] for k in range((len(s4EOG) + n - 1) // n)]
remEOG = [remEOG[k * n:(k + 1) * n] for k in range((len(remEOG) + n - 1) // n)]

# Combine samples from each signal for each sleep stage
# Use nonrem in 3-category classification only
# use S1, S2, S3, S4 in 6-category classification only

for i in range(len(wakeEEG_f)):
    wake.append(wakeEEG_f[i]+wakeEEG_p[i]+wakeEOG[i])
    
for i in range(len(s1EEG_f)):
    s1.append(s1EEG_f[i]+s1EEG_p[i]+s1EOG[i])
    
for i in range(len(s2EEG_f)):
    s2.append(s2EEG_f[i]+s2EEG_p[i]+s2EOG[i])

for i in range(len(s3EEG_f)):
    s3.append(s3EEG_f[i]+s3EEG_p[i]+s3EOG[i])

for i in range(len(s4EEG_f)):
    s4.append(s4EEG_f[i]+s4EEG_p[i]+s4EOG[i])

for i in range(len(nonrem_EEG_f)):
    nonrem.append(nonrem_EEG_f[i]+nonrem_EEG_p[i]+nonrem_EOG[i])
    
for i in range(len(remEEG_f)):
    rem.append(remEEG_f[i]+remEEG_p[i]+remEOG[i])

# Combine all sleep data

# For 6-category classification
X = wake+s1+s2+s3+s4+rem

# For 3-category classification
X = wake+nonrem+rem

# Create arrays with labels for each of the samples

# For 6-category classification, 0-5 corresponds to wake, S1, S2, S3, S4, and REM
y = [0]*len(wake)+[1]*len(s1)+[2]*len(s2)+[3]*len(s3)+[4]*len(s4)+[5]*len(rem)

# For 3-category classification, 0-2 corresponds to wake, non-REM, and REM
y = [0]*len(wake)+[1]*len(nonrem)+[2]*len(rem)
___________________________________________________________________________________________________

# **FOR TRAINING AND VALIDATION ONLY**

# Perform LDA fit_transform for kNN and SVM only. Do not use for CNN.
lda = LinearDiscriminantAnalysis()
X = lda.fit_transform(X,y)

# Format data for kNN and SVM only
# The shape of the arrays should be (#_of_samples, #_of_outputs - 1)
X = np.array(X)
y = np.array(y)

# Format data for CNN only
# The shape of the arrays should be (#_of_samples, #_of_features, 1)
# The number of features is the total number of data points in a 30-second sample
X = np.dstack(X)
X = X.transpose()
y = to_categorical(y)

# Separate data into training ("train") and validation ("test") sets.
# Proportion of data used for training is set in "train_size". The rest will be used for validation.
# Random state can also be set. 
trainX, testX, trainy, testy = train_test_split(X, y, train_size=0.80, random_state=42)

# Output results
___________________________________________________________________________________________________

# **FOR TESTING ONLY**

# Format data for kNN and SVM only
# The shape of the arrays should be (#_of_samples, #_of_features)
# The number of features is the total number of data points in a 30-second sample
X = np.array(X)
y = np.array(y)

# Format data for CNN only
# The shape of the arrays should be (#_of_samples, #_of_features, 1)
# The number of features is the total number of data points in a 30-second sample
X = np.dstack(X)
X = X.transpose()
y = to_categorical(y)

# Perform LDA transform (using previously fitted LDA) for kNN and SVM only. Provide path to LDA. Do not use for CNN.
# After this, the shape of the arrays should be (#_of_samples, #_of_outputs - 1)
lda = joblib.load(PATH_LDA)
X = lda.transform(X)

# Load trained model by providing its path.
model = joblib.load(PATH_MODEL)

___________________________________________________________________________________________________

# **DISPLAYING RESULTS**

# For training results, X = trainX and y = trainy
# For validation results, X = testX and y = testy
# For testing results, leave X and y as is

# Output results for kNN and SVM only

# Store predictions
predy = model.predict(X)

# Output mean accuracy, confusion matrix, and classification report (precision, recall, f1 score, accuracy)
print(model.score(X, y))
print(confusion_matrix(y, predy))
print(classification_report(y, predy))

# Output results for CNN only

# Output score = (loss, accuracy)
score = model.evaluate(X, y)
print(score)

# Store and reformat predictions and labels
predy = model.predict(X)
predy = np.argmax(predy,axis=1)
y = np.argmax(y,axis=1)

# Output confusion matrix and classification report (precision, recall, f1 score, accuracy)
print(confusion_matrix(y, predy))
print(classification_report(y, predy))


